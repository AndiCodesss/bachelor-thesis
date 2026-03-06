"""Adaptive transaction cost model for NQ futures backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.framework.data.constants import (
    ADAPTIVE_COST_SESSION_MULTIPLIER,
    ADAPTIVE_COST_SPREAD_MULTIPLIER,
    ADAPTIVE_COST_VOL_LOOKBACK,
    ADAPTIVE_COST_VOL_MULTIPLIER,
    ADAPTIVE_COST_VOLUME_DISCOUNT,
    COMMISSION_RT,
    SLIPPAGE_PER_SIDE,
    TICK_VALUE,
)


@dataclass(frozen=True)
class CostModel:
    """Per-trade adaptive cost model. Real slippage varies with spread, volatility, and liquidity."""

    commission_rt: float = COMMISSION_RT
    base_slippage_per_side: float = SLIPPAGE_PER_SIDE
    vol_multiplier: float = ADAPTIVE_COST_VOL_MULTIPLIER
    spread_multiplier: float = ADAPTIVE_COST_SPREAD_MULTIPLIER
    volume_discount: float = ADAPTIVE_COST_VOLUME_DISCOUNT
    session_multiplier: float = ADAPTIVE_COST_SESSION_MULTIPLIER

    def estimate_cost_rt(
        self,
        spread_ticks: float,
        volatility_z: float,
        volume_z: float,
        session_progress: float,
    ) -> float:
        """Estimate round-trip cost for a single trade given market conditions.

        Args:
            spread_ticks: bid-ask spread in ticks (1 tick = 0.25 pts)
            volatility_z: z-score of recent bar volatility vs rolling window
            volume_z: z-score of recent bar volume vs rolling window
            session_progress: fraction of RTH elapsed, 0.0 (open) to 1.0 (close)

        Returns:
            Estimated round-trip cost in dollars per contract.
        """
        # Liquidity U-shape: wider spreads near session open/close
        session_penalty = 4.0 * (session_progress - 0.5) ** 2  # 0 at mid, 1 at edges
        slippage_per_side = (
            self.base_slippage_per_side
            + self.vol_multiplier * max(0.0, volatility_z) * TICK_VALUE
            + self.spread_multiplier * max(0.0, spread_ticks - 1.0) * TICK_VALUE
            - self.volume_discount * max(0.0, volume_z) * TICK_VALUE
            + self.session_multiplier * session_penalty * TICK_VALUE
        )
        # Floor at half base slippage — can't get cheaper than that
        slippage_per_side = max(self.base_slippage_per_side * 0.5, slippage_per_side)
        total = self.commission_rt + 2.0 * slippage_per_side
        # Absolute floor: at minimum you pay the commission
        return max(self.commission_rt, total)

    @staticmethod
    def flat() -> CostModel:
        """Return a cost model with defaults that yield ~$14.50 at neutral inputs."""
        return CostModel()


def compute_adaptive_costs(
    trades: pl.DataFrame,
    bars: pl.DataFrame,
    cost_model: CostModel | None = None,
) -> pl.DataFrame:
    """Attach per-trade adaptive cost estimates to a trades DataFrame.

    For each trade, finds the nearest bar (by entry_time) via asof join and
    computes spread, volatility z-score, volume z-score, and session progress
    to feed into the cost model.

    Args:
        trades: DataFrame with TRADE_SCHEMA columns (entry_time, exit_time, etc.)
        bars: DataFrame with ts_event, close, volume, and optionally ask_price/bid_price columns.
        cost_model: CostModel instance; defaults to CostModel.flat().

    Returns:
        trades DataFrame with added column "adaptive_cost_rt" (Float64).
    """
    if cost_model is None:
        cost_model = CostModel.flat()

    if len(trades) == 0:
        return trades.with_columns(pl.lit(0.0).alias("adaptive_cost_rt"))

    # Normalize join keys to the framework's canonical timestamp precision.
    # This keeps adaptive-cost attachment robust even if a caller provides
    # trades with microsecond datetimes while bars use nanoseconds.
    join_dtype = pl.Datetime("ns", "UTC")
    trades = trades.with_columns([
        pl.col("entry_time").cast(join_dtype).alias("entry_time"),
        pl.col("entry_time").cast(join_dtype).alias("_entry_time_join"),
        pl.col("exit_time").cast(join_dtype).alias("exit_time"),
    ])

    # Prepare bars with required derived columns
    bars_sorted = bars.with_columns(
        pl.col("ts_event").cast(join_dtype)
    ).sort("ts_event")

    # Spread in ticks: if ask_price/bid_price available, compute; otherwise assume 1 tick
    if "ask_price" in bars_sorted.columns and "bid_price" in bars_sorted.columns:
        bars_sorted = bars_sorted.with_columns(
            ((pl.col("ask_price") - pl.col("bid_price")) / 0.25).alias("_spread_ticks")
        )
    else:
        bars_sorted = bars_sorted.with_columns(
            pl.lit(1.0).alias("_spread_ticks")
        )

    # Ensure volume column exists
    if "volume" not in bars_sorted.columns:
        bars_sorted = bars_sorted.with_columns(pl.lit(1.0).alias("volume"))

    # Bar returns for volatility computation
    bars_sorted = bars_sorted.with_columns(
        pl.col("close").pct_change().alias("_bar_return")
    )

    # Rolling volatility z-score (std of returns, z-scored over lookback window)
    bars_sorted = bars_sorted.with_columns(
        pl.col("_bar_return")
        .rolling_std(ADAPTIVE_COST_VOL_LOOKBACK, min_samples=2)
        .alias("_rolling_vol")
    ).with_columns([
        pl.col("_rolling_vol")
        .rolling_mean(ADAPTIVE_COST_VOL_LOOKBACK, min_samples=2)
        .alias("_vol_mean"),
        pl.col("_rolling_vol")
        .rolling_std(ADAPTIVE_COST_VOL_LOOKBACK, min_samples=2)
        .alias("_vol_std"),
    ]).with_columns(
        pl.when(pl.col("_vol_std") > 0)
        .then((pl.col("_rolling_vol") - pl.col("_vol_mean")) / pl.col("_vol_std"))
        .otherwise(0.0)
        .alias("_volatility_z")
    )

    # Rolling volume z-score
    bars_sorted = bars_sorted.with_columns([
        pl.col("volume")
        .rolling_mean(ADAPTIVE_COST_VOL_LOOKBACK, min_samples=2)
        .alias("_vol_rolling_mean"),
        pl.col("volume")
        .rolling_std(ADAPTIVE_COST_VOL_LOOKBACK, min_samples=2)
        .alias("_vol_rolling_std"),
    ]).with_columns(
        pl.when(pl.col("_vol_rolling_std") > 0)
        .then((pl.col("volume") - pl.col("_vol_rolling_mean")) / pl.col("_vol_rolling_std"))
        .otherwise(0.0)
        .alias("_volume_z")
    )

    # Session progress: fraction of RTH elapsed (convert UTC → Eastern first)
    bars_sorted = bars_sorted.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.hour().alias("_hour"),
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.minute().alias("_minute"),
    ).with_columns(
        (pl.col("_hour") * 60 + pl.col("_minute")).alias("_minute_of_day"),
    ).with_columns(
        # RTH bars use [0, 1] progress. Non-RTH bars use neutral midpoint (0.5).
        pl.when((pl.col("_minute_of_day") >= 570) & (pl.col("_minute_of_day") <= 960))
        .then((pl.col("_minute_of_day") - 570) / 390.0)
        .otherwise(0.5)
        .alias("_session_progress")
    )

    # Select only the columns needed for the asof join
    bar_features = bars_sorted.select([
        pl.col("ts_event").alias("_ts_event_join"),
        pl.col("_spread_ticks"),
        pl.col("_volatility_z"),
        pl.col("_volume_z"),
        pl.col("_session_progress"),
    ])

    # Asof join: for each trade entry_time, find the nearest preceding bar
    trades_sorted = trades.sort("_entry_time_join")
    joined = trades_sorted.join_asof(
        bar_features,
        left_on="_entry_time_join",
        right_on="_ts_event_join",
        strategy="backward",
    )

    # Fill nulls from unmatched joins (e.g., trades before first bar)
    joined = joined.with_columns([
        pl.col("_spread_ticks").fill_null(1.0),
        pl.col("_volatility_z").fill_null(0.0),
        pl.col("_volume_z").fill_null(0.0),
        pl.col("_session_progress").fill_null(0.5),
    ])

    # Compute adaptive cost per trade via map_elements on struct
    cost_model_ref = cost_model
    joined = joined.with_columns(
        pl.struct([
            "_spread_ticks",
            "_volatility_z",
            "_volume_z",
            "_session_progress",
        ])
        .map_elements(
            lambda row: cost_model_ref.estimate_cost_rt(
                spread_ticks=row["_spread_ticks"],
                volatility_z=row["_volatility_z"],
                volume_z=row["_volume_z"],
                session_progress=row["_session_progress"],
            ),
            return_dtype=pl.Float64,
        )
        .alias("adaptive_cost_rt")
    )

    # Drop temporary columns
    temp_cols = [c for c in joined.columns if c.startswith("_")]
    result = joined.drop(temp_cols)

    # Restore original sort order by re-sorting on entry_time
    return result.sort("entry_time")
