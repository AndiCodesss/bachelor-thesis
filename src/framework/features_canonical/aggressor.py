"""Aggressor features: CVD, trade intensity, large lot activity."""
import polars as pl

from src.framework.data.constants import (
    TICK_SIZE,
    CVD_ALIGNMENT_WINDOW,
    CVD_ALIGNMENT_MIN_SAMPLES,
    CVD_DIVERGENCE_LOOKBACK,
)


def compute_aggressor_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute aggressor-side features from pre-aggregated bar data.

    Expects bars with columns: ts_event, close, buy_volume, sell_volume,
    trade_count, large_buy_volume, large_sell_volume, volume.
    """
    if len(bars) == 0:
        return pl.DataFrame(schema={
            "ts_event": pl.Datetime("ns", "UTC"),
            "volume_delta": pl.Int64,
            "cvd": pl.Int64,
            "cvd_divergence_6": pl.Float64,
            "cvd_slope_3": pl.Float64,
            "cvd_slope_6": pl.Float64,
            "cvd_slope_12": pl.Float64,
            "cvd_accel_3": pl.Float64,
            "cvd_accel_6": pl.Float64,
            "cvd_price_alignment": pl.Float64,
            "buy_sell_ratio": pl.Float64,
            "trade_intensity": pl.Float64,
            "relative_intensity": pl.Float64,
            "large_lot_fraction": pl.Float64,
            "large_lot_imbalance": pl.Float64,
        })

    # Compute total volumes from bar columns
    bars = bars.select([
        "ts_event", "close", "buy_volume", "sell_volume",
        "trade_count", "large_buy_volume", "large_sell_volume", "volume",
    ])

    bars = bars.with_columns([
        (pl.col("buy_volume").cast(pl.UInt64) + pl.col("sell_volume").cast(pl.UInt64))
        .alias("total_volume"),
        (pl.col("large_buy_volume").cast(pl.UInt64) + pl.col("large_sell_volume").cast(pl.UInt64))
        .alias("large_total_volume"),
    ])

    bars = bars.sort("ts_event")

    # Extract date for daily CVD reset + bar index for day-boundary nulling
    bars = bars.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_date"),
    )
    bars = bars.with_columns(
        pl.int_range(pl.len()).over("_date").alias("_bar_idx_in_day"),
    )

    idx = pl.col("_bar_idx_in_day")

    # Volume delta: net aggressive flow per bar
    bars = bars.with_columns(
        (pl.col("buy_volume").cast(pl.Int64) - pl.col("sell_volume").cast(pl.Int64))
        .alias("volume_delta"),
    )

    # CVD: cumulative volume delta, reset daily
    bars = bars.with_columns(
        pl.col("volume_delta").cum_sum().over("_date").alias("cvd"),
    )

    # Price return for divergence check (use .over("_date") to prevent cross-day shift)
    bars = bars.with_columns(
        pl.when(idx >= 1).then(
            (pl.col("close") - pl.col("close").shift(1).over("_date"))
            / (pl.col("close").shift(1).over("_date") + TICK_SIZE)
        ).otherwise(None)
        .alias("_price_return"),
    )

    # CVD divergence: price and CVD moving opposite directions over N bars
    div_lb = CVD_DIVERGENCE_LOOKBACK
    bars = bars.with_columns([
        pl.when(idx >= div_lb).then(
            pl.col("close") - pl.col("close").shift(div_lb).over("_date")
        ).otherwise(None).alias("_price_chg_6"),
        pl.when(idx >= div_lb).then(
            pl.col("cvd") - pl.col("cvd").shift(div_lb).over("_date")
        ).otherwise(None).alias("_cvd_chg_6"),
    ])

    bars = bars.with_columns(
        pl.when(
            (pl.col("_price_chg_6").is_not_null()) & (pl.col("_cvd_chg_6").is_not_null())
            & (pl.col("_price_chg_6") * pl.col("_cvd_chg_6") < 0)
        )
        .then(1)
        .otherwise(0)
        .cast(pl.Float64)
        .alias("cvd_divergence_6"),
    )

    # CVD slope: average CVD change per bar over N bars — buying/selling momentum
    # Use .over("_date") shifts to prevent cross-day contamination
    # Guard with idx >= N so slope uses exactly N-bar deltas.
    cvd_f = pl.col("cvd").cast(pl.Float64)
    bars = bars.with_columns([
        pl.when(idx >= 3).then((cvd_f - cvd_f.shift(3).over("_date")) / 3.0).otherwise(None)
        .alias("cvd_slope_3"),
        pl.when(idx >= 6).then((cvd_f - cvd_f.shift(6).over("_date")) / 6.0).otherwise(None)
        .alias("cvd_slope_6"),
        pl.when(idx >= 12).then((cvd_f - cvd_f.shift(12).over("_date")) / 12.0).otherwise(None)
        .alias("cvd_slope_12"),
    ])

    # CVD acceleration: is buying/selling pressure speeding up or slowing down?
    bars = bars.with_columns([
        pl.when(idx >= 6)
        .then(pl.col("cvd_slope_3") - pl.col("cvd_slope_3").shift(3).over("_date")).otherwise(None)
        .alias("cvd_accel_3"),
        pl.when(idx >= 12)
        .then(pl.col("cvd_slope_6") - pl.col("cvd_slope_6").shift(6).over("_date")).otherwise(None)
        .alias("cvd_accel_6"),
    ])

    # CVD-price alignment: rolling correlation of close returns vs CVD changes
    # +1 = healthy trend, -1 = divergence/exhaustion, 0 = decoupled
    # Use .over("_date") shift to prevent cross-day contamination
    bars = bars.with_columns(
        pl.when(idx >= 1).then(cvd_f - cvd_f.shift(1).over("_date")).otherwise(None)
        .alias("_cvd_change"),
    )
    bars = bars.with_columns(
        pl.when(idx >= CVD_ALIGNMENT_WINDOW)
        .then(
            pl.rolling_corr(
                pl.col("_price_return"),
                pl.col("_cvd_change"),
                window_size=CVD_ALIGNMENT_WINDOW,
                min_samples=CVD_ALIGNMENT_MIN_SAMPLES,
            )
        ).otherwise(None)
        .alias("cvd_price_alignment"),
    )

    # Buy/sell ratio: fraction of volume from aggressive buyers
    bars = bars.with_columns(
        (pl.col("buy_volume").cast(pl.Float64) / (pl.col("total_volume") + 1).cast(pl.Float64))
        .alias("buy_sell_ratio"),
    )

    # Trade intensity: number of trades in bar (raw count)
    bars = bars.with_columns(
        pl.col("trade_count").cast(pl.Float64).alias("trade_intensity"),
    )

    # Relative intensity: current vs 24-bar average -- surges precede breakouts
    intensity_ma = pl.col("trade_intensity").rolling_mean(window_size=24, min_samples=1).over("_date")
    bars = bars.with_columns(
        pl.when(intensity_ma > 1e-12)
        .then(pl.col("trade_intensity") / intensity_ma)
        .otherwise(0.0)
        .alias("relative_intensity"),
    )

    # Large lot fraction: institutional participation rate
    bars = bars.with_columns(
        (pl.col("large_total_volume").cast(pl.Float64) / (pl.col("total_volume").cast(pl.Float64) + 1))
        .alias("large_lot_fraction"),
    )

    # Large lot imbalance: net institutional direction
    bars = bars.with_columns(
        ((pl.col("large_buy_volume").cast(pl.Int64) - pl.col("large_sell_volume").cast(pl.Int64)).cast(pl.Float64)
         / (pl.col("large_total_volume").cast(pl.Float64) + 1))
        .alias("large_lot_imbalance"),
    )

    # Drop intermediate columns
    bars = bars.drop([
        "_date", "_bar_idx_in_day", "_price_return", "_price_chg_6", "_cvd_chg_6",
        "_cvd_change", "close", "buy_volume", "sell_volume", "total_volume",
        "trade_count", "large_buy_volume", "large_sell_volume", "large_total_volume",
        "volume",
    ])

    return bars
