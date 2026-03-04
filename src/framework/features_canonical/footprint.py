"""Footprint (Deep Chart) features: stacked imbalances, zero prints, unfinished business, delta heat."""
import polars as pl

from src.framework.data.constants import DELTA_Z_LOOKBACK, ZERO_PRINT_LOOKBACK, EXTREME_AGGRESSION_RATIO


def compute_footprint_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute Deep Chart features from footprint-enriched bars.

    Expects bars with footprint columns from bars.py: stacked_imbalance_count,
    stacked_imbalance_direction, zero_print_count, zero_print_ratio,
    unfinished_high, unfinished_low, max_level_volume, volume_at_high,
    volume_at_low, plus standard: ts_event, open, close, volume,
    buy_volume, sell_volume, bar_duration_ns, high, low.
    """
    if len(bars) == 0:
        return _empty_result()

    df = bars.select([
        "ts_event", "open", "high", "low", "close", "volume",
        "buy_volume", "sell_volume", "bar_duration_ns",
        "stacked_imbalance_count", "stacked_imbalance_direction",
        "zero_print_count", "zero_print_ratio",
        "unfinished_high", "unfinished_low",
        "max_level_volume", "volume_at_high", "volume_at_low",
        "buy_vol_at_high", "sell_vol_at_high",
        "buy_vol_at_low", "sell_vol_at_low",
    ]).sort("ts_event")
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_date"),
    )

    # --- Stacked imbalance features ---
    # Signed strength: count * direction
    df = df.with_columns(
        (pl.col("stacked_imbalance_count").cast(pl.Int32)
         * pl.col("stacked_imbalance_direction").cast(pl.Int32))
        .cast(pl.Float64)
        .alias("stacked_imb_strength"),
    )

    df = df.with_columns(
        pl.col("stacked_imb_strength")
        .rolling_mean(window_size=3, min_samples=1)
        .over("_date")
        .alias("stacked_imb_ma3"),
    )

    # Streak: consecutive bars with same-sign stacked imbalances
    # Positive strength → bullish streak, negative → bearish, zero resets
    sign = pl.when(pl.col("stacked_imb_strength") > 0).then(1) \
             .when(pl.col("stacked_imb_strength") < 0).then(-1) \
             .otherwise(0)
    prev_sign = sign.shift(1).over("_date").fill_null(0)
    # Reset group_id when sign changes
    df = df.with_columns(
        (sign != prev_sign).cum_sum().over("_date").alias("_streak_group"),
        sign.alias("_sign"),
    )
    df = df.with_columns(
        pl.when(pl.col("_sign") != 0)
        .then(pl.int_range(pl.len()).over(["_date", "_streak_group"]) + 1)
        .otherwise(0)
        .cast(pl.Float64)
        .alias("stacked_imb_streak"),
    )

    # --- Zero print features ---
    # Expansion: current zero_print_count > rolling 75th percentile
    df = df.with_columns(
        pl.col("zero_print_count").cast(pl.Float64)
        .rolling_quantile(quantile=0.75, window_size=ZERO_PRINT_LOOKBACK, min_samples=1)
        .over("_date")
        .alias("_zp_p75"),
    )
    df = df.with_columns(
        pl.when(pl.col("zero_print_count").cast(pl.Float64) > pl.col("_zp_p75"))
        .then(1)
        .otherwise(0)
        .cast(pl.Float64)
        .alias("zero_print_expansion"),
    )

    # Direction: sign of (close - open) when zero prints exist
    df = df.with_columns(
        pl.when(pl.col("zero_print_count") > 0)
        .then(
            pl.when(pl.col("close") > pl.col("open")).then(1.0)
            .when(pl.col("close") < pl.col("open")).then(-1.0)
            .otherwise(0.0)
        )
        .otherwise(0.0)
        .alias("zero_print_direction"),
    )

    # --- Unfinished business features ---
    # Bars since last unfinished high/low (NaN before first occurrence)
    df = df.with_columns([
        _bars_since_flag("unfinished_high", group_col="_date").alias("bars_since_unfinished_high"),
        _bars_since_flag("unfinished_low", group_col="_date").alias("bars_since_unfinished_low"),
    ])

    # --- Delta intensity (heat) ---
    # Volume delta: buy - sell
    df = df.with_columns(
        (pl.col("buy_volume").cast(pl.Int64) - pl.col("sell_volume").cast(pl.Int64))
        .cast(pl.Float64)
        .alias("_vol_delta"),
    )

    # Delta per second
    df = df.with_columns(
        (pl.col("_vol_delta") / (pl.col("bar_duration_ns") / 1e9 + 1e-9))
        .alias("delta_per_second"),
    )

    # Z-score of delta_per_second
    df = df.with_columns(
        _zscore_expr("delta_per_second", DELTA_Z_LOOKBACK, group_col="_date")
        .alias("delta_intensity_z"),
    )

    # Heat: absolute z-score — magnitude regardless of direction
    df = df.with_columns(
        pl.col("delta_intensity_z").abs().alias("delta_heat"),
    )

    # --- Volume concentration within bar ---
    df = df.with_columns([
        # How concentrated was volume at a single price level
        (pl.col("max_level_volume").cast(pl.Float64)
         / (pl.col("volume").cast(pl.Float64) + 1.0))
        .alias("max_level_vol_ratio"),
        # Activity at bar extremes (high + low vs total)
        ((pl.col("volume_at_high").cast(pl.Float64)
          + pl.col("volume_at_low").cast(pl.Float64))
         / (pl.col("volume").cast(pl.Float64) + 1.0))
        .alias("high_low_vol_ratio"),
    ])

    # --- Buy/sell aggression at bar extremes ---
    df = df.with_columns([
        # "Who dominated at the high — buyers chasing or sellers defending"
        (pl.col("buy_vol_at_high").cast(pl.Float64)
         / (pl.col("buy_vol_at_high").cast(pl.Float64)
            + pl.col("sell_vol_at_high").cast(pl.Float64) + 1.0))
        .alias("extreme_buy_ratio_high"),

        # "Who dominated at the low — sellers pushing or buyers absorbing"
        (pl.col("buy_vol_at_low").cast(pl.Float64)
         / (pl.col("buy_vol_at_low").cast(pl.Float64)
            + pl.col("sell_vol_at_low").cast(pl.Float64) + 1.0))
        .alias("extreme_buy_ratio_low"),

        # "Sellers aggressive at high = absorption / potential reversal down"
        pl.when(pl.col("sell_vol_at_high").cast(pl.Float64)
                > pl.col("buy_vol_at_high").cast(pl.Float64) * EXTREME_AGGRESSION_RATIO)
        .then(1.0).otherwise(0.0)
        .alias("extreme_aggression_high"),

        # "Buyers aggressive at low = absorption / potential reversal up"
        pl.when(pl.col("buy_vol_at_low").cast(pl.Float64)
                > pl.col("sell_vol_at_low").cast(pl.Float64) * EXTREME_AGGRESSION_RATIO)
        .then(1.0).otherwise(0.0)
        .alias("extreme_aggression_low"),
    ])

    # Pass through footprint columns as features with fp_ prefix to avoid
    # name collisions with bar columns that other modules also pass through
    df = df.with_columns([
        pl.col("zero_print_ratio").cast(pl.Float64).alias("fp_zero_print_ratio"),
        pl.col("unfinished_high").cast(pl.Float64).alias("fp_unfinished_high"),
        pl.col("unfinished_low").cast(pl.Float64).alias("fp_unfinished_low"),
    ])

    # Select output columns
    output_cols = [
        "ts_event",
        "stacked_imb_strength",
        "stacked_imb_ma3",
        "stacked_imb_streak",
        "fp_zero_print_ratio",
        "zero_print_expansion",
        "zero_print_direction",
        "fp_unfinished_high",
        "fp_unfinished_low",
        "bars_since_unfinished_high",
        "bars_since_unfinished_low",
        "delta_per_second",
        "delta_intensity_z",
        "delta_heat",
        "max_level_vol_ratio",
        "high_low_vol_ratio",
        "extreme_buy_ratio_high",
        "extreme_buy_ratio_low",
        "extreme_aggression_high",
        "extreme_aggression_low",
    ]

    return df.select(output_cols)


def _zscore_expr(col_name: str, lookback: int, group_col: str | None = None) -> pl.Expr:
    """Rolling z-score: (x - mean) / std over lookback window."""
    col = pl.col(col_name)
    mean = col.rolling_mean(window_size=lookback, min_samples=2)
    std = col.rolling_std(window_size=lookback, min_samples=2)
    if group_col is not None:
        mean = mean.over(group_col)
        std = std.over(group_col)
    return pl.when(std > 0).then((col - mean) / std).otherwise(0.0)


def _bars_since_flag(col_name: str, group_col: str | None = None) -> pl.Expr:
    """Count bars since the last time col_name was 1.

    Uses cumulative sum as group marker: each time the flag is 1, a new group
    starts. Within each group, row number gives bars since flag.
    Returns null for bars before the first flag ever fires.
    """
    flag = pl.col(col_name).cast(pl.UInt32)
    idx = pl.int_range(pl.len())
    if group_col is not None:
        idx = idx.over(group_col)
    last_flag_idx = (
        pl.when(flag == 1).then(idx).otherwise(None)
        .forward_fill()
    )
    if group_col is not None:
        last_flag_idx = last_flag_idx.over(group_col)
    # Group 0 = before any flag fired -> null (consistent missing-value semantics)
    return (
        pl.when(last_flag_idx.is_not_null())
        .then((idx - last_flag_idx).cast(pl.Float64))
        .otherwise(None)
    )


def _empty_result() -> pl.DataFrame:
    schema = {
        "ts_event": pl.Datetime("ns", "UTC"),
        "stacked_imb_strength": pl.Float64,
        "stacked_imb_ma3": pl.Float64,
        "stacked_imb_streak": pl.Float64,
        "fp_zero_print_ratio": pl.Float64,
        "zero_print_expansion": pl.Float64,
        "zero_print_direction": pl.Float64,
        "fp_unfinished_high": pl.Float64,
        "fp_unfinished_low": pl.Float64,
        "bars_since_unfinished_high": pl.Float64,
        "bars_since_unfinished_low": pl.Float64,
        "delta_per_second": pl.Float64,
        "delta_intensity_z": pl.Float64,
        "delta_heat": pl.Float64,
        "max_level_vol_ratio": pl.Float64,
        "high_low_vol_ratio": pl.Float64,
        "extreme_buy_ratio_high": pl.Float64,
        "extreme_buy_ratio_low": pl.Float64,
        "extreme_aggression_high": pl.Float64,
        "extreme_aggression_low": pl.Float64,
    }
    return pl.DataFrame(schema=schema)
