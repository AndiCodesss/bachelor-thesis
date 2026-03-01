"""Scalping features for the NQ Order Flow Scalping Strategy.

Features designed for volume/tick bars where bar_duration varies:
- Delta divergence (reversal signal)
- Effort vs result (absorption detection)
- Intensity z-score (breakout trigger)
"""
import polars as pl


# Rolling windows for scalping features
DIVERGENCE_WINDOW = 10
INTENSITY_ZSCORE_WINDOW = 20
INTENSITY_SPIKE_THRESHOLD = 2.5
ABSORPTION_VOL_QUANTILE = 0.90
ABSORPTION_RANGE_QUANTILE = 0.20
ABSORPTION_LOOKBACK = 24


def compute_scalping_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute scalping-specific features from pre-aggregated bars.

    Expects bars with: ts_event, open, high, low, close, volume,
    buy_volume, sell_volume, trade_count, bar_duration_ns.
    """
    if len(bars) == 0:
        return pl.DataFrame(schema={
            "ts_event": pl.Datetime("ns", "UTC"),
            "delta_divergence_bull": pl.Float64,
            "delta_divergence_bear": pl.Float64,
            "effort_vs_result": pl.Float64,
            "absorption_signal": pl.Float64,
            "intensity": pl.Float64,
            "intensity_z": pl.Float64,
            "tape_speed_spike": pl.Float64,
        })

    result = bars.select(["ts_event", "open", "high", "low", "close", "volume",
                          "buy_volume", "sell_volume", "trade_count", "bar_duration_ns"]).sort("ts_event")

    # --- Volume delta ---
    result = result.with_columns(
        (pl.col("buy_volume").cast(pl.Int64) - pl.col("sell_volume").cast(pl.Int64))
        .alias("_volume_delta"),
    )

    # --- Delta Divergence (reversal signal) ---
    # Bullish: price makes lower low, but delta makes higher low
    result = result.with_columns([
        pl.col("close").rolling_min(window_size=DIVERGENCE_WINDOW, min_samples=1).alias("_min_price"),
        pl.col("_volume_delta").rolling_min(window_size=DIVERGENCE_WINDOW, min_samples=1).alias("_min_delta"),
        pl.col("close").rolling_max(window_size=DIVERGENCE_WINDOW, min_samples=1).alias("_max_price"),
        pl.col("_volume_delta").rolling_max(window_size=DIVERGENCE_WINDOW, min_samples=1).alias("_max_delta"),
    ])

    result = result.with_columns([
        # Bullish divergence: new price low but delta not confirming
        pl.when(
            (pl.col("close") < pl.col("_min_price").shift(1)) &
            (pl.col("_volume_delta") > pl.col("_min_delta").shift(1))
        ).then(1.0).otherwise(0.0).alias("delta_divergence_bull"),

        # Bearish divergence: new price high but delta not confirming
        pl.when(
            (pl.col("close") > pl.col("_max_price").shift(1)) &
            (pl.col("_volume_delta") < pl.col("_max_delta").shift(1))
        ).then(1.0).otherwise(0.0).alias("delta_divergence_bear"),
    ])

    # --- Effort vs Result (absorption detection) ---
    # High volume trapped in small range = liquidity absorption
    result = result.with_columns(
        (pl.col("volume").cast(pl.Float64) / (pl.col("high") - pl.col("low") + 1e-9))
        .alias("effort_vs_result"),
    )

    # Absorption signal: volume > rolling 90th percentile AND
    # range < rolling 20th percentile (using only prior bars).
    result = result.with_columns(
        (pl.col("high") - pl.col("low")).alias("_bar_range"),
    )
    result = result.with_columns([
        pl.col("volume").cast(pl.Float64)
        .rolling_quantile(
            quantile=ABSORPTION_VOL_QUANTILE,
            window_size=ABSORPTION_LOOKBACK,
            min_samples=5,
        )
        .shift(1)
        .alias("_vol_q90_prev"),
        pl.col("_bar_range")
        .rolling_quantile(
            quantile=ABSORPTION_RANGE_QUANTILE,
            window_size=ABSORPTION_LOOKBACK,
            min_samples=5,
        )
        .shift(1)
        .alias("_range_q20_prev"),
    ])

    result = result.with_columns(
        pl.when(
            pl.col("_vol_q90_prev").is_not_null()
            & pl.col("_range_q20_prev").is_not_null()
            & (pl.col("volume").cast(pl.Float64) > pl.col("_vol_q90_prev"))
            & (pl.col("_bar_range") < pl.col("_range_q20_prev"))
        )
        .then(1.0)
        .otherwise(0.0)
        .alias("absorption_signal"),
    )

    # --- Trade Intensity Z-Score (breakout trigger) ---
    # Intensity = trades per second (adapts to bar duration)
    result = result.with_columns(
        (pl.col("trade_count").cast(pl.Float64) / (pl.col("bar_duration_ns") / 1e9 + 1e-9))
        .alias("intensity"),
    )

    result = result.with_columns([
        pl.col("intensity").rolling_mean(window_size=INTENSITY_ZSCORE_WINDOW, min_samples=2)
        .alias("_intensity_mean"),
        pl.col("intensity").rolling_std(window_size=INTENSITY_ZSCORE_WINDOW, min_samples=2)
        .alias("_intensity_std"),
    ])

    result = result.with_columns(
        pl.when(pl.col("_intensity_std") > 1e-9)
        .then((pl.col("intensity") - pl.col("_intensity_mean")) / pl.col("_intensity_std"))
        .otherwise(0.0)
        .alias("intensity_z"),
    )

    # Tape speed spike: binary flag when intensity is abnormally high
    result = result.with_columns(
        pl.when(pl.col("intensity_z") > INTENSITY_SPIKE_THRESHOLD)
        .then(1.0).otherwise(0.0).alias("tape_speed_spike"),
    )

    # Select only scalping feature columns
    return result.select([
        "ts_event",
        "delta_divergence_bull",
        "delta_divergence_bear",
        "effort_vs_result",
        "absorption_signal",
        "intensity",
        "intensity_z",
        "tape_speed_spike",
    ])
