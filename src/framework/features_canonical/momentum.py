"""Momentum and price features from pre-aggregated bar data."""
import polars as pl


def compute_momentum_features(bars: pl.DataFrame) -> pl.DataFrame:
    # Sort by timestamp and derive per-session partition key for lag/rolling ops.
    bars = bars.sort("ts_event").with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_date"),
    )
    bars = bars.with_columns(
        pl.int_range(pl.len()).over("_date").alias("_bar_idx_in_day"),
    )
    prev_close = pl.col("close").shift(1).over("_date")
    idx = pl.col("_bar_idx_in_day")

    # Compute return and momentum features
    bars = bars.with_columns([
        # return_1bar: "Single-bar return"
        pl.when(idx == 0)
        .then(0.0)
        .when(prev_close.abs() > 1e-12)
        .then((pl.col("close") - prev_close) / prev_close)
        .otherwise(None)
        .alias("return_1bar"),

        # vwap_deviation: "Price above VWAP suggests bullish sentiment"
        ((pl.col("close") - pl.col("vwap")) / (pl.col("vwap") + 0.0001)).alias("vwap_deviation"),

        # high_low_range: "Bar range as volatility proxy"
        ((pl.col("high") - pl.col("low")) / (pl.col("close") + 0.0001)).alias("high_low_range"),
    ])

    # Add rolling features
    bars = bars.with_columns([
        # return_5bar: "Short-term momentum"
        pl.col("return_1bar").rolling_sum(window_size=5, min_samples=1).over("_date").alias("return_5bar"),

        # return_12bar: "Medium-term momentum (1hr at 5m bars)"
        pl.col("return_1bar").rolling_sum(window_size=12, min_samples=1).over("_date").alias("return_12bar"),

        # vwap_deviation_ma5: "Persistent VWAP deviation signals trend"
        pl.col("vwap_deviation").rolling_mean(window_size=5, min_samples=1).over("_date").alias("vwap_deviation_ma5"),

        # range_ma5: "Smoothed volatility"
        pl.col("high_low_range").rolling_mean(window_size=5, min_samples=1).over("_date").alias("range_ma5"),

        # volume_ma5: "Average volume baseline"
        pl.col("volume").rolling_mean(window_size=5, min_samples=1).over("_date").alias("volume_ma5"),
    ])

    # Add volume-based features
    bars = bars.with_columns([
        # volume_ratio: "Volume surge detection — breakout precursor"
        (pl.col("volume") / (pl.col("volume_ma5") + 0.0001)).alias("volume_ratio"),

        # close_position: "Where price closed within bar range — bullish if near high"
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 0.0001)).alias("close_position"),

        # momentum_volume: "Volume-weighted momentum — conviction behind moves"
        (pl.col("return_1bar") * pl.col("volume")).alias("momentum_volume"),
    ])

    # Handle edge case: when high == low, set close_position to 0.5
    bars = bars.with_columns([
        pl.when((pl.col("high") - pl.col("low")).abs() < 0.0001)
        .then(0.5)
        .otherwise(pl.col("close_position"))
        .alias("close_position")
    ])

    # --- Wick ratios: "Rejection at extremes signals trapped traders" ---
    range_expr = pl.col("high") - pl.col("low") + 1e-9
    zero_range = (pl.col("high") - pl.col("low")).abs() < 0.0001

    bars = bars.with_columns([
        # Upper wick ratio: "Rejection at highs = bearish trap"
        pl.when(zero_range).then(0.0)
        .otherwise(
            (pl.col("high") - pl.max_horizontal("open", "close")) / range_expr
        ).alias("upper_wick_ratio"),

        # Lower wick ratio: "Rejection at lows = bullish trap"
        pl.when(zero_range).then(0.0)
        .otherwise(
            (pl.min_horizontal("open", "close") - pl.col("low")) / range_expr
        ).alias("lower_wick_ratio"),

        # Body ratio: "Small body + big wicks = doji = indecision"
        pl.when(zero_range).then(0.0)
        .otherwise(
            (pl.col("close") - pl.col("open")).abs() / range_expr
        ).alias("body_ratio"),
    ])

    # Select ts_event + OHLCV + vwap (needed downstream by pipeline/labels)
    # plus all computed feature columns
    output_cols = [
        "ts_event",
        "open", "high", "low", "close", "volume", "vwap",
        "return_1bar",
        "vwap_deviation",
        "high_low_range",
        "return_5bar",
        "return_12bar",
        "vwap_deviation_ma5",
        "range_ma5",
        "volume_ma5",
        "volume_ratio",
        "close_position",
        "momentum_volume",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "body_ratio",
    ]

    return bars.select(output_cols)
