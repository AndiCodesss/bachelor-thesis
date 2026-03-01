"""Orderflow features from pre-aggregated bar data."""
import polars as pl

from src.framework.data.constants import TICK_SIZE


def compute_orderflow_features(bars: pl.DataFrame) -> pl.DataFrame:
    # Compute derived features
    bars = bars.with_columns([
        # Delta for order flow (compute before rolling)
        (pl.col("buy_volume").cast(pl.Int64) - pl.col("sell_volume").cast(pl.Int64))
        .alias("volume_delta"),

        # Volume imbalance: "Measures directional aggression — imbalance predicts short-term price movement"
        ((pl.col("buy_volume").cast(pl.Int64) - pl.col("sell_volume").cast(pl.Int64)) /
         (pl.col("buy_volume") + pl.col("sell_volume") + 1).cast(pl.Float64)).alias("volume_imbalance"),

        # Trade intensity: "Sudden bursts of trading precede breakouts"
        (pl.col("trade_count").cast(pl.Float64) / (pl.col("bar_duration_ns") / 1e9 + 1e-9)).alias("trade_intensity"),

        # Large trade ratio: "High concentration of large trades predicts trend continuation"
        (pl.col("large_trade_count").cast(pl.Float64) / (pl.col("trade_count") + 1).cast(pl.Float64)).alias("large_trade_ratio"),
    ])

    # Sort by timestamp for proper rolling operations
    bars = bars.sort("ts_event")

    # Rolling features
    bars = bars.with_columns([
        # Order flow imbalance: "Sustained one-sided flow indicates institutional activity"
        pl.col("volume_delta")
        .rolling_sum(window_size=5, min_samples=1)
        .alias("order_flow_imbalance"),
    ])

    # --- CVD slope vs price slope divergence: "Flow-price disconnect signals exhaustion" ---
    bars = bars.with_columns([
        pl.col("volume_delta").rolling_sum(window_size=3, min_samples=1).alias("_cvd_slope_3"),
        pl.col("volume_delta").rolling_sum(window_size=6, min_samples=1).alias("_cvd_slope_6"),
        (pl.col("close") - pl.col("close").shift(3)).alias("_price_slope_3"),
        (pl.col("close") - pl.col("close").shift(6)).alias("_price_slope_6"),
    ])

    bars = bars.with_columns([
        pl.when(
            pl.col("_price_slope_3").is_not_null()
            & (pl.col("_cvd_slope_3").sign() != pl.col("_price_slope_3").sign())
        ).then(1.0).otherwise(0.0).alias("cvd_price_divergence_3"),

        pl.when(
            pl.col("_price_slope_6").is_not_null()
            & (pl.col("_cvd_slope_6").sign() != pl.col("_price_slope_6").sign())
        ).then(1.0).otherwise(0.0).alias("cvd_price_divergence_6"),
    ])

    # --- Absorption factor: "High volume + small range = hidden supply/demand" ---
    bars = bars.with_columns(
        (pl.col("high") - pl.col("low")).alias("_bar_range"),
    )

    bars = bars.with_columns(
        (pl.col("volume").cast(pl.Float64) * pl.col("volume_delta").abs().cast(pl.Float64)
         / (pl.col("_bar_range") / TICK_SIZE + 1)).alias("absorption_factor"),
    )

    bars = bars.with_columns([
        pl.col("volume").cast(pl.Float64).rolling_quantile(0.9, window_size=12, min_samples=1)
        .alias("_vol_90pct"),
        pl.col("_bar_range").rolling_quantile(0.2, window_size=12, min_samples=1)
        .alias("_range_20pct"),
    ])

    bars = bars.with_columns(
        pl.when(
            (pl.col("volume").cast(pl.Float64) > pl.col("_vol_90pct"))
            & (pl.col("_bar_range") < pl.col("_range_20pct"))
        ).then(1.0).otherwise(0.0).alias("absorption_signal"),
    )

    # --- Orderflow ratio: "Extreme one-sidedness signals institutional sweep" ---
    bars = bars.with_columns([
        (pl.max_horizontal("buy_volume", "sell_volume").cast(pl.Float64)
         / (pl.col("volume") + 1).cast(pl.Float64)).alias("orderflow_ratio"),
    ])

    # Select only ts_event + computed feature columns
    output_cols = [
        "ts_event",
        "volume_imbalance",
        "trade_intensity",
        "large_trade_ratio",
        "order_flow_imbalance",
        "cvd_price_divergence_3",
        "cvd_price_divergence_6",
        "absorption_factor",
        "absorption_signal",
        "orderflow_ratio",
    ]

    return bars.select(output_cols)
