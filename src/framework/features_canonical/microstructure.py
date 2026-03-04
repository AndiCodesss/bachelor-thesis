"""Microstructure features from pre-aggregated bar data."""
import polars as pl

from src.framework.data.constants import (
    TAPE_SPEED_SPIKE_Z, WHIP_Z_THRESHOLD, RECOIL_LOW, RECOIL_HIGH,
)


def compute_microstructure_features(bars: pl.DataFrame) -> pl.DataFrame:
    # Map bar columns to microstructure naming
    bars = bars.with_columns([
        pl.col("msg_count").alias("message_count"),
        pl.col("trade_count").alias("trade_count_micro"),
    ])

    # Compute derived features
    bars = bars.with_columns([
        # Cancel/trade ratio: "High cancel/trade ratio indicates noise vs real flow"
        (pl.col("cancel_count") / (pl.col("trade_count_micro") + 1)).alias("cancel_trade_ratio"),

        # Modify/trade ratio: "Frequent modifications suggest HFT activity"
        (pl.col("modify_count") / (pl.col("trade_count_micro") + 1)).alias("modify_trade_ratio"),

        # Mean trade interval: "Clustered arrivals indicate informed trading"
        # Approximate from bar_duration_ns / (trade_count - 1), converted to microseconds
        pl.when(pl.col("trade_count_micro") > 1)
        .then(
            (pl.col("bar_duration_ns").cast(pl.Float64) / (pl.col("trade_count_micro") - 1)) / 1000.0
        )
        .otherwise(None)
        .alias("mean_trade_interval_us"),

        # Mean message latency (ns) — aggregated in bars.py from (ts_recv - ts_event)
        pl.col("latency_mean"),
    ])

    # Sort by timestamp for proper rolling operations
    bars = bars.sort("ts_event")
    bars = bars.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_date"),
    )

    # Add rolling features (computed after bar aggregation, looking backward only)
    prev_cancel = pl.col("cancel_count").shift(1).over("_date")
    bars = bars.with_columns([
        # Message intensity MA: "Smoothed activity level"
        pl.col("message_count")
        .rolling_mean(window_size=5, min_samples=1)
        .over("_date")
        .alias("message_intensity_ma5"),

        # Cancel rate change: "Sudden increase in cancels precedes volatility"
        pl.when(prev_cancel > 0)
        .then(
            (pl.col("cancel_count") - prev_cancel)
            / prev_cancel
        )
        .otherwise(None)
        .alias("cancel_rate_change"),
    ])

    # --- Tape speed: "Fast tape precedes breakout moves" ---
    bars = bars.with_columns(
        (pl.col("trade_count_micro").cast(pl.Float64)
         / (pl.col("bar_duration_ns") / 1e6 + 1e-9)).alias("tape_speed"),
    )

    _ts_mean = pl.col("tape_speed").rolling_mean(window_size=24, min_samples=2).over("_date")
    _ts_std = pl.col("tape_speed").rolling_std(window_size=24, min_samples=2).over("_date")
    bars = bars.with_columns(
        pl.when(_ts_std > 0)
        .then((pl.col("tape_speed") - _ts_mean) / _ts_std)
        .otherwise(0.0)
        .alias("tape_speed_z"),
    )

    bars = bars.with_columns(
        pl.when(pl.col("tape_speed_z") > TAPE_SPEED_SPIKE_Z)
        .then(1.0).otherwise(0.0).alias("tape_speed_spike"),
    )

    # --- Price velocity: "Fast price movement signals institutional urgency" ---
    bars = bars.with_columns(
        ((pl.col("close").cast(pl.Float64) - pl.col("open").cast(pl.Float64)).abs()
         / (pl.col("bar_duration_ns") / 1e9 + 1e-9)).alias("price_velocity"),
    )

    _pv_mean = pl.col("price_velocity").rolling_mean(window_size=24, min_samples=2).over("_date")
    _pv_std = pl.col("price_velocity").rolling_std(window_size=24, min_samples=2).over("_date")
    bars = bars.with_columns(
        pl.when(_pv_std > 0)
        .then((pl.col("price_velocity") - _pv_mean) / _pv_std)
        .otherwise(0.0)
        .alias("price_velocity_z"),
    )

    bars = bars.with_columns(
        pl.when(pl.col("price_velocity_z") > WHIP_Z_THRESHOLD)
        .then(1.0).otherwise(0.0).alias("is_whip"),
    )

    # --- Recoil: "How much price retraces after a whip bar" ---
    bars = bars.with_columns([
        pl.col("is_whip").shift(1).over("_date").fill_null(0.0).alias("_prev_is_whip"),
        (pl.col("close").shift(1).over("_date") - pl.col("open").shift(1).over("_date")).abs().alias("_prev_range"),
    ])

    bars = bars.with_columns(
        pl.when(pl.col("_prev_is_whip") == 1.0)
        .then(
            (pl.col("close").shift(1).over("_date") - pl.col("close")).abs()
            / (pl.col("_prev_range") + 1e-9)
        )
        .otherwise(0.0)
        .alias("recoil_pct"),
    )

    bars = bars.with_columns(
        pl.when((pl.col("recoil_pct") > RECOIL_LOW) & (pl.col("recoil_pct") < RECOIL_HIGH))
        .then(1.0).otherwise(0.0).alias("recoil_50pct"),
    )

    # Select only ts_event + computed feature columns
    output_cols = [
        "ts_event",
        "cancel_trade_ratio",
        "modify_trade_ratio",
        "mean_trade_interval_us",
        "latency_mean",
        "message_intensity_ma5",
        "cancel_rate_change",
        "tape_speed",
        "tape_speed_z",
        "tape_speed_spike",
        "price_velocity",
        "price_velocity_z",
        "is_whip",
        "recoil_pct",
        "recoil_50pct",
    ]

    return bars.select(output_cols)
