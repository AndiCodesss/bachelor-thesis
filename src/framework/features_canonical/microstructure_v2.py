"""Advanced L1 microstructure features from pre-aggregated bar data.

References:
- Easley et al. (2012): VPIN
- Cartea & Jaimungal (2016): Trade arrival asymmetry
- Hautsch & Huang (2012): Cancel/trade ratios
- Cao et al. (2009): Weighted book imbalance
- Stoikov (2018): Micro-price
"""
import polars as pl


def compute_microstructure_v2_features(bars: pl.DataFrame) -> pl.DataFrame:
    # Cast to signed to prevent u32 underflow in subtraction
    bars = bars.with_columns([
        pl.col("buy_volume").cast(pl.Int64),
        pl.col("sell_volume").cast(pl.Int64),
    ])

    bars = bars.with_columns([
        # Feature: trade_arrival_imbalance (Cartea & Jaimungal 2016)
        # Volume-based imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol), bounded [-1, 1]
        pl.when((pl.col("buy_volume") + pl.col("sell_volume")) > 0)
        .then(
            (pl.col("buy_volume") - pl.col("sell_volume")).cast(pl.Float64) /
            (pl.col("buy_volume") + pl.col("sell_volume")).cast(pl.Float64)
        )
        .otherwise(None)
        .alias("trade_arrival_imbalance"),

        # Feature: vpin (Easley et al. 2012)
        # |buy_vol - sell_vol| / total_vol, bounded [0, 1]
        pl.when((pl.col("buy_volume") + pl.col("sell_volume")) > 0)
        .then(
            (pl.col("buy_volume") - pl.col("sell_volume")).abs().cast(pl.Float64) /
            (pl.col("buy_volume") + pl.col("sell_volume")).cast(pl.Float64)
        )
        .otherwise(None)
        .alias("vpin_raw"),
    ])

    # VPIN rolling window (Easley et al.)
    bars = bars.sort("ts_event").with_columns([
        pl.col("vpin_raw").rolling_mean(window_size=20, min_samples=1).alias("vpin")
    ]).drop("vpin_raw")

    # Feature: cancel_ratio (Hautsch & Huang 2012)
    # Approximation: uses total cancel_count / trade_count (per-side split not available in bars)
    bars = bars.with_columns([
        (pl.col("cancel_count").cast(pl.Float64) / (pl.col("trade_count") + 1).cast(pl.Float64))
        .alias("bid_cancel_ratio"),

        (pl.col("cancel_count").cast(pl.Float64) / (pl.col("trade_count") + 1).cast(pl.Float64))
        .alias("ask_cancel_ratio"),
    ])

    # Feature: weighted_book_imbalance (Cao et al. 2009)
    bars = bars.with_columns([
        (
            ((pl.col("bid_size") * pl.col("bid_count")) - (pl.col("ask_size") * pl.col("ask_count"))) /
            ((pl.col("bid_size") * pl.col("bid_count")) + (pl.col("ask_size") * pl.col("ask_count")) + 1)
        ).alias("weighted_book_imbalance"),
    ])

    # Feature: micro_price_momentum (Stoikov 2018)
    bars = bars.with_columns([
        (
            (pl.col("bid_price") * pl.col("ask_size").cast(pl.Float64) +
             pl.col("ask_price") * pl.col("bid_size").cast(pl.Float64)) /
            (pl.col("bid_size").cast(pl.Float64) + pl.col("ask_size").cast(pl.Float64) + 1e-9)
        ).alias("micro_price"),
    ])

    bars = bars.with_columns([
        ((pl.col("micro_price") - pl.col("micro_price").shift(1)) /
         (pl.col("micro_price").shift(1) + 1e-9)).alias("micro_price_momentum"),
    ])

    # Cast features to Float64
    result = bars.select([
        "ts_event",
        pl.col("trade_arrival_imbalance").cast(pl.Float64),
        pl.col("vpin").cast(pl.Float64),
        pl.col("bid_cancel_ratio").cast(pl.Float64),
        pl.col("ask_cancel_ratio").cast(pl.Float64),
        pl.col("weighted_book_imbalance").cast(pl.Float64),
        pl.col("micro_price_momentum").cast(pl.Float64),
    ])

    return result
