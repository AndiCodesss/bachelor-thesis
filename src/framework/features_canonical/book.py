"""Order book features from pre-aggregated bar data."""
import polars as pl


def compute_book_features(bars: pl.DataFrame) -> pl.DataFrame:
    # Compute derived features
    bars = bars.with_columns([
        # Mid price: fair value estimate
        ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("mid_price"),

        # Spread: wider spread = less liquidity, higher uncertainty
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
    ])

    bars = bars.with_columns([
        # Spread in basis points: normalized spread for cross-session comparison
        pl.when(pl.col("mid_price").abs() > 1e-12)
        .then(pl.col("spread") / pl.col("mid_price") * 10000.0)
        .otherwise(None)
        .alias("spread_bps"),

        # Book imbalance: bid-heavy book predicts upward pressure
        ((pl.col("bid_size").cast(pl.Int64) - pl.col("ask_size").cast(pl.Int64)) /
         (pl.col("bid_size").cast(pl.Int64) + pl.col("ask_size").cast(pl.Int64) + 1).cast(pl.Float64)).alias("book_imbalance"),

        # Depth ratio: more bid orders = stronger support
        (pl.col("bid_count").cast(pl.Float64) / (pl.col("ask_count").cast(pl.Float64) + 1.0)).alias("depth_ratio"),
    ])

    # Sort by timestamp to ensure proper rolling window calculations
    bars = bars.sort("ts_event")
    bars = bars.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_date"),
    )

    # Add rolling features (look backward only)
    prev_mid = pl.col("mid_price").shift(1).over("_date")
    idx = pl.int_range(pl.len()).over("_date")
    bars = bars.with_columns([
        # Smoothed directional pressure signal
        pl.col("book_imbalance").rolling_mean(window_size=5, min_samples=1).over("_date").alias("book_imbalance_ma5"),

        # Increasing spread vol signals regime change
        pl.col("spread").rolling_std(window_size=5, min_samples=1).over("_date").fill_null(0.0).alias("spread_volatility"),

        # Bar-to-bar return with explicit zero guard on prior mid-price.
        pl.when(idx == 0)
        .then(0.0)
        .when(prev_mid > 0)
        .then((pl.col("mid_price") - prev_mid) / prev_mid)
        .otherwise(None)
        .alias("mid_price_return"),
    ])

    # Short-term momentum: rolling 5-bar sum of returns
    bars = bars.with_columns([
        pl.col("mid_price_return").rolling_sum(window_size=5, min_samples=1).over("_date").alias("mid_price_return_5"),
    ])

    # Select final columns in logical order
    return bars.select([
        "ts_event",
        "bid_price",
        "ask_price",
        "mid_price",
        "spread",
        "spread_bps",
        "book_imbalance",
        "book_imbalance_ma5",
        "depth_ratio",
        "spread_volatility",
        "mid_price_return",
        "mid_price_return_5",
    ])
