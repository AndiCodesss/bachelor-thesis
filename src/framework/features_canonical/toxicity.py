"""Toxicity features: VPIN (Volume-Synchronized Probability of Informed Trading)."""
import polars as pl

# VPIN parameters
VPIN_WINDOW = 20  # rolling window of bars for VPIN calculation
VPIN_ZSCORE_LOOKBACK = 100  # bars for z-score normalization
VPIN_ZSCORE_MIN_SAMPLES = max(10, VPIN_ZSCORE_LOOKBACK // 4)
VPIN_RISK_OFF_THRESHOLD = 0.7  # binary flag threshold


def compute_toxicity_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute VPIN from pre-aggregated bar data.

    Each bar's imbalance = |buy_volume - sell_volume| / (buy_volume + sell_volume + 1).
    Rolling VPIN = rolling_mean(imbalance, window=20).

    Expects bars with columns: ts_event, buy_volume, sell_volume.
    """
    if len(bars) == 0:
        return pl.DataFrame(schema={
            "ts_event": pl.Datetime("ns", "UTC"),
            "vpin": pl.Float64,
            "vpin_zscore": pl.Float64,
            "vpin_risk_off": pl.Int8,
        })

    result = bars.select(["ts_event", "buy_volume", "sell_volume"]).sort("ts_event")
    result = result.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_date")
    )

    # Per-bar absolute imbalance normalized by total volume
    result = result.with_columns(
        ((pl.col("buy_volume").cast(pl.Float64) - pl.col("sell_volume").cast(pl.Float64)).abs()
         / (pl.col("buy_volume").cast(pl.Float64) + pl.col("sell_volume").cast(pl.Float64) + 1.0))
        .alias("_abs_imbalance")
    )

    # Rolling VPIN
    result = result.with_columns(
        pl.col("_abs_imbalance")
        .rolling_mean(window_size=VPIN_WINDOW, min_samples=1)
        .over("_date")
        .alias("vpin")
    )

    # Z-score over lookback
    result = result.with_columns(
        _zscore_expr("vpin", VPIN_ZSCORE_LOOKBACK, group_col="_date").alias("vpin_zscore")
    )

    # Risk-off flag
    result = result.with_columns(
        pl.when(pl.col("vpin") > VPIN_RISK_OFF_THRESHOLD)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("vpin_risk_off")
    )

    result = result.drop(["buy_volume", "sell_volume", "_abs_imbalance", "_date"])

    return result


def _zscore_expr(col_name: str, lookback: int, group_col: str | None = None) -> pl.Expr:
    """Rolling z-score expression: (x - mean) / std over lookback window."""
    col = pl.col(col_name)
    min_samples = max(2, min(lookback, VPIN_ZSCORE_MIN_SAMPLES))
    mean = col.rolling_mean(window_size=lookback, min_samples=min_samples)
    std = col.rolling_std(window_size=lookback, min_samples=min_samples)
    if group_col is not None:
        mean = mean.over(group_col)
        std = std.over(group_col)
    return (
        pl.when(std.is_null())
        .then(None)
        .when(std > 0)
        .then((col - mean) / std)
        .otherwise(0.0)
    )
