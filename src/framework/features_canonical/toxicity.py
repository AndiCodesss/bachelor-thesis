"""Toxicity features: VPIN (Volume-Synchronized Probability of Informed Trading)."""
import polars as pl

# VPIN parameters
VPIN_WINDOW = 20  # rolling window of bars for VPIN calculation
VPIN_ZSCORE_LOOKBACK = 100  # bars for z-score normalization
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
        .alias("vpin")
    )

    # Z-score over lookback
    result = result.with_columns(
        _zscore_expr("vpin", VPIN_ZSCORE_LOOKBACK).alias("vpin_zscore")
    )

    # Risk-off flag
    result = result.with_columns(
        pl.when(pl.col("vpin") > VPIN_RISK_OFF_THRESHOLD)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("vpin_risk_off")
    )

    result = result.drop(["buy_volume", "sell_volume", "_abs_imbalance"])

    return result


def _zscore_expr(col_name: str, lookback: int) -> pl.Expr:
    """Rolling z-score expression: (x - mean) / std over lookback window."""
    col = pl.col(col_name)
    mean = col.rolling_mean(window_size=lookback, min_samples=2)
    std = col.rolling_std(window_size=lookback, min_samples=2)
    return pl.when(std > 0).then((col - mean) / std).otherwise(0.0)
