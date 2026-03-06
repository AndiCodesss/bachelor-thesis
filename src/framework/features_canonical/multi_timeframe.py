"""Multi-timeframe features: aggregate 1m features into 5m bars.

Captures intra-bar path dynamics — a 5m bar that went up then down looks
different from one that went down then up, even if close is the same.
"""
import polars as pl

# Columns to exclude from feature selection (duplicated here to avoid circular
# import with builder.py which imports this module)
_EXCLUDE_COLUMNS = {
    "ts_event", "ts_close", "bar_duration_ns",
    # Raw prices (non-stationary)
    "open", "high", "low", "close", "vwap", "bid_price", "ask_price", "mid_price",
    "poc_price", "va_high", "va_low", "rolling_poc", "rolling_va_high", "rolling_va_low",
    # Labels
    "fwd_return_1bar", "fwd_return_3bar", "fwd_return_5bar",
    "fwd_return_6bar", "fwd_return_10bar", "fwd_return_12bar",
    "label_1bar", "label_3bar", "label_5bar",
}

# Default aggregation operations
DEFAULT_AGG_OPS = ["mean", "std", "last", "delta", "min", "max"]

# Default number of most-variable features to aggregate
DEFAULT_TOP_N = 20

# Fixed set of 1m features to aggregate — ensures consistent schema across all files.
# Selected by highest variance across the full TRAIN set (Feb 2026 diagnostic).
# Using a fixed list prevents per-file variance selection from producing different columns.
FIXED_MTF_FEATURES = [
    "cancel_count", "buy_volume", "volume", "trade_count",
    "message_count", "sell_volume", "add_count", "volume_ma5",
    "message_intensity_ma5", "modify_count", "cvd", "order_flow_imbalance",
    "volume_delta", "mean_trade_interval_us", "latency_mean",
    "vwap_deviation", "weighted_book_imbalance", "trade_count_micro",
    "trade_intensity", "buy_sell_ratio",
]


def _parse_bar_minutes(bar_size: str) -> int:
    """Convert bar_size string to minutes. Only supports minute-based sizes."""
    if bar_size.endswith("m"):
        return int(bar_size[:-1])
    if bar_size.endswith("h"):
        return int(bar_size[:-1]) * 60
    raise ValueError(f"Unsupported bar_size for MTF: {bar_size}")


def compute_multi_timeframe_features(
    lf: pl.LazyFrame,
    bar_size: str = "5m",
    source_bar_size: str = "1m",
    top_n: int = DEFAULT_TOP_N,
    agg_ops: list[str] | None = None,
) -> pl.DataFrame:
    """Aggregate 1m features into 5m bars via statistical summaries.

    Builds the full feature matrix at source_bar_size resolution, selects the
    top_n most variable numeric features, and aggregates them into bar_size
    windows using multiple statistical operations.
    """
    if agg_ops is None:
        agg_ops = list(DEFAULT_AGG_OPS)

    target_mins = _parse_bar_minutes(bar_size)
    source_mins = _parse_bar_minutes(source_bar_size)
    assert target_mins > source_mins, (
        f"bar_size ({bar_size}) must be larger than source_bar_size ({source_bar_size})"
    )

    # Aggregate ticks into 1m bars first, then compute features
    from src.framework.data.bars import aggregate_time_bars
    from src.framework.features_canonical.orderflow import compute_orderflow_features
    from src.framework.features_canonical.book import compute_book_features
    from src.framework.features_canonical.microstructure import compute_microstructure_features
    from src.framework.features_canonical.microstructure_v2 import compute_microstructure_v2_features
    from src.framework.features_canonical.momentum import compute_momentum_features
    from src.framework.features_canonical.aggressor import compute_aggressor_features
    from src.framework.features_canonical.toxicity import compute_toxicity_features
    from src.framework.features_canonical.statistical import compute_statistical_features
    from src.framework.features_canonical.volume_profile import compute_volume_profile_features

    # Build 1m bars from raw ticks
    bars_1m = aggregate_time_bars(lf, source_bar_size)

    if len(bars_1m) == 0:
        return pl.DataFrame({"ts_event": pl.Series([], dtype=pl.Datetime("ns", "UTC"))})

    # Compute each module at 1m resolution from bars
    momentum_1m = compute_momentum_features(bars_1m)
    orderflow_1m = compute_orderflow_features(bars_1m)
    book_1m = compute_book_features(bars_1m)
    micro_1m = compute_microstructure_features(bars_1m)
    micro_v2_1m = compute_microstructure_v2_features(bars_1m)
    aggressor_1m = compute_aggressor_features(bars_1m)
    toxicity_1m = compute_toxicity_features(bars_1m)
    statistical_1m = compute_statistical_features(bars_1m)
    volume_profile_1m = compute_volume_profile_features(bars_1m)

    # Join all on ts_event using momentum as base
    df_1m = momentum_1m
    for module_feats, suffix in [
        (orderflow_1m, "_of"),
        (book_1m, "_bk"),
        (micro_1m, "_mc"),
        (micro_v2_1m, "_mv2"),
        (aggressor_1m, "_ag"),
        (toxicity_1m, "_tx"),
        (statistical_1m, "_st"),
        (volume_profile_1m, "_vp"),
    ]:
        df_1m = df_1m.join(module_feats, on="ts_event", how="left", suffix=suffix)

    if len(df_1m) == 0:
        return pl.DataFrame({"ts_event": pl.Series([], dtype=pl.Datetime("ns", "UTC"))})

    # Use fixed feature list for consistent schema across all files.
    available_cols = set(df_1m.columns)
    top_features = [f for f in FIXED_MTF_FEATURES if f in available_cols]

    if len(top_features) == 0:
        # Fallback for schema drift: pick top-N numeric non-excluded columns by variance.
        numeric_candidates = [
            name for name, dtype in df_1m.schema.items()
            if name not in _EXCLUDE_COLUMNS and name != "ts_event" and dtype.is_numeric()
        ]
        if len(numeric_candidates) == 0:
            return pl.DataFrame({"ts_event": pl.Series([], dtype=pl.Datetime("ns", "UTC"))})

        variance_row = df_1m.select([
            pl.col(name).cast(pl.Float64).var().alias(name)
            for name in numeric_candidates
        ]).row(0, named=True)

        def _variance_key(name: str) -> tuple[int, float, str]:
            value = variance_row.get(name)
            if value is None:
                return (1, 0.0, name)
            val = float(value)
            if val != val or val == float("inf") or val == float("-inf"):
                return (1, 0.0, name)
            return (0, -val, name)

        ranked = sorted(numeric_candidates, key=_variance_key)
        n_pick = max(1, min(int(top_n), len(ranked)))
        top_features = ranked[:n_pick]

    # Build aggregation expressions
    agg_exprs = []
    for feat in top_features:
        for op in agg_ops:
            col_name = f"m1f_{feat}__{op}"
            if op == "mean":
                agg_exprs.append(pl.col(feat).mean().alias(col_name))
            elif op == "std":
                agg_exprs.append(pl.col(feat).std().alias(col_name))
            elif op == "last":
                agg_exprs.append(pl.col(feat).last().alias(col_name))
            elif op == "delta":
                agg_exprs.append(
                    (pl.col(feat).last() - pl.col(feat).first()).alias(col_name)
                )
            elif op == "min":
                agg_exprs.append(pl.col(feat).min().alias(col_name))
            elif op == "max":
                agg_exprs.append(pl.col(feat).max().alias(col_name))
            elif op == "slope":
                agg_exprs.append(
                    pl.when(pl.len() > 1)
                    .then(
                        (pl.col(feat).last() - pl.col(feat).first())
                        / (pl.len() - 1).cast(pl.Float64)
                    )
                    .otherwise(0.0)
                    .alias(col_name)
                )

    # Aggregate 1m bars into target bar_size windows
    mtf_bars = (
        df_1m.sort("ts_event")
        .group_by_dynamic(
            "ts_event",
            every=bar_size,
            period=bar_size,
            closed="left",
            label="left",
        )
        .agg(agg_exprs)
    )

    return mtf_bars
