"""Feature pipeline — combines all feature modules into a unified matrix.

Cache builder: aggregates target day's raw ticks to bars, loads prior-day
warmup bars from a dedicated bar cache (untrimmed), then computes features.
This keeps peak memory to 1 raw file at a time.
"""
import gc
import re
from datetime import date as _date_cls
from pathlib import Path
import polars as pl
from src.framework.data.bars import aggregate_time_bars, aggregate_volume_bars, aggregate_tick_bars
from src.framework.features_canonical.orderflow import compute_orderflow_features
from src.framework.features_canonical.book import compute_book_features
from src.framework.features_canonical.microstructure import compute_microstructure_features
from src.framework.features_canonical.microstructure_v2 import compute_microstructure_v2_features
from src.framework.features_canonical.momentum import compute_momentum_features
from src.framework.features_canonical.labels import compute_labels
from src.framework.features_canonical.toxicity import compute_toxicity_features
from src.framework.features_canonical.volume_profile import compute_volume_profile_features
from src.framework.features_canonical.statistical import compute_statistical_features
from src.framework.features_canonical.aggressor import compute_aggressor_features
from src.framework.features_canonical.footprint import compute_footprint_features
from src.framework.features_canonical.opening_range import compute_opening_range_features
from src.framework.features_canonical.ohlcv_indicators import compute_ohlcv_indicators
from src.framework.features_canonical.pipeline import compute_pipeline_features
from src.framework.data.constants import RESULTS_DIR
from src.framework.data.loader import filter_rth, filter_eth

CACHE_DIR = RESULTS_DIR / "cache"

# Number of prior calendar-day files to prepend for rolling-window warmup.
# Weekends/holidays produce 0 bars, so we use 4 files to guarantee >= 200 bars
# of actual trading data (3 trading days * ~78 bars = ~234 bars), covering
# the longest rolling window (SMA-200).
WARMUP_CONTEXT_DAYS = 4

_DATE_RE = re.compile(r"nq_(\d{4})-(\d{2})-(\d{2})\.parquet$")

# Define which columns are labels (not features)
LABEL_COLUMNS = [
    "fwd_return_1bar",
    "fwd_return_3bar",
    "fwd_return_5bar",
    "fwd_return_6bar",
    "fwd_return_10bar",
    "fwd_return_12bar",
    "label_1bar",
    "label_3bar",
    "label_5bar"
]

# Raw non-stationary price columns — not valid ML features (absolute levels change over time)
RAW_PRICE_COLUMNS = [
    "open", "high", "low", "close", "vwap", "bid_price", "ask_price", "mid_price",
    "poc_price", "va_high", "va_low", "rolling_poc", "rolling_va_high", "rolling_va_low",
    "prev_day_high", "prev_day_low", "prev_day_close",
    "prev_day_poc", "prev_day_vah", "prev_day_val",
    "or_high", "or_low",
]

# Session-level VP features — use full-day data = lookahead bias for bars before session end
SESSION_VP_COLUMNS = [
    "poc_distance", "poc_distance_raw", "va_width", "position_in_va",
    "vp_skew", "vp_kurtosis", "hvn_count",
]

# Bar metadata columns from bars.py — not ML features
BAR_META_COLUMNS = [
    "ts_close", "bar_duration_ns",
]

# Raw flow columns optionally passed through for strategy logic; not ML features
RAW_FLOW_COLUMNS = [
    "trade_count",
    "buy_volume",
    "sell_volume",
    "large_trade_count",
    "large_buy_volume",
    "large_sell_volume",
    "whale_trade_count_30",
    "whale_buy_volume_30",
    "whale_sell_volume_30",
]

# Columns to exclude from feature matrix (identifiers + labels + raw prices + lookahead + bar meta)
NON_FEATURE_COLUMNS = (
    ["ts_event"]
    + BAR_META_COLUMNS
    + RAW_FLOW_COLUMNS
    + RAW_PRICE_COLUMNS
    + SESSION_VP_COLUMNS
    + LABEL_COLUMNS
)


def build_feature_matrix(
    lf: pl.LazyFrame,
    bar_size: str = "5m",
    bar_type: str = "time",
    bar_threshold: int | None = None,
    include_bar_columns: bool = False,
) -> pl.DataFrame:
    """Combine all feature modules into a unified feature matrix ready for ML.

    1. Aggregates raw ticks into bars (time, volume, or tick)
    2. Passes bars to all feature modules
    3. Joins everything on ts_event
    4. Drops rows with null labels

    Args:
        lf: LazyFrame with raw MBP1 tick data
        bar_size: Bar period for time bars (e.g., "5m", "1m")
        bar_type: "time" | "volume" | "tick"
        bar_threshold: Volume or tick count for non-time bars
        include_bar_columns: If True, include raw bar flow columns required by
            strategy-level modules (e.g., scalping feature recomputation).

    Returns:
        DataFrame with ts_event, all features, OHLCV, and labels
    """
    # Step 1: Aggregate ticks into bars
    if bar_type == "time":
        bars = aggregate_time_bars(lf, bar_size)
    elif bar_type == "volume":
        if bar_threshold is None or bar_threshold <= 0:
            raise ValueError("bar_threshold must be > 0 for volume bars")
        bars = aggregate_volume_bars(lf, bar_threshold)
    elif bar_type == "tick":
        if bar_threshold is None or bar_threshold <= 0:
            raise ValueError("bar_threshold must be > 0 for tick bars")
        bars = aggregate_tick_bars(lf, bar_threshold)
    else:
        raise ValueError(f"Unknown bar_type: {bar_type}. Use 'time', 'volume', or 'tick'")

    if len(bars) == 0:
        return pl.DataFrame()

    # Step 2: Compute all feature modules from bars
    momentum_feats = compute_momentum_features(bars)
    orderflow_feats = compute_orderflow_features(bars)
    book_feats = compute_book_features(bars)
    microstructure_feats = compute_microstructure_features(bars)
    microstructure_v2_feats = compute_microstructure_v2_features(bars)
    toxicity_feats = compute_toxicity_features(bars)
    volume_profile_feats = compute_volume_profile_features(bars)
    statistical_feats = compute_statistical_features(bars)
    aggressor_feats = compute_aggressor_features(bars)
    footprint_feats = compute_footprint_features(bars)
    opening_range_feats = compute_opening_range_features(bars)
    labels_feats = compute_labels(bars)
    ohlcv_indicator_feats = compute_ohlcv_indicators(bars)

    # Step 3: Join all on ts_event, momentum as anchor
    result = momentum_feats

    for feats, suffix in [
        (orderflow_feats, "_orderflow"),
        (book_feats, "_book"),
        (microstructure_feats, "_micro"),
        (microstructure_v2_feats, "_micro_v2"),
        (toxicity_feats, "_toxicity"),
        (volume_profile_feats, "_vp"),
        (statistical_feats, "_stat"),
        (aggressor_feats, "_aggressor"),
        (footprint_feats, "_footprint"),
        (opening_range_feats, "_or"),
        (labels_feats, "_labels"),
        (ohlcv_indicator_feats, "_ohlcv"),
    ]:
        result = result.join(feats, on="ts_event", how="left", suffix=suffix)

    # Drop duplicate close column from labels (keep momentum's close)
    if "close_labels" in result.columns:
        result = result.drop("close_labels")

    # Optional strategy passthrough columns from the raw bar aggregation.
    # These are intentionally excluded from default ML feature workflows.
    if include_bar_columns:
        passthrough_cols = [
            c for c in (
                "ts_close",
                "bar_duration_ns",
                "trade_count",
                "buy_volume",
                "sell_volume",
                "large_trade_count",
                "large_buy_volume",
                "large_sell_volume",
                "whale_trade_count_30",
                "whale_buy_volume_30",
                "whale_sell_volume_30",
            )
            if c in bars.columns and c not in result.columns
        ]
        if passthrough_cols:
            result = result.join(
                bars.select(["ts_event", *passthrough_cols]),
                on="ts_event",
                how="left",
            )

    # Step 4: Compute interaction/regime features (needs joined matrix)
    result = compute_pipeline_features(result)

    # Step 5: Apply caller-facing row filters (labels + warmup rows).
    return _apply_output_filters(result)


def get_feature_columns(df_or_lf) -> list[str]:
    """Return feature column names (excludes ts_event, labels, close, bar metadata)."""
    if isinstance(df_or_lf, pl.DataFrame):
        all_cols = df_or_lf.columns
    else:
        all_cols = df_or_lf.collect_schema().names()

    return [c for c in all_cols if c not in NON_FEATURE_COLUMNS]


def _aggregate_bars(
    lf: pl.LazyFrame,
    bar_size: str,
    bar_type: str,
    bar_threshold: int | None,
) -> pl.DataFrame:
    """Aggregate raw ticks into bars."""
    if bar_type == "time":
        return aggregate_time_bars(lf, bar_size)
    elif bar_type == "volume":
        if bar_threshold is None or bar_threshold <= 0:
            raise ValueError("bar_threshold must be > 0 for volume bars")
        return aggregate_volume_bars(lf, bar_threshold)
    elif bar_type == "tick":
        if bar_threshold is None or bar_threshold <= 0:
            raise ValueError("bar_threshold must be > 0 for tick bars")
        return aggregate_tick_bars(lf, bar_threshold)
    raise ValueError(f"Unknown bar_type: {bar_type}")


def _compute_features_from_bars(
    bars: pl.DataFrame,
    include_bar_columns: bool = False,
) -> pl.DataFrame:
    """Compute all features from pre-aggregated bars.

    Same logic as build_feature_matrix steps 2-6, but operates on bars
    directly instead of raw tick data.
    """
    if len(bars) == 0:
        return pl.DataFrame()

    momentum_feats = compute_momentum_features(bars)
    result = momentum_feats

    for feats, suffix in [
        (compute_orderflow_features(bars), "_orderflow"),
        (compute_book_features(bars), "_book"),
        (compute_microstructure_features(bars), "_micro"),
        (compute_microstructure_v2_features(bars), "_micro_v2"),
        (compute_toxicity_features(bars), "_toxicity"),
        (compute_volume_profile_features(bars), "_vp"),
        (compute_statistical_features(bars), "_stat"),
        (compute_aggressor_features(bars), "_aggressor"),
        (compute_footprint_features(bars), "_footprint"),
        (compute_opening_range_features(bars), "_or"),
        (compute_labels(bars), "_labels"),
        (compute_ohlcv_indicators(bars), "_ohlcv"),
    ]:
        result = result.join(feats, on="ts_event", how="left", suffix=suffix)

    if "close_labels" in result.columns:
        result = result.drop("close_labels")

    if include_bar_columns:
        passthrough_cols = [
            c for c in _BAR_COLUMNS
            if c in bars.columns and c not in result.columns and c != "ts_event"
        ]
        if passthrough_cols:
            result = result.join(
                bars.select(["ts_event", *passthrough_cols]),
                on="ts_event", how="left",
            )

    result = compute_pipeline_features(result)
    return result


def load_cached_matrix(
    parquet_path: Path,
    bar_size: str = "5m",
    bar_type: str = "time",
    bar_threshold: int | None = None,
    include_bar_columns: bool = False,
    required_columns: list[str] | None = None,
    session_filter: str = "rth",
    warmup_days: int = WARMUP_CONTEXT_DAYS,
) -> pl.DataFrame:
    """Load feature matrix from cache, or build with warmup context.

    Build approach (memory-efficient):
      1. Load target day's bars from dedicated bar cache (or aggregate raw ticks)
      2. Load prior-day warmup bars from dedicated bar cache
         (falls back to raw tick aggregation if no cache exists yet)
      3. Compute features on combined bars → filter to target day → feature-cache

    Only 1 raw parquet file is in memory at a time.

    Args:
        session_filter: "rth" for Regular Trading Hours, "eth" for Extended.
        warmup_days: Number of prior calendar-day files for rolling-window warmup.
    """
    cache_dir = _feature_cache_dir(bar_size, bar_type, bar_threshold, session_filter)
    cache_path = cache_dir / parquet_path.name

    cache_required = _required_cache_columns(include_bar_columns, required_columns)

    if cache_path.exists():
        cached = pl.read_parquet(cache_path)
        if len(cached) == 0:
            return cached
        missing = [c for c in cache_required if c not in cached.columns]
        if not missing and not _is_legacy_filtered_cache(cached):
            return _apply_output_filters(cached)

    session_fn = filter_eth if session_filter == "eth" else filter_rth
    target_date = _extract_date_from_filename(parquet_path.name)
    bars_cache_dir = _bar_cache_dir(bar_size, bar_type, bar_threshold, session_filter)

    # Phase 1: Load target day's bars (cached or raw aggregation)
    current_bars = _load_or_build_bars(
        parquet_path, bars_cache_dir, bar_size, bar_type, bar_threshold, session_fn,
    )

    if len(current_bars) == 0:
        return pl.DataFrame()

    # Phase 2: Load warmup bars from dedicated bar cache (or raw fallback)
    context_files = _find_context_files(parquet_path, warmup_days)
    warmup_bars = [
        bars for fp in context_files
        if len(bars := _load_warmup_bars(
            fp, bars_cache_dir, bar_size, bar_type, bar_threshold, session_fn,
        )) > 0
    ]

    if warmup_bars:
        combined_bars = pl.concat([*warmup_bars, current_bars])
    else:
        combined_bars = current_bars

    # Phase 3: Compute unfiltered features on combined bars
    df = _compute_features_from_bars(combined_bars, include_bar_columns)
    del combined_bars
    gc.collect()

    # Filter to target date only (strip warmup rows)
    if target_date is not None and context_files and len(df) > 0:
        df = df.filter(
            pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date() == target_date
        )

    if len(df) > 0:
        missing = [c for c in cache_required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns after build for {parquet_path.name}: {missing}",
            )

    # Cache non-empty results (unfiltered)
    if len(df) > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)

    return _apply_output_filters(df)


def _build_bars_from_raw(
    raw_path: Path,
    bar_size: str,
    bar_type: str,
    bar_threshold: int | None,
    session_fn,
) -> pl.DataFrame:
    """Aggregate raw ticks into bars (no caching of intermediate bars)."""
    lf = session_fn(pl.scan_parquet(str(raw_path)))
    bars = _aggregate_bars(lf, bar_size, bar_type, bar_threshold)
    del lf
    gc.collect()
    return bars


def _required_cache_columns(
    include_bar_columns: bool,
    required_columns: list[str] | None,
) -> list[str]:
    """Columns that must be present to accept an existing feature cache file."""
    required = set(required_columns or [])
    if include_bar_columns:
        required.update(_BAR_COLUMNS)
    return sorted(required)


def _apply_output_filters(df: pl.DataFrame) -> pl.DataFrame:
    """Apply caller-facing row filters on top of unfiltered cached features."""
    if len(df) == 0:
        return df

    # Drop rows with null labels (forward return edge effects).
    df = df.filter(
        pl.all_horizontal(pl.col(c).is_not_null() for c in LABEL_COLUMNS)
    )

    # Drop warmup rows where rolling features have not yet filled.
    # sma_ratio_8 requires 8 bars of context — with cross-day warmup, this
    # should not drop target-day bars except at early dataset start.
    if "sma_ratio_8" in df.columns:
        return df.filter(pl.col("sma_ratio_8").is_not_null())
    return df.filter(pl.col("return_1bar").is_not_null())


def _is_legacy_filtered_cache(df: pl.DataFrame) -> bool:
    """Detect old cache format that stored already-filtered rows.

    Legacy caches dropped label-null tail rows before write. Unfiltered caches
    should keep those rows, so for normal trading days we expect at least one
    null in the forward-label columns.
    """
    if len(df) <= 12:
        return False

    # Legacy format 1: cache written after label filtering (no label-null tail).
    if all(c in df.columns for c in LABEL_COLUMNS):
        if all(df[c].null_count() == 0 for c in LABEL_COLUMNS):
            return True

    # Legacy format 2: OR breakout flags encoded as 0.0 before OR is ready.
    # Current semantics: when OR is not ready (or_width is null), breakout flags
    # are also null (unknown), not zero.
    or_cols = {"or_width", "or_broken_up", "or_broken_down"}
    if or_cols.issubset(df.columns):
        mismatch = df.filter(
            pl.col("or_width").is_null()
            & (
                pl.col("or_broken_up").is_not_null()
                | pl.col("or_broken_down").is_not_null()
            )
        ).height
        if mismatch > 0:
            return True

    return False


# Bar columns needed for feature computation (input to _compute_features_from_bars)
_BAR_COLUMNS = [
    "ts_event", "ts_close", "bar_duration_ns",
    "open", "high", "low", "close", "volume", "vwap",
    "trade_count", "buy_volume", "sell_volume",
    "large_trade_count", "large_buy_volume", "large_sell_volume",
    "whale_trade_count_30", "whale_buy_volume_30", "whale_sell_volume_30",
    "bid_price", "ask_price", "bid_size", "ask_size", "bid_count", "ask_count",
    "msg_count", "add_count", "cancel_count", "modify_count", "latency_mean",
    "stacked_imbalance_count", "stacked_imbalance_direction",
    "zero_print_count", "zero_print_ratio",
    "unfinished_high", "unfinished_low", "max_level_volume",
    "volume_at_high", "volume_at_low",
    "buy_vol_at_high", "sell_vol_at_high", "buy_vol_at_low", "sell_vol_at_low",
    "vap_prices", "vap_volumes",
]


def _load_warmup_bars(
    raw_path: Path,
    bars_cache_dir: Path,
    bar_size: str,
    bar_type: str,
    bar_threshold: int | None,
    session_fn,
) -> pl.DataFrame:
    """Load untrimmed warmup bars from dedicated bar cache.

    Falls back to raw aggregation when no cache exists or columns are missing.
    """
    return _load_or_build_bars(
        raw_path, bars_cache_dir, bar_size, bar_type, bar_threshold, session_fn,
    )


def _load_or_build_bars(
    raw_path: Path,
    bars_cache_dir: Path,
    bar_size: str,
    bar_type: str,
    bar_threshold: int | None,
    session_fn,
) -> pl.DataFrame:
    """Load untrimmed bars from bar cache, or build from raw and persist."""
    cache_path = bars_cache_dir / raw_path.name
    if cache_path.exists():
        available = set(pl.read_parquet_schema(cache_path))
        if all(c in available for c in _BAR_COLUMNS):
            return pl.read_parquet(cache_path, columns=_BAR_COLUMNS)

    bars = _build_bars_from_raw(raw_path, bar_size, bar_type, bar_threshold, session_fn)
    if len(bars) == 0:
        return bars

    missing_cols = [c for c in _BAR_COLUMNS if c not in bars.columns]
    if missing_cols:
        raise ValueError(f"Bar aggregation missing required columns: {missing_cols}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bars.select(_BAR_COLUMNS).write_parquet(cache_path)
    return bars.select(_BAR_COLUMNS)


def _bar_cache_dir(
    bar_size: str,
    bar_type: str,
    bar_threshold: int | None,
    session_filter: str,
) -> Path:
    """Directory holding untrimmed aggregated bars for warmup reuse."""
    return _feature_cache_dir(bar_size, bar_type, bar_threshold, session_filter) / "_bars"


def _feature_cache_dir(
    bar_size: str,
    bar_type: str,
    bar_threshold: int | None,
    session_filter: str,
) -> Path:
    """Cache directory: one folder per bar type (e.g. eth_tick_610/)."""
    if bar_type == "time":
        name = bar_size
    elif bar_type == "volume":
        name = f"vol_{bar_threshold}"
    elif bar_type == "tick":
        name = f"{bar_type}_{bar_threshold}"
    else:
        name = bar_size
    if session_filter == "eth":
        name = f"eth_{name}"
    return CACHE_DIR / name


def _extract_date_from_filename(name: str) -> _date_cls | None:
    """Extract trading date from parquet filename like nq_2023-03-01.parquet."""
    m = _DATE_RE.search(name)
    if m is None:
        return None
    return _date_cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _find_context_files(target_path: Path, n_days: int) -> list[Path]:
    """Find the *n_days* preceding daily parquet files for cross-day warmup."""
    if n_days <= 0:
        return []
    data_root = target_path.parent.parent
    if not data_root.is_dir():
        return []
    all_files = sorted(data_root.rglob("nq_*.parquet"))
    target_resolved = target_path.resolve()
    target_idx: int | None = None
    for i, f in enumerate(all_files):
        if f.resolve() == target_resolved:
            target_idx = i
            break
    if target_idx is None or target_idx == 0:
        return []
    start = max(0, target_idx - n_days)
    return all_files[start:target_idx]


# Default bar configurations for NQ E-mini futures
BAR_CONFIGS = [
    {"bar_type": "tick",   "bar_size": "5m", "bar_threshold": 610},
    {"bar_type": "volume", "bar_size": "5m", "bar_threshold": 2000},
    {"bar_type": "time",   "bar_size": "1m", "bar_threshold": None},
]


def build_full_cache(
    split: str = "validate",
    bar_configs: list[dict] | None = None,
    session_filter: str = "eth",
    file_limit: int | None = None,
    bar_filter: str | None = None,
) -> dict:
    """Build feature cache for all files in a split across all bar configs.

    Iterates chronologically through every file × bar config, calling
    load_cached_matrix for each. Once a day is cached, its bar columns
    are read directly for warmup context on subsequent files.

    Args:
        split: Data split to cache ("validate", "train").
        bar_configs: List of bar config dicts. Defaults to BAR_CONFIGS.
        session_filter: "eth" or "rth".
        file_limit: Process only first N files (for testing).
        bar_filter: Single bar label like "tick_610", "vol_2000", "1m".

    Returns:
        Dict with total, completed, errors counts.
    """
    from src.framework.api import ExecutionMode, get_split_files, set_execution_mode
    from src.framework.data.loader import get_execution_mode

    if get_execution_mode() is None:
        set_execution_mode(ExecutionMode.RESEARCH)
    files = get_split_files(split)

    if file_limit is not None:
        files = files[:file_limit]

    configs = bar_configs or BAR_CONFIGS
    if bar_filter:
        configs = [bc for bc in configs if _bar_label(bc) == bar_filter]
        if not configs:
            available = [_bar_label(bc) for bc in (bar_configs or BAR_CONFIGS)]
            raise ValueError(f"Unknown bar_filter '{bar_filter}'. Available: {available}")

    total = len(files) * len(configs)
    completed = 0
    errors = 0

    print(f"Cache build: {len(files)} files x {len(configs)} bar configs = {total} tasks")
    print(f"Session: {session_filter}, Warmup: {WARMUP_CONTEXT_DAYS} days")
    print("=" * 70, flush=True)

    for bc in configs:
        label = _bar_label(bc)
        cache_dir = _feature_cache_dir(bc["bar_size"], bc["bar_type"], bc["bar_threshold"], session_filter)
        print(f"\n--- {label} → {cache_dir.name}/ ---", flush=True)

        for fp in files:
            completed += 1
            feat_path = cache_dir / fp.name

            if feat_path.exists():
                cached = pl.read_parquet(feat_path)
                print(f"  [{completed}/{total}] {fp.name} -> {len(cached)} rows (cached)", flush=True)
                continue

            try:
                df = load_cached_matrix(
                    fp,
                    bar_size=bc["bar_size"],
                    bar_type=bc["bar_type"],
                    bar_threshold=bc["bar_threshold"],
                    include_bar_columns=True,
                    session_filter=session_filter,
                )
                print(
                    f"  [{completed}/{total}] {fp.name} -> {len(df)} rows, "
                    f"{len(df.columns)} cols",
                    flush=True,
                )
            except Exception as e:
                errors += 1
                print(f"  [{completed}/{total}] {fp.name} -> ERROR: {e}", flush=True)

    print(f"\n{'=' * 70}")
    print(f"Done! {completed - errors}/{completed} in {total} tasks")
    if errors:
        print(f"ERRORS: {errors}")

    return {"total": total, "completed": completed, "errors": errors}


def _bar_label(bc: dict) -> str:
    """Human-readable bar config label: tick_610, vol_2000, 1m."""
    if bc["bar_type"] == "time":
        return bc["bar_size"]
    if bc["bar_type"] == "volume":
        return f"vol_{bc['bar_threshold']}"
    return f"{bc['bar_type']}_{bc['bar_threshold']}"
