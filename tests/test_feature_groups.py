"""Tests for A/B experiment feature group filtering."""

import polars as pl
import pytest

from research.lib.feature_groups import (
    OHLCV_FEATURE_COLUMNS,
    filter_feature_group,
)
from src.framework.features_canonical.builder import NON_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_matrix() -> pl.DataFrame:
    """Create a minimal feature matrix with OHLCV + MBP1 + non-feature columns."""
    n = 5
    data = {"ts_event": pl.Series(range(n), dtype=pl.Int64)}

    # Add a few non-feature columns
    data["close"] = pl.Series([100.0] * n)
    data["fwd_return_1bar"] = pl.Series([0.01] * n)

    # Add some OHLCV features
    for col in ["sma_ratio_8", "rsi_14", "return_1bar", "or_width"]:
        data[col] = pl.Series([1.0] * n)

    # Add some MBP1 features
    for col in ["order_flow_imbalance", "book_imbalance", "spread_bps", "cvd"]:
        data[col] = pl.Series([0.5] * n)

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# OHLCV_FEATURE_COLUMNS integrity
# ---------------------------------------------------------------------------


def test_ohlcv_columns_is_frozenset():
    """OHLCV set must be immutable."""
    assert isinstance(OHLCV_FEATURE_COLUMNS, frozenset)


def test_ohlcv_columns_count():
    """Exactly 57 OHLCV feature columns."""
    assert len(OHLCV_FEATURE_COLUMNS) == 57


def test_ohlcv_columns_no_overlap_with_non_features():
    """OHLCV feature columns must not include non-feature columns (labels, raw prices)."""
    overlap = OHLCV_FEATURE_COLUMNS & set(NON_FEATURE_COLUMNS)
    assert overlap == set(), f"OHLCV columns overlap with non-features: {overlap}"


def test_ohlcv_columns_no_mbp1_prefixes():
    """OHLCV features must not include multi-timeframe MBP1 aggregates."""
    mtf_cols = {c for c in OHLCV_FEATURE_COLUMNS if c.startswith("m1f_")}
    assert mtf_cols == set(), f"MTF columns in OHLCV set: {mtf_cols}"


def test_ohlcv_known_indicators_present():
    """Key indicators from the proposal must be in the OHLCV set."""
    expected = {
        "sma_ratio_8", "sma_ratio_21", "sma_ratio_50", "sma_ratio_200",
        "rsi_14", "atr_norm_14", "bb_bandwidth_20", "bb_pctb_20",
        "macd_norm", "adx_14", "obv_slope_14",
    }
    missing = expected - OHLCV_FEATURE_COLUMNS
    assert missing == set(), f"Missing from OHLCV set: {missing}"


def test_ohlcv_excludes_microstructure():
    """Known MBP1 features must NOT be in the OHLCV set."""
    mbp1_examples = {
        "order_flow_imbalance", "book_imbalance", "spread_bps",
        "cvd", "vpin", "absorption_factor", "micro_price_momentum",
        "trade_arrival_imbalance", "weighted_book_imbalance",
    }
    overlap = mbp1_examples & OHLCV_FEATURE_COLUMNS
    assert overlap == set(), f"MBP1 features in OHLCV set: {overlap}"


# ---------------------------------------------------------------------------
# filter_feature_group
# ---------------------------------------------------------------------------


def test_filter_all_returns_unchanged():
    """group='all' returns the full DataFrame."""
    df = _make_dummy_matrix()
    result = filter_feature_group(df, "all")
    assert result.columns == df.columns
    assert len(result) == len(df)


def test_filter_ohlcv_removes_mbp1_features():
    """group='ohlcv' removes MBP1 feature columns."""
    df = _make_dummy_matrix()
    result = filter_feature_group(df, "ohlcv")

    for col in ["order_flow_imbalance", "book_imbalance", "spread_bps", "cvd"]:
        assert col not in result.columns, f"MBP1 column '{col}' should be removed"


def test_filter_ohlcv_keeps_ohlcv_features():
    """group='ohlcv' keeps OHLCV feature columns."""
    df = _make_dummy_matrix()
    result = filter_feature_group(df, "ohlcv")

    for col in ["sma_ratio_8", "rsi_14", "return_1bar", "or_width"]:
        assert col in result.columns, f"OHLCV column '{col}' should be kept"


def test_filter_ohlcv_keeps_non_features():
    """group='ohlcv' keeps non-feature columns (ts_event, close, labels)."""
    df = _make_dummy_matrix()
    result = filter_feature_group(df, "ohlcv")

    assert "ts_event" in result.columns
    assert "close" in result.columns
    assert "fwd_return_1bar" in result.columns


def test_filter_ohlcv_drop_non_features():
    """keep_non_features=False drops non-feature columns too."""
    df = _make_dummy_matrix()
    result = filter_feature_group(df, "ohlcv", keep_non_features=False)

    assert "close" not in result.columns
    assert "fwd_return_1bar" not in result.columns
    # But OHLCV features remain
    assert "rsi_14" in result.columns


def test_filter_invalid_group_raises():
    """Invalid group name raises ValueError."""
    df = _make_dummy_matrix()
    with pytest.raises(ValueError, match="Unknown feature_group"):
        filter_feature_group(df, "invalid")


def test_filter_preserves_row_count():
    """Filtering only removes columns, not rows."""
    df = _make_dummy_matrix()
    result = filter_feature_group(df, "ohlcv")
    assert len(result) == len(df)


def test_filter_preserves_column_order():
    """Filtered columns maintain original order from the DataFrame."""
    df = _make_dummy_matrix()
    result = filter_feature_group(df, "ohlcv")

    original_order = [c for c in df.columns if c in result.columns]
    assert result.columns == original_order


# ---------------------------------------------------------------------------
# Integration: real data (requires NQ_DATA_PATH)
# ---------------------------------------------------------------------------


def test_ohlcv_columns_subset_of_real_features():
    """Every OHLCV column name must exist in the actual feature matrix."""
    from src.framework.api import get_split_files, load_cached_matrix
    from src.framework.features_canonical.builder import get_feature_columns

    files = get_split_files("train")
    df = load_cached_matrix(files[1], bar_size="5m")  # files[0] may be Sunday (no bars)
    real_features = set(get_feature_columns(df))

    missing = OHLCV_FEATURE_COLUMNS - real_features
    assert missing == set(), f"OHLCV columns not in real features: {missing}"


def test_filter_ohlcv_real_data_column_count():
    """Filtered real data should have exactly 57 OHLCV feature columns."""
    from src.framework.api import get_split_files, load_cached_matrix
    from src.framework.features_canonical.builder import get_feature_columns

    files = get_split_files("train")
    df = load_cached_matrix(files[1], bar_size="5m")  # files[0] may be Sunday (no bars)
    filtered = filter_feature_group(df, "ohlcv")

    feat_cols = get_feature_columns(filtered)
    assert len(feat_cols) == 57, f"Expected 57 OHLCV features, got {len(feat_cols)}: {feat_cols}"
