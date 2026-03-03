"""Tests for feature matrix builder."""

import pytest
import polars as pl
from datetime import datetime
from pathlib import Path
import src.framework.features_canonical.builder as builder_mod
from src.framework.features_canonical.builder import build_feature_matrix, get_feature_columns, LABEL_COLUMNS
from src.framework.data.loader import get_parquet_files


def test_build_feature_matrix_columns():
    """Verify feature matrix contains expected columns from all modules."""
    # Load single file
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)

    # Build feature matrix
    result = build_feature_matrix(lf, bar_size="5m")

    assert len(result) > 0, "Feature matrix should not be empty"

    # Check for ts_event
    assert "ts_event" in result.columns, "Missing ts_event"

    # Check for OHLCV from momentum
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in result.columns, f"Missing OHLCV column: {col}"

    # Check for some orderflow features
    for col in ["volume_imbalance", "order_flow_imbalance", "absorption_factor"]:
        assert col in result.columns, f"Missing orderflow column: {col}"
    # Cached pipeline should also include advanced orderflow context features.
    for col in ["ofi_impulse_z", "value_acceptance_rate_8", "structure_break_score"]:
        assert col in result.columns, f"Missing advanced orderflow context column: {col}"

    # Check for some book features
    for col in ["bid_price", "ask_price", "spread", "book_imbalance"]:
        assert col in result.columns, f"Missing book column: {col}"

    # Check for some microstructure features
    for col in ["cancel_trade_ratio", "tape_speed", "price_velocity"]:
        assert col in result.columns, f"Missing microstructure column: {col}"

    # Check for some momentum features
    for col in ["return_1bar", "vwap", "return_5bar"]:
        assert col in result.columns, f"Missing momentum column: {col}"

    # Check for labels
    for col in LABEL_COLUMNS:
        assert col in result.columns, f"Missing label column: {col}"


def test_build_feature_matrix_no_label_nulls():
    """Verify no null values in label columns after build."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)

    result = build_feature_matrix(lf, bar_size="5m")

    # Check that no label columns have nulls
    for label_col in LABEL_COLUMNS:
        null_count = result[label_col].null_count()
        assert null_count == 0, f"Label column '{label_col}' has {null_count} nulls"


def test_build_feature_matrix_sorted():
    """Verify output is sorted by timestamp."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)

    result = build_feature_matrix(lf, bar_size="5m")

    timestamps = result["ts_event"].to_list()
    assert timestamps == sorted(timestamps), "Feature matrix not sorted by ts_event"


def test_get_feature_columns():
    """Verify get_feature_columns returns correct list."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)

    result_df = build_feature_matrix(lf, bar_size="5m")

    # Test with DataFrame
    feature_cols_df = get_feature_columns(result_df)
    assert isinstance(feature_cols_df, list), "Should return a list"
    assert len(feature_cols_df) > 0, "Should have at least some features"

    # Verify no excluded columns
    for col in ["ts_event", "close"] + LABEL_COLUMNS:
        assert col not in feature_cols_df, f"Excluded column '{col}' should not be in feature list"

    # Verify some expected features are included
    for col in ["volume_imbalance", "book_imbalance", "return_1bar"]:
        assert col in feature_cols_df, f"Expected feature '{col}' not in list"


def test_get_feature_columns_no_labels():
    """Verify labels are excluded from feature columns."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)

    result = build_feature_matrix(lf, bar_size="5m")
    feature_cols = get_feature_columns(result)

    # Check that NONE of the label columns are in features
    for label_col in LABEL_COLUMNS:
        assert label_col not in feature_cols, \
            f"Label column '{label_col}' should NOT be in feature columns"


def test_build_feature_matrix_synthetic():
    """Test feature matrix builder with synthetic data."""
    # Create minimal synthetic data for one 5m bar
    synthetic_data = pl.DataFrame({
        "ts_event": [datetime(2024, 7, 15, 13, 30, 0, i) for i in range(20)],
        "ts_recv": [datetime(2024, 7, 15, 13, 30, 0, i + 1000) for i in range(20)],
        "action": (["T"] * 10 + ["A", "C", "M"] * 3 + ["T"]),  # Mix of actions
        "side": (["A", "B"] * 10),  # Alternating buy/sell
        "price": [18000.0 + i * 0.25 for i in range(20)],  # Incrementing prices
        "size": [5] * 20,
        "bid_px_00": [17999.75] * 20,
        "ask_px_00": [18000.00] * 20,
        "bid_sz_00": [50] * 20,
        "ask_sz_00": [45] * 20,
        "bid_ct_00": [10] * 20,
        "ask_ct_00": [8] * 20,
        "depth": [0] * 20,
        "flags": [0] * 20,
        "sequence": list(range(20)),
        "ts_in_delta": [1000] * 20,
        "rtype": [0] * 20,
        "publisher_id": [1] * 20,
        "instrument_id": [1000] * 20,
        "symbol": ["NQ.c.0"] * 20,
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("ts_recv").dt.replace_time_zone("UTC")
    )

    lf = synthetic_data.lazy()
    result = build_feature_matrix(lf, bar_size="5m")

    # For 1 bar of data, forward returns will be null, so result should be empty
    # after the label null filter — this is expected behavior
    assert len(result) == 0, "Single bar should produce empty matrix (no valid labels)"


def test_build_feature_matrix_multiple_bars_synthetic():
    """Test with enough synthetic bars to have valid labels."""
    # Create 20 bars worth of synthetic trade data (enough for 12-bar forward return)
    from datetime import timedelta
    timestamps = []
    base_time = datetime(2024, 7, 15, 13, 30, 0)
    for bar_idx in range(20):
        # 10 trades per 5-minute bar
        bar_time = base_time + timedelta(minutes=bar_idx * 5)
        timestamps.extend([bar_time] * 10)

    synthetic_data = pl.DataFrame({
        "ts_event": timestamps,
        "ts_recv": timestamps,  # Simplified
        "action": ["T"] * len(timestamps),
        "side": (["A", "B"] * (len(timestamps) // 2 + 1))[:len(timestamps)],
        "price": [18000.0 + i * 0.25 for i in range(len(timestamps))],
        "size": [5] * len(timestamps),
        "bid_px_00": [17999.75] * len(timestamps),
        "ask_px_00": [18000.00] * len(timestamps),
        "bid_sz_00": [50] * len(timestamps),
        "ask_sz_00": [45] * len(timestamps),
        "bid_ct_00": [10] * len(timestamps),
        "ask_ct_00": [8] * len(timestamps),
        "depth": [0] * len(timestamps),
        "flags": [0] * len(timestamps),
        "sequence": list(range(len(timestamps))),
        "ts_in_delta": [1000] * len(timestamps),
        "rtype": [0] * len(timestamps),
        "publisher_id": [1] * len(timestamps),
        "instrument_id": [1000] * len(timestamps),
        "symbol": ["NQ.c.0"] * len(timestamps),
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("ts_recv").dt.replace_time_zone("UTC")
    )

    lf = synthetic_data.lazy()
    result = build_feature_matrix(lf, bar_size="5m")

    # Should have some bars (20 - 12 = 8 bars minimum after dropping edge nulls)
    assert len(result) > 0, "Should produce bars with valid labels"

    # Verify no nulls in labels
    for label_col in LABEL_COLUMNS:
        null_count = result[label_col].null_count()
        assert null_count == 0, f"Label '{label_col}' should have no nulls"

    # Verify we can extract features
    feature_cols = get_feature_columns(result)
    assert len(feature_cols) > 20, "Should have many features from all modules"


def test_build_feature_matrix_include_bar_columns_synthetic():
    """include_bar_columns=True should expose raw flow/bar internals for strategies."""
    from datetime import timedelta

    timestamps = []
    base_time = datetime(2024, 7, 15, 13, 30, 0)
    for bar_idx in range(20):
        bar_time = base_time + timedelta(minutes=bar_idx * 5)
        timestamps.extend([bar_time] * 10)

    synthetic_data = pl.DataFrame({
        "ts_event": timestamps,
        "ts_recv": timestamps,
        "action": ["T"] * len(timestamps),
        "side": (["A", "B"] * (len(timestamps) // 2 + 1))[:len(timestamps)],
        "price": [18000.0 + i * 0.25 for i in range(len(timestamps))],
        "size": [5] * len(timestamps),
        "bid_px_00": [17999.75] * len(timestamps),
        "ask_px_00": [18000.00] * len(timestamps),
        "bid_sz_00": [50] * len(timestamps),
        "ask_sz_00": [45] * len(timestamps),
        "bid_ct_00": [10] * len(timestamps),
        "ask_ct_00": [8] * len(timestamps),
        "depth": [0] * len(timestamps),
        "flags": [0] * len(timestamps),
        "sequence": list(range(len(timestamps))),
        "ts_in_delta": [1000] * len(timestamps),
        "rtype": [0] * len(timestamps),
        "publisher_id": [1] * len(timestamps),
        "instrument_id": [1000] * len(timestamps),
        "symbol": ["NQ.c.0"] * len(timestamps),
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("ts_recv").dt.replace_time_zone("UTC")
    )

    lf = synthetic_data.lazy()
    result = build_feature_matrix(lf, bar_size="5m", include_bar_columns=True)

    for col in ("bar_duration_ns", "trade_count", "buy_volume", "sell_volume"):
        assert col in result.columns, f"Missing strategy passthrough column: {col}"

    feature_cols = get_feature_columns(result)
    for col in ("trade_count", "buy_volume", "sell_volume"):
        assert col not in feature_cols, f"Raw flow column should not be ML feature: {col}"



def test_build_feature_matrix_real_data_end_to_end():
    """End-to-end test with real data from train split."""
    files = get_parquet_files("train")
    # Use second file (files[0] may be Sunday with no bars)
    lf = pl.scan_parquet(str(files[1]))

    # Build feature matrix
    result = build_feature_matrix(lf, bar_size="5m")

    # Basic assertions
    assert len(result) > 0, "Should produce bars"
    assert len(result) > 50, "Should have reasonable number of bars for a full day"

    # Verify shape
    assert result.shape[0] > 0, "Should have rows"
    assert result.shape[1] > 40, "Should have many columns from all modules"

    # Verify no nulls in labels
    for label_col in LABEL_COLUMNS:
        null_count = result[label_col].null_count()
        assert null_count == 0, f"Label '{label_col}' has nulls in real data"

    # Verify timestamps are in order
    timestamps = result["ts_event"].to_list()
    assert timestamps == sorted(timestamps), "Timestamps not sorted"

    # Verify features can be extracted
    feature_cols = get_feature_columns(result)
    assert len(feature_cols) > 30, "Should have substantial number of features"

    # Verify we can separate X and y
    X_cols = feature_cols
    y_cols = LABEL_COLUMNS

    # No overlap between features and labels
    assert len(set(X_cols) & set(y_cols)) == 0, "Features and labels should not overlap"

    # Together they should account for most columns (except ts_event + raw prices + session VP)
    from src.framework.features_canonical.builder import RAW_PRICE_COLUMNS, SESSION_VP_COLUMNS, BAR_META_COLUMNS
    all_cols = set(result.columns)
    feature_label_cols = set(X_cols) | set(y_cols)
    excluded = {"ts_event"} | set(RAW_PRICE_COLUMNS) | set(SESSION_VP_COLUMNS) | set(BAR_META_COLUMNS)
    remaining = all_cols - feature_label_cols - excluded
    assert len(remaining) == 0, f"Unexpected remaining columns: {remaining}"


def test_build_feature_matrix_no_feature_nulls():
    """Verify no null values in feature columns after warmup."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)

    result = build_feature_matrix(lf, bar_size="5m")
    feature_cols = get_feature_columns(result)

    # Check that no feature columns have nulls (except known nullable features)
    # mean_trade_interval_us: null when bar has 0 or 1 trades
    # cancel_rate_change: null when previous bar has 0 cancels
    # regime_autocorr: null during warmup (rolling_corr needs min_samples + lag)
    # latency_mean: always null (placeholder — needs per-event ts_recv - ts_event)
    # OR features: null during 30-min OR formation period
    # gap_open: null on first session day (no previous session)
    nullable_features = {
        "mean_trade_interval_us", "cancel_rate_change", "regime_autocorr",
        "latency_mean",
        # Tape speed / price velocity Z-scores: null on first bar (rolling_std needs min 2 samples)
        "tape_speed_z", "price_velocity_z",
        "or_width", "position_in_or", "or_broken_up", "or_broken_down",
        "gap_open",
        # Previous session: null on first session (no previous day data)
        "dist_prev_high", "dist_prev_low",
        # Previous session VP: null on first session (no previous day VP)
        "dist_prev_poc", "dist_prev_vah", "dist_prev_val", "prev_day_va_position",
        # Failed auction: null when prev_day_vah/val are null (first session)
        "failed_auction_bull", "failed_auction_bear", "failed_auction_score",
        # CVD dynamics: shift-based warmup nulls
        "cvd_slope_3", "cvd_slope_6", "cvd_slope_12",
        "cvd_accel_3", "cvd_accel_6", "cvd_price_alignment",
        # LVN/HVN distance: NaN when no HVN/LVN exists in rolling window
        "dist_nearest_hvn", "dist_nearest_lvn",
        # Swing VP: NaN when no breakout or no LVN/HVN in swing profile
        "swing_poc_dist", "swing_lvn_dist", "swing_hvn_dist",
        "swing_va_position", "bars_since_breakout",
        # Accumulation: rolling_std warmup nulls
        "range_compression", "range_compression_z",
        # OHLCV indicators: rolling/ewm warmup nulls (SMA-200 needs 200 bars, etc.)
        "sma_ratio_8", "sma_ratio_21", "sma_ratio_50", "sma_ratio_200",
        "ema_ratio_8", "ema_ratio_21", "ema_ratio_50",
        "rsi_14", "atr_norm_14",
        "bb_bandwidth_20", "bb_pctb_20",
        "macd_norm", "macd_signal_norm", "macd_hist_norm",
        "stoch_k_14", "stoch_d_14",
        "adx_14", "plus_di_14", "minus_di_14",
        "obv_slope_14",
    }
    for col in feature_cols:
        if col in nullable_features:
            continue
        # MTF features may have nulls at edges (partial 5m windows)
        if col.startswith("m1f_"):
            continue
        null_count = result[col].null_count()
        assert null_count == 0, f"Feature '{col}' has {null_count} nulls"


def test_build_feature_matrix_no_infinities():
    """Verify no infinite values in any feature column."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)

    result = build_feature_matrix(lf, bar_size="5m")
    feature_cols = get_feature_columns(result)

    for col in feature_cols:
        if result[col].dtype in [pl.Float32, pl.Float64]:
            inf_count = result.filter(pl.col(col).is_infinite()).height
            assert inf_count == 0, f"Feature '{col}' has {inf_count} infinite values"


def test_raw_prices_excluded_from_features():
    """Verify raw non-stationary prices are not in feature columns."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)

    result = build_feature_matrix(lf, bar_size="5m")
    feature_cols = get_feature_columns(result)

    from src.framework.features_canonical.builder import RAW_PRICE_COLUMNS
    for col in RAW_PRICE_COLUMNS:
        assert col not in feature_cols, f"Raw price '{col}' should not be a feature"


def test_session_vp_excluded_from_features():
    """Verify session-level VP features (lookahead) are excluded from feature columns."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)

    result = build_feature_matrix(lf, bar_size="5m")
    feature_cols = get_feature_columns(result)

    from src.framework.features_canonical.builder import SESSION_VP_COLUMNS
    for col in SESSION_VP_COLUMNS:
        if col in result.columns:
            assert col not in feature_cols, f"Session VP '{col}' should not be a feature"


def test_build_feature_matrix_tick_bars_synthetic():
    """Tick bars should build end-to-end with strategy passthrough columns."""
    # 300 trades with threshold=10 => 30 bars, enough for forward-label windows
    timestamps = [datetime(2024, 7, 15, 13, 30, 0, i) for i in range(300)]
    synthetic_data = pl.DataFrame({
        "ts_event": timestamps,
        "ts_recv": timestamps,
        "action": ["T"] * len(timestamps),
        "side": (["A", "B"] * (len(timestamps) // 2 + 1))[:len(timestamps)],
        "price": [18000.0 + i * 0.25 for i in range(len(timestamps))],
        "size": [5] * len(timestamps),
        "bid_px_00": [17999.75] * len(timestamps),
        "ask_px_00": [18000.00] * len(timestamps),
        "bid_sz_00": [50] * len(timestamps),
        "ask_sz_00": [45] * len(timestamps),
        "bid_ct_00": [10] * len(timestamps),
        "ask_ct_00": [8] * len(timestamps),
        "depth": [0] * len(timestamps),
        "flags": [0] * len(timestamps),
        "sequence": list(range(len(timestamps))),
        "ts_in_delta": [1000] * len(timestamps),
        "rtype": [0] * len(timestamps),
        "publisher_id": [1] * len(timestamps),
        "instrument_id": [1000] * len(timestamps),
        "symbol": ["NQ.c.0"] * len(timestamps),
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("ts_recv").dt.replace_time_zone("UTC"),
    )

    result = build_feature_matrix(
        synthetic_data.lazy(),
        bar_type="tick",
        bar_threshold=10,
        include_bar_columns=True,
    )
    assert len(result) > 0, "Tick-bar feature matrix should not be empty"

    for col in ("bar_duration_ns", "trade_count", "buy_volume", "sell_volume"):
        assert col in result.columns, f"Missing tick strategy passthrough column: {col}"


def test_build_feature_matrix_tick_threshold_validation():
    """Tick bars should reject non-positive threshold values."""
    synthetic_data = pl.DataFrame({
        "ts_event": [datetime(2024, 7, 15, 13, 30, 0)],
        "ts_recv": [datetime(2024, 7, 15, 13, 30, 0)],
        "action": ["T"],
        "side": ["A"],
        "price": [18000.0],
        "size": [1],
        "bid_px_00": [17999.75],
        "ask_px_00": [18000.00],
        "bid_sz_00": [50],
        "ask_sz_00": [45],
        "bid_ct_00": [10],
        "ask_ct_00": [8],
        "depth": [0],
        "flags": [0],
        "sequence": [1],
        "ts_in_delta": [1000],
        "rtype": [0],
        "publisher_id": [1],
        "instrument_id": [1000],
        "symbol": ["NQ.c.0"],
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("ts_recv").dt.replace_time_zone("UTC"),
    )

    with pytest.raises(ValueError, match="bar_threshold must be > 0 for tick bars"):
        build_feature_matrix(synthetic_data.lazy(), bar_type="tick", bar_threshold=0)


def test_load_cached_matrix_stores_unfiltered_cache_and_returns_filtered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Cache should keep label-null tail rows; returned frame should filter them."""
    files = get_parquet_files("train")
    file_path = files[1]  # first full trading day

    monkeypatch.setattr(builder_mod, "CACHE_DIR", tmp_path)

    returned = builder_mod.load_cached_matrix(
        file_path,
        bar_size="5m",
        bar_type="time",
        session_filter="eth",
        include_bar_columns=False,
    )
    assert len(returned) > 0
    for col in LABEL_COLUMNS:
        assert returned[col].null_count() == 0, f"Returned frame should filter null labels: {col}"

    cache_path = builder_mod._feature_cache_dir("5m", "time", None, "eth") / file_path.name
    cached = pl.read_parquet(cache_path)
    assert len(cached) >= len(returned)
    assert any(cached[col].null_count() > 0 for col in LABEL_COLUMNS), \
        "Cached frame should retain unfiltered label-null tail rows"

    # Cache hit should still return filtered output (filters applied at read time).
    returned_cached = builder_mod.load_cached_matrix(
        file_path,
        bar_size="5m",
        bar_type="time",
        session_filter="eth",
        include_bar_columns=False,
    )
    for col in LABEL_COLUMNS:
        assert returned_cached[col].null_count() == 0, \
            f"Cache-hit output should filter null labels: {col}"


def test_load_cached_matrix_rebuilds_stale_cache_for_include_bar_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Existing cache without bar passthrough columns must be rebuilt when requested."""
    files = get_parquet_files("train")
    file_path = files[1]

    monkeypatch.setattr(builder_mod, "CACHE_DIR", tmp_path)

    _ = builder_mod.load_cached_matrix(
        file_path,
        bar_size="5m",
        bar_type="time",
        session_filter="eth",
        include_bar_columns=False,
    )

    returned = builder_mod.load_cached_matrix(
        file_path,
        bar_size="5m",
        bar_type="time",
        session_filter="eth",
        include_bar_columns=True,
    )
    required = {
        "ts_close",
        "bar_duration_ns",
        "trade_count",
        "buy_volume",
        "sell_volume",
        "vap_prices",
        "vap_volumes",
    }
    missing = required - set(returned.columns)
    assert not missing, f"Rebuilt output missing required bar columns: {sorted(missing)}"
