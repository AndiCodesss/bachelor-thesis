"""Tests for interaction, regime, time-of-day, and previous session features."""

import polars as pl
from datetime import datetime

from src.framework.features_canonical.pipeline import compute_pipeline_features, REQUIRED_COLUMNS


def _make_matrix(n_rows, **overrides):
    """Build a synthetic joined feature matrix with sensible defaults."""
    defaults = {
        "range_ma5": [0.001] * n_rows,
        "return_1bar": [0.0005] * n_rows,
        "order_flow_imbalance": [100.0] * n_rows,
        "spread_bps": [1.5] * n_rows,
        "book_imbalance": [0.2] * n_rows,
        "volume_delta": [50] * n_rows,
    }
    defaults.update(overrides)
    return pl.DataFrame(defaults)


def _make_matrix_with_time(n_bars_per_day=5, n_days=2, base_price=21000.0):
    """Build a synthetic joined matrix with UTC timestamps during RTH.

    Creates bars starting at 14:30 UTC (09:30 ET) each day.
    """
    rows = []
    datetime(2024, 7, 15)
    for day in range(n_days):
        for bar in range(n_bars_per_day):
            # 14:30 UTC = 09:30 ET (EST+5, summer EDT+4 but July is EDT)
            # July: EDT = UTC-4, so 09:30 ET = 13:30 UTC
            hour = 13 + (bar * 5) // 60
            minute = 30 + (bar * 5) % 60
            if minute >= 60:
                hour += 1
                minute -= 60
            ts = datetime(2024, 7, 15 + day, hour, minute, 0)

            price = base_price + day * 50 + bar * 2.0
            rows.append({
                "ts_event": ts,
                "open": price - 1.0,
                "high": price + 2.0,
                "low": price - 2.0,
                "close": price,
                "range_ma5": 0.001,
                "return_1bar": 0.0005,
                "order_flow_imbalance": 100.0,
                "spread_bps": 1.5,
                "book_imbalance": 0.2,
                "volume_delta": 50,
            })

    df = pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )
    return df


def _make_orderflow_context_matrix(n_rows=20):
    rows = []
    for i in range(n_rows):
        close = 21000.0 + i
        row = {
            "open": close - 0.25,
            "high": close + 0.75,
            "low": close - 0.75,
            "close": close,
            "volume": 100.0,
            "range_ma5": 0.001,
            "return_1bar": 0.0005,
            "order_flow_imbalance": 100.0,
            "volume_imbalance": 0.15,
            "spread_bps": 1.5,
            "book_imbalance": 0.2,
            "volume_delta": 50.0,
            "rolling_va_position": 0.5,
            "whale_buy_volume_30": 20.0,
            "whale_sell_volume_30": 10.0,
        }
        rows.append(row)

    # Create one bullish structure break near VAL and one bearish rejection near VAH.
    rows[-2]["rolling_va_position"] = 0.02
    rows[-2]["low"] = rows[-2]["close"] - 1.5
    rows[-2]["order_flow_imbalance"] = 150.0
    rows[-2]["volume_imbalance"] = 0.25

    rows[-1]["rolling_va_position"] = 0.98
    rows[-1]["high"] = rows[-1]["close"] + 1.5
    rows[-1]["order_flow_imbalance"] = -180.0
    rows[-1]["volume_imbalance"] = -0.35
    rows[-1]["whale_buy_volume_30"] = 10.0
    rows[-1]["whale_sell_volume_30"] = 35.0
    return pl.DataFrame(rows)


def test_regime_vol_relative_constant():
    """Constant volatility -> ratio = 1.0 after warmup."""
    df = _make_matrix(60)
    result = compute_pipeline_features(df)

    assert "regime_vol_relative" in result.columns
    # After warmup, constant range_ma5 / median(range_ma5) = 1.0
    vals = result["regime_vol_relative"].to_list()
    # All bars should be 1.0 (constant / constant)
    for i, v in enumerate(vals):
        assert v is not None, f"Null at row {i}"
        assert abs(v - 1.0) < 1e-6, f"Expected 1.0 at row {i}, got {v}"


def test_regime_vol_relative_increasing():
    """Increasing volatility -> ratio > 1.0 for later bars."""
    n = 60
    # Linearly increasing volatility
    range_vals = [0.001 + i * 0.0001 for i in range(n)]
    df = _make_matrix(n, range_ma5=range_vals)
    result = compute_pipeline_features(df)

    vals = result["regime_vol_relative"].to_list()
    # Last bar should have ratio > 1.0 (current > historical median)
    assert vals[-1] > 1.0, f"Expected > 1.0 for last bar, got {vals[-1]}"
    # First bar: 0.001 / median([0.001]) = 1.0
    assert abs(vals[0] - 1.0) < 1e-6


def test_regime_autocorr_trending():
    """Consistent positive returns -> positive autocorrelation."""
    n = 30
    # All positive returns (trending up)
    returns = [0.001] * n
    df = _make_matrix(n, return_1bar=returns)
    result = compute_pipeline_features(df)

    # With constant returns, correlation is undefined (zero variance)
    # Use slightly varying positive returns instead
    returns = [0.001 + i * 0.00001 for i in range(n)]
    df = _make_matrix(n, return_1bar=returns)
    result = compute_pipeline_features(df)

    vals = result["regime_autocorr"].to_list()
    # After warmup (need REGIME_AUTOCORR_WINDOW bars + 1 for lag), should be positive
    last_val = vals[-1]
    assert last_val is not None, "Autocorr should not be null at end"
    assert last_val > 0, f"Expected positive autocorr for trending returns, got {last_val}"


def test_regime_autocorr_alternating():
    """Alternating returns -> negative autocorrelation."""
    n = 30
    # Alternating positive/negative returns (mean-reverting)
    returns = [0.001 if i % 2 == 0 else -0.001 for i in range(n)]
    df = _make_matrix(n, return_1bar=returns)
    result = compute_pipeline_features(df)

    vals = result["regime_autocorr"].to_list()
    last_val = vals[-1]
    assert last_val is not None, "Autocorr should not be null at end"
    assert last_val < 0, f"Expected negative autocorr for alternating returns, got {last_val}"


def test_ix_ofi_x_vol_product():
    """ix_ofi_x_vol = order_flow_imbalance * range_ma5."""
    df = _make_matrix(5, order_flow_imbalance=[200.0] * 5, range_ma5=[0.002] * 5)
    result = compute_pipeline_features(df)

    vals = result["ix_ofi_x_vol"].to_list()
    for v in vals:
        assert abs(v - (200.0 * 0.002)) < 1e-9, f"Expected {200.0 * 0.002}, got {v}"


def test_ix_ofi_x_spread_product():
    """ix_ofi_x_spread = order_flow_imbalance * spread_bps."""
    df = _make_matrix(5, order_flow_imbalance=[-50.0] * 5, spread_bps=[3.0] * 5)
    result = compute_pipeline_features(df)

    vals = result["ix_ofi_x_spread"].to_list()
    for v in vals:
        assert abs(v - (-50.0 * 3.0)) < 1e-9, f"Expected {-50.0 * 3.0}, got {v}"


def test_ix_bbo_imb_x_vdelta_product():
    """ix_bbo_imb_x_vdelta = book_imbalance * volume_delta."""
    df = _make_matrix(5, book_imbalance=[0.5] * 5, volume_delta=[100] * 5)
    result = compute_pipeline_features(df)

    vals = result["ix_bbo_imb_x_vdelta"].to_list()
    for v in vals:
        assert abs(v - (0.5 * 100)) < 1e-9, f"Expected {0.5 * 100}, got {v}"


def test_null_propagation():
    """Null input features produce null output (Polars default for arithmetic)."""
    df = pl.DataFrame({
        "range_ma5": [0.001, None, 0.001],
        "return_1bar": [0.001, 0.001, None],
        "order_flow_imbalance": [100.0, None, 100.0],
        "spread_bps": [1.5, 1.5, 1.5],
        "book_imbalance": [0.2, 0.2, None],
        "volume_delta": [50, 50, 50],
    })
    result = compute_pipeline_features(df)

    # Row 1: null range_ma5 -> null regime_vol_relative, null ofi -> null ix_ofi_*
    assert result["regime_vol_relative"][1] is None
    assert result["ix_ofi_x_vol"][1] is None
    assert result["ix_ofi_x_spread"][1] is None

    # Row 2: null book_imbalance -> null ix_bbo_imb_x_vdelta
    assert result["ix_bbo_imb_x_vdelta"][2] is None


def test_output_columns_exist():
    """All expected pipeline columns should be present in output."""
    df = _make_matrix(10)
    result = compute_pipeline_features(df)

    expected = [
        "regime_vol_relative",
        "regime_autocorr",
        "ix_ofi_x_vol",
        "ix_ofi_x_spread",
        "ix_bbo_imb_x_vdelta",
    ]
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"


def test_orderflow_context_columns_present():
    df = _make_orderflow_context_matrix(24)
    result = compute_pipeline_features(df)
    expected = [
        "ofi_impulse_z",
        "whale_imbalance_ratio_30",
        "whale_participation_rate_30",
        "whale_ofi_alignment",
        "value_acceptance_rate_8",
        "va_rejection_score",
        "price_discovery_score",
        "structure_break_score",
    ]
    for col in expected:
        assert col in result.columns, f"Missing orderflow context column: {col}"


def test_orderflow_context_known_ratios():
    df = _make_orderflow_context_matrix(24)
    result = compute_pipeline_features(df)
    # whale imbalance ratio = (20 - 10) / (20 + 10 + 1) = 10/31
    assert abs(float(result["whale_imbalance_ratio_30"][0]) - (10.0 / 31.0)) < 1e-9
    # whale participation = (20 + 10) / (100 + 1) = 30/101
    assert abs(float(result["whale_participation_rate_30"][0]) - (30.0 / 101.0)) < 1e-9


def test_orderflow_context_rejection_and_structure_signs():
    df = _make_orderflow_context_matrix(24)
    result = compute_pipeline_features(df)
    # Near VAH + negative aligned flow should score bearish rejection.
    assert float(result["va_rejection_score"][-1]) < 0.0
    # With rising closes, last-2 row should break prior structure with positive flow.
    assert float(result["structure_break_score"][-2]) > 0.0


def test_no_intermediate_columns_leaked():
    """Temporary _return_lag1 should not appear in output."""
    df = _make_matrix(10)
    result = compute_pipeline_features(df)
    assert "_return_lag1" not in result.columns


def test_input_columns_preserved():
    """All original input columns should still be present."""
    df = _make_matrix(10)
    result = compute_pipeline_features(df)
    for col in REQUIRED_COLUMNS:
        assert col in result.columns, f"Input column lost: {col}"


def test_row_count_preserved():
    """Output should have the same number of rows as input."""
    n = 25
    df = _make_matrix(n)
    result = compute_pipeline_features(df)
    assert len(result) == n, f"Expected {n} rows, got {len(result)}"


def test_single_row():
    """Single-row input should not crash."""
    df = _make_matrix(1)
    result = compute_pipeline_features(df)
    assert len(result) == 1
    # regime_vol_relative: 0.001 / median([0.001]) = 1.0
    assert abs(result["regime_vol_relative"][0] - 1.0) < 1e-6
    # regime_autocorr: null (not enough samples)
    assert result["regime_autocorr"][0] is None


# ---- Time-of-day feature tests ----

def test_minutes_since_open_at_open():
    """09:30 ET = 0 minutes since open."""
    # July 2024 is EDT (UTC-4), so 09:30 ET = 13:30 UTC
    df = _make_matrix_with_time(n_bars_per_day=1, n_days=1)
    result = compute_pipeline_features(df)

    assert "minutes_since_open" in result.columns
    assert result["minutes_since_open"][0] == 0.0


def test_minutes_since_open_after_5_bars():
    """Each bar is 5 minutes apart, so bar 3 (index 2) = 10 minutes."""
    df = _make_matrix_with_time(n_bars_per_day=5, n_days=1)
    result = compute_pipeline_features(df)

    mins = result["minutes_since_open"].to_list()
    assert mins[0] == 0.0
    assert mins[1] == 5.0
    assert mins[2] == 10.0
    assert mins[3] == 15.0
    assert mins[4] == 20.0


def test_session_progress_bounded():
    """session_progress should be 0.0 at open and increase."""
    df = _make_matrix_with_time(n_bars_per_day=5, n_days=1)
    result = compute_pipeline_features(df)

    progress = result["session_progress"].to_list()
    assert abs(progress[0] - 0.0) < 1e-9  # at open
    assert progress[-1] > progress[0]  # increases
    for p in progress:
        assert p >= 0.0


def test_is_power_hour_first_90_min():
    """Power hour = first 90 minutes (09:30-11:00 ET)."""
    # Create bars covering more time: every 30 min for 3 hours
    rows = []
    for i in range(6):  # 0, 30, 60, 90, 120, 150 minutes after open
        hour = 13 + (30 * i) // 60
        minute = 30 + (30 * i) % 60
        if minute >= 60:
            hour += 1
            minute -= 60
        ts = datetime(2024, 7, 15, hour, minute, 0)
        rows.append({
            "ts_event": ts,
            "open": 21000.0, "high": 21002.0, "low": 20998.0, "close": 21000.0,
            "range_ma5": 0.001, "return_1bar": 0.0005,
            "order_flow_imbalance": 100.0, "spread_bps": 1.5,
            "book_imbalance": 0.2, "volume_delta": 50,
        })

    df = pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )
    result = compute_pipeline_features(df)

    ph = result["is_power_hour"].to_list()
    # min 0 (09:30): power hour
    assert ph[0] == 1.0
    # min 30 (10:00): power hour
    assert ph[1] == 1.0
    # min 60 (10:30): power hour
    assert ph[2] == 1.0
    # min 90 (11:00): NOT power hour (end is exclusive)
    assert ph[3] == 0.0
    # min 120 (11:30): not power hour
    assert ph[4] == 0.0


def test_no_time_features_without_ts_event():
    """When ts_event is absent, time features should not be added."""
    df = _make_matrix(5)
    result = compute_pipeline_features(df)
    assert "minutes_since_open" not in result.columns
    assert "is_power_hour" not in result.columns
    assert "session_progress" not in result.columns


# ---- Previous session feature tests ----

def test_prev_day_values_null_for_first_session():
    """First session has no previous day data (no lookahead)."""
    df = _make_matrix_with_time(n_bars_per_day=3, n_days=1)
    result = compute_pipeline_features(df)

    for col in ["prev_day_high", "prev_day_low", "prev_day_close"]:
        assert col in result.columns
        for i in range(len(result)):
            assert result[col][i] is None, f"{col} should be null on first day, bar {i}"


def test_prev_day_values_from_previous_session():
    """Second session should have previous day's high/low/close."""
    df = _make_matrix_with_time(n_bars_per_day=3, n_days=2, base_price=21000.0)
    result = compute_pipeline_features(df)

    # Day 1 prices: 21000, 21002, 21004
    # Day 1 high = max(high) = 21004 + 2 = 21006
    # Day 1 low = min(low) = 21000 - 2 = 20998
    # Day 1 close = last close = 21004

    # Day 2 bars should have prev_day values from Day 1
    for i in range(3, 6):  # day 2 bars
        assert result["prev_day_high"][i] is not None, f"prev_day_high null at bar {i}"
        assert result["prev_day_high"][i] == 21006.0
        assert result["prev_day_low"][i] == 20998.0
        assert result["prev_day_close"][i] == 21004.0


def test_gap_open_calculation():
    """gap_open = (today_open - prev_close) / prev_close."""
    df = _make_matrix_with_time(n_bars_per_day=3, n_days=2, base_price=21000.0)
    result = compute_pipeline_features(df)

    # Day 1 close = 21004, Day 2 first open = 21050 - 1 = 21049
    # gap = (21049 - 21004) / 21004
    assert result["gap_open"][0] is None  # first day
    expected_gap = (21049.0 - 21004.0) / 21004.0
    actual_gap = result["gap_open"][3]
    assert actual_gap is not None
    assert abs(actual_gap - expected_gap) < 1e-6


def test_dist_prev_high_sign():
    """dist_prev_high positive when close > prev high, negative below."""
    df = _make_matrix_with_time(n_bars_per_day=3, n_days=2, base_price=21000.0)
    result = compute_pipeline_features(df)

    # Day 2, bar 0: close = 21050, prev_day_high = 21006
    # So dist_prev_high = (21050 - 21006) / 0.001 > 0
    assert result["dist_prev_high"][3] > 0


def test_dist_prev_low_sign():
    """dist_prev_low positive when close > prev low."""
    df = _make_matrix_with_time(n_bars_per_day=3, n_days=2, base_price=21000.0)
    result = compute_pipeline_features(df)

    # All day 2 closes above day 1 low (20998)
    for i in range(3, 6):
        assert result["dist_prev_low"][i] > 0


def test_prev_session_no_intermediate_columns():
    """Intermediate _session_date and _ts_eastern should not leak."""
    df = _make_matrix_with_time(n_bars_per_day=3, n_days=2)
    result = compute_pipeline_features(df)

    assert "_session_date" not in result.columns
    assert "_ts_eastern" not in result.columns


def test_row_count_preserved_with_time():
    """Row count must match input even with time features."""
    df = _make_matrix_with_time(n_bars_per_day=5, n_days=3)
    result = compute_pipeline_features(df)
    assert len(result) == 15


# ---- Accumulation detection + micro-stall tests ----

def _make_accum_matrix(n_rows, **overrides):
    """Build a matrix with high_low_range, absorption_signal, and recoil_50pct."""
    defaults = {
        "range_ma5": [0.001] * n_rows,
        "return_1bar": [0.0005] * n_rows,
        "order_flow_imbalance": [100.0] * n_rows,
        "spread_bps": [1.5] * n_rows,
        "book_imbalance": [0.2] * n_rows,
        "volume_delta": [50] * n_rows,
        "high_low_range": [0.001] * n_rows,
        "absorption_signal": [0.0] * n_rows,
        "recoil_50pct": [0.0] * n_rows,
    }
    defaults.update(overrides)
    return pl.DataFrame(defaults)


def test_accum_columns_present():
    """Accumulation feature columns should be present when high_low_range exists."""
    df = _make_accum_matrix(20)
    result = compute_pipeline_features(df)

    for col in ["range_compression", "range_compression_z",
                "volatility_compression_1bar",
                "post_absorption_contraction",
                "post_recoil_stall"]:
        assert col in result.columns, f"Missing column: {col}"


def test_accum_columns_absent_without_high_low_range():
    """Without high_low_range, accumulation features should not be added."""
    df = _make_matrix(10)
    result = compute_pipeline_features(df)
    assert "range_compression" not in result.columns
    assert "post_absorption_contraction" not in result.columns


def test_range_compression_constant_range():
    """Constant bar range => std=0, so range_compression ~ 0 (or null for first bar)."""
    df = _make_accum_matrix(10, high_low_range=[0.002] * 10)
    result = compute_pipeline_features(df)

    # With constant range, rolling_std = 0, so CV = 0/mean = 0
    rc = result["range_compression"].to_list()
    # First bar has null std (need min 2 samples), rest should be 0
    for i in range(2, len(rc)):
        if rc[i] is not None:
            assert abs(rc[i]) < 1e-6, f"Expected ~0 range_compression at bar {i}, got {rc[i]}"


def test_range_compression_varying_range():
    """Varying bar range should have positive range_compression."""
    ranges = [0.001, 0.003, 0.001, 0.003, 0.001, 0.003, 0.001, 0.003, 0.001, 0.003]
    df = _make_accum_matrix(10, high_low_range=ranges)
    result = compute_pipeline_features(df)

    rc = result["range_compression"].to_list()
    # After warmup, CV should be positive (high variation)
    valid = [v for v in rc if v is not None]
    assert len(valid) > 0
    assert valid[-1] > 0, f"Expected positive range_compression, got {valid[-1]}"


def test_range_compression_z_negative_for_tight():
    """When current range_compression is below its rolling mean, z-score is negative."""
    # Build: first 20 bars with high variance, then 5 bars with low variance
    ranges = ([0.001, 0.005] * 10) + [0.002] * 5
    df = _make_accum_matrix(25, high_low_range=ranges)
    result = compute_pipeline_features(df)

    z = result["range_compression_z"].to_list()
    # Last bars have constant range (low CV) after a period of high CV
    # z-score should be negative (below historical mean)
    valid_last = [v for v in z[20:] if v is not None]
    if len(valid_last) > 0:
        assert valid_last[-1] < 0, f"Expected negative z for tight contraction, got {valid_last[-1]}"


def test_volatility_compression_1bar_known_answer():
    """volatility_compression_1bar = current_range / rolling_mean(range, 6)."""
    # Constant range of 0.002
    df = _make_accum_matrix(10, high_low_range=[0.002] * 10)
    result = compute_pipeline_features(df)

    vc = result["volatility_compression_1bar"].to_list()
    # 0.002 / mean([0.002]*6) = 0.002 / 0.002 = 1.0
    for i in range(6, len(vc)):
        if vc[i] is not None:
            assert abs(vc[i] - 1.0) < 1e-6, f"Expected 1.0 at bar {i}, got {vc[i]}"


def test_volatility_compression_compressed_bar():
    """A bar with half the average range should have volatility_compression ~0.5."""
    ranges = [0.004] * 8 + [0.002, 0.004]  # bar 8 has half-size range
    df = _make_accum_matrix(10, high_low_range=ranges)
    result = compute_pipeline_features(df)

    vc = result["volatility_compression_1bar"][8]
    # mean of [0.004]*6 = 0.004, bar 8 range = 0.002, ratio = 0.5
    assert vc is not None
    assert abs(vc - 0.5) < 0.05, f"Expected ~0.5, got {vc}"


def test_post_absorption_contraction_fires():
    """post_absorption_contraction = 1 when absorption in last 3 bars AND tight range."""
    n = 15
    ranges = [0.003] * n  # baseline range
    absorption = [0.0] * n
    # Absorption at bar 8
    absorption[8] = 1.0
    # Tight range at bar 9 (within 3-bar lookback of absorption at 8)
    ranges[9] = 0.0005  # well below median

    df = _make_accum_matrix(n, high_low_range=ranges, absorption_signal=absorption)
    result = compute_pipeline_features(df)

    pac = result["post_absorption_contraction"].to_list()
    # Bar 9: absorption was at bar 8 (1 bar ago), range is tight
    assert pac[9] == 1.0, f"Expected contraction at bar 9, got {pac[9]}"


def test_post_absorption_contraction_no_absorption():
    """No absorption => post_absorption_contraction always 0."""
    df = _make_accum_matrix(10, high_low_range=[0.001] * 10)
    result = compute_pipeline_features(df)

    pac = result["post_absorption_contraction"].to_list()
    assert all(v == 0.0 for v in pac), "Expected all 0.0 without absorption"


def test_post_absorption_contraction_wide_range():
    """Absorption but wide range (not tight) => no contraction."""
    n = 10
    ranges = [0.001] * n
    absorption = [0.0] * n
    absorption[5] = 1.0
    ranges[6] = 0.005  # much wider than median

    df = _make_accum_matrix(n, high_low_range=ranges, absorption_signal=absorption)
    result = compute_pipeline_features(df)

    assert result["post_absorption_contraction"][6] == 0.0


def test_accum_row_count_preserved():
    """Row count must be preserved with accumulation features."""
    n = 20
    df = _make_accum_matrix(n)
    result = compute_pipeline_features(df)
    assert len(result) == n


def test_accum_binary_features_are_binary():
    """post_absorption_contraction must be 0.0 or 1.0."""
    ranges = [0.001 + i * 0.0001 for i in range(20)]
    absorption = [1.0 if i % 5 == 0 else 0.0 for i in range(20)]

    df = _make_accum_matrix(20, high_low_range=ranges,
                            absorption_signal=absorption)
    result = compute_pipeline_features(df)

    vals = result["post_absorption_contraction"].to_list()
    for i, v in enumerate(vals):
        assert v in (0.0, 1.0), f"post_absorption_contraction[{i}] = {v}, expected 0 or 1"


# ---- post_recoil_stall tests ----

def test_post_recoil_stall_fires():
    """post_recoil_stall = 1 when recoil_50pct == 1.0 AND vol_compression < 0.75."""
    n = 15
    ranges = [0.004] * n
    recoil = [0.0] * n
    # Bar 8: recoil active, bar range half of average = 0.5 compression
    recoil[8] = 1.0
    ranges[8] = 0.002  # 0.002 / mean([0.004]*6) = 0.5 < 0.75

    df = _make_accum_matrix(n, high_low_range=ranges, recoil_50pct=recoil)
    result = compute_pipeline_features(df)

    prs = result["post_recoil_stall"].to_list()
    assert prs[8] == 1.0, f"Expected stall at bar 8, got {prs[8]}"


def test_post_recoil_stall_no_recoil():
    """No recoil => post_recoil_stall always 0."""
    df = _make_accum_matrix(10, high_low_range=[0.001] * 10, recoil_50pct=[0.0] * 10)
    result = compute_pipeline_features(df)

    prs = result["post_recoil_stall"].to_list()
    assert all(v == 0.0 for v in prs), "Expected all 0.0 without recoil"


def test_post_recoil_stall_wide_bar():
    """Recoil but wide bar (vol_compression >= 0.75) => no stall."""
    n = 10
    ranges = [0.004] * n
    recoil = [0.0] * n
    recoil[8] = 1.0
    ranges[8] = 0.004  # 0.004 / 0.004 = 1.0 >= 0.75

    df = _make_accum_matrix(n, high_low_range=ranges, recoil_50pct=recoil)
    result = compute_pipeline_features(df)

    assert result["post_recoil_stall"][8] == 0.0


def test_post_recoil_stall_is_binary():
    """post_recoil_stall must be 0.0 or 1.0."""
    ranges = [0.004] * 20
    recoil = [1.0 if i % 4 == 0 else 0.0 for i in range(20)]
    # Make some bars compressed
    for i in range(20):
        if i % 4 == 0:
            ranges[i] = 0.001  # compressed

    df = _make_accum_matrix(20, high_low_range=ranges, recoil_50pct=recoil)
    result = compute_pipeline_features(df)

    vals = result["post_recoil_stall"].to_list()
    for i, v in enumerate(vals):
        assert v in (0.0, 1.0), f"post_recoil_stall[{i}] = {v}, expected 0 or 1"
