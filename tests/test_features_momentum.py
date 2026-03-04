"""Tests for momentum feature engineering."""

import polars as pl
import pytest
from datetime import datetime
from src.framework.features_canonical.momentum import compute_momentum_features
from src.framework.data.bars import aggregate_time_bars


def _make_bars(ohlcv_list):
    """Helper: build a bars DataFrame from list of (ts, open, high, low, close, volume, vwap) tuples."""
    return pl.DataFrame({
        "ts_event": [r[0] for r in ohlcv_list],
        "ts_close": [r[0] for r in ohlcv_list],
        "bar_duration_ns": [300_000_000_000] * len(ohlcv_list),
        "open": [r[1] for r in ohlcv_list],
        "high": [r[2] for r in ohlcv_list],
        "low": [r[3] for r in ohlcv_list],
        "close": [r[4] for r in ohlcv_list],
        "volume": pl.Series([r[5] for r in ohlcv_list], dtype=pl.UInt32),
        "vwap": [r[6] for r in ohlcv_list],
        "trade_count": pl.Series([10] * len(ohlcv_list), dtype=pl.UInt32),
        "buy_volume": pl.Series([5] * len(ohlcv_list), dtype=pl.UInt32),
        "sell_volume": pl.Series([5] * len(ohlcv_list), dtype=pl.UInt32),
        "large_trade_count": pl.Series([1] * len(ohlcv_list), dtype=pl.UInt32),
        "large_buy_volume": pl.Series([0] * len(ohlcv_list), dtype=pl.UInt32),
        "large_sell_volume": pl.Series([0] * len(ohlcv_list), dtype=pl.UInt32),
        "bid_price": [99.0] * len(ohlcv_list),
        "ask_price": [101.0] * len(ohlcv_list),
        "bid_size": pl.Series([10] * len(ohlcv_list), dtype=pl.UInt32),
        "ask_size": pl.Series([10] * len(ohlcv_list), dtype=pl.UInt32),
        "bid_count": pl.Series([5] * len(ohlcv_list), dtype=pl.UInt32),
        "ask_count": pl.Series([5] * len(ohlcv_list), dtype=pl.UInt32),
        "msg_count": pl.Series([100] * len(ohlcv_list), dtype=pl.UInt32),
        "add_count": pl.Series([50] * len(ohlcv_list), dtype=pl.UInt32),
        "cancel_count": pl.Series([30] * len(ohlcv_list), dtype=pl.UInt32),
        "modify_count": pl.Series([10] * len(ohlcv_list), dtype=pl.UInt32),
    }).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))


@pytest.mark.slow
def test_compute_momentum_features_basic():
    """Test momentum features on real data sample."""
    from src.framework.data.loader import get_parquet_files, filter_rth

    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)
    lf_rth = filter_rth(lf)
    bars = aggregate_time_bars(lf_rth, bar_size="5m")

    result = compute_momentum_features(bars)

    assert len(result) > 0, "Result DataFrame should not be empty"

    expected_columns = [
        "ts_event",
        "open", "high", "low", "close", "volume", "vwap",
        "return_1bar", "return_5bar", "return_12bar",
        "vwap_deviation", "vwap_deviation_ma5",
        "high_low_range", "range_ma5",
        "volume_ma5", "volume_ratio",
        "close_position",
        "momentum_volume"
    ]

    for col in expected_columns:
        assert col in result.columns, f"Expected column '{col}' not found"


def test_ohlcv_bars_synthetic():
    """Test OHLCV values pass through correctly with known-answer data."""
    bars = _make_bars([
        # (ts, open, high, low, close, volume, vwap)
        (datetime(2024, 7, 15, 14, 0, 0), 100.0, 102.0, 98.0, 101.0, 70, 100.0),
        (datetime(2024, 7, 15, 14, 5, 0), 101.0, 105.0, 99.0, 104.0, 75, 102.0),
    ])

    result = compute_momentum_features(bars)

    assert len(result) == 2, f"Expected 2 bars, got {len(result)}"

    # Bar 1
    assert result["open"][0] == 100.0
    assert result["high"][0] == 102.0
    assert result["low"][0] == 98.0
    assert result["close"][0] == 101.0
    assert result["volume"][0] == 70

    # Bar 2
    assert result["open"][1] == 101.0
    assert result["high"][1] == 105.0
    assert result["low"][1] == 99.0
    assert result["close"][1] == 104.0
    assert result["volume"][1] == 75


def test_vwap_calculation():
    """Test VWAP passes through from bars."""
    expected_vwap = 5000.0 / 30.0
    bars = _make_bars([
        (datetime(2024, 7, 15, 14, 0, 0), 100.0, 200.0, 100.0, 200.0, 30, expected_vwap),
    ])

    result = compute_momentum_features(bars)
    actual_vwap = result["vwap"][0]
    assert abs(actual_vwap - expected_vwap) < 0.01, f"Expected VWAP={expected_vwap:.2f}, got {actual_vwap:.2f}"


def test_return_calculation():
    """Test return computation with known values."""
    bars = _make_bars([
        (datetime(2024, 7, 15, 14, 0, 0), 100.0, 100.0, 100.0, 100.0, 10, 100.0),
        (datetime(2024, 7, 15, 14, 5, 0), 105.0, 105.0, 105.0, 105.0, 10, 105.0),
        (datetime(2024, 7, 15, 14, 10, 0), 100.0, 100.0, 100.0, 100.0, 10, 100.0),
    ])

    result = compute_momentum_features(bars)

    # Bar 1: return_1bar should be null (no previous bar)
    assert result["return_1bar"][0] is None or pl.Series([result["return_1bar"][0]]).is_null()[0]

    # Bar 2: return = (105-100)/100 = 0.05
    assert abs(result["return_1bar"][1] - 0.05) < 0.001

    # Bar 3: return = (100-105)/105 ~ -0.0476
    assert abs(result["return_1bar"][2] - (-0.047619)) < 0.001


def test_close_position_edge_cases():
    """Test close_position handles edge case where high == low."""
    bars = _make_bars([
        (datetime(2024, 7, 15, 14, 0, 0), 100.0, 100.0, 100.0, 100.0, 20, 100.0),
    ])

    result = compute_momentum_features(bars)
    assert result["close_position"][0] == 0.5, \
        f"close_position should be 0.5 when high==low, got {result['close_position'][0]}"


@pytest.mark.slow
def test_no_nulls_in_features():
    """Verify no NaN/null in computed features (except first bar for lagged features)."""
    from src.framework.data.loader import get_parquet_files, filter_rth

    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)
    lf_rth = filter_rth(lf)
    bars = aggregate_time_bars(lf_rth, bar_size="5m")

    result = compute_momentum_features(bars)

    non_lagged_features = [
        "open", "high", "low", "close", "volume", "vwap",
        "vwap_deviation", "high_low_range", "close_position"
    ]

    if len(result) > 1:
        for col in non_lagged_features:
            null_count = result[col][1:].null_count()
            assert null_count == 0, f"Column '{col}' has {null_count} nulls after first bar"


def test_rolling_sum_not_cumulative():
    """Verify rolling_sum uses window, not cumulative sum."""
    timestamps = [datetime(2024, 7, 15, 14, i * 5, 0) for i in range(10)]
    prices = [100.0 * (1.01 ** i) for i in range(10)]

    bars = _make_bars([
        (timestamps[i], prices[i], prices[i], prices[i], prices[i], 10, prices[i])
        for i in range(10)
    ])

    result = compute_momentum_features(bars)

    if len(result) >= 8:
        return_5bar_at_6 = result["return_5bar"][5]
        return_5bar_at_8 = result["return_5bar"][7]

        assert abs(return_5bar_at_8 - return_5bar_at_6) < 0.01, \
            f"return_5bar should use rolling window, not cumulative. Bar 6: {return_5bar_at_6:.4f}, Bar 8: {return_5bar_at_8:.4f}"


def test_wick_ratio_known_answer():
    """Wick ratios with known OHLC: open=100, high=110, low=90, close=105."""
    bars = _make_bars([
        # (ts, open, high, low, close, volume, vwap)
        (datetime(2024, 7, 15, 14, 0, 0), 100.0, 110.0, 90.0, 105.0, 100, 100.0),
    ])

    result = compute_momentum_features(bars)
    row = result.row(0, named=True)

    # range = 110 - 90 = 20
    # upper_wick = 110 - max(100, 105) = 110 - 105 = 5
    # lower_wick = min(100, 105) - 90 = 100 - 90 = 10
    # body = abs(105 - 100) = 5
    assert abs(row["upper_wick_ratio"] - 5.0 / 20.0) < 0.001, \
        f"upper_wick_ratio: expected 0.25, got {row['upper_wick_ratio']}"
    assert abs(row["lower_wick_ratio"] - 10.0 / 20.0) < 0.001, \
        f"lower_wick_ratio: expected 0.50, got {row['lower_wick_ratio']}"
    assert abs(row["body_ratio"] - 5.0 / 20.0) < 0.001, \
        f"body_ratio: expected 0.25, got {row['body_ratio']}"

    # Sum should equal 1.0
    total = row["upper_wick_ratio"] + row["lower_wick_ratio"] + row["body_ratio"]
    assert abs(total - 1.0) < 0.001, f"Ratios should sum to 1.0, got {total}"


def test_wick_ratio_bearish_bar():
    """Bearish bar: open > close. Upper wick = high - open, lower wick = close - low."""
    bars = _make_bars([
        (datetime(2024, 7, 15, 14, 0, 0), 108.0, 110.0, 90.0, 92.0, 100, 100.0),
    ])

    result = compute_momentum_features(bars)
    row = result.row(0, named=True)

    # range = 20, upper_wick = 110 - max(108, 92) = 110 - 108 = 2
    # lower_wick = min(108, 92) - 90 = 92 - 90 = 2
    # body = abs(92 - 108) = 16
    assert abs(row["upper_wick_ratio"] - 2.0 / 20.0) < 0.001
    assert abs(row["lower_wick_ratio"] - 2.0 / 20.0) < 0.001
    assert abs(row["body_ratio"] - 16.0 / 20.0) < 0.001


def test_wick_ratio_doji():
    """Doji bar: open == close. Body ratio should be 0."""
    bars = _make_bars([
        (datetime(2024, 7, 15, 14, 0, 0), 100.0, 105.0, 95.0, 100.0, 100, 100.0),
    ])

    result = compute_momentum_features(bars)
    row = result.row(0, named=True)

    # range = 10, body = 0, upper_wick = 5, lower_wick = 5
    assert abs(row["body_ratio"]) < 0.001, "Doji should have body_ratio ~0"
    assert abs(row["upper_wick_ratio"] - 0.5) < 0.001
    assert abs(row["lower_wick_ratio"] - 0.5) < 0.001


def test_wick_ratio_zero_range():
    """When high == low (zero range), all ratios should be 0."""
    bars = _make_bars([
        (datetime(2024, 7, 15, 14, 0, 0), 100.0, 100.0, 100.0, 100.0, 100, 100.0),
    ])

    result = compute_momentum_features(bars)
    row = result.row(0, named=True)

    assert row["upper_wick_ratio"] == 0.0, "Zero range: upper_wick_ratio should be 0"
    assert row["lower_wick_ratio"] == 0.0, "Zero range: lower_wick_ratio should be 0"
    assert row["body_ratio"] == 0.0, "Zero range: body_ratio should be 0"


@pytest.mark.slow
def test_wick_ratio_columns_in_shape_test():
    """Verify wick ratio columns are present in real data output."""
    from src.framework.data.loader import get_parquet_files, filter_rth

    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)
    lf_rth = filter_rth(lf)
    bars = aggregate_time_bars(lf_rth, bar_size="5m")

    result = compute_momentum_features(bars)

    for col in ["upper_wick_ratio", "lower_wick_ratio", "body_ratio"]:
        assert col in result.columns, f"Missing column: {col}"
        assert result[col].null_count() == 0, f"Nulls in {col}"
        assert (result[col] >= 0).all(), f"Negative values in {col}"
        assert (result[col] <= 1.0001).all(), f"Values > 1 in {col}"
