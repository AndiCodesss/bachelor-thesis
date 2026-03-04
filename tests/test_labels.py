"""Tests for forward return label computation."""

import polars as pl
import pytest
from datetime import datetime, timedelta
from src.framework.features_canonical.labels import compute_labels


def _make_bars(timestamps, close_prices):
    """Helper: build a bars DataFrame with ts_event and close."""
    return pl.DataFrame({
        "ts_event": timestamps,
        "close": [float(p) for p in close_prices],
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )


def test_forward_returns_known_values():
    """Test forward return computation with known prices."""
    base_time = datetime(2024, 7, 15, 14, 0, 0)
    timestamps = [base_time + timedelta(minutes=i*5) for i in range(13)]
    close_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 113]

    bars = _make_bars(timestamps, close_prices)
    result = compute_labels(bars)

    assert len(result) == 13, f"Expected 13 bars, got {len(result)}"

    # Bar 0: close=100, next close=102
    expected_fwd_1 = (102 - 100) / 100
    actual_fwd_1 = result["fwd_return_1bar"][0]
    assert abs(actual_fwd_1 - expected_fwd_1) < 0.0001

    # Bar 0: close=100, close at bar 3 = 103
    expected_fwd_3 = (103 - 100) / 100
    actual_fwd_3 = result["fwd_return_3bar"][0]
    assert abs(actual_fwd_3 - expected_fwd_3) < 0.0001

    # Bar 0: close=100, close at bar 6 = 106
    expected_fwd_6 = (106 - 100) / 100
    actual_fwd_6 = result["fwd_return_6bar"][0]
    assert abs(actual_fwd_6 - expected_fwd_6) < 0.0001

    # Bar 0: close=100, close at bar 12 = 113
    expected_fwd_12 = (113 - 100) / 100
    actual_fwd_12 = result["fwd_return_12bar"][0]
    assert abs(actual_fwd_12 - expected_fwd_12) < 0.0001

    # Bar 5: close=104, next close=106
    expected_fwd_1_bar5 = (106 - 104) / 104
    actual_fwd_1_bar5 = result["fwd_return_1bar"][5]
    assert abs(actual_fwd_1_bar5 - expected_fwd_1_bar5) < 0.0001


def test_forward_returns_null_when_close_is_zero():
    timestamps = [
        datetime(2024, 7, 15, 14, 0, 0),
        datetime(2024, 7, 15, 14, 5, 0),
    ]
    bars = _make_bars(timestamps, [0.0, 1.0])
    result = compute_labels(bars)

    assert result["fwd_return_1bar"][0] is None


def test_classification_labels():
    """Test that classification labels match forward return sign."""
    timestamps = [datetime(2024, 7, 15, 14, i * 5, 0) for i in range(6)]
    close_prices = [100, 102, 101, 103, 99, 104]

    bars = _make_bars(timestamps, close_prices)
    result = compute_labels(bars)

    assert result["label_1bar"][0] == 1  # 100 -> 102 (up)
    assert result["fwd_return_1bar"][0] > 0
    assert result["label_1bar"][1] == 0  # 102 -> 101 (down)
    assert result["fwd_return_1bar"][1] < 0
    assert result["label_1bar"][2] == 1  # 101 -> 103 (up)
    assert result["label_1bar"][3] == 0  # 103 -> 99 (down)

    assert result["label_3bar"][0] == 1  # 100 -> 103 (up)
    assert result["label_3bar"][1] == 0  # 102 -> 99 (down)


def test_last_bars_have_null_forward_returns():
    """Test that last N bars have null forward returns where N = horizon."""
    base_time = datetime(2024, 7, 15, 14, 0, 0)
    timestamps = [base_time + timedelta(minutes=i*5) for i in range(15)]
    close_prices = list(range(100, 115))

    bars = _make_bars(timestamps, close_prices)
    result = compute_labels(bars)

    # Last bar (index 14) should have null for all forward returns
    assert result["fwd_return_1bar"][14] is None
    assert result["fwd_return_3bar"][14] is None

    # Last 3 bars should have null fwd_return_3bar
    for i in [12, 13, 14]:
        assert result["fwd_return_3bar"][i] is None

    # Last 6 bars should have null fwd_return_6bar
    for i in range(9, 15):
        assert result["fwd_return_6bar"][i] is None

    # Last 12 bars should have null fwd_return_12bar
    for i in range(3, 15):
        assert result["fwd_return_12bar"][i] is None

    # First bar should have non-null fwd_return_1bar
    assert result["fwd_return_1bar"][0] is not None


@pytest.mark.slow
def test_real_data_not_empty():
    """Test labels computation on real data sample."""
    from src.framework.data.loader import get_parquet_files, filter_rth
    from src.framework.data.bars import aggregate_time_bars

    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[1]))  # files[0] may be Sunday (no bars)
    lf_rth = filter_rth(lf)
    bars = aggregate_time_bars(lf_rth, "5m")

    result = compute_labels(bars)

    assert len(result) > 0

    expected_columns = [
        "ts_event", "close",
        "fwd_return_1bar", "fwd_return_3bar", "fwd_return_5bar",
        "fwd_return_6bar", "fwd_return_10bar", "fwd_return_12bar",
        "label_1bar", "label_3bar", "label_5bar",
    ]
    for col in expected_columns:
        assert col in result.columns, f"Expected column '{col}' not found"

    non_null_1bar = result["fwd_return_1bar"].drop_nulls().len()
    assert non_null_1bar > 0

    non_null_labels = result.filter(pl.col("fwd_return_1bar").is_not_null())
    if len(non_null_labels) > 0:
        unique_label_1bar = non_null_labels["label_1bar"].unique().sort()
        assert len(unique_label_1bar) <= 2
        for val in unique_label_1bar:
            assert val in [0, 1]


def test_fwd_return_5bar_known_values():
    """Test fwd_return_5bar with hand-computed values."""
    base_time = datetime(2024, 7, 15, 14, 0, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(8)]
    close_prices = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0]

    bars = _make_bars(timestamps, close_prices)
    result = compute_labels(bars)
    assert len(result) == 8

    # Bar 0: close=100, bar 5 close=103 -> fwd_return_5bar = (103-100)/100 = 0.03
    actual_0 = result["fwd_return_5bar"][0]
    assert actual_0 is not None
    assert abs(actual_0 - 0.03) < 1e-6

    # Bar 1: close=101, bar 6 close=97 -> fwd_return_5bar = (97-101)/101
    actual_1 = result["fwd_return_5bar"][1]
    assert actual_1 is not None
    expected_1 = (97.0 - 101.0) / 101.0
    assert abs(actual_1 - expected_1) < 1e-6

    # Bar 2: close=99, bar 7 close=104 -> fwd_return_5bar = (104-99)/99
    actual_2 = result["fwd_return_5bar"][2]
    assert actual_2 is not None
    expected_2 = (104.0 - 99.0) / 99.0
    assert abs(actual_2 - expected_2) < 1e-6

    # Bar 3+: only 4 bars remaining, fwd_return_5bar should be null (day boundary)
    for i in range(3, 8):
        val = result["fwd_return_5bar"][i]
        assert val is None, f"Bar {i} fwd_return_5bar should be null, got {val}"

    assert result["label_5bar"][0] == 1  # positive return
    assert result["label_5bar"][1] == 0  # negative return


def test_cross_day_forward_returns_nulled():
    """Forward returns must not cross day boundaries."""
    day1_base = datetime(2024, 7, 15, 14, 0, 0)
    day1_ts = [day1_base + timedelta(minutes=i * 5) for i in range(3)]
    day2_base = datetime(2024, 7, 16, 14, 0, 0)
    day2_ts = [day2_base + timedelta(minutes=i * 5) for i in range(3)]

    timestamps = day1_ts + day2_ts
    prices = [100.0, 101.0, 102.0, 200.0, 201.0, 202.0]

    bars = _make_bars(timestamps, prices)
    result = compute_labels(bars)

    # Day 1 last bar (bar 2): 0 bars remaining, fwd_return_1bar should be null
    assert result["fwd_return_1bar"][2] is None

    # Day 1 bar 0: 2 bars remaining, fwd_return_3bar should be null
    assert result["fwd_return_3bar"][0] is None

    # Day 1 bar 0: fwd_return_1bar should be valid within day
    val_1 = result["fwd_return_1bar"][0]
    assert val_1 is not None
    expected = (101.0 - 100.0) / 100.0
    assert abs(val_1 - expected) < 1e-6


def test_label_dtype():
    """Verify labels are UInt8 type."""
    timestamps = [datetime(2024, 7, 15, 14, i * 5, 0) for i in range(5)]
    close_prices = [100, 102, 101, 103, 105]

    bars = _make_bars(timestamps, close_prices)
    result = compute_labels(bars)

    assert result["label_1bar"].dtype == pl.UInt8
    assert result["label_3bar"].dtype == pl.UInt8
    assert result["label_5bar"].dtype == pl.UInt8
