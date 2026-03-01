"""Tests for opening range feature computation."""

import polars as pl
from datetime import datetime, timedelta

from src.framework.features_canonical.opening_range import compute_opening_range_features


def _make_bars(n_bars, start_hour=9, start_min=30, bar_minutes=5,
               date=datetime(2024, 7, 15),
               base_price=21000.0, range_size=5.0):
    """Create synthetic bar data with UTC timestamps for Eastern RTH."""
    rows = []
    # ET to UTC offset: assume EDT (UTC-4) for summer dates
    utc_offset_hours = 4
    for i in range(n_bars):
        total_min = start_hour * 60 + start_min + i * bar_minutes
        h = total_min // 60
        m = total_min % 60
        # Convert ET time to UTC
        ts = datetime(date.year, date.month, date.day, h + utc_offset_hours, m)
        rows.append({
            "ts_event": ts,
            "open": base_price + i * 0.5,
            "high": base_price + i * 0.5 + range_size,
            "low": base_price + i * 0.5 - range_size,
            "close": base_price + i * 0.5 + 1.0,
        })
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )


def test_or_null_during_forming_period():
    """OR features are null during the first 30 minutes (OR formation)."""
    # 5m bars: 6 bars in OR period (0-25 min), bar at 30 min is first post-OR
    bars = _make_bars(12, bar_minutes=5)
    result = compute_opening_range_features(bars)

    # First 6 bars (minutes 0,5,10,15,20,25) should have null OR features
    for i in range(6):
        assert result["or_width"][i] is None, f"Bar {i} should have null or_width"
        assert result["position_in_or"][i] is None, f"Bar {i} should have null position_in_or"

    # Bar 6 (minute 30) should have non-null OR features
    assert result["or_width"][6] is not None, "Bar 6 (30 min) should have or_width"
    assert result["position_in_or"][6] is not None, "Bar 6 should have position_in_or"


def test_or_width_known_answer():
    """OR width = (or_high - or_low) / close."""
    # Create bars where OR period has known high and low
    rows = []
    utc_offset = 4  # EDT
    date = datetime(2024, 7, 15)
    # OR bars: 09:30-09:55 ET (6 bars at 5m)
    for i in range(6):
        h, m = divmod(9 * 60 + 30 + i * 5, 60)
        ts = datetime(date.year, date.month, date.day, h + utc_offset, m)
        rows.append({
            "ts_event": ts,
            "open": 21000.0,
            "high": 21010.0 if i == 2 else 21005.0,  # OR high = 21010
            "low": 20990.0 if i == 4 else 20995.0,    # OR low = 20990
            "close": 21000.0,
        })
    # Post-OR bar at 10:00
    ts = datetime(date.year, date.month, date.day, 10 + utc_offset, 0)
    rows.append({
        "ts_event": ts,
        "open": 21005.0,
        "high": 21008.0,
        "low": 21002.0,
        "close": 21005.0,
    })
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )
    result = compute_opening_range_features(df)

    # OR high = 21010, OR low = 20990, close at bar 6 = 21005
    expected_width = (21010.0 - 20990.0) / (21005.0 + 1e-9)
    assert abs(result["or_width"][6] - expected_width) < 1e-6


def test_position_in_or_known_answer():
    """position_in_or = (close - or_low) / (or_high - or_low)."""
    rows = []
    utc_offset = 4
    date = datetime(2024, 7, 15)
    # OR bars with known range: high=21020, low=21000
    for i in range(6):
        h, m = divmod(9 * 60 + 30 + i * 5, 60)
        ts = datetime(date.year, date.month, date.day, h + utc_offset, m)
        rows.append({
            "ts_event": ts,
            "open": 21010.0,
            "high": 21020.0 if i == 0 else 21015.0,
            "low": 21000.0 if i == 0 else 21005.0,
            "close": 21010.0,
        })
    # Post-OR bar closing at midpoint
    ts = datetime(date.year, date.month, date.day, 10 + utc_offset, 0)
    rows.append({
        "ts_event": ts,
        "open": 21010.0, "high": 21015.0, "low": 21005.0,
        "close": 21010.0,  # midpoint of [21000, 21020]
    })
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )
    result = compute_opening_range_features(df)

    # position = (21010 - 21000) / (21020 - 21000) = 0.5
    assert abs(result["position_in_or"][6] - 0.5) < 1e-6


def test_or_broken_up():
    """or_broken_up = 1 when close > or_high."""
    rows = []
    utc_offset = 4
    date = datetime(2024, 7, 15)
    # OR bars: range [20990, 21010]
    for i in range(6):
        h, m = divmod(9 * 60 + 30 + i * 5, 60)
        ts = datetime(date.year, date.month, date.day, h + utc_offset, m)
        rows.append({
            "ts_event": ts,
            "open": 21000.0,
            "high": 21010.0 if i == 0 else 21005.0,
            "low": 20990.0 if i == 0 else 20995.0,
            "close": 21000.0,
        })
    # Post-OR: close above OR high
    ts = datetime(date.year, date.month, date.day, 10 + utc_offset, 0)
    rows.append({
        "ts_event": ts,
        "open": 21012.0, "high": 21015.0, "low": 21008.0,
        "close": 21012.0,  # > 21010
    })
    # Post-OR: close within OR
    ts = datetime(date.year, date.month, date.day, 10 + utc_offset, 5)
    rows.append({
        "ts_event": ts,
        "open": 21005.0, "high": 21008.0, "low": 21002.0,
        "close": 21005.0,  # within [20990, 21010]
    })
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )
    result = compute_opening_range_features(df)

    assert result["or_broken_up"][6] == 1.0, "Close > OR high should be broken_up=1"
    assert result["or_broken_up"][7] == 0.0, "Close within OR should be broken_up=0"


def test_or_broken_down():
    """or_broken_down = 1 when close < or_low."""
    rows = []
    utc_offset = 4
    date = datetime(2024, 7, 15)
    for i in range(6):
        h, m = divmod(9 * 60 + 30 + i * 5, 60)
        ts = datetime(date.year, date.month, date.day, h + utc_offset, m)
        rows.append({
            "ts_event": ts,
            "open": 21000.0,
            "high": 21010.0 if i == 0 else 21005.0,
            "low": 20990.0 if i == 0 else 20995.0,
            "close": 21000.0,
        })
    # Post-OR: close below OR low
    ts = datetime(date.year, date.month, date.day, 10 + utc_offset, 0)
    rows.append({
        "ts_event": ts,
        "open": 20985.0, "high": 20992.0, "low": 20980.0,
        "close": 20985.0,  # < 20990
    })
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )
    result = compute_opening_range_features(df)

    assert result["or_broken_down"][6] == 1.0, "Close < OR low should be broken_down=1"


def test_multi_day():
    """OR computed independently per session date."""
    rows = []
    utc_offset = 4
    for day_offset in range(2):
        date = datetime(2024, 7, 15 + day_offset)
        for i in range(8):  # 6 OR + 2 post-OR
            h, m = divmod(9 * 60 + 30 + i * 5, 60)
            ts = datetime(date.year, date.month, date.day, h + utc_offset, m)
            price = 21000.0 + day_offset * 100  # different base per day
            rows.append({
                "ts_event": ts,
                "open": price,
                "high": price + 10.0,
                "low": price - 10.0,
                "close": price + 1.0,
            })
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )
    result = compute_opening_range_features(df)

    assert len(result) == 16
    # Day 1 post-OR bars should have different OR width than day 2
    # (both same range_size in this test, but different close prices)
    day1_width = result["or_width"][6]
    day2_width = result["or_width"][14]
    assert day1_width is not None
    assert day2_width is not None
    # They should differ slightly due to different close prices
    assert day1_width != day2_width


def test_empty_dataframe():
    """Empty input produces empty output with correct schema."""
    df = pl.DataFrame(schema={
        "ts_event": pl.Datetime("ns", "UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
    })
    result = compute_opening_range_features(df)
    assert len(result) == 0
    assert "or_width" in result.columns


def test_row_count_preserved():
    """Output row count matches input."""
    bars = _make_bars(20)
    result = compute_opening_range_features(bars)
    assert len(result) == 20


def test_output_columns():
    """Verify expected output columns."""
    bars = _make_bars(10)
    result = compute_opening_range_features(bars)
    expected = ["ts_event", "or_width", "position_in_or", "or_broken_up", "or_broken_down"]
    for col in expected:
        assert col in result.columns, f"Missing: {col}"


def test_no_intermediate_columns():
    """No internal helper columns in output."""
    bars = _make_bars(10)
    result = compute_opening_range_features(bars)
    for col in result.columns:
        assert not col.startswith("_"), f"Intermediate column leaked: {col}"


def test_1m_bars_or_period():
    """With 1m bars, first 30 bars are in OR period."""
    bars = _make_bars(35, bar_minutes=1)
    result = compute_opening_range_features(bars)

    # First 30 bars (minutes 0-29) should have null features
    for i in range(30):
        assert result["or_width"][i] is None, f"Bar {i} (1m) should be null during OR"

    # Bar 30 (minute 30) should be non-null
    assert result["or_width"][30] is not None, "Bar 30 (1m) should have OR features"
