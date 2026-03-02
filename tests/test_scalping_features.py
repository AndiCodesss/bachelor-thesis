"""Tests for scalping features."""
import polars as pl
from datetime import datetime
from src.framework.features_canonical.scalping import compute_scalping_features


def _make_bars(n=20):
    """Create synthetic bars for testing scalping features."""
    datetime(2024, 7, 15, 13, 30, 0)
    bars = pl.DataFrame({
        "ts_event": [datetime(2024, 7, 15, 13 + (30 + i) // 60, (30 + i) % 60, 0) for i in range(n)],
        "open": [15000.0 + i * 0.5 for i in range(n)],
        "high": [15001.0 + i * 0.5 for i in range(n)],
        "low": [14999.0 + i * 0.5 for i in range(n)],
        "close": [15000.0 + i * 0.5 for i in range(n)],
        "volume": [100] * n,
        "buy_volume": [60] * n,
        "sell_volume": [40] * n,
        "trade_count": [50] * n,
        "bar_duration_ns": [60_000_000_000] * n,  # 1 minute each
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )
    return bars


def test_scalping_features_schema():
    bars = _make_bars()
    result = compute_scalping_features(bars)
    expected = {"ts_event", "delta_divergence_bull", "delta_divergence_bear",
                "effort_vs_result", "absorption_signal", "intensity",
                "intensity_z", "tape_speed_spike"}
    assert expected == set(result.columns)


def test_scalping_features_no_nulls():
    bars = _make_bars()
    result = compute_scalping_features(bars)
    # All features should have values (no nulls except possibly first few warmup bars)
    for col in result.columns:
        if col == "ts_event":
            continue
        null_count = result[col].null_count()
        assert null_count < len(result), f"Column {col} is all null"


def test_effort_vs_result_known_answer():
    """High volume + low range = high effort/result ratio."""
    bars = pl.DataFrame({
        "ts_event": [datetime(2024, 7, 15, 13, 30, 0)],
        "open": [15000.0],
        "high": [15000.25],  # range = 0.25
        "low": [15000.0],
        "close": [15000.0],
        "volume": [500],
        "buy_volume": [300],
        "sell_volume": [200],
        "trade_count": [100],
        "bar_duration_ns": [60_000_000_000],
    }).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))

    result = compute_scalping_features(bars)
    evr = result["effort_vs_result"][0]
    # 500 / 0.25 = 2000
    assert abs(evr - 2000.0) < 1.0


def test_intensity_known_answer():
    """100 trades in 60 seconds = 1.667 trades/sec."""
    bars = pl.DataFrame({
        "ts_event": [datetime(2024, 7, 15, 13, 30, 0)],
        "open": [15000.0],
        "high": [15001.0],
        "low": [14999.0],
        "close": [15000.0],
        "volume": [200],
        "buy_volume": [100],
        "sell_volume": [100],
        "trade_count": [100],
        "bar_duration_ns": [60_000_000_000],  # 60 seconds
    }).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))

    result = compute_scalping_features(bars)
    intensity = result["intensity"][0]
    assert abs(intensity - 100 / 60.0) < 0.01


def test_absorption_signal_detects_high_vol_low_range():
    """Create bars where one bar has abnormally high volume and low range."""
    n = 50
    volumes = [100] * n
    highs = [15001.0] * n
    lows = [14999.0] * n  # range = 2.0

    # Bar 25: huge volume, tiny range
    volumes[25] = 1000
    highs[25] = 15000.10
    lows[25] = 15000.00  # range = 0.10

    bars = pl.DataFrame({
        "ts_event": [datetime(2024, 7, 15, 13 + (30 + i) // 60, (30 + i) % 60, 0) for i in range(n)],
        "open": [15000.0] * n,
        "high": highs,
        "low": lows,
        "close": [15000.0] * n,
        "volume": volumes,
        "buy_volume": [v // 2 for v in volumes],
        "sell_volume": [v // 2 for v in volumes],
        "trade_count": [50] * n,
        "bar_duration_ns": [60_000_000_000] * n,
    }).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))

    result = compute_scalping_features(bars)
    # Bar 25 should have absorption_signal = 1
    assert result["absorption_signal"][25] == 1.0


def test_empty_bars():
    bars = pl.DataFrame(schema={
        "ts_event": pl.Datetime("ns", "UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.UInt32,
        "buy_volume": pl.UInt32,
        "sell_volume": pl.UInt32,
        "trade_count": pl.UInt32,
        "bar_duration_ns": pl.Int64,
    })
    result = compute_scalping_features(bars)
    assert len(result) == 0
    assert "intensity_z" in result.columns
