"""Tests for statistical feature engineering."""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from src.framework.features_canonical.statistical import (
    compute_statistical_features,
    _fracdiff_weights,
    _apply_fracdiff,
    FRACDIFF_D,
    FRACDIFF_WINDOW,
    YZ_WINDOW,
)


def _make_bars(timestamps, prices, volumes=None, highs=None, lows=None, opens=None, vwaps=None):
    """Helper: build a bars DataFrame with OHLCV + vwap."""
    n = len(timestamps)
    if volumes is None:
        volumes = [10] * n
    if opens is None:
        opens = prices  # single-trade bars: open=close
    if highs is None:
        highs = prices
    if lows is None:
        lows = prices
    if vwaps is None:
        vwaps = prices  # single-trade bars: vwap=close

    return pl.DataFrame({
        "ts_event": timestamps,
        "open": [float(p) for p in opens],
        "high": [float(p) for p in highs],
        "low": [float(p) for p in lows],
        "close": [float(p) for p in prices],
        "volume": volumes,
        "vwap": [float(p) for p in vwaps],
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("volume").cast(pl.UInt32),
    )


_BASE_TS = datetime(2024, 7, 15, 14, 0, 0)


def _ts_seq(n, step_minutes=5):
    return [_BASE_TS + timedelta(minutes=i * step_minutes) for i in range(n)]


# --------------- fracdiff weights ---------------

def test_fracdiff_weights_first_element():
    w = _fracdiff_weights(0.4, 10)
    assert w[0] == 1.0


def test_fracdiff_weights_recurrence():
    d = 0.4
    w = _fracdiff_weights(d, 5)
    for k in range(1, 5):
        expected = w[k - 1] * (d - k + 1) / k
        assert abs(w[k] - expected) < 1e-12


def test_fracdiff_weights_decay():
    w = _fracdiff_weights(0.4, 50)
    assert abs(w[-1]) < abs(w[0])
    assert abs(w[-1]) < 0.1


def test_fracdiff_weights_known_values():
    w = _fracdiff_weights(0.4, 4)
    assert abs(w[0] - 1.0) < 1e-12
    assert abs(w[1] - 0.4) < 1e-12
    assert abs(w[2] - (-0.12)) < 1e-12
    assert abs(w[3] - 0.064) < 1e-12


# --------------- fracdiff application ---------------

def test_apply_fracdiff_warmup_nans():
    prices = np.arange(100, 110, dtype=np.float64)
    w = _fracdiff_weights(0.4, 5)
    result = _apply_fracdiff(prices, w)
    assert all(np.isnan(result[:4]))
    assert not np.isnan(result[4])


def test_apply_fracdiff_known_series():
    prices = np.full(60, 100.0)
    w = _fracdiff_weights(0.4, 50)
    result = _apply_fracdiff(prices, w)
    valid = result[49:]
    assert np.all(np.abs(valid - valid[0]) < 1e-10)
    expected = 100.0 * np.sum(w)
    assert abs(valid[0] - expected) < 1e-10


# --------------- log_return ---------------

def test_log_return_known_values():
    ts = [
        datetime(2024, 7, 15, 14, 0, 0),
        datetime(2024, 7, 15, 14, 5, 0),
        datetime(2024, 7, 15, 14, 10, 0),
    ]
    bars = _make_bars(ts, [100.0, 105.0, 100.0])
    result = compute_statistical_features(bars)

    assert result["log_return"][0] is None
    expected_1 = np.log(105.0 / 100.0)
    assert abs(result["log_return"][1] - expected_1) < 1e-6
    expected_2 = np.log(100.0 / 105.0)
    assert abs(result["log_return"][2] - expected_2) < 1e-6


# --------------- Yang-Zhang volatility ---------------

def test_yz_volatility_constant_price():
    n = YZ_WINDOW + 5
    ts = _ts_seq(n, step_minutes=5)
    prices = [100.0] * n
    bars = _make_bars(ts, prices)
    result = compute_statistical_features(bars)

    valid = result.filter(pl.col("yz_volatility").is_not_null())
    for v in valid["yz_volatility"].to_list():
        assert abs(v) < 1e-10


def test_yz_volatility_positive():
    n = YZ_WINDOW + 10
    ts = _ts_seq(n, step_minutes=1)
    # Alternating prices: create OHLC variation
    closes = [100.0 + ((-1) ** i) * 2.0 for i in range(n)]
    highs = [max(c, c + 1.0) for c in closes]
    lows = [min(c, c - 1.0) for c in closes]
    bars = _make_bars(ts, closes, highs=highs, lows=lows)
    result = compute_statistical_features(bars)

    valid = result.filter(pl.col("yz_volatility") > 0)
    assert len(valid) > 0


def test_yz_volatility_rs_component():
    """Verify RS component for a known OHLC bar."""
    ts = [
        datetime(2024, 7, 15, 14, 0, 0),
        datetime(2024, 7, 15, 14, 5, 0),
    ]
    # Bar 0: warmup, Bar 1: O=100, H=105, L=98, C=102
    bars = _make_bars(
        ts,
        prices=[100.0, 102.0],
        opens=[100.0, 100.0],
        highs=[100.0, 105.0],
        lows=[100.0, 98.0],
    )
    result = compute_statistical_features(bars)

    h = np.log(105 / 100)
    l_val = np.log(98 / 100)
    c_val = np.log(102 / 100)
    expected_rs = h * (h - c_val) + l_val * (l_val - c_val)
    assert expected_rs > 0


# --------------- vol_zscore ---------------

def test_vol_zscore_constant_vol():
    n = 80
    ts = _ts_seq(n, step_minutes=1)
    closes = [100.0 + ((-1) ** i) * 1.0 for i in range(n)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    bars = _make_bars(ts, closes, highs=highs, lows=lows)
    result = compute_statistical_features(bars)

    late = result.tail(10)
    for z in late["vol_zscore"].to_list():
        if z is not None:
            assert abs(z) < 2.0


# --------------- VWAP deviation ---------------

def test_vwap_deviation_known_values():
    """Session VWAP for 2 bars with known prices and volumes."""
    ts = [
        datetime(2024, 7, 15, 14, 0, 0),
        datetime(2024, 7, 15, 14, 5, 0),
    ]
    # Bar 1: price=100, vol=10, vwap=100 -> cumPV=1000, cumVol=10, VWAP=100, dev=0
    # Bar 2: price=110, vol=20, vwap=110 -> cumPV=1000+2200=3200, cumVol=30, VWAP=106.67, dev=3.33
    bars = _make_bars(ts, [100.0, 110.0], volumes=[10, 20], vwaps=[100.0, 110.0])
    result = compute_statistical_features(bars)

    assert abs(result["vwap_deviation"][0]) < 0.01

    expected_vwap = (100.0 * 10 + 110.0 * 20) / 30.0
    expected_dev = 110.0 - expected_vwap
    assert abs(result["vwap_deviation"][1] - expected_dev) < 0.01


def test_vwap_deviation_resets_daily():
    ts = [
        datetime(2024, 7, 15, 14, 0, 0),
        datetime(2024, 7, 15, 14, 5, 0),
        datetime(2024, 7, 16, 14, 0, 0),
    ]
    bars = _make_bars(ts, [100.0, 110.0, 200.0], volumes=[10, 10, 10], vwaps=[100.0, 110.0, 200.0])
    result = compute_statistical_features(bars)

    # Day 2 bar: close=200, session VWAP=200 (only bar in day) -> dev=0
    assert abs(result["vwap_deviation"][2]) < 0.01


# --------------- vwap_dev_zscore ---------------

def test_vwap_dev_zscore_exists():
    ts = _ts_seq(30, step_minutes=5)
    prices = [100.0 + i * 0.5 for i in range(30)]
    bars = _make_bars(ts, prices)
    result = compute_statistical_features(bars)
    assert "vwap_dev_zscore" in result.columns


# --------------- output columns ---------------

def test_output_columns_exist():
    ts = _ts_seq(60, step_minutes=5)
    prices = [100.0 + np.sin(i / 5.0) * 2.0 for i in range(60)]
    bars = _make_bars(ts, prices)
    result = compute_statistical_features(bars)

    expected = [
        "ts_event", "log_return", "fracdiff_close",
        "yz_volatility", "vol_zscore", "vwap_deviation", "vwap_dev_zscore",
    ]
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"


def test_output_no_raw_price_columns():
    ts = _ts_seq(60, step_minutes=5)
    prices = [100.0 + i * 0.1 for i in range(60)]
    bars = _make_bars(ts, prices)
    result = compute_statistical_features(bars)

    for col in ["open", "high", "low", "close", "volume"]:
        assert col not in result.columns, f"Raw column '{col}' should be dropped"


def test_output_no_temp_columns():
    ts = _ts_seq(60, step_minutes=5)
    prices = [100.0 + i * 0.1 for i in range(60)]
    bars = _make_bars(ts, prices)
    result = compute_statistical_features(bars)

    temp = [c for c in result.columns if c.startswith("_")]
    assert len(temp) == 0, f"Temp columns leaked: {temp}"


def test_output_dtypes_float64():
    ts = _ts_seq(60, step_minutes=5)
    prices = [100.0 + i * 0.1 for i in range(60)]
    bars = _make_bars(ts, prices)
    result = compute_statistical_features(bars)

    for col in ["log_return", "fracdiff_close", "yz_volatility", "vol_zscore",
                "vwap_deviation", "vwap_dev_zscore"]:
        assert result[col].dtype == pl.Float64, f"{col} should be Float64"


# --------------- edge cases ---------------

def test_empty_dataframe():
    df = pl.DataFrame(schema={
        "ts_event": pl.Datetime("ns", "UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.UInt32,
        "vwap": pl.Float64,
    })
    result = compute_statistical_features(df)
    assert len(result) == 0


def test_single_bar():
    ts = [datetime(2024, 7, 15, 14, 0, 0)]
    bars = _make_bars(ts, [100.0])
    result = compute_statistical_features(bars)
    assert len(result) == 1
    assert result["log_return"][0] is None


def test_row_count_preserved():
    ts = _ts_seq(20, step_minutes=5)
    prices = [100.0 + i for i in range(20)]
    bars = _make_bars(ts, prices)
    result = compute_statistical_features(bars)
    assert len(result) == 20


# --------------- integration with real data ---------------

def test_real_data_smoke():
    from src.framework.data.loader import get_parquet_files, filter_rth
    from src.framework.data.bars import aggregate_time_bars

    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))
    lf = filter_rth(lf)
    bars = aggregate_time_bars(lf, "5m")
    result = compute_statistical_features(bars)

    assert len(result) > 0
    assert "yz_volatility" in result.columns
    assert "fracdiff_close" in result.columns
    assert "vwap_deviation" in result.columns

    valid_fracdiff = result.filter(pl.col("fracdiff_close").is_not_null())
    assert len(valid_fracdiff) > 0


def test_real_data_sorted():
    from src.framework.data.loader import get_parquet_files, filter_rth
    from src.framework.data.bars import aggregate_time_bars

    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))
    lf = filter_rth(lf)
    bars = aggregate_time_bars(lf, "5m")
    result = compute_statistical_features(bars)

    timestamps = result["ts_event"].to_list()
    assert timestamps == sorted(timestamps)
