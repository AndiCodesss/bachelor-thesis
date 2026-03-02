"""Tests for OHLCV technical indicator features."""

import polars as pl
from datetime import datetime, timedelta

from src.framework.features_canonical.ohlcv_indicators import (
    compute_ohlcv_indicators,
    OHLCV_INDICATOR_COLUMNS,
)


_BASE_TS = datetime(2024, 7, 15, 14, 0, 0)


def _ts_seq(n, step_minutes=5):
    return [_BASE_TS + timedelta(minutes=i * step_minutes) for i in range(n)]


def _make_bars(timestamps, closes, opens=None, highs=None, lows=None, volumes=None):
    """Build minimal bars DataFrame for OHLCV indicators."""
    n = len(timestamps)
    if opens is None:
        opens = closes
    if highs is None:
        highs = closes
    if lows is None:
        lows = closes
    if volumes is None:
        volumes = [100] * n

    return pl.DataFrame({
        "ts_event": timestamps,
        "open": [float(o) for o in opens],
        "high": [float(h) for h in highs],
        "low": [float(l) for l in lows],
        "close": [float(c) for c in closes],
        "volume": volumes,
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("volume").cast(pl.UInt32),
    )


# --------------- output shape & columns ---------------

def test_output_columns_present():
    """All 20 indicator columns + ts_event are present."""
    n = 220  # enough for SMA-200 warmup
    ts = _ts_seq(n)
    closes = [100.0 + 0.1 * i for i in range(n)]
    bars = _make_bars(ts, closes)
    result = compute_ohlcv_indicators(bars)

    assert "ts_event" in result.columns
    for col in OHLCV_INDICATOR_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"


def test_output_no_raw_price_columns():
    """Raw OHLCV columns must not appear in output."""
    n = 50
    bars = _make_bars(_ts_seq(n), [100.0 + i for i in range(n)])
    result = compute_ohlcv_indicators(bars)

    for col in ["open", "high", "low", "close", "volume"]:
        assert col not in result.columns, f"Raw column '{col}' should be dropped"


def test_output_no_temp_columns():
    """No internal temp columns (prefixed with _) should leak."""
    n = 50
    bars = _make_bars(_ts_seq(n), [100.0 + i for i in range(n)])
    result = compute_ohlcv_indicators(bars)

    temp = [c for c in result.columns if c.startswith("_")]
    assert len(temp) == 0, f"Temp columns leaked: {temp}"


def test_output_dtypes_float64():
    """All indicator columns should be Float64."""
    n = 50
    bars = _make_bars(_ts_seq(n), [100.0 + i for i in range(n)])
    result = compute_ohlcv_indicators(bars)

    for col in OHLCV_INDICATOR_COLUMNS:
        assert result[col].dtype == pl.Float64, f"{col} should be Float64, got {result[col].dtype}"


def test_row_count_preserved():
    """Output has same number of rows as input."""
    n = 50
    bars = _make_bars(_ts_seq(n), [100.0] * n)
    result = compute_ohlcv_indicators(bars)
    assert len(result) == n


# --------------- empty & edge cases ---------------

def test_empty_dataframe():
    """Empty input returns empty DataFrame with correct schema."""
    df = pl.DataFrame(schema={
        "ts_event": pl.Datetime("ns", "UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.UInt32,
    })
    result = compute_ohlcv_indicators(df)
    assert len(result) == 0
    assert "ts_event" in result.columns
    for col in OHLCV_INDICATOR_COLUMNS:
        assert col in result.columns


def test_single_bar():
    """Single bar should not crash."""
    bars = _make_bars([_BASE_TS], [100.0])
    result = compute_ohlcv_indicators(bars)
    assert len(result) == 1


# --------------- SMA / EMA ratios ---------------

def test_sma_ratio_constant_price():
    """Constant price -> all SMA ratios = 0."""
    n = 210
    bars = _make_bars(_ts_seq(n), [100.0] * n)
    result = compute_ohlcv_indicators(bars)

    for col in ["sma_ratio_8", "sma_ratio_21", "sma_ratio_50", "sma_ratio_200"]:
        valid = result.filter(pl.col(col).is_not_null())
        if len(valid) > 0:
            last = valid[col][-1]
            assert abs(last) < 1e-10, f"{col} should be 0 for constant price, got {last}"


def test_ema_ratio_constant_price():
    """Constant price -> all EMA ratios = 0."""
    n = 60
    bars = _make_bars(_ts_seq(n), [100.0] * n)
    result = compute_ohlcv_indicators(bars)

    for col in ["ema_ratio_8", "ema_ratio_21", "ema_ratio_50"]:
        valid = result.filter(pl.col(col).is_not_null())
        if len(valid) > 0:
            last = valid[col][-1]
            assert abs(last) < 1e-10, f"{col} should be 0 for constant price, got {last}"


def test_sma_ratio_uptrend():
    """In uptrend, close > SMA -> ratios positive."""
    n = 210
    closes = [100.0 + i for i in range(n)]
    bars = _make_bars(_ts_seq(n), closes)
    result = compute_ohlcv_indicators(bars)

    last = result.row(-1, named=True)
    for col in ["sma_ratio_8", "sma_ratio_21", "sma_ratio_50", "sma_ratio_200"]:
        if last[col] is not None:
            assert last[col] > 0, f"{col} should be positive in uptrend, got {last[col]}"


def test_sma_ratio_known_answer():
    """SMA(8) known-answer: 8 bars [100..107], SMA=103.5, ratio=(107/103.5)-1."""
    n = 8
    closes = [100.0 + i for i in range(n)]
    bars = _make_bars(_ts_seq(n), closes)
    result = compute_ohlcv_indicators(bars)

    expected_sma = sum(closes) / 8
    expected_ratio = 107.0 / expected_sma - 1
    actual = result["sma_ratio_8"][-1]
    assert abs(actual - expected_ratio) < 1e-6, f"Expected {expected_ratio}, got {actual}"


# --------------- RSI ---------------

def test_rsi_constant_price():
    """Constant price -> RSI edge case, should not crash."""
    n = 30
    bars = _make_bars(_ts_seq(n), [100.0] * n)
    compute_ohlcv_indicators(bars)


def test_rsi_all_gains():
    """Monotonically increasing -> RSI close to 100."""
    n = 30
    closes = [100.0 + i for i in range(n)]
    bars = _make_bars(_ts_seq(n), closes)
    result = compute_ohlcv_indicators(bars)

    last_rsi = result["rsi_14"][-1]
    assert last_rsi is not None
    assert last_rsi > 90, f"RSI should be near 100 for all gains, got {last_rsi}"


def test_rsi_all_losses():
    """Monotonically decreasing -> RSI close to 0."""
    n = 30
    closes = [200.0 - i for i in range(n)]
    bars = _make_bars(_ts_seq(n), closes)
    result = compute_ohlcv_indicators(bars)

    last_rsi = result["rsi_14"][-1]
    assert last_rsi is not None
    assert last_rsi < 10, f"RSI should be near 0 for all losses, got {last_rsi}"


def test_rsi_bounded():
    """RSI must be in [0, 100]."""
    n = 50
    import math
    closes = [100.0 + 10 * math.sin(i / 3.0) for i in range(n)]
    bars = _make_bars(_ts_seq(n), closes)
    result = compute_ohlcv_indicators(bars)

    valid = result.filter(pl.col("rsi_14").is_not_null())
    assert (valid["rsi_14"] >= -0.01).all(), "RSI has values below 0"
    assert (valid["rsi_14"] <= 100.01).all(), "RSI has values above 100"


# --------------- ATR ---------------

def test_atr_constant_price():
    """Constant OHLC -> ATR = 0."""
    n = 30
    bars = _make_bars(_ts_seq(n), [100.0] * n)
    result = compute_ohlcv_indicators(bars)

    valid = result.filter(pl.col("atr_norm_14").is_not_null())
    if len(valid) > 0:
        last = valid["atr_norm_14"][-1]
        assert abs(last) < 1e-10, f"ATR should be 0 for constant price, got {last}"


def test_atr_positive():
    """ATR should be positive when there's price movement."""
    n = 30
    closes = [100.0 + i for i in range(n)]
    highs = [c + 2 for c in closes]
    lows = [c - 2 for c in closes]
    bars = _make_bars(_ts_seq(n), closes, highs=highs, lows=lows)
    result = compute_ohlcv_indicators(bars)

    valid = result.filter(pl.col("atr_norm_14").is_not_null())
    assert len(valid) > 0
    assert (valid["atr_norm_14"] > 0).all(), "ATR should be positive with price movement"


def test_atr_normalized():
    """ATR normalized by close should be scale-free."""
    n = 30
    closes_low = [100.0 + i for i in range(n)]
    closes_high = [1000.0 + i * 10 for i in range(n)]

    highs_low = [c + 2 for c in closes_low]
    lows_low = [c - 2 for c in closes_low]
    highs_high = [c + 20 for c in closes_high]
    lows_high = [c - 20 for c in closes_high]

    bars_low = _make_bars(_ts_seq(n), closes_low, highs=highs_low, lows=lows_low)
    bars_high = _make_bars(_ts_seq(n), closes_high, highs=highs_high, lows=lows_high)

    result_low = compute_ohlcv_indicators(bars_low)
    result_high = compute_ohlcv_indicators(bars_high)

    atr_low = result_low.filter(pl.col("atr_norm_14").is_not_null())["atr_norm_14"][-1]
    atr_high = result_high.filter(pl.col("atr_norm_14").is_not_null())["atr_norm_14"][-1]
    assert abs(atr_low - atr_high) < 0.01, \
        f"Normalized ATR should be similar: {atr_low:.4f} vs {atr_high:.4f}"


# --------------- Bollinger Bands ---------------

def test_bb_pctb_constant_price():
    """Constant price -> %B undefined (bandwidth=0), but should not crash."""
    n = 30
    bars = _make_bars(_ts_seq(n), [100.0] * n)
    compute_ohlcv_indicators(bars)


def test_bb_bandwidth_positive():
    """Bandwidth should be positive with price variance."""
    n = 30
    import math
    closes = [100.0 + 5 * math.sin(i / 3.0) for i in range(n)]
    bars = _make_bars(_ts_seq(n), closes)
    result = compute_ohlcv_indicators(bars)

    valid = result.filter(pl.col("bb_bandwidth_20").is_not_null())
    if len(valid) > 0:
        assert (valid["bb_bandwidth_20"] > 0).all(), "Bandwidth should be positive"


def test_bb_pctb_midband():
    """When close oscillates around mean, %B should average near 0.5."""
    n = 40
    import math
    closes = [100.0 + 5 * math.sin(2 * math.pi * i / 20) for i in range(n)]
    bars = _make_bars(_ts_seq(n), closes)
    result = compute_ohlcv_indicators(bars)

    valid = result.filter(pl.col("bb_pctb_20").is_not_null())
    if len(valid) > 5:
        mean_pctb = valid["bb_pctb_20"].mean()
        assert 0.2 < mean_pctb < 0.8, f"Mean %B should be near 0.5, got {mean_pctb}"


# --------------- MACD ---------------

def test_macd_constant_price():
    """Constant price -> MACD = 0, signal = 0, histogram = 0."""
    n = 40
    bars = _make_bars(_ts_seq(n), [100.0] * n)
    result = compute_ohlcv_indicators(bars)

    for col in ["macd_norm", "macd_signal_norm", "macd_hist_norm"]:
        valid = result.filter(pl.col(col).is_not_null())
        if len(valid) > 0:
            last = valid[col][-1]
            assert abs(last) < 1e-10, f"{col} should be 0 for constant price, got {last}"


def test_macd_uptrend_positive():
    """In strong uptrend, MACD line should be positive (EMA12 > EMA26)."""
    n = 40
    closes = [100.0 + i for i in range(n)]
    bars = _make_bars(_ts_seq(n), closes)
    result = compute_ohlcv_indicators(bars)

    last = result["macd_norm"][-1]
    assert last is not None
    assert last > 0, f"MACD should be positive in uptrend, got {last}"


# --------------- Stochastic ---------------

def test_stoch_monotonic_up():
    """Monotonically increasing -> %K should be 100 (close = highest high)."""
    n = 20
    closes = [100.0 + i for i in range(n)]
    highs = closes[:]
    lows = closes[:]
    bars = _make_bars(_ts_seq(n), closes, highs=highs, lows=lows)
    result = compute_ohlcv_indicators(bars)

    valid = result.filter(pl.col("stoch_k_14").is_not_null())
    if len(valid) > 0:
        last_k = valid["stoch_k_14"][-1]
        assert abs(last_k - 100.0) < 0.01, f"%K should be 100 in monotonic uptrend, got {last_k}"


def test_stoch_bounded():
    """Stochastic %K and %D should be in [0, 100]."""
    n = 50
    import math
    closes = [100.0 + 10 * math.sin(i / 3.0) for i in range(n)]
    highs = [c + 2 for c in closes]
    lows = [c - 2 for c in closes]
    bars = _make_bars(_ts_seq(n), closes, highs=highs, lows=lows)
    result = compute_ohlcv_indicators(bars)

    for col in ["stoch_k_14", "stoch_d_14"]:
        valid = result.filter(pl.col(col).is_not_null())
        assert (valid[col] >= -0.01).all(), f"{col} has values below 0"
        assert (valid[col] <= 100.01).all(), f"{col} has values above 100"


# --------------- ADX ---------------

def test_adx_bounded():
    """ADX and DI values should be in [0, 100]."""
    n = 50
    import math
    closes = [100.0 + 10 * math.sin(i / 3.0) for i in range(n)]
    highs = [c + 3 for c in closes]
    lows = [c - 3 for c in closes]
    bars = _make_bars(_ts_seq(n), closes, highs=highs, lows=lows)
    result = compute_ohlcv_indicators(bars)

    for col in ["adx_14", "plus_di_14", "minus_di_14"]:
        valid = result.filter(pl.col(col).is_not_null())
        if len(valid) > 0:
            assert (valid[col] >= -0.01).all(), f"{col} has values below 0"
            assert (valid[col] <= 100.01).all(), f"{col} has values above 100"


def test_adx_trending_market():
    """Strong trend -> ADX should be high (>25)."""
    n = 50
    closes = [100.0 + 2 * i for i in range(n)]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    bars = _make_bars(_ts_seq(n), closes, highs=highs, lows=lows)
    result = compute_ohlcv_indicators(bars)

    valid = result.filter(pl.col("adx_14").is_not_null())
    if len(valid) > 5:
        last_adx = valid["adx_14"][-1]
        assert last_adx > 25, f"ADX should be >25 in strong trend, got {last_adx}"


# --------------- OBV ---------------

def test_obv_slope_constant_price():
    """Constant price -> OBV direction undefined, slope should be 0 or near 0."""
    n = 30
    bars = _make_bars(_ts_seq(n), [100.0] * n)
    result = compute_ohlcv_indicators(bars)

    valid = result.filter(pl.col("obv_slope_14").is_not_null())
    if len(valid) > 0:
        last = valid["obv_slope_14"][-1]
        assert abs(last) < 0.01, f"OBV slope should be ~0 for constant price, got {last}"


def test_obv_slope_uptrend():
    """Monotonic uptrend -> all volumes added -> OBV growing -> positive slope."""
    n = 30
    closes = [100.0 + i for i in range(n)]
    bars = _make_bars(_ts_seq(n), closes)
    result = compute_ohlcv_indicators(bars)

    valid = result.filter(pl.col("obv_slope_14").is_not_null())
    if len(valid) > 0:
        last = valid["obv_slope_14"][-1]
        assert last > 0, f"OBV slope should be positive in uptrend, got {last}"
