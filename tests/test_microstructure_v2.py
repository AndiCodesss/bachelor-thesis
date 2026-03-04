"""Tests for microstructure_v2 features."""
import polars as pl
import pytest
from datetime import datetime
from src.framework.features_canonical.microstructure_v2 import compute_microstructure_v2_features


def _make_bars(rows):
    """Helper: build bars DataFrame from list of dicts."""
    n = len(rows)
    return pl.DataFrame({
        "ts_event": [r["ts_event"] for r in rows],
        "ts_close": [r["ts_event"] for r in rows],
        "bar_duration_ns": [60_000_000_000] * n,
        "open": [20000.0] * n,
        "high": [20001.0] * n,
        "low": [19999.0] * n,
        "close": [20000.25] * n,
        "volume": pl.Series([r.get("volume", 100) for r in rows], dtype=pl.UInt32),
        "vwap": [20000.0] * n,
        "trade_count": pl.Series([r.get("trade_count", 10) for r in rows], dtype=pl.UInt32),
        "buy_volume": pl.Series([r["buy_volume"] for r in rows], dtype=pl.UInt32),
        "sell_volume": pl.Series([r["sell_volume"] for r in rows], dtype=pl.UInt32),
        "large_trade_count": pl.Series([0] * n, dtype=pl.UInt32),
        "large_buy_volume": pl.Series([0] * n, dtype=pl.UInt32),
        "large_sell_volume": pl.Series([0] * n, dtype=pl.UInt32),
        "bid_price": [r.get("bid_price", 20000.0) for r in rows],
        "ask_price": [r.get("ask_price", 20000.25) for r in rows],
        "bid_size": pl.Series([r.get("bid_size", 10) for r in rows], dtype=pl.UInt32),
        "ask_size": pl.Series([r.get("ask_size", 8) for r in rows], dtype=pl.UInt32),
        "bid_count": pl.Series([r.get("bid_count", 5) for r in rows], dtype=pl.UInt32),
        "ask_count": pl.Series([r.get("ask_count", 4) for r in rows], dtype=pl.UInt32),
        "msg_count": pl.Series([r.get("msg_count", 200) for r in rows], dtype=pl.UInt32),
        "add_count": pl.Series([r.get("add_count", 100) for r in rows], dtype=pl.UInt32),
        "cancel_count": pl.Series([r.get("cancel_count", 50) for r in rows], dtype=pl.UInt32),
        "modify_count": pl.Series([r.get("modify_count", 20) for r in rows], dtype=pl.UInt32),
    }).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))


@pytest.fixture
def sample_bars():
    """Create sample bars for testing."""
    return _make_bars([
        {"ts_event": datetime(2024, 1, 2, 9, 30 + i), "buy_volume": 50, "sell_volume": 50}
        for i in range(6)
    ])


def test_microstructure_v2_output_shape(sample_bars):
    """Test that output has correct shape and columns."""
    result = compute_microstructure_v2_features(sample_bars)

    assert len(result) == 6

    expected_cols = [
        "ts_event",
        "trade_arrival_imbalance",
        "vpin",
        "cancel_ratio",
        "weighted_book_imbalance",
        "micro_price_momentum"
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_trade_arrival_imbalance_bounded(sample_bars):
    """Test trade arrival imbalance is bounded [-1, 1]."""
    result = compute_microstructure_v2_features(sample_bars)
    non_null = result["trade_arrival_imbalance"].drop_nulls()
    assert (non_null >= -1.0).all()
    assert (non_null <= 1.0).all()


def test_trade_arrival_imbalance_known_value():
    """Hand-computed: buy_vol=15, sell_vol=5 -> (15-5)/(15+5) = 0.5"""
    bars = _make_bars([{
        "ts_event": datetime(2024, 1, 2, 9, 30, 0),
        "buy_volume": 15,
        "sell_volume": 5,
    }])

    result = compute_microstructure_v2_features(bars)
    actual = result["trade_arrival_imbalance"][0]
    assert abs(actual - 0.5) < 1e-6, f"Expected 0.5, got {actual}"


def test_vpin_known_value():
    """Hand-computed: buy_vol=15, sell_vol=5, VPIN = |15-5|/20 = 0.5"""
    bars = _make_bars([{
        "ts_event": datetime(2024, 1, 2, 9, 30, 0),
        "buy_volume": 15,
        "sell_volume": 5,
    }])

    result = compute_microstructure_v2_features(bars)
    actual = result["vpin"][0]
    assert abs(actual - 0.5) < 1e-6, f"Expected 0.5, got {actual}"


def test_vpin_values(sample_bars):
    """Test VPIN is bounded [0, 1]."""
    result = compute_microstructure_v2_features(sample_bars)
    non_null = result["vpin"].drop_nulls()
    assert (non_null >= 0).all()
    assert (non_null <= 1).all()


def test_cancel_ratio(sample_bars):
    """Test cancel ratio is non-negative."""
    result = compute_microstructure_v2_features(sample_bars)
    non_null = result["cancel_ratio"].drop_nulls()
    assert (non_null >= 0).all()


def test_weighted_book_imbalance_known_value():
    """Hand-computed: bid_sz=10, bid_ct=5, ask_sz=8, ask_ct=4
    bid_weight = 10*5 = 50, ask_weight = 8*4 = 32
    imbalance = (50-32)/(50+32+1) = 18/83 ~ 0.2169
    """
    bars = _make_bars([{
        "ts_event": datetime(2024, 1, 2, 9, 30, 0),
        "buy_volume": 15,
        "sell_volume": 5,
        "bid_size": 10,
        "ask_size": 8,
        "bid_count": 5,
        "ask_count": 4,
    }])

    result = compute_microstructure_v2_features(bars)
    expected = (50 - 32) / (50 + 32 + 1)
    actual = result["weighted_book_imbalance"][0]
    assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"


def test_weighted_book_imbalance_bounded(sample_bars):
    """Test weighted book imbalance is bounded [-1, 1]."""
    result = compute_microstructure_v2_features(sample_bars)
    non_null = result["weighted_book_imbalance"].drop_nulls()
    assert (non_null >= -1.01).all()
    assert (non_null <= 1.01).all()


def test_weighted_book_imbalance_negative_when_ask_side_dominates():
    """Unsigned inputs must not underflow when ask weight exceeds bid weight."""
    bars = _make_bars([{
        "ts_event": datetime(2024, 1, 2, 9, 30, 0),
        "buy_volume": 10,
        "sell_volume": 10,
        "bid_size": 1,
        "bid_count": 1,
        "ask_size": 10,
        "ask_count": 10,
    }])

    result = compute_microstructure_v2_features(bars)
    val = result["weighted_book_imbalance"][0]
    assert val < 0
    assert abs(val - ((1 - 100) / (1 + 100 + 1))) < 1e-9


def test_micro_price_momentum_first_bar_null():
    """First bar has no previous bar, so micro_price_momentum should be null."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 1, 2, 9, 30), "buy_volume": 10, "sell_volume": 10},
        {"ts_event": datetime(2024, 1, 2, 9, 31), "buy_volume": 10, "sell_volume": 10},
    ])

    result = compute_microstructure_v2_features(bars)
    assert result["micro_price_momentum"][0] is None


def test_micro_price_momentum_known_value():
    """Hand-computed: equal bid/ask sizes -> micro_price = mid_price.
    Bar 0: mid = (20000 + 20000.25)/2 = 20000.125
    Bar 1: mid = (20000.50 + 20000.75)/2 = 20000.625
    momentum = (20000.625 - 20000.125) / 20000.125
    """
    bars = _make_bars([
        {"ts_event": datetime(2024, 1, 2, 9, 30), "buy_volume": 10, "sell_volume": 10,
         "bid_price": 20000.0, "ask_price": 20000.25, "bid_size": 10, "ask_size": 10},
        {"ts_event": datetime(2024, 1, 2, 9, 31), "buy_volume": 10, "sell_volume": 10,
         "bid_price": 20000.50, "ask_price": 20000.75, "bid_size": 10, "ask_size": 10},
        {"ts_event": datetime(2024, 1, 2, 9, 32), "buy_volume": 10, "sell_volume": 10,
         "bid_price": 20001.0, "ask_price": 20001.25, "bid_size": 10, "ask_size": 10},
    ])

    result = compute_microstructure_v2_features(bars)

    # Bar 0 micro = (20000*10 + 20000.25*10)/(10+10+1e-9) ~ 20000.125
    # Bar 1 micro = (20000.50*10 + 20000.75*10)/(10+10+1e-9) ~ 20000.625
    expected = (20000.625 - 20000.125) / 20000.125
    actual = result["micro_price_momentum"][1]
    assert actual is not None
    assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"


def test_empty_bar_handling():
    """Test handling of bars with zero volume."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 1, 2, 9, 30), "buy_volume": 5, "sell_volume": 3},
        {"ts_event": datetime(2024, 1, 2, 9, 35), "buy_volume": 0, "sell_volume": 0, "trade_count": 0},
    ])

    result = compute_microstructure_v2_features(bars)
    assert len(result) == 2
    # Second bar with zero volume should have null trade_arrival_imbalance
    assert result["trade_arrival_imbalance"][1] is None
    # vpin uses rolling_mean which propagates from previous bars — not null
    assert result["vpin"][1] is not None


def test_vpin_rolling_window():
    """Test VPIN uses rolling window correctly."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 1, 2, 9, 30 + i), "buy_volume": 100, "sell_volume": 0}
        for i in range(20)
    ])

    result = compute_microstructure_v2_features(bars)
    # All buys, vpin_raw = 1.0 for every bar, rolling mean = 1.0
    assert result["vpin"].max() > 0.8
