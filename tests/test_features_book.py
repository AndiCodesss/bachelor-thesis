"""Tests for order book feature computation."""

import pytest
import polars as pl
from datetime import datetime
from src.framework.features_canonical.book import compute_book_features


def _make_bars(rows):
    """Helper: build bars DataFrame from list of dicts with book-level fields."""
    n = len(rows)
    return pl.DataFrame({
        "ts_event": [r["ts_event"] for r in rows],
        "ts_close": [r["ts_event"] for r in rows],
        "bar_duration_ns": [300_000_000_000] * n,
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": pl.Series([100] * n, dtype=pl.UInt32),
        "vwap": [100.25] * n,
        "trade_count": pl.Series([10] * n, dtype=pl.UInt32),
        "buy_volume": pl.Series([50] * n, dtype=pl.UInt32),
        "sell_volume": pl.Series([50] * n, dtype=pl.UInt32),
        "large_trade_count": pl.Series([1] * n, dtype=pl.UInt32),
        "large_buy_volume": pl.Series([0] * n, dtype=pl.UInt32),
        "large_sell_volume": pl.Series([0] * n, dtype=pl.UInt32),
        "bid_price": [r["bid_price"] for r in rows],
        "ask_price": [r["ask_price"] for r in rows],
        "bid_size": pl.Series([r["bid_size"] for r in rows], dtype=pl.UInt32),
        "ask_size": pl.Series([r["ask_size"] for r in rows], dtype=pl.UInt32),
        "bid_count": pl.Series([r["bid_count"] for r in rows], dtype=pl.UInt32),
        "ask_count": pl.Series([r["ask_count"] for r in rows], dtype=pl.UInt32),
        "msg_count": pl.Series([100] * n, dtype=pl.UInt32),
        "add_count": pl.Series([50] * n, dtype=pl.UInt32),
        "cancel_count": pl.Series([20] * n, dtype=pl.UInt32),
        "modify_count": pl.Series([10] * n, dtype=pl.UInt32),
    }).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))


def test_compute_book_features_synthetic():
    """Test book features with synthetic known-answer data."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 1, 15, 10, 0, 0), "bid_price": 100.5, "ask_price": 101.0,
         "bid_size": 15, "ask_size": 12, "bid_count": 4, "ask_count": 3},
        {"ts_event": datetime(2024, 1, 15, 10, 5, 0), "bid_price": 101.5, "ask_price": 102.0,
         "bid_size": 16, "ask_size": 13, "bid_count": 3, "ask_count": 2},
        {"ts_event": datetime(2024, 1, 15, 10, 10, 0), "bid_price": 102.5, "ask_price": 103.0,
         "bid_size": 20, "ask_size": 16, "bid_count": 4, "ask_count": 3},
    ])

    result = compute_book_features(bars)

    assert len(result) == 3, f"Expected 3 bars, got {len(result)}"

    # Bar 1: bid=100.5, ask=101.0, mid=100.75, spread=0.5
    assert result["bid_price"][0] == 100.5
    assert result["ask_price"][0] == 101.0
    assert result["mid_price"][0] == 100.75
    assert result["spread"][0] == 0.5

    # Bar 2: bid=101.5, ask=102.0, mid=101.75, spread=0.5
    assert result["bid_price"][1] == 101.5
    assert result["ask_price"][1] == 102.0
    assert result["mid_price"][1] == 101.75
    assert result["spread"][1] == 0.5

    # Bar 3: bid=102.5, ask=103.0, mid=102.75, spread=0.5
    assert result["bid_price"][2] == 102.5
    assert result["ask_price"][2] == 103.0
    assert result["mid_price"][2] == 102.75
    assert result["spread"][2] == 0.5

    # spread_bps = 0.5 / 100.75 * 10000 ~ 49.627
    assert abs(result["spread_bps"][0] - 49.627) < 1.0

    # book_imbalance = (15 - 12) / (15 + 12 + 1) = 3 / 28
    expected_imbalance = (15 - 12) / (15 + 12 + 1)
    assert abs(result["book_imbalance"][0] - expected_imbalance) < 0.001

    # depth_ratio = 4 / (3 + 1) = 1.0
    assert result["depth_ratio"][0] == 1.0

    # mid_price_return: null for first bar
    assert result["mid_price_return"][0] is None

    # mid_price_return for bar 2: (101.75 - 100.75) / 100.75
    expected_return = (101.75 - 100.75) / 100.75
    assert abs(result["mid_price_return"][1] - expected_return) < 0.00001


def test_compute_book_features_has_expected_columns():
    """Verify output has all expected feature columns."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 1, 15, 10, i * 5, 0), "bid_price": 100.0 + i * 0.25,
         "ask_price": 100.5 + i * 0.25, "bid_size": 10 + i, "ask_size": 8 + i,
         "bid_count": 2 + i % 3, "ask_count": 2 + i % 2}
        for i in range(5)
    ])

    result = compute_book_features(bars)

    expected_columns = [
        "ts_event",
        "bid_price",
        "ask_price",
        "mid_price",
        "spread",
        "spread_bps",
        "book_imbalance",
        "book_imbalance_ma5",
        "depth_ratio",
        "spread_volatility",
        "mid_price_return",
        "mid_price_return_5",
    ]

    for col in expected_columns:
        assert col in result.columns, f"Expected column '{col}' not found"

    assert len(result.columns) == len(expected_columns), f"Extra columns found: {set(result.columns) - set(expected_columns)}"


def test_compute_book_features_no_nulls_in_base_features():
    """Verify that base features (non-rolling) have no nulls."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 1, 15, 10, i * 5, 0), "bid_price": 100.0 + i * 0.25,
         "ask_price": 100.5 + i * 0.25, "bid_size": 10 + i, "ask_size": 8 + i,
         "bid_count": 2 + i % 3, "ask_count": 2 + i % 2}
        for i in range(5)
    ])

    result = compute_book_features(bars)

    base_features = [
        "ts_event", "bid_price", "ask_price", "mid_price", "spread",
        "spread_bps", "book_imbalance", "depth_ratio",
    ]

    for col in base_features:
        null_count = result[col].null_count()
        assert null_count == 0, f"Column '{col}' has {null_count} nulls, expected 0"


def test_compute_book_features_rolling_features_have_expected_nulls():
    """Verify rolling features have expected nulls."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 1, 15, 10, i * 5, 0), "bid_price": 100.0 + i * 0.1,
         "ask_price": 100.5 + i * 0.1, "bid_size": 10 + i, "ask_size": 8 + i,
         "bid_count": 2 + i % 3, "ask_count": 2 + i % 2}
        for i in range(10)
    ])

    result = compute_book_features(bars)

    assert result["book_imbalance_ma5"].null_count() == 0
    assert result["spread_volatility"].null_count() <= 1
    assert result["mid_price_return"].null_count() == 1
    assert result["mid_price_return_5"].null_count() == 1


def test_compute_book_features_output_shape_preserved():
    """Verify number of output bars matches input bar count."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 1, 15, 10, i * 5, 0), "bid_price": 100.0,
         "ask_price": 100.5, "bid_size": 10, "ask_size": 8, "bid_count": 2, "ask_count": 2}
        for i in range(6)
    ])

    result = compute_book_features(bars)
    assert len(result) == 6, f"Expected 6 bars, got {len(result)}"
