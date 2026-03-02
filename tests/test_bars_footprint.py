"""Tests for footprint aggregation pass in bars.py."""

import polars as pl
import numpy as np
from datetime import datetime

from src.framework.data.bars import (
    aggregate_time_bars,
    aggregate_volume_bars,
    _compute_footprint_for_group,
)


# --- Helpers ---------------------------------------------------------------

BOOK_DEFAULTS = {
    "bid_px_00": 21000.0,
    "ask_px_00": 21000.50,
    "bid_sz_00": 10,
    "ask_sz_00": 8,
    "bid_ct_00": 3,
    "ask_ct_00": 2,
}

DTYPES = {
    "ts_event": pl.Datetime("ns", "UTC"),
    "action": pl.Utf8,
    "side": pl.Utf8,
    "size": pl.UInt32,
    "price": pl.Float64,
    "bid_px_00": pl.Float64,
    "ask_px_00": pl.Float64,
    "bid_sz_00": pl.UInt32,
    "ask_sz_00": pl.UInt32,
    "bid_ct_00": pl.UInt32,
    "ask_ct_00": pl.UInt32,
    "ts_recv": pl.Datetime("ns", "UTC"),
}


def _row(ts, action, side, size, price, **overrides):
    r = {
        "ts_event": ts,
        "action": action,
        "side": side,
        "size": size,
        "price": price,
        "ts_recv": ts,
        **BOOK_DEFAULTS,
    }
    r.update(overrides)
    return r


def _to_lf(rows):
    df = pl.DataFrame(rows, schema_overrides=DTYPES)
    return df.lazy()


# --- Tests: _compute_footprint_for_group -----------------------------------

def test_footprint_empty_group():
    """Empty trades should produce all-zero footprint stats."""
    stats = _compute_footprint_for_group(
        np.array([]), np.array([], dtype=np.uint64), []
    )
    assert stats["stacked_imbalance_count"] == 0
    assert stats["zero_print_count"] == 0
    assert stats["unfinished_high"] == 0
    assert stats["unfinished_low"] == 0
    assert stats["max_level_volume"] == 0


def test_footprint_single_trade():
    """Single trade: no stacks, no zero prints, volume = that trade."""
    prices = np.array([21000.0])
    sizes = np.array([10], dtype=np.uint64)
    sides = ["A"]
    stats = _compute_footprint_for_group(prices, sizes, sides)

    assert stats["stacked_imbalance_count"] == 0
    assert stats["zero_print_count"] == 0
    assert stats["zero_print_ratio"] == 0.0
    # Single level = high = low, only buy volume at high -> no unfinished (need both sides)
    assert stats["unfinished_high"] == 0
    assert stats["unfinished_low"] == 0
    assert stats["max_level_volume"] == 10
    assert stats["volume_at_high"] == 10
    assert stats["volume_at_low"] == 10


def test_footprint_unfinished_business():
    """Unfinished business: both buy and sell volume at high/low."""
    # High at 21001.0: buy=10, sell=5 -> unfinished_high=1
    # Low at 21000.0: only sell -> unfinished_low=0
    prices = np.array([21001.0, 21001.0, 21000.0])
    sizes = np.array([10, 5, 8], dtype=np.uint64)
    sides = ["A", "B", "B"]
    stats = _compute_footprint_for_group(prices, sizes, sides)

    assert stats["unfinished_high"] == 1, "Both sides traded at high"
    assert stats["unfinished_low"] == 0, "Only sells at low"


def test_footprint_unfinished_both():
    """Both high and low have two-sided action."""
    prices = np.array([21000.0, 21000.0, 21001.0, 21001.0])
    sizes = np.array([10, 5, 8, 3], dtype=np.uint64)
    sides = ["A", "B", "A", "B"]
    stats = _compute_footprint_for_group(prices, sizes, sides)

    assert stats["unfinished_high"] == 1
    assert stats["unfinished_low"] == 1


def test_footprint_zero_prints():
    """Zero prints: levels in [low, high] with no volume."""
    # Trades at 21000.0 and 21001.0 (4 tick levels apart)
    # Levels: 21000.0, 21000.25, 21000.50, 21000.75, 21001.0
    # Only 21000.0 and 21001.0 have volume => 3 zero prints
    prices = np.array([21000.0, 21001.0])
    sizes = np.array([10, 5], dtype=np.uint64)
    sides = ["A", "B"]
    stats = _compute_footprint_for_group(prices, sizes, sides)

    assert stats["zero_print_count"] == 3, f"Expected 3 gaps, got {stats['zero_print_count']}"
    # 3 out of 5 levels have zero prints
    assert abs(stats["zero_print_ratio"] - 3.0 / 5.0) < 1e-6


def test_footprint_no_zero_prints():
    """Consecutive tick levels with volume -> zero prints = 0."""
    prices = np.array([21000.0, 21000.25, 21000.50])
    sizes = np.array([10, 5, 8], dtype=np.uint64)
    sides = ["A", "B", "A"]
    stats = _compute_footprint_for_group(prices, sizes, sides)

    assert stats["zero_print_count"] == 0
    assert stats["zero_print_ratio"] == 0.0


def test_footprint_max_level_volume():
    """max_level_volume = highest total volume at any single level."""
    prices = np.array([21000.0, 21000.0, 21000.25, 21000.25])
    sizes = np.array([10, 5, 20, 3], dtype=np.uint64)
    sides = ["A", "B", "A", "B"]
    stats = _compute_footprint_for_group(prices, sizes, sides)

    # 21000.0: 10+5=15, 21000.25: 20+3=23
    assert stats["max_level_volume"] == 23
    assert stats["volume_at_high"] == 23  # high is 21000.25
    assert stats["volume_at_low"] == 15   # low is 21000.0


def test_footprint_stacked_imbalances_buy():
    """3+ consecutive levels with buy diagonal imbalance -> stacked count."""
    # Build 4 consecutive levels where buy@level >> sell@(level-tick)
    # Level 0: 21000.00 sell=1 buy=0
    # Level 1: 21000.25 sell=1 buy=10 -> buy@1 vs sell@0: 10 vs 1 = 10x (imbalance)
    # Level 2: 21000.50 sell=1 buy=10 -> buy@2 vs sell@1: 10 vs 1 = 10x (imbalance)
    # Level 3: 21000.75 sell=1 buy=10 -> buy@3 vs sell@2: 10 vs 1 = 10x (imbalance)
    # 3 consecutive buy imbalances -> 1 stacked buy
    prices = np.array([21000.0, 21000.25, 21000.50, 21000.75,
                       21000.0, 21000.25, 21000.50, 21000.75])
    sizes = np.array([1, 10, 10, 10,
                      0, 1, 1, 1], dtype=np.uint64)
    # First 4 are buys ("A"), next 4 are sells ("B")
    sides = ["B", "A", "A", "A",
             "B", "B", "B", "B"]
    stats = _compute_footprint_for_group(prices, sizes, sides)

    assert stats["stacked_imbalance_count"] == 1, (
        f"Expected exactly 1 stacked buy imbalance, got {stats['stacked_imbalance_count']}"
    )
    assert stats["stacked_imbalance_direction"] == 1, "Should be buy-dominant"


def test_footprint_no_stacked_imbalances():
    """Balanced volume at all levels -> no stacked imbalances."""
    prices = np.array([21000.0, 21000.0, 21000.25, 21000.25,
                       21000.50, 21000.50])
    sizes = np.array([10, 10, 10, 10, 10, 10], dtype=np.uint64)
    sides = ["A", "B", "A", "B", "A", "B"]
    stats = _compute_footprint_for_group(prices, sizes, sides)

    assert stats["stacked_imbalance_count"] == 0


# --- Tests: time bars footprint integration ---------------------------------

def test_time_bars_have_footprint_columns():
    """Time bars output includes all footprint columns."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        _row(ts(0), "T", "A", 10, 21000.0),
        _row(ts(1), "T", "B", 5, 21000.25),
        _row(ts(2), "T", "A", 8, 21000.50),
    ]
    lf = _to_lf(rows)
    result = aggregate_time_bars(lf, bar_size="5m")

    footprint_cols = [
        "stacked_imbalance_count", "stacked_imbalance_direction",
        "zero_print_count", "zero_print_ratio",
        "unfinished_high", "unfinished_low",
        "max_level_volume", "volume_at_high", "volume_at_low",
    ]
    for col in footprint_cols:
        assert col in result.columns, f"Missing footprint column: {col}"


def test_time_bars_footprint_values():
    """Verify footprint values on a known trade sequence."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        # Bar: trades at 21000.0 (buy=10), 21000.25 (sell=5), 21000.50 (buy=8)
        _row(ts(0), "T", "A", 10, 21000.0),
        _row(ts(1), "T", "B", 5, 21000.25),
        _row(ts(2), "T", "A", 8, 21000.50),
    ]
    lf = _to_lf(rows)
    result = aggregate_time_bars(lf, bar_size="5m")
    bar = result.row(0, named=True)

    # 3 consecutive levels, all have volume -> 0 zero prints
    assert bar["zero_print_count"] == 0
    assert bar["zero_print_ratio"] == 0.0

    # max_level_volume = max(10, 5, 8) = 10
    assert bar["max_level_volume"] == 10

    # volume_at_high = 21000.50 has buy=8 only -> 8
    assert bar["volume_at_high"] == 8
    # volume_at_low = 21000.0 has buy=10 only -> 10
    assert bar["volume_at_low"] == 10

    # Unfinished: high (21000.50) only has buys -> not unfinished
    assert bar["unfinished_high"] == 0
    # Low (21000.0) only has buys -> not unfinished
    assert bar["unfinished_low"] == 0


def test_time_bars_footprint_unfinished():
    """Both sides traded at bar high/low -> unfinished business."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        # Trades at bar high (21001.0): buy and sell
        _row(ts(0), "T", "A", 10, 21001.0),
        _row(ts(1), "T", "B", 5, 21001.0),
        # Trades at bar low (21000.0): buy and sell
        _row(ts(2), "T", "A", 3, 21000.0),
        _row(ts(3), "T", "B", 8, 21000.0),
    ]
    lf = _to_lf(rows)
    result = aggregate_time_bars(lf, bar_size="5m")
    bar = result.row(0, named=True)

    assert bar["unfinished_high"] == 1
    assert bar["unfinished_low"] == 1


def test_volume_bars_have_footprint_columns():
    """Volume bars output includes all footprint columns."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        _row(ts(0), "T", "A", 10, 21000.0),
        _row(ts(1), "T", "B", 10, 21000.25),
        _row(ts(2), "T", "A", 10, 21000.50),
    ]
    lf = _to_lf(rows)
    result = aggregate_volume_bars(lf, volume_threshold=20)

    footprint_cols = [
        "stacked_imbalance_count", "stacked_imbalance_direction",
        "zero_print_count", "zero_print_ratio",
        "unfinished_high", "unfinished_low",
        "max_level_volume", "volume_at_high", "volume_at_low",
    ]
    for col in footprint_cols:
        assert col in result.columns, f"Missing footprint column: {col}"


def test_volume_bars_footprint_values():
    """Verify footprint values computed per volume bucket."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        # Bucket 0: cumvol 1-20
        _row(ts(0), "T", "A", 10, 21000.0),
        _row(ts(1), "T", "B", 10, 21000.0),
        # Bucket 1: cumvol 21-30
        _row(ts(2), "T", "A", 10, 21001.0),
    ]
    lf = _to_lf(rows)
    result = aggregate_volume_bars(lf, volume_threshold=20)

    assert len(result) == 2

    # Bucket 0: both buy and sell at same level -> unfinished
    bar0 = result.row(0, named=True)
    assert bar0["unfinished_high"] == 1  # same level is both high and low
    assert bar0["unfinished_low"] == 1
    assert bar0["max_level_volume"] == 20  # 10+10 at same level

    # Bucket 1: single trade, single level
    bar1 = result.row(1, named=True)
    assert bar1["max_level_volume"] == 10
