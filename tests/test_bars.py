"""Tests for shared bar aggregation layer (src/data/bars.py)."""

import polars as pl
from datetime import datetime

from src.framework.data.bars import (
    aggregate_time_bars,
    aggregate_volume_bars,
    aggregate_tick_bars,
    WHALE_LOT_THRESHOLD,
    _output_columns,
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
    """Build a single MBP1 event dict."""
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
    """Convert list of row dicts to a UTC-aware LazyFrame."""
    df = pl.DataFrame(rows, schema_overrides=DTYPES)
    return df.lazy()


def _make_simple_session():
    """Build a two-bar session with known trade and book values.

    Bar 1 (10:00-10:04): 3 trades + 1 book add
    Bar 2 (10:05-10:09): 2 trades + 1 cancel
    """
    def ts(m, s=0):
        return datetime(2024, 7, 15, 10, m, s)
    rows = [
        # --- Bar 1 ---
        _row(ts(0, 0), "A", "N", 0, 0.0, bid_px_00=21000.0, ask_px_00=21000.50),       # book add
        _row(ts(0, 1), "T", "A", 5, 21001.0),                                            # buy trade
        _row(ts(1, 0), "T", "B", 3, 21000.0),                                            # sell trade
        _row(ts(2, 0), "T", "A", 12, 21002.0),                                           # large buy
        # --- Bar 2 ---
        _row(ts(5, 0), "C", "N", 0, 0.0, bid_px_00=21001.0, ask_px_00=21001.50),        # cancel
        _row(ts(6, 0), "T", "A", 4, 21003.0),                                            # buy
        _row(ts(7, 0), "T", "B", 15, 21000.50, bid_px_00=21000.50, ask_px_00=21001.25), # large sell
    ]
    return _to_lf(rows)


# --- Tests: time bars schema -----------------------------------------------

def test_time_bars_schema():
    """All canonical columns must be present in time bar output."""
    lf = _make_simple_session()
    result = aggregate_time_bars(lf, bar_size="5m")

    expected = set(_output_columns())
    actual = set(result.columns)
    missing = expected - actual
    assert not missing, f"Missing columns: {missing}"


def test_time_bars_column_order():
    """Columns should appear in canonical order."""
    lf = _make_simple_session()
    result = aggregate_time_bars(lf, bar_size="5m")

    canonical = _output_columns()
    # All canonical columns present, in order (extra columns allowed after)
    actual_order = [c for c in result.columns if c in canonical]
    assert actual_order == canonical


# --- Tests: time bars known values -----------------------------------------

def test_time_bars_values():
    """Verify OHLCV, buy/sell volumes, and message counts on synthetic data."""
    lf = _make_simple_session()
    result = aggregate_time_bars(lf, bar_size="5m")

    assert len(result) == 2, f"Expected 2 bars, got {len(result)}"

    bar1 = result.row(0, named=True)
    bar2 = result.row(1, named=True)

    # --- Bar 1: trades at 21001.0, 21000.0, 21002.0 with sizes 5, 3, 12 ---
    assert bar1["open"] == 21001.0
    assert bar1["high"] == 21002.0
    assert bar1["low"] == 21000.0
    assert bar1["close"] == 21002.0
    assert bar1["volume"] == 20  # 5 + 3 + 12
    assert bar1["trade_count"] == 3

    # VWAP = (21001*5 + 21000*3 + 21002*12) / 20 = (105005 + 63000 + 252024) / 20
    expected_vwap = (21001.0 * 5 + 21000.0 * 3 + 21002.0 * 12) / 20
    assert abs(bar1["vwap"] - expected_vwap) < 0.01

    # Buy volume: side=="A" -> 5 + 12 = 17
    assert bar1["buy_volume"] == 17
    # Sell volume: side=="B" -> 3
    assert bar1["sell_volume"] == 3

    # Large lots (>= 10): trade at size=12 is a buy
    assert bar1["large_trade_count"] == 1
    assert bar1["large_buy_volume"] == 12
    assert bar1["large_sell_volume"] == 0
    assert bar1["whale_trade_count_30"] == 0
    assert bar1["whale_buy_volume_30"] == 0
    assert bar1["whale_sell_volume_30"] == 0

    # --- Bar 2: trades at 21003.0, 21000.50 with sizes 4, 15 ---
    assert bar2["open"] == 21003.0
    assert bar2["high"] == 21003.0
    assert bar2["low"] == 21000.50
    assert bar2["close"] == 21000.50
    assert bar2["volume"] == 19  # 4 + 15
    assert bar2["trade_count"] == 2
    assert bar2["buy_volume"] == 4
    assert bar2["sell_volume"] == 15
    assert bar2["large_trade_count"] == 1
    assert bar2["large_buy_volume"] == 0
    assert bar2["large_sell_volume"] == 15
    assert bar2["whale_trade_count_30"] == 0
    assert bar2["whale_buy_volume_30"] == 0
    assert bar2["whale_sell_volume_30"] == 0


def test_time_bars_vap_histogram():
    """Per-bar VAP arrays should match executed trade distribution."""
    lf = _make_simple_session()
    result = aggregate_time_bars(lf, bar_size="5m")

    bar1 = result.row(0, named=True)
    bar2 = result.row(1, named=True)

    assert bar1["vap_prices"] == [21000.0, 21001.0, 21002.0]
    assert bar1["vap_volumes"] == [3, 5, 12]

    assert bar2["vap_prices"] == [21000.5, 21003.0]
    assert bar2["vap_volumes"] == [15, 4]

    assert sum(bar1["vap_volumes"]) == bar1["volume"]
    assert sum(bar2["vap_volumes"]) == bar2["volume"]


def test_time_bars_book_snapshot():
    """Book columns reflect last snapshot in each bar."""
    lf = _make_simple_session()
    result = aggregate_time_bars(lf, bar_size="5m")

    bar2 = result.row(1, named=True)
    # Last event in bar 2 is the sell trade at ts(7,0) which has bid=21000.50, ask=21001.25
    assert bar2["bid_price"] == 21000.50
    assert bar2["ask_price"] == 21001.25


def test_time_bars_activity_counts():
    """Message counts reflect all event types in each bar."""
    lf = _make_simple_session()
    result = aggregate_time_bars(lf, bar_size="5m")

    bar1 = result.row(0, named=True)
    bar2 = result.row(1, named=True)

    # Bar 1: 1 add + 3 trades = 4 messages
    assert bar1["msg_count"] == 4
    assert bar1["add_count"] == 1
    assert bar1["cancel_count"] == 0
    assert bar1["modify_count"] == 0

    # Bar 2: 1 cancel + 2 trades = 3 messages
    assert bar2["msg_count"] == 3
    assert bar2["add_count"] == 0
    assert bar2["cancel_count"] == 1
    assert bar2["modify_count"] == 0


def test_time_bars_bar_duration():
    """bar_duration_ns should be ts_close - ts_event in nanoseconds."""
    lf = _make_simple_session()
    result = aggregate_time_bars(lf, bar_size="5m")

    bar1 = result.row(0, named=True)
    # Bar 1: last trade at 10:02:00, first trade at 10:00:01
    # ts_event is bar open (10:00:00 from group_by_dynamic), ts_close is last trade ts
    assert bar1["bar_duration_ns"] >= 0


def test_time_bars_no_nulls_in_ohlcv():
    """OHLCV columns should never be null when trades exist."""
    lf = _make_simple_session()
    result = aggregate_time_bars(lf, bar_size="5m")

    for col in ["open", "high", "low", "close", "volume", "vwap"]:
        assert result[col].null_count() == 0, f"Null found in {col}"


def test_time_bars_sorted():
    """Output must be sorted by ts_event."""
    lf = _make_simple_session()
    result = aggregate_time_bars(lf, bar_size="5m")

    timestamps = result["ts_event"].to_list()
    assert timestamps == sorted(timestamps)


# --- Tests: volume bars basic ----------------------------------------------

def test_volume_bars_basic():
    """Volume bars should close when cumulative volume reaches threshold."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        # 5 trades, each size=10 => total=50, with threshold=20: buckets at 20, 40, 50
        _row(ts(0), "T", "A", 10, 21000.0),   # cumvol=10,  bucket 0
        _row(ts(1), "T", "B", 10, 21001.0),   # cumvol=20,  bucket 0
        _row(ts(2), "T", "A", 10, 21002.0),   # cumvol=30,  bucket 1
        _row(ts(3), "T", "B", 10, 21003.0),   # cumvol=40,  bucket 1
        _row(ts(4), "T", "A", 10, 21004.0),   # cumvol=50,  bucket 2
    ]
    lf = _to_lf(rows)
    result = aggregate_volume_bars(lf, volume_threshold=20)

    # bucket_id = (cumvol - 1) // 20: [0,0,1,1,2] => 3 bars
    assert len(result) == 3, f"Expected 3 volume bars, got {len(result)}"

    # Bar 0: trades at 21000, 21001, volume=20
    assert result["open"][0] == 21000.0
    assert result["close"][0] == 21001.0
    assert result["volume"][0] == 20

    # Bar 1: trades at 21002, 21003, volume=20
    assert result["open"][1] == 21002.0
    assert result["close"][1] == 21003.0
    assert result["volume"][1] == 20

    # Bar 2: single trade at 21004, volume=10
    assert result["open"][2] == 21004.0
    assert result["close"][2] == 21004.0
    assert result["volume"][2] == 10


def test_volume_bars_schema_matches_time_bars():
    """Volume bars should have the same columns as time bars."""
    lf = _make_simple_session()
    time_result = aggregate_time_bars(lf, bar_size="5m")
    vol_result = aggregate_volume_bars(lf, volume_threshold=10)

    time_cols = set(time_result.columns)
    vol_cols = set(vol_result.columns)

    # All canonical columns present in both
    canonical = set(_output_columns())
    missing_time = canonical - time_cols
    missing_vol = canonical - vol_cols
    assert not missing_time, f"Time bars missing: {missing_time}"
    assert not missing_vol, f"Volume bars missing: {missing_vol}"


def test_volume_bars_total_volume():
    """Sum of all bar volumes should equal total trade volume."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    sizes = [7, 3, 12, 8, 5, 15, 2, 10, 6, 4]
    total = sum(sizes)

    rows = [
        _row(ts(i), "T", "A" if i % 2 == 0 else "B", s, 21000.0 + i)
        for i, s in enumerate(sizes)
    ]
    lf = _to_lf(rows)
    result = aggregate_volume_bars(lf, volume_threshold=20)

    bar_vol_sum = result["volume"].sum()
    assert bar_vol_sum == total, f"Expected total volume {total}, got {bar_vol_sum}"


def test_volume_bars_buy_sell_volume():
    """Buy + sell volume per bar should equal bar volume."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        _row(ts(0), "T", "A", 10, 21000.0),
        _row(ts(1), "T", "B", 5,  21001.0),
        _row(ts(2), "T", "A", 8,  21002.0),
        _row(ts(3), "T", "B", 12, 21003.0),
    ]
    lf = _to_lf(rows)
    result = aggregate_volume_bars(lf, volume_threshold=15)

    for i in range(len(result)):
        row = result.row(i, named=True)
        assert row["buy_volume"] + row["sell_volume"] == row["volume"], (
            f"Bar {i}: buy({row['buy_volume']}) + sell({row['sell_volume']}) != vol({row['volume']})"
        )


def test_volume_bars_vap_histogram():
    """Volume bars should carry per-bar VAP arrays with matching volume sums."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        _row(ts(0), "T", "A", 10, 21000.0),   # bucket 0
        _row(ts(1), "T", "B", 10, 21001.0),   # bucket 0
        _row(ts(2), "T", "A", 10, 21002.0),   # bucket 1
    ]
    lf = _to_lf(rows)
    result = aggregate_volume_bars(lf, volume_threshold=20)

    bar0 = result.row(0, named=True)
    bar1 = result.row(1, named=True)

    assert bar0["vap_prices"] == [21000.0, 21001.0]
    assert bar0["vap_volumes"] == [10, 10]
    assert sum(bar0["vap_volumes"]) == bar0["volume"]

    assert bar1["vap_prices"] == [21002.0]
    assert bar1["vap_volumes"] == [10]
    assert sum(bar1["vap_volumes"]) == bar1["volume"]


def test_volume_bars_empty_input():
    """Empty trade data should return empty DataFrame with correct schema."""
    rows = [
        # Only book events, no trades
        _row(datetime(2024, 7, 15, 10, 0, 0), "A", "N", 0, 0.0),
    ]
    lf = _to_lf(rows)
    # Filter inside volume bars removes non-trades, leaving empty
    result = aggregate_volume_bars(lf, volume_threshold=100)

    assert len(result) == 0
    assert set(result.columns) == set(_output_columns())


def test_volume_bars_single_trade():
    """Single trade should produce one volume bar."""
    rows = [
        _row(datetime(2024, 7, 15, 10, 0, 0), "T", "A", 5, 21000.0),
    ]
    lf = _to_lf(rows)
    result = aggregate_volume_bars(lf, volume_threshold=100)

    assert len(result) == 1
    assert result["open"][0] == 21000.0
    assert result["volume"][0] == 5


def test_volume_bars_large_lots():
    """Large lot counts should work correctly in volume bars."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        _row(ts(0), "T", "A", 15, 21000.0),   # large buy
        _row(ts(1), "T", "B", 3,  21001.0),   # small sell
        _row(ts(2), "T", "A", 2,  21002.0),   # small buy
    ]
    lf = _to_lf(rows)
    result = aggregate_volume_bars(lf, volume_threshold=100)

    assert len(result) == 1
    assert result["large_trade_count"][0] == 1
    assert result["large_buy_volume"][0] == 15
    assert result["large_sell_volume"][0] == 0
    assert result["whale_trade_count_30"][0] == 0
    assert result["whale_buy_volume_30"][0] == 0
    assert result["whale_sell_volume_30"][0] == 0


def test_volume_bars_whale_lots():
    """Whale counts should only include trades >= 30 contracts."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        _row(ts(0), "T", "A", WHALE_LOT_THRESHOLD, 21000.0),       # whale buy
        _row(ts(1), "T", "B", WHALE_LOT_THRESHOLD + 5, 21001.0),   # whale sell
        _row(ts(2), "T", "A", WHALE_LOT_THRESHOLD - 1, 21002.0),   # non-whale
    ]
    lf = _to_lf(rows)
    result = aggregate_volume_bars(lf, volume_threshold=200)

    assert len(result) == 1
    bar = result.row(0, named=True)

    assert bar["whale_trade_count_30"] == 2
    assert bar["whale_buy_volume_30"] == WHALE_LOT_THRESHOLD
    assert bar["whale_sell_volume_30"] == WHALE_LOT_THRESHOLD + 5


# --- Tests: tick bars basic -------------------------------------------------

def test_tick_bars_basic():
    """Tick bars should close after fixed trade count, independent of trade size."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        _row(ts(0), "T", "A", 10, 21000.0),   # tick 1 -> bucket 0
        _row(ts(1), "T", "B", 10, 21001.0),   # tick 2 -> bucket 0
        _row(ts(2), "T", "A", 10, 21002.0),   # tick 3 -> bucket 1
        _row(ts(3), "T", "B", 10, 21003.0),   # tick 4 -> bucket 1
        _row(ts(4), "T", "A", 10, 21004.0),   # tick 5 -> bucket 2
    ]
    lf = _to_lf(rows)
    result = aggregate_tick_bars(lf, tick_threshold=2)

    # bucket_id = row_index // 2: [0,0,1,1,2] => 3 bars
    assert len(result) == 3, f"Expected 3 tick bars, got {len(result)}"

    assert result["trade_count"].to_list() == [2, 2, 1]
    assert result["open"][0] == 21000.0
    assert result["close"][0] == 21001.0
    assert result["open"][1] == 21002.0
    assert result["close"][1] == 21003.0
    assert result["open"][2] == 21004.0
    assert result["close"][2] == 21004.0


def test_tick_bars_whale_trade_counts_as_one_tick():
    """A large block trade contributes one tick, not a full bar by itself."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        _row(ts(0), "T", "A", 1000, 21000.0),   # whale trade = 1 tick
        _row(ts(1), "T", "B", 5, 21000.25),     # tick 2 closes bar 0
        _row(ts(2), "T", "A", 6, 21000.50),     # bar 1
    ]
    lf = _to_lf(rows)
    result = aggregate_tick_bars(lf, tick_threshold=2)

    assert len(result) == 2
    bar0 = result.row(0, named=True)
    bar1 = result.row(1, named=True)

    assert bar0["trade_count"] == 2
    assert bar0["volume"] == 1005
    assert bar0["open"] == 21000.0
    assert bar0["close"] == 21000.25

    assert bar1["trade_count"] == 1
    assert bar1["volume"] == 6


def test_tick_bars_schema_matches_time_bars():
    """Tick bars should expose the same canonical schema as time bars."""
    lf = _make_simple_session()
    time_result = aggregate_time_bars(lf, bar_size="5m")
    tick_result = aggregate_tick_bars(lf, tick_threshold=2)

    canonical = set(_output_columns())
    assert canonical.issubset(set(time_result.columns))
    assert canonical.issubset(set(tick_result.columns))


def test_tick_bars_vap_histogram():
    """Tick bars should carry per-bar VAP arrays with matching volume sums."""
    def ts(s):
        return datetime(2024, 7, 15, 10, 0, s)
    rows = [
        _row(ts(0), "T", "A", 10, 21000.0),   # bucket 0
        _row(ts(1), "T", "B", 10, 21001.0),   # bucket 0
        _row(ts(2), "T", "A", 10, 21002.0),   # bucket 1
    ]
    lf = _to_lf(rows)
    result = aggregate_tick_bars(lf, tick_threshold=2)

    bar0 = result.row(0, named=True)
    bar1 = result.row(1, named=True)

    assert bar0["vap_prices"] == [21000.0, 21001.0]
    assert bar0["vap_volumes"] == [10, 10]
    assert sum(bar0["vap_volumes"]) == bar0["volume"]

    assert bar1["vap_prices"] == [21002.0]
    assert bar1["vap_volumes"] == [10]
    assert sum(bar1["vap_volumes"]) == bar1["volume"]
