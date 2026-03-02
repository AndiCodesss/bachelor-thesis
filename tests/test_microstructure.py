"""Tests for microstructure feature engineering."""

import polars as pl
from datetime import datetime, timedelta
from src.framework.features_canonical.microstructure import compute_microstructure_features
from src.framework.data.bars import aggregate_time_bars
from src.framework.data.loader import get_parquet_files, filter_rth


def _make_bars(rows):
    """Helper: build bars DataFrame from list of dicts with activity-level fields."""
    n = len(rows)
    return pl.DataFrame({
        "ts_event": [r["ts_event"] for r in rows],
        "ts_close": [r["ts_event"] for r in rows],
        "bar_duration_ns": [r.get("bar_duration_ns", 300_000_000_000) for r in rows],
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": pl.Series([100] * n, dtype=pl.UInt32),
        "vwap": [100.25] * n,
        "trade_count": pl.Series([r["trade_count"] for r in rows], dtype=pl.UInt32),
        "buy_volume": pl.Series([50] * n, dtype=pl.UInt32),
        "sell_volume": pl.Series([50] * n, dtype=pl.UInt32),
        "large_trade_count": pl.Series([1] * n, dtype=pl.UInt32),
        "large_buy_volume": pl.Series([0] * n, dtype=pl.UInt32),
        "large_sell_volume": pl.Series([0] * n, dtype=pl.UInt32),
        "bid_price": [99.75] * n,
        "ask_price": [100.25] * n,
        "bid_size": pl.Series([10] * n, dtype=pl.UInt32),
        "ask_size": pl.Series([10] * n, dtype=pl.UInt32),
        "bid_count": pl.Series([5] * n, dtype=pl.UInt32),
        "ask_count": pl.Series([5] * n, dtype=pl.UInt32),
        "msg_count": pl.Series([r["msg_count"] for r in rows], dtype=pl.UInt32),
        "add_count": pl.Series([r["add_count"] for r in rows], dtype=pl.UInt32),
        "cancel_count": pl.Series([r["cancel_count"] for r in rows], dtype=pl.UInt32),
        "modify_count": pl.Series([r["modify_count"] for r in rows], dtype=pl.UInt32),
        "latency_mean": [r.get("latency_mean", None) for r in rows],
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("latency_mean").cast(pl.Float64),
    )


def test_compute_microstructure_features_shape():
    """Verify compute_microstructure_features produces expected output columns."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))
    lf = filter_rth(lf).head(100000)
    bars = aggregate_time_bars(lf, bar_size="5m")

    result = compute_microstructure_features(bars)

    assert len(result) > 0, "Result should not be empty"

    expected_columns = [
        "ts_event",
        "cancel_trade_ratio",
        "modify_trade_ratio",
        "mean_trade_interval_us",
        "latency_mean",
        "message_intensity_ma5",
        "cancel_rate_change",
        "tape_speed",
        "tape_speed_z",
        "tape_speed_spike",
        "price_velocity",
        "price_velocity_z",
        "is_whip",
        "recoil_pct",
        "recoil_50pct",
    ]

    for col in expected_columns:
        assert col in result.columns, f"Expected column '{col}' not found"


def test_microstructure_synthetic_known_answer():
    """Test with synthetic data where we know expected outputs."""
    bars = _make_bars([{
        "ts_event": datetime(2024, 7, 15, 14, 30, 0),
        "msg_count": 20,
        "add_count": 10,
        "cancel_count": 5,
        "modify_count": 2,
        "trade_count": 3,
        "bar_duration_ns": 60_000_000_000,  # 60 seconds
    }])

    result = compute_microstructure_features(bars)

    assert len(result) == 1, f"Expected 1 bar, got {len(result)}"

    row = result.row(0, named=True)

    # cancel_trade_ratio = 5 / (3 + 1) = 1.25
    expected_cancel_trade_ratio = 5.0 / (3 + 1)
    assert abs(row["cancel_trade_ratio"] - expected_cancel_trade_ratio) < 0.01

    # modify_trade_ratio = 2 / (3 + 1) = 0.5
    expected_modify_trade_ratio = 2.0 / (3 + 1)
    assert abs(row["modify_trade_ratio"] - expected_modify_trade_ratio) < 0.01

    # mean_trade_interval_us = (60e9 / (3 - 1)) / 1000 = 30e6 / 1000 = 30_000_000 us
    expected_interval = (60_000_000_000 / 2.0) / 1000.0
    assert abs(row["mean_trade_interval_us"] - expected_interval) < 1.0

    # latency_mean is null (not available from aggregated bars)
    assert row["latency_mean"] is None

    # message_intensity_ma5 = 20.0 (single bar)
    assert row["message_intensity_ma5"] == 20.0

    # cancel_rate_change: null (first bar)
    assert row["cancel_rate_change"] is None


def test_microstructure_no_trades_bar():
    """Test bar with no trade events."""
    bars = _make_bars([{
        "ts_event": datetime(2024, 7, 15, 14, 30, 0),
        "msg_count": 10,
        "add_count": 10,
        "cancel_count": 0,
        "modify_count": 0,
        "trade_count": 0,
        "bar_duration_ns": 300_000_000_000,
    }])

    result = compute_microstructure_features(bars)

    assert len(result) == 1
    row = result.row(0, named=True)

    # cancel_trade_ratio = 0 / (0 + 1) = 0
    assert abs(row["cancel_trade_ratio"] - 0.0) < 0.01
    # mean_trade_interval_us: null when trade_count <= 1
    assert row["mean_trade_interval_us"] is None


def test_microstructure_multiple_bars():
    """Test multiple bars to verify rolling features work correctly."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 7, 15, 14, 30, 0), "msg_count": 15, "add_count": 10,
         "cancel_count": 5, "modify_count": 0, "trade_count": 0, "bar_duration_ns": 300_000_000_000},
        {"ts_event": datetime(2024, 7, 15, 14, 35, 0), "msg_count": 20, "add_count": 10,
         "cancel_count": 10, "modify_count": 0, "trade_count": 0, "bar_duration_ns": 300_000_000_000},
        {"ts_event": datetime(2024, 7, 15, 14, 40, 0), "msg_count": 25, "add_count": 10,
         "cancel_count": 15, "modify_count": 0, "trade_count": 0, "bar_duration_ns": 300_000_000_000},
    ])

    result = compute_microstructure_features(bars)

    assert len(result) == 3

    # cancel_rate_change: Bar 0: null, Bar 1: (10-5)/5=1.0, Bar 2: (15-10)/10=0.5
    assert result.row(0, named=True)["cancel_rate_change"] is None
    assert abs(result.row(1, named=True)["cancel_rate_change"] - 1.0) < 0.01
    assert abs(result.row(2, named=True)["cancel_rate_change"] - 0.5) < 0.01


def test_microstructure_non_empty():
    """Assert non-empty DataFrame after feature computation on real data."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))
    lf = filter_rth(lf)
    bars = aggregate_time_bars(lf, bar_size="5m")

    result = compute_microstructure_features(bars)

    assert len(result) > 0, "Empty DataFrame after compute_microstructure_features"
    assert result["cancel_trade_ratio"].null_count() == 0, "Null values found in cancel_trade_ratio"


def test_tape_speed_known_answer():
    """Tape speed = trade_count / (bar_duration_ns / 1e6)."""
    bars = _make_bars([{
        "ts_event": datetime(2024, 7, 15, 14, 30, 0),
        "msg_count": 20,
        "add_count": 10,
        "cancel_count": 5,
        "modify_count": 2,
        "trade_count": 100,
        "bar_duration_ns": 300_000_000_000,  # 300 seconds = 300e6 ms
    }])

    result = compute_microstructure_features(bars)
    row = result.row(0, named=True)

    # tape_speed = 100 / (300e9 / 1e6 + 1e-9) = 100 / 300000 = 0.000333
    expected = 100.0 / (300_000_000_000 / 1e6 + 1e-9)
    assert abs(row["tape_speed"] - expected) < 1e-8, \
        f"tape_speed {row['tape_speed']} != {expected}"


def test_tape_speed_spike_detection():
    """Tape speed spike fires when Z-score > 2.5."""
    # Create 25 bars: 24 normal, 1 extreme spike
    base_row = {
        "msg_count": 20, "add_count": 10, "cancel_count": 5,
        "modify_count": 2, "trade_count": 50,
        "bar_duration_ns": 300_000_000_000,
    }
    rows = []
    for i in range(24):
        r = dict(base_row)
        r["ts_event"] = datetime(2024, 7, 15, 10, 0, 0) + timedelta(minutes=5 * i)
        rows.append(r)

    # Spike bar: 10x more trades
    spike = dict(base_row)
    spike["ts_event"] = datetime(2024, 7, 15, 10, 0, 0) + timedelta(minutes=5 * 24)
    spike["trade_count"] = 500
    rows.append(spike)

    bars = _make_bars(rows)
    result = compute_microstructure_features(bars)

    last = result.row(-1, named=True)
    assert last["tape_speed_spike"] == 1.0, "Spike bar should trigger tape_speed_spike"

    # Normal bars should not trigger
    first = result.row(5, named=True)
    assert first["tape_speed_spike"] == 0.0, "Normal bar should not trigger spike"


def test_price_velocity_known_answer():
    """Price velocity = abs(close - open) / (bar_duration / 1e9)."""
    bars = _make_bars([{
        "ts_event": datetime(2024, 7, 15, 14, 30, 0),
        "msg_count": 20, "add_count": 10, "cancel_count": 5,
        "modify_count": 2, "trade_count": 10,
        "bar_duration_ns": 60_000_000_000,  # 60 seconds
    }])
    # Set open=100, close=105 -> abs diff = 5
    bars = bars.with_columns([
        pl.lit(100.0).alias("open"),
        pl.lit(105.0).alias("close"),
    ])

    result = compute_microstructure_features(bars)
    row = result.row(0, named=True)

    # price_velocity = abs(105 - 100) / (60e9 / 1e9 + 1e-9) = 5 / 60 = 0.0833
    expected = 5.0 / (60_000_000_000 / 1e9 + 1e-9)
    assert abs(row["price_velocity"] - expected) < 1e-6, \
        f"price_velocity {row['price_velocity']} != {expected}"


def test_recoil_after_whip():
    """Recoil should compute retracement after a whip bar."""
    # Build 25 bars: 24 calm, then a whip, then a recoil bar
    base_row = {
        "msg_count": 20, "add_count": 10, "cancel_count": 5,
        "modify_count": 2, "trade_count": 10,
        "bar_duration_ns": 300_000_000_000,
    }
    rows = []
    opens = []
    closes = []
    for i in range(24):
        r = dict(base_row)
        r["ts_event"] = datetime(2024, 7, 15, 10, 0, 0) + timedelta(minutes=5 * i)
        rows.append(r)
        opens.append(100.0)
        closes.append(100.25)  # Tiny move

    # Whip bar: huge price move
    whip = dict(base_row)
    whip["ts_event"] = datetime(2024, 7, 15, 10, 0, 0) + timedelta(minutes=5 * 24)
    rows.append(whip)
    opens.append(100.0)
    closes.append(120.0)  # 20 point move

    # Recoil bar: partial retrace
    recoil = dict(base_row)
    recoil["ts_event"] = datetime(2024, 7, 15, 10, 0, 0) + timedelta(minutes=5 * 25)
    rows.append(recoil)
    opens.append(120.0)
    closes.append(110.0)  # Retraces 10 of 20 = 50%

    bars = _make_bars(rows)
    bars = bars.with_columns([
        pl.Series("open", opens),
        pl.Series("close", closes),
    ])

    result = compute_microstructure_features(bars)

    whip_row = result.row(24, named=True)
    assert whip_row["is_whip"] == 1.0, "Bar with 20pt move should be a whip"

    recoil_row = result.row(25, named=True)
    # recoil_pct = abs(120 - 110) / (abs(120 - 100) + 1e-9) = 10/20 = 0.5
    assert abs(recoil_row["recoil_pct"] - 0.5) < 0.01, \
        f"recoil_pct should be ~0.5, got {recoil_row['recoil_pct']}"
    assert recoil_row["recoil_50pct"] == 1.0, "50% recoil should trigger recoil_50pct"


def test_microstructure_intermediates_dropped():
    """Ensure intermediate columns are not in the output."""
    bars = _make_bars([{
        "ts_event": datetime(2024, 7, 15, 14, 30, 0),
        "msg_count": 20, "add_count": 10, "cancel_count": 5,
        "modify_count": 2, "trade_count": 10,
    }])

    result = compute_microstructure_features(bars)

    intermediates = ["_prev_is_whip", "_prev_range"]
    for col in intermediates:
        assert col not in result.columns, f"Intermediate column '{col}' leaked into output"
