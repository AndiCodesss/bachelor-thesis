"""Tests for orderflow feature computation."""

import polars as pl
from datetime import datetime
from src.framework.features_canonical.orderflow import compute_orderflow_features
from src.framework.data.bars import aggregate_time_bars
from src.framework.data.loader import get_parquet_files


def _make_bars(rows):
    """Helper: build a bars DataFrame from list of dicts with trade-level fields."""
    n = len(rows)
    return pl.DataFrame({
        "ts_event": [r["ts_event"] for r in rows],
        "ts_close": [r["ts_event"] for r in rows],
        "bar_duration_ns": [r.get("bar_duration_ns", 300_000_000_000) for r in rows],
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": pl.Series([r.get("volume", 100) for r in rows], dtype=pl.UInt32),
        "vwap": [100.25] * n,
        "trade_count": pl.Series([r["trade_count"] for r in rows], dtype=pl.UInt32),
        "buy_volume": pl.Series([r["buy_volume"] for r in rows], dtype=pl.UInt32),
        "sell_volume": pl.Series([r["sell_volume"] for r in rows], dtype=pl.UInt32),
        "large_trade_count": pl.Series([r["large_trade_count"] for r in rows], dtype=pl.UInt32),
        "large_buy_volume": pl.Series([0] * n, dtype=pl.UInt32),
        "large_sell_volume": pl.Series([0] * n, dtype=pl.UInt32),
        "bid_price": [99.75] * n,
        "ask_price": [100.25] * n,
        "bid_size": pl.Series([10] * n, dtype=pl.UInt32),
        "ask_size": pl.Series([10] * n, dtype=pl.UInt32),
        "bid_count": pl.Series([5] * n, dtype=pl.UInt32),
        "ask_count": pl.Series([5] * n, dtype=pl.UInt32),
        "msg_count": pl.Series([200] * n, dtype=pl.UInt32),
        "add_count": pl.Series([50] * n, dtype=pl.UInt32),
        "cancel_count": pl.Series([30] * n, dtype=pl.UInt32),
        "modify_count": pl.Series([10] * n, dtype=pl.UInt32),
    }).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))


def test_orderflow_features_shape():
    """Verify feature computation returns expected columns."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))
    bars = aggregate_time_bars(lf, bar_size="5m")

    result = compute_orderflow_features(bars)

    assert len(result) > 0, "Should produce at least one bar"

    expected_columns = [
        "ts_event",
        "volume_imbalance",
        "trade_intensity",
        "large_trade_ratio",
        "order_flow_imbalance",
        "cvd_price_divergence_3",
        "cvd_price_divergence_6",
        "absorption_factor",
        "absorption_signal",
        "orderflow_ratio",
    ]

    for col in expected_columns:
        assert col in result.columns, f"Expected column '{col}' not found"


def test_orderflow_no_nulls():
    """Verify no null values in computed features."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))
    bars = aggregate_time_bars(lf, bar_size="5m")

    result = compute_orderflow_features(bars)
    # Check only the derived feature columns (not bar_duration_ns etc.)
    feature_cols = [
        "volume_imbalance", "trade_intensity", "large_trade_ratio", "order_flow_imbalance",
        "cvd_price_divergence_3", "cvd_price_divergence_6",
        "absorption_factor", "absorption_signal",
        "orderflow_ratio",
    ]
    for col in feature_cols:
        null_count = result[col].null_count()
        assert null_count == 0, f"Found {null_count} nulls in {col}"


def test_orderflow_synthetic_known_answer():
    """Test orderflow features with synthetic data and known answers."""
    bars = _make_bars([
        {
            "ts_event": datetime(2024, 7, 15, 13, 30, 0),
            "trade_count": 10,
            "buy_volume": 60,
            "sell_volume": 40,
            "large_trade_count": 10,
            "bar_duration_ns": 300_000_000_000,  # 5 min = 300s
        },
        {
            "ts_event": datetime(2024, 7, 15, 13, 35, 0),
            "trade_count": 5,
            "buy_volume": 20,
            "sell_volume": 30,
            "large_trade_count": 5,
            "bar_duration_ns": 300_000_000_000,
        },
    ])

    result = compute_orderflow_features(bars)

    assert len(result) == 2, f"Expected 2 bars, got {len(result)}"

    bar1 = result.row(0, named=True)

    # Volume imbalance = (60 - 40) / (60 + 40 + 1) = 20 / 101
    expected_imb1 = (60 - 40) / (60 + 40 + 1)
    assert abs(bar1["volume_imbalance"] - expected_imb1) < 0.001

    # Trade intensity = 10 / (300_000_000_000 / 1e9 + 1e-9) = 10 / 300 = 0.0333
    expected_intensity1 = 10 / 300.0
    assert abs(bar1["trade_intensity"] - expected_intensity1) < 0.001

    expected_ratio1 = 10 / 11.0
    assert abs(bar1["large_trade_ratio"] - expected_ratio1) < 0.001

    # OFI for bar 1: delta = 60 - 40 = 20
    assert bar1["order_flow_imbalance"] == 20

    bar2 = result.row(1, named=True)

    # OFI for bar 2: rolling sum of (20, -10) = 10
    assert bar2["order_flow_imbalance"] == 10


def test_orderflow_volume_imbalance_range():
    """Verify volume_imbalance stays within [-1, 1] range."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))
    bars = aggregate_time_bars(lf, bar_size="5m")

    result = compute_orderflow_features(bars)

    min_imb = result["volume_imbalance"].min()
    max_imb = result["volume_imbalance"].max()

    assert min_imb >= -1.0, f"volume_imbalance below -1: {min_imb}"
    assert max_imb <= 1.0, f"volume_imbalance above 1: {max_imb}"


def test_orderflow_features_sorted():
    """Verify output is sorted by timestamp."""
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))
    bars = aggregate_time_bars(lf, bar_size="5m")

    result = compute_orderflow_features(bars)

    timestamps = result["ts_event"].to_list()
    assert timestamps == sorted(timestamps), "Output not sorted by ts_event"


def test_cvd_price_divergence_known_answer():
    """CVD divergence should fire when volume delta and price move in opposite directions."""
    # 4 bars: price rising, delta negative for bars 3-4 -> divergence at bar 4 (lookback 3)
    bars = _make_bars([
        {"ts_event": datetime(2024, 7, 15, 13, 30, 0), "trade_count": 10,
         "buy_volume": 60, "sell_volume": 40, "large_trade_count": 1, "volume": 100},
        {"ts_event": datetime(2024, 7, 15, 13, 35, 0), "trade_count": 10,
         "buy_volume": 30, "sell_volume": 70, "large_trade_count": 1, "volume": 100},
        {"ts_event": datetime(2024, 7, 15, 13, 40, 0), "trade_count": 10,
         "buy_volume": 20, "sell_volume": 80, "large_trade_count": 1, "volume": 100},
        {"ts_event": datetime(2024, 7, 15, 13, 45, 0), "trade_count": 10,
         "buy_volume": 25, "sell_volume": 75, "large_trade_count": 1, "volume": 100},
    ])
    # Override close to create rising price
    bars = bars.with_columns([
        pl.Series("close", [100.0, 101.0, 102.0, 103.0]),
    ])

    result = compute_orderflow_features(bars)

    # Bar 4 (idx 3): price_slope_3 = 103 - 100 = 3.0 (positive)
    # CVD_3 = rolling_sum(delta, 3) = (-40) + (-60) + (-50) = -150 (negative)
    # Signs differ -> divergence = 1.0
    bar4 = result.row(3, named=True)
    assert bar4["cvd_price_divergence_3"] == 1.0, "Should detect divergence at bar 4"

    # Bar 1 (idx 0): price_slope_3 is null (no shift(3) data) -> divergence = 0.0
    bar1 = result.row(0, named=True)
    assert bar1["cvd_price_divergence_3"] == 0.0, "First bars should be 0 (no lookback)"


def test_absorption_factor_known_answer():
    """Absorption factor = volume * abs(delta) / (range/TICK_SIZE + 1)."""
    bars = _make_bars([{
        "ts_event": datetime(2024, 7, 15, 13, 30, 0),
        "trade_count": 10,
        "buy_volume": 70,
        "sell_volume": 30,
        "large_trade_count": 1,
        "volume": 100,
    }])
    # Set high=101.0, low=99.0 (range=2.0), volume=100, delta=70-30=40
    bars = bars.with_columns([
        pl.lit(101.0).alias("high"),
        pl.lit(99.0).alias("low"),
    ])

    result = compute_orderflow_features(bars)
    row = result.row(0, named=True)

    # absorption_factor = 100 * 40 / (2.0 / 0.25 + 1) = 4000 / 9 = 444.44
    expected = 100.0 * 40.0 / (2.0 / 0.25 + 1)
    assert abs(row["absorption_factor"] - expected) < 0.01, \
        f"absorption_factor {row['absorption_factor']} != {expected}"


def test_orderflow_ratio_known_answer():
    """Orderflow ratio = max(buy, sell) / (volume + 1)."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 7, 15, 13, 30, 0), "trade_count": 10,
         "buy_volume": 90, "sell_volume": 10, "large_trade_count": 1, "volume": 100},
        {"ts_event": datetime(2024, 7, 15, 13, 35, 0), "trade_count": 10,
         "buy_volume": 50, "sell_volume": 50, "large_trade_count": 1, "volume": 100},
    ])

    result = compute_orderflow_features(bars)

    bar1 = result.row(0, named=True)
    # max(90, 10) / (100 + 1) = 90/101 = 0.891
    expected_ratio_1 = 90.0 / 101.0
    assert abs(bar1["orderflow_ratio"] - expected_ratio_1) < 0.001

    bar2 = result.row(1, named=True)
    # max(50, 50) / (100 + 1) = 50/101 = 0.495
    expected_ratio_2 = 50.0 / 101.0
    assert abs(bar2["orderflow_ratio"] - expected_ratio_2) < 0.001


def test_orderflow_intermediates_dropped():
    """Ensure intermediate columns (volume_delta, _cvd_slope_*, etc.) are not in output."""
    bars = _make_bars([
        {"ts_event": datetime(2024, 7, 15, 13, 30, 0), "trade_count": 10,
         "buy_volume": 60, "sell_volume": 40, "large_trade_count": 1},
    ])

    result = compute_orderflow_features(bars)

    intermediates = ["volume_delta", "_cvd_slope_3", "_cvd_slope_6",
                     "_price_slope_3", "_price_slope_6", "_bar_range",
                     "_vol_90pct", "_range_20pct"]
    for col in intermediates:
        assert col not in result.columns, f"Intermediate column '{col}' leaked into output"
