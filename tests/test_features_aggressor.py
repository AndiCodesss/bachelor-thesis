"""Tests for aggressor feature computation."""

import polars as pl
from datetime import datetime

from src.framework.features_canonical.aggressor import compute_aggressor_features


def _make_bar(ts, buy_vol, sell_vol, trade_count, large_buy_vol=0, large_sell_vol=0, close=21000.0):
    """Helper: create a single bar row matching bars.py schema."""
    return {
        "ts_event": ts,
        "close": close,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "volume": buy_vol + sell_vol,
        "trade_count": trade_count,
        "large_buy_volume": large_buy_vol,
        "large_sell_volume": large_sell_vol,
    }


def _make_bars_df(rows):
    """Build a UTC-aware bars DataFrame from row dicts."""
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("buy_volume").cast(pl.UInt32),
        pl.col("sell_volume").cast(pl.UInt32),
        pl.col("volume").cast(pl.UInt32),
        pl.col("trade_count").cast(pl.UInt32),
        pl.col("large_buy_volume").cast(pl.UInt32),
        pl.col("large_sell_volume").cast(pl.UInt32),
    )


def test_cvd_basic():
    """CVD should be cumulative buy-sell delta, reset daily."""
    rows = [
        # Bar 1: buy=30, sell=10 => delta=+20
        _make_bar(datetime(2024, 7, 15, 13, 30, 0), buy_vol=30, sell_vol=10, trade_count=4),
        # Bar 2: buy=5, sell=15 => delta=-10
        _make_bar(datetime(2024, 7, 15, 13, 35, 0), buy_vol=5, sell_vol=15, trade_count=2),
        # Bar 3: buy=20, sell=0 => delta=+20
        _make_bar(datetime(2024, 7, 15, 13, 40, 0), buy_vol=20, sell_vol=0, trade_count=1),
    ]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    assert len(result) == 3
    # CVD: cumulative within day: +20, +10 (20-10), +30 (20-10+20)
    cvd = result["cvd"].to_list()
    assert cvd == [20, 10, 30], f"Expected [20, 10, 30], got {cvd}"


def test_cvd_daily_reset():
    """CVD cumsum resets at day boundary."""
    rows = [
        # Day 1: buy=20, sell=0 => delta=+20, CVD=+20
        _make_bar(datetime(2024, 7, 15, 13, 30, 0), buy_vol=20, sell_vol=0, trade_count=1),
        # Day 2: buy=0, sell=10 => delta=-10, CVD=-10 (reset)
        _make_bar(datetime(2024, 7, 16, 13, 30, 0), buy_vol=0, sell_vol=10, trade_count=1),
    ]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    assert len(result) == 2
    cvd = result["cvd"].to_list()
    assert cvd[0] == 20, f"Day 1 CVD should be 20, got {cvd[0]}"
    assert cvd[1] == -10, f"Day 2 CVD should be -10 (reset), got {cvd[1]}"


def test_cvd_divergence():
    """Divergence = 1 when price and CVD move opposite over 6 bars."""
    rows = []
    for i in range(7):
        ts = datetime(2024, 7, 15, 13, i * 5, 0)
        price = 21000.0 + i * 10  # price goes up
        # All sells => CVD goes down
        rows.append(_make_bar(ts, buy_vol=0, sell_vol=10, trade_count=1, close=price))

    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    assert len(result) == 7
    div = result["cvd_divergence_6"].to_list()
    assert div[6] == 1.0, f"Expected divergence=1 at bar 6, got {div[6]}"
    for i in range(6):
        assert div[i] == 0.0, f"Expected divergence=0 at bar {i}, got {div[i]}"


def test_cvd_convergence():
    """Convergence = 0 when price and CVD move same direction."""
    rows = []
    for i in range(7):
        ts = datetime(2024, 7, 15, 13, i * 5, 0)
        price = 21000.0 + i * 10  # price goes up
        # All buys => CVD also goes up
        rows.append(_make_bar(ts, buy_vol=10, sell_vol=0, trade_count=1, close=price))

    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    div = result["cvd_divergence_6"].to_list()
    assert div[6] == 0.0, f"Expected convergence=0 at bar 6, got {div[6]}"


def test_buy_sell_ratio():
    """buy_sell_ratio = buy_volume / (total_volume + 1)."""
    rows = [_make_bar(datetime(2024, 7, 15, 13, 30, 0), buy_vol=30, sell_vol=20, trade_count=2)]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    # buy=30, total=50, ratio = 30/51
    expected = 30.0 / 51.0
    actual = result["buy_sell_ratio"][0]
    assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"


def test_trade_intensity_and_relative():
    """trade_intensity = trade count; relative_intensity = current / MA(24)."""
    rows = [_make_bar(datetime(2024, 7, 15, 13, 30, 0), buy_vol=40, sell_vol=0, trade_count=8)]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    assert result["trade_intensity"][0] == 8.0
    # Single bar => MA(24) with min_samples=1 = 8.0, so relative = 1.0
    assert abs(result["relative_intensity"][0] - 1.0) < 1e-6


def test_large_lot_fraction():
    """large_lot_fraction = large_vol / (total_vol + 1)."""
    # large_buy=15, large_sell=12, large_total=27, total=35
    rows = [_make_bar(
        datetime(2024, 7, 15, 13, 30, 0),
        buy_vol=18, sell_vol=17, trade_count=4,
        large_buy_vol=15, large_sell_vol=12,
    )]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    expected = 27.0 / 36.0  # +1 in denominator
    actual = result["large_lot_fraction"][0]
    assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"


def test_large_lot_imbalance():
    """large_lot_imbalance = (large_buy - large_sell) / (large_total + 1)."""
    # large_buy=15, large_sell=12, large_total=27
    rows = [_make_bar(
        datetime(2024, 7, 15, 13, 30, 0),
        buy_vol=17, sell_vol=12, trade_count=3,
        large_buy_vol=15, large_sell_vol=12,
    )]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    expected = (15 - 12) / 28.0  # +1 in denominator
    actual = result["large_lot_imbalance"][0]
    assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"


def test_empty_dataframe():
    """Empty input should produce empty output without errors."""
    df = pl.DataFrame(schema={
        "ts_event": pl.Datetime("ns", "UTC"),
        "close": pl.Float64,
        "buy_volume": pl.UInt32,
        "sell_volume": pl.UInt32,
        "volume": pl.UInt32,
        "trade_count": pl.UInt32,
        "large_buy_volume": pl.UInt32,
        "large_sell_volume": pl.UInt32,
    })
    result = compute_aggressor_features(df)
    assert len(result) == 0


def test_output_columns():
    """Verify expected output columns are present."""
    rows = [_make_bar(datetime(2024, 7, 15, 13, 30, 0), buy_vol=10, sell_vol=0, trade_count=1)]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    expected_cols = [
        "ts_event",
        "volume_delta",
        "cvd",
        "cvd_divergence_6",
        "cvd_slope_3",
        "cvd_slope_6",
        "cvd_slope_12",
        "cvd_accel_3",
        "cvd_accel_6",
        "cvd_price_alignment",
        "buy_sell_ratio",
        "trade_intensity",
        "relative_intensity",
        "large_lot_fraction",
        "large_lot_imbalance",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_no_intermediate_columns_leaked():
    """Intermediate columns like _date, close should not appear in output."""
    rows = [_make_bar(datetime(2024, 7, 15, 13, 30, 0), buy_vol=10, sell_vol=0, trade_count=1)]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    forbidden = ["_date", "_price_return", "_price_chg_6", "_cvd_chg_6", "_cvd_change",
                 "close", "buy_volume", "sell_volume", "total_volume",
                 "trade_count", "large_buy_volume", "large_sell_volume", "large_total_volume"]
    for col in forbidden:
        assert col not in result.columns, f"Intermediate column leaked: {col}"


def test_output_sorted_by_timestamp():
    """Output must be sorted by ts_event."""
    rows = []
    for i in range(5):
        ts = datetime(2024, 7, 15, 13, i * 5, 0)
        rows.append(_make_bar(ts, buy_vol=10, sell_vol=0, trade_count=1))
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    timestamps = result["ts_event"].to_list()
    assert timestamps == sorted(timestamps)


def test_all_sells_bar():
    """Bar with only sells: buy_sell_ratio ~ 0, volume_delta < 0."""
    rows = [_make_bar(datetime(2024, 7, 15, 13, 30, 0), buy_vol=0, sell_vol=50, trade_count=5)]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    assert result["buy_sell_ratio"][0] < 0.02  # 0/51
    assert result["volume_delta"][0] == -50


def test_all_buys_bar():
    """Bar with only buys: buy_sell_ratio ~ 1, volume_delta > 0."""
    rows = [_make_bar(datetime(2024, 7, 15, 13, 30, 0), buy_vol=50, sell_vol=0, trade_count=5)]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    assert result["buy_sell_ratio"][0] > 0.95  # 50/51
    assert result["volume_delta"][0] == 50


# ---- CVD dynamics tests ----

def test_cvd_slope_3_known_answer():
    """cvd_slope_3 = (cvd[i] - cvd[i-3]) / 3 for constant buying."""
    rows = []
    for i in range(7):
        ts = datetime(2024, 7, 15, 13, i * 5, 0)
        # Each bar: buy=10, sell=0 => delta=+10
        rows.append(_make_bar(ts, buy_vol=10, sell_vol=0, trade_count=1))
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    # CVD: 10, 20, 30, 40, 50, 60, 70
    # slope_3[3] = (40 - 10) / 3 = 10.0
    # slope_3[4] = (50 - 20) / 3 = 10.0
    # slope_3[5] = (60 - 30) / 3 = 10.0
    assert result["cvd_slope_3"][3] == 10.0
    assert result["cvd_slope_3"][4] == 10.0
    assert result["cvd_slope_3"][5] == 10.0
    # First 3 bars: null (idx < 3)
    for i in range(3):
        assert result["cvd_slope_3"][i] is None


def test_cvd_slope_6_known_answer():
    """cvd_slope_6 = (cvd[i] - cvd[i-6]) / 6."""
    rows = []
    for i in range(8):
        ts = datetime(2024, 7, 15, 13, i * 5, 0)
        rows.append(_make_bar(ts, buy_vol=10, sell_vol=0, trade_count=1))
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    # CVD: 10, 20, 30, 40, 50, 60, 70, 80
    # slope_6[6] = (70 - 10) / 6 = 10.0
    # slope_6[7] = (80 - 20) / 6 = 10.0
    assert result["cvd_slope_6"][6] == 10.0
    assert result["cvd_slope_6"][7] == 10.0
    for i in range(6):
        assert result["cvd_slope_6"][i] is None


def test_cvd_accel_constant_slope():
    """With constant CVD increments, acceleration should be 0."""
    rows = []
    for i in range(14):
        total_min = i * 5
        ts = datetime(2024, 7, 15, 13 + total_min // 60, total_min % 60, 0)
        rows.append(_make_bar(ts, buy_vol=10, sell_vol=0, trade_count=1))
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    # Constant slope => accel = slope_3[i] - slope_3[i-3] = 10 - 10 = 0
    accel = result["cvd_accel_3"].to_list()
    # accel_3 valid from idx >= 6. First 6 bars null.
    for i in range(6):
        assert accel[i] is None
    assert accel[6] == 0.0


def test_cvd_accel_accelerating():
    """With increasing buy volume, CVD accelerates and accel > 0."""
    rows = []
    for i in range(14):
        total_min = i * 5
        ts = datetime(2024, 7, 15, 13 + total_min // 60, total_min % 60, 0)
        # Increasing buy volume: 10, 20, 30, ...
        rows.append(_make_bar(ts, buy_vol=10 * (i + 1), sell_vol=0, trade_count=1))
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    # CVD grows quadratically => slope increases => accel > 0
    accel = result["cvd_accel_3"].to_list()
    last_valid = [a for a in accel if a is not None]
    assert last_valid[-1] > 0, f"Expected positive accel, got {last_valid[-1]}"


def test_cvd_price_alignment_correlated():
    """When price and CVD move together with variance, alignment positive."""
    rows = []
    for i in range(20):
        total_min = i * 5
        ts = datetime(2024, 7, 15, 13 + total_min // 60, total_min % 60, 0)
        # Varying price moves and varying buy volumes (both trending up but with variance)
        price = 21000.0 + i * 10 + (i % 3) * 5  # sawtooth-ish up
        buy = 10 + i * 2 + (i % 3) * 3  # correlated with price moves
        sell = 5
        rows.append(_make_bar(ts, buy_vol=buy, sell_vol=sell, trade_count=1, close=price))
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    alignment = result["cvd_price_alignment"].to_list()
    valid = [a for a in alignment if a is not None and a != float('inf') and a != float('-inf')]
    assert len(valid) > 0
    assert valid[-1] > 0.0, f"Expected positive alignment, got {valid[-1]}"


def test_cvd_price_alignment_divergent():
    """When price up but CVD down, alignment negative."""
    rows = []
    for i in range(20):
        total_min = i * 5
        ts = datetime(2024, 7, 15, 13 + total_min // 60, total_min % 60, 0)
        price = 21000.0 + i * 10 + (i % 3) * 5  # price up
        # sell > buy and increasing sells => CVD goes down, anti-correlated with price
        sell = 10 + i * 2 + (i % 3) * 3
        buy = 5
        rows.append(_make_bar(ts, buy_vol=buy, sell_vol=sell, trade_count=1, close=price))
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    alignment = result["cvd_price_alignment"].to_list()
    valid = [a for a in alignment if a is not None and a != float('inf') and a != float('-inf')]
    assert len(valid) > 0
    assert valid[-1] < 0.0, f"Expected negative alignment, got {valid[-1]}"


def test_cvd_slope_null_for_single_bar():
    """Single bar: all slope and accel features should be null."""
    rows = [_make_bar(datetime(2024, 7, 15, 13, 30, 0), buy_vol=10, sell_vol=0, trade_count=1)]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    assert result["cvd_slope_3"][0] is None
    assert result["cvd_slope_6"][0] is None
    assert result["cvd_slope_12"][0] is None
    assert result["cvd_accel_3"][0] is None
    assert result["cvd_accel_6"][0] is None


def test_cvd_slope_no_cross_day_contamination():
    """CVD slope must not leak across day boundaries in multi-day data."""
    rows = []
    # Day 1: 5 bars with heavy buying (CVD goes up fast)
    for i in range(5):
        ts = datetime(2024, 7, 15, 13, 30 + i * 5, 0)
        rows.append(_make_bar(ts, buy_vol=100, sell_vol=0, trade_count=10))
    # Day 2: 5 bars with light selling
    for i in range(5):
        ts = datetime(2024, 7, 16, 13, 30 + i * 5, 0)
        rows.append(_make_bar(ts, buy_vol=0, sell_vol=5, trade_count=1))
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    # First 3 bars of day 2 (idx 5-7) should have null cvd_slope_3
    # because idx >= 3 guard uses per-day bar index (0-based within day 2)
    for i in range(5, 8):
        assert result["cvd_slope_3"][i] is None, (
            f"Bar {i} (day 2) should have null cvd_slope_3, got {result['cvd_slope_3'][i]}"
        )


def test_cvd_day_boundary_after_warmup_drop():
    """After dropping bar 0, first 2 visible bars should have null cvd_slope_3."""
    rows = []
    for i in range(10):
        total_min = 30 + i * 5
        ts = datetime(2024, 7, 15, 13 + total_min // 60, total_min % 60, 0)
        rows.append(_make_bar(ts, buy_vol=10, sell_vol=0, trade_count=1))
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    # Simulate warmup filter: drop bar 0, re-index
    result_filtered = result.slice(1)  # drop first row
    # First 2 bars (original idx 1, 2) should be null
    for i in range(2):
        assert result_filtered["cvd_slope_3"][i] is None, (
            f"Post-warmup bar {i} should have null cvd_slope_3, got {result_filtered['cvd_slope_3'][i]}"
        )


def test_relative_intensity_resets_at_day_boundary():
    rows = [
        _make_bar(datetime(2024, 7, 15, 15, 50, 0), buy_vol=10, sell_vol=0, trade_count=1),
        _make_bar(datetime(2024, 7, 15, 15, 55, 0), buy_vol=10, sell_vol=0, trade_count=100),
        _make_bar(datetime(2024, 7, 16, 9, 30, 0), buy_vol=10, sell_vol=0, trade_count=1),
    ]
    df = _make_bars_df(rows)
    result = compute_aggressor_features(df)

    # First bar of a new day should use that day's MA only -> 1 / 1 = 1.
    assert result["relative_intensity"][2] == 1.0
