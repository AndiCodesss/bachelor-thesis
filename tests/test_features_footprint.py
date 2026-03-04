"""Tests for footprint feature computation."""

import polars as pl
from datetime import datetime, timedelta

from src.framework.features_canonical.footprint import compute_footprint_features


def _make_bar(ts, open_=21000.0, high=21001.0, low=20999.0, close=21000.5,
              volume=100, buy_vol=60, sell_vol=40, bar_duration_ns=300_000_000_000,
              stacked_count=0, stacked_dir=0,
              zero_count=0, zero_ratio=0.0,
              unfinished_high=0, unfinished_low=0,
              max_level_vol=30, vol_at_high=15, vol_at_low=12,
              buy_vol_at_high=10, sell_vol_at_high=5,
              buy_vol_at_low=4, sell_vol_at_low=8):
    return {
        "ts_event": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "bar_duration_ns": bar_duration_ns,
        "stacked_imbalance_count": stacked_count,
        "stacked_imbalance_direction": stacked_dir,
        "zero_print_count": zero_count,
        "zero_print_ratio": zero_ratio,
        "unfinished_high": unfinished_high,
        "unfinished_low": unfinished_low,
        "max_level_volume": max_level_vol,
        "volume_at_high": vol_at_high,
        "volume_at_low": vol_at_low,
        "buy_vol_at_high": buy_vol_at_high,
        "sell_vol_at_high": sell_vol_at_high,
        "buy_vol_at_low": buy_vol_at_low,
        "sell_vol_at_low": sell_vol_at_low,
    }


def _make_bars_df(rows):
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("volume").cast(pl.UInt32),
        pl.col("buy_volume").cast(pl.UInt32),
        pl.col("sell_volume").cast(pl.UInt32),
        pl.col("stacked_imbalance_count").cast(pl.UInt32),
        pl.col("stacked_imbalance_direction").cast(pl.Int8),
        pl.col("zero_print_count").cast(pl.UInt32),
        pl.col("unfinished_high").cast(pl.UInt8),
        pl.col("unfinished_low").cast(pl.UInt8),
        pl.col("max_level_volume").cast(pl.UInt32),
        pl.col("volume_at_high").cast(pl.UInt32),
        pl.col("volume_at_low").cast(pl.UInt32),
        pl.col("buy_vol_at_high").cast(pl.UInt32),
        pl.col("sell_vol_at_high").cast(pl.UInt32),
        pl.col("buy_vol_at_low").cast(pl.UInt32),
        pl.col("sell_vol_at_low").cast(pl.UInt32),
    )


# --- Tests: stacked imbalance features ------------------------------------

def test_stacked_imb_strength():
    """stacked_imb_strength = count * direction (signed)."""
    rows = [
        _make_bar(datetime(2024, 7, 15, 10, 0), stacked_count=3, stacked_dir=1),
        _make_bar(datetime(2024, 7, 15, 10, 5), stacked_count=2, stacked_dir=-1),
        _make_bar(datetime(2024, 7, 15, 10, 10), stacked_count=0, stacked_dir=0),
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    strength = result["stacked_imb_strength"].to_list()
    assert strength[0] == 3.0, f"Expected 3*1=3, got {strength[0]}"
    assert strength[1] == -2.0, f"Expected 2*-1=-2, got {strength[1]}"
    assert strength[2] == 0.0, f"Expected 0, got {strength[2]}"


def test_stacked_imb_ma3():
    """stacked_imb_ma3 = 3-bar rolling mean of signed strength."""
    rows = [
        _make_bar(datetime(2024, 7, 15, 10, 0), stacked_count=3, stacked_dir=1),   # +3
        _make_bar(datetime(2024, 7, 15, 10, 5), stacked_count=0, stacked_dir=0),   # 0
        _make_bar(datetime(2024, 7, 15, 10, 10), stacked_count=3, stacked_dir=-1),  # -3
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    ma3 = result["stacked_imb_ma3"].to_list()
    # Bar 0: mean([3]) = 3.0
    assert abs(ma3[0] - 3.0) < 1e-6
    # Bar 1: mean([3, 0]) = 1.5
    assert abs(ma3[1] - 1.5) < 1e-6
    # Bar 2: mean([3, 0, -3]) = 0.0
    assert abs(ma3[2] - 0.0) < 1e-6


def test_stacked_imb_streak():
    """Streak counts consecutive same-sign stacked imbalances."""
    rows = [
        _make_bar(datetime(2024, 7, 15, 10, i*5), stacked_count=c, stacked_dir=d)
        for i, (c, d) in enumerate([
            (1, 1),   # bullish -> streak=1
            (2, 1),   # bullish -> streak=2
            (1, 1),   # bullish -> streak=3
            (0, 0),   # zero    -> streak=0
            (1, -1),  # bearish -> streak=1
        ])
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    streak = result["stacked_imb_streak"].to_list()
    assert streak[0] == 1.0
    assert streak[1] == 2.0
    assert streak[2] == 3.0
    assert streak[3] == 0.0
    assert streak[4] == 1.0


# --- Tests: zero print features -------------------------------------------

def test_zero_print_expansion():
    """Expansion flag = 1 when count > rolling 75th percentile."""
    base = datetime(2024, 7, 15, 10, 0)
    rows = []
    for i in range(15):
        zc = 1 if i < 12 else 0
        rows.append(_make_bar(
            base + timedelta(minutes=i*5),
            zero_count=zc, zero_ratio=0.1,
        ))
    # Add a bar with high zero prints
    rows.append(_make_bar(
        base + timedelta(minutes=75),
        zero_count=5, zero_ratio=0.8,
    ))
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    # Last bar has 5 zero prints vs rolling P75 of ~1 -> expansion=1
    assert result["zero_print_expansion"][-1] == 1.0


def test_zero_print_direction():
    """Direction = sign of (close-open) when zero prints exist, else 0."""
    rows = [
        # Up bar with zero prints
        _make_bar(datetime(2024, 7, 15, 10, 0), open_=21000.0, close=21005.0,
                  zero_count=3, zero_ratio=0.5),
        # Down bar with zero prints
        _make_bar(datetime(2024, 7, 15, 10, 5), open_=21005.0, close=21000.0,
                  zero_count=2, zero_ratio=0.3),
        # Bar with no zero prints
        _make_bar(datetime(2024, 7, 15, 10, 10), open_=21000.0, close=21002.0,
                  zero_count=0, zero_ratio=0.0),
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    directions = result["zero_print_direction"].to_list()
    assert directions[0] == 1.0, "Up bar with zero prints -> +1"
    assert directions[1] == -1.0, "Down bar with zero prints -> -1"
    assert directions[2] == 0.0, "No zero prints -> 0"


# --- Tests: unfinished business features -----------------------------------

def test_bars_since_unfinished():
    """bars_since_unfinished counts bars since last flag=1, null before first flag."""
    rows = [
        _make_bar(datetime(2024, 7, 15, 10, 0), unfinished_high=0),
        _make_bar(datetime(2024, 7, 15, 10, 5), unfinished_high=0),
        _make_bar(datetime(2024, 7, 15, 10, 10), unfinished_high=1),  # flag
        _make_bar(datetime(2024, 7, 15, 10, 15), unfinished_high=0),  # 1 bar since
        _make_bar(datetime(2024, 7, 15, 10, 20), unfinished_high=0),  # 2 bars since
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    bars_since = result["bars_since_unfinished_high"].to_list()
    # Before first flag: null
    assert bars_since[0] is None, "Before first flag should be null"
    assert bars_since[1] is None, "Before first flag should be null"
    # After the flag at bar 2, counts should be 0 (at flag), 1, 2
    assert bars_since[2] == 0.0, "At flag bar, count should be 0"
    assert bars_since[3] == 1.0, "1 bar after flag"
    assert bars_since[4] == 2.0, "2 bars after flag"


def test_unfinished_passthrough():
    """unfinished_high and unfinished_low pass through as Float64 features."""
    rows = [
        _make_bar(datetime(2024, 7, 15, 10, 0), unfinished_high=1, unfinished_low=0),
        _make_bar(datetime(2024, 7, 15, 10, 5), unfinished_high=0, unfinished_low=1),
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    assert result["fp_unfinished_high"][0] == 1.0
    assert result["fp_unfinished_high"][1] == 0.0
    assert result["fp_unfinished_low"][0] == 0.0
    assert result["fp_unfinished_low"][1] == 1.0


# --- Tests: delta intensity (heat) ----------------------------------------

def test_delta_per_second():
    """delta_per_second = (buy - sell) / (bar_duration_ns / 1e9)."""
    rows = [
        _make_bar(datetime(2024, 7, 15, 10, 0),
                  buy_vol=80, sell_vol=20,
                  bar_duration_ns=60_000_000_000),  # 60 seconds
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    # delta = 80-20 = 60, duration = 60s => delta_per_second = 1.0
    assert abs(result["delta_per_second"][0] - 1.0) < 1e-6


def test_delta_heat_is_abs_zscore():
    """delta_heat = abs(delta_intensity_z)."""
    base = datetime(2024, 7, 15, 10, 0)
    rows = []
    for i in range(30):
        rows.append(_make_bar(
            base + timedelta(minutes=i*5),
            buy_vol=60 + i*2, sell_vol=40,
            bar_duration_ns=300_000_000_000,
        ))
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    for i in range(len(result)):
        z = result["delta_intensity_z"][i]
        heat = result["delta_heat"][i]
        if z is not None and heat is not None:
            assert abs(heat - abs(z)) < 1e-9


# --- Tests: volume concentration ------------------------------------------

def test_max_level_vol_ratio():
    """max_level_vol_ratio = max_level_volume / (volume + 1)."""
    rows = [
        _make_bar(datetime(2024, 7, 15, 10, 0),
                  volume=100, max_level_vol=50),
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    expected = 50.0 / 101.0
    assert abs(result["max_level_vol_ratio"][0] - expected) < 1e-6


def test_high_low_vol_ratio():
    """high_low_vol_ratio = (vol_at_high + vol_at_low) / (volume + 1)."""
    rows = [
        _make_bar(datetime(2024, 7, 15, 10, 0),
                  volume=100, vol_at_high=25, vol_at_low=15),
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    expected = (25.0 + 15.0) / 101.0
    assert abs(result["high_low_vol_ratio"][0] - expected) < 1e-6


# --- Tests: edge cases -----------------------------------------------------

def test_empty_dataframe():
    """Empty input produces empty output with correct schema."""
    df = pl.DataFrame(schema={
        "ts_event": pl.Datetime("ns", "UTC"),
        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
        "close": pl.Float64, "volume": pl.UInt32,
        "buy_volume": pl.UInt32, "sell_volume": pl.UInt32,
        "bar_duration_ns": pl.Int64,
        "stacked_imbalance_count": pl.UInt32,
        "stacked_imbalance_direction": pl.Int8,
        "zero_print_count": pl.UInt32, "zero_print_ratio": pl.Float64,
        "unfinished_high": pl.UInt8, "unfinished_low": pl.UInt8,
        "max_level_volume": pl.UInt32, "volume_at_high": pl.UInt32,
        "volume_at_low": pl.UInt32,
        "buy_vol_at_high": pl.UInt32, "sell_vol_at_high": pl.UInt32,
        "buy_vol_at_low": pl.UInt32, "sell_vol_at_low": pl.UInt32,
    })
    result = compute_footprint_features(df)
    assert len(result) == 0
    assert "stacked_imb_strength" in result.columns


def test_single_row():
    """Single-row input should produce single-row output."""
    rows = [_make_bar(datetime(2024, 7, 15, 10, 0))]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)
    assert len(result) == 1


def test_all_zeros():
    """All-zero footprint columns should produce all-zero features."""
    rows = [_make_bar(datetime(2024, 7, 15, 10, 0),
                      stacked_count=0, stacked_dir=0,
                      zero_count=0, zero_ratio=0.0,
                      unfinished_high=0, unfinished_low=0,
                      max_level_vol=0, vol_at_high=0, vol_at_low=0)]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    assert result["stacked_imb_strength"][0] == 0.0
    assert result["zero_print_expansion"][0] == 0.0
    assert result["zero_print_direction"][0] == 0.0


def test_row_count_preserved():
    """Output row count matches input row count."""
    base = datetime(2024, 7, 15, 10, 0)
    n = 20
    rows = [_make_bar(base + timedelta(minutes=i*5)) for i in range(n)]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)
    assert len(result) == n


def test_output_columns():
    """Verify expected output columns are present."""
    rows = [_make_bar(datetime(2024, 7, 15, 10, 0))]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    expected_cols = [
        "ts_event",
        "stacked_imb_strength",
        "stacked_imb_ma3",
        "stacked_imb_streak",
        "fp_zero_print_ratio",
        "zero_print_expansion",
        "zero_print_direction",
        "fp_unfinished_high",
        "fp_unfinished_low",
        "bars_since_unfinished_high",
        "bars_since_unfinished_low",
        "delta_per_second",
        "delta_intensity_z",
        "delta_heat",
        "max_level_vol_ratio",
        "high_low_vol_ratio",
        "extreme_buy_ratio_high",
        "extreme_buy_ratio_low",
        "extreme_aggression_high",
        "extreme_aggression_low",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_no_intermediate_columns():
    """No internal helper columns should leak into output."""
    rows = [_make_bar(datetime(2024, 7, 15, 10, i*5)) for i in range(5)]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    for col in result.columns:
        assert not col.startswith("_"), f"Intermediate column leaked: {col}"
    assert "open" not in result.columns
    assert "close" not in result.columns
    assert "volume" not in result.columns


def test_output_sorted():
    """Output is sorted by ts_event."""
    rows = [_make_bar(datetime(2024, 7, 15, 10, i*5)) for i in range(10)]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    timestamps = result["ts_event"].to_list()
    assert timestamps == sorted(timestamps)


def test_no_nulls_in_output():
    """No null values in output after sufficient warmup."""
    base = datetime(2024, 7, 15, 10, 0)
    rows = [_make_bar(base + timedelta(minutes=i*5),
                      stacked_count=i % 3, stacked_dir=1 if i % 3 > 0 else 0,
                      zero_count=i % 2, zero_ratio=0.1 * (i % 2),
                      unfinished_high=1 if i == 0 else 0,
                      unfinished_low=1 if i == 0 else 0,
                      buy_vol=60 + i, sell_vol=40 + i)
            for i in range(30)]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    for col in result.columns:
        if col == "ts_event":
            continue
        null_count = result[col].null_count()
        assert null_count == 0, f"Feature '{col}' has {null_count} nulls"


# --- Tests: buy/sell aggression at extremes --------------------------------

def test_extreme_buy_ratio_high():
    """extreme_buy_ratio_high = buy_vol_at_high / (buy + sell + 1)."""
    rows = [
        _make_bar(datetime(2024, 7, 15, 10, 0),
                  buy_vol_at_high=20, sell_vol_at_high=5),
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    expected = 20.0 / (20.0 + 5.0 + 1.0)
    assert abs(result["extreme_buy_ratio_high"][0] - expected) < 1e-6


def test_extreme_buy_ratio_low():
    """extreme_buy_ratio_low = buy_vol_at_low / (buy + sell + 1)."""
    rows = [
        _make_bar(datetime(2024, 7, 15, 10, 0),
                  buy_vol_at_low=3, sell_vol_at_low=12),
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    expected = 3.0 / (3.0 + 12.0 + 1.0)
    assert abs(result["extreme_buy_ratio_low"][0] - expected) < 1e-6


def test_extreme_aggression_high():
    """extreme_aggression_high = 1 when sellers > 2x buyers at high."""
    rows = [
        # Sellers aggressive: sell=30 > buy=10 * 2
        _make_bar(datetime(2024, 7, 15, 10, 0),
                  buy_vol_at_high=10, sell_vol_at_high=30),
        # Balanced: sell=12 < buy=10 * 2
        _make_bar(datetime(2024, 7, 15, 10, 5),
                  buy_vol_at_high=10, sell_vol_at_high=12),
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    assert result["extreme_aggression_high"][0] == 1.0, "sell>2*buy at high -> 1"
    assert result["extreme_aggression_high"][1] == 0.0, "sell<2*buy at high -> 0"


def test_extreme_aggression_low():
    """extreme_aggression_low = 1 when buyers > 2x sellers at low."""
    rows = [
        # Buyers aggressive: buy=25 > sell=10 * 2
        _make_bar(datetime(2024, 7, 15, 10, 0),
                  buy_vol_at_low=25, sell_vol_at_low=10),
        # Balanced: buy=15 < sell=10 * 2
        _make_bar(datetime(2024, 7, 15, 10, 5),
                  buy_vol_at_low=15, sell_vol_at_low=10),
    ]
    df = _make_bars_df(rows)
    result = compute_footprint_features(df)

    assert result["extreme_aggression_low"][0] == 1.0, "buy>2*sell at low -> 1"
    assert result["extreme_aggression_low"][1] == 0.0, "buy<2*sell at low -> 0"
