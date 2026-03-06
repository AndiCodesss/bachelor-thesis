"""Tests for backtest metrics computation."""

import polars as pl
import pytest
import numpy as np
from datetime import datetime, timedelta
from src.framework.backtest.metrics import compute_daily_pnl_series, compute_metrics
from src.framework.data.constants import TOTAL_COST_RT


def test_known_answer_five_trades():
    """Test metrics with 5 hand-computed trades (3 winners, 2 losers)."""
    # Spread trades across 3 days for meaningful daily Sharpe computation
    day1 = datetime(2025, 2, 1, 10, 0, 0)
    day2 = datetime(2025, 2, 2, 10, 0, 0)
    day3 = datetime(2025, 2, 3, 10, 0, 0)

    trades = pl.DataFrame({
        "entry_time": [
            day1,
            day1 + timedelta(hours=1),
            day2,
            day2 + timedelta(hours=1),
            day3,
        ],
        "exit_time": [
            day1 + timedelta(minutes=30),
            day1 + timedelta(hours=1, minutes=45),
            day2 + timedelta(minutes=20),
            day2 + timedelta(hours=1, minutes=15),
            day3 + timedelta(minutes=60),
        ],
        "entry_price": [18000.0, 18050.0, 18020.0, 18100.0, 18000.0],
        "exit_price": [18010.0, 18040.0, 18005.0, 18110.0, 18025.0],
        "direction": [1, -1, 1, -1, 1],  # long, short, long, short, long
        "size": [1, 1, 1, 1, 1],
    })

    # Hand-computed expected values:
    # Trade 1: Long 18000->18010, +10 pts = 10/0.25*5 = $200 gross, $185.50 net
    # Trade 2: Short 18050->18040, +10 pts = 10/0.25*5 = $200 gross, $185.50 net
    # Trade 3: Long 18020->18005, -15 pts = -15/0.25*5 = -$300 gross, -$314.50 net
    # Trade 4: Short 18100->18110, -10 pts = -10/0.25*5 = -$200 gross, -$214.50 net
    # Trade 5: Long 18000->18025, +25 pts = 25/0.25*5 = $500 gross, $485.50 net

    expected_gross_pnl = 200 + 200 - 300 - 200 + 500  # = $400
    expected_total_costs = 5 * TOTAL_COST_RT  # = 5 * 14.50 = $72.50
    expected_net_pnl = expected_gross_pnl - expected_total_costs  # = 400 - 72.50 = $327.50

    expected_win_count = 3  # Trades 1, 2, 5
    expected_loss_count = 2  # Trades 3, 4
    expected_win_rate = 3 / 5  # = 0.6

    expected_avg_trade_pnl = expected_net_pnl / 5  # = 327.50 / 5 = $65.50

    expected_max_win = 485.50  # Trade 5
    expected_max_loss = -314.50  # Trade 3

    # Profit factor: (185.50 + 185.50 + 485.50) / (314.50 + 214.50)
    # = 856.50 / 529.00 = 1.619...
    gross_wins = 185.50 + 185.50 + 485.50  # = 856.50
    gross_losses = 314.50 + 214.50  # = 529.00
    expected_profit_factor = gross_wins / gross_losses  # ~1.619

    # Equity curve: 0 -> 185.50 -> 371.00 -> 56.50 -> -158.00 -> 327.50
    # Running max:  0 -> 185.50 -> 371.00 -> 371.00 -> 371.00 -> 371.00
    # Drawdowns:    0 -> 0 -> 0 -> 314.50 -> 529.00 -> 43.50
    # Max drawdown = $529.00
    expected_max_drawdown = 529.00
    expected_max_drawdown_pct = (529.00 / (100_000.0 + 371.00)) * 100  # ~0.527%

    # Holding times: 30, 45, 20, 15, 60 minutes
    expected_avg_holding_time = (30 + 45 + 20 + 15 + 60) / 5  # = 34 minutes
    expected_avg_bars_held = expected_avg_holding_time / 5  # = 6.8 bars

    metrics = compute_metrics(trades)

    assert metrics["trade_count"] == 5
    assert metrics["win_count"] == expected_win_count
    assert metrics["loss_count"] == expected_loss_count
    assert abs(metrics["win_rate"] - expected_win_rate) < 1e-6

    assert abs(metrics["gross_pnl"] - expected_gross_pnl) < 1e-6
    assert abs(metrics["total_costs"] - expected_total_costs) < 1e-6
    assert abs(metrics["net_pnl"] - expected_net_pnl) < 1e-6
    assert abs(metrics["avg_trade_pnl"] - expected_avg_trade_pnl) < 1e-6

    assert abs(metrics["max_win"] - expected_max_win) < 1e-6
    assert abs(metrics["max_loss"] - expected_max_loss) < 1e-6

    assert abs(metrics["profit_factor"] - expected_profit_factor) < 0.01

    assert abs(metrics["max_drawdown"] - expected_max_drawdown) < 1e-6
    assert abs(metrics["max_drawdown_pct"] - expected_max_drawdown_pct) < 0.1

    assert abs(metrics["avg_holding_time_min"] - expected_avg_holding_time) < 1e-6
    assert abs(metrics["avg_bars_held"] - expected_avg_bars_held) < 1e-6

    # Sharpe should be non-zero and reasonable for this mixed win/loss set
    assert metrics["sharpe_ratio"] != 0.0


def test_empty_trades():
    """Test with empty trades DataFrame returns all zeros."""
    trades = pl.DataFrame({
        "entry_time": [],
        "exit_time": [],
        "entry_price": [],
        "exit_price": [],
        "direction": [],
        "size": [],
    }).with_columns([
        pl.col("entry_time").cast(pl.Datetime),
        pl.col("exit_time").cast(pl.Datetime),
        pl.col("entry_price").cast(pl.Float64),
        pl.col("exit_price").cast(pl.Float64),
        pl.col("direction").cast(pl.Int8),
        pl.col("size").cast(pl.Int32),
    ])

    metrics = compute_metrics(trades)

    assert metrics["trade_count"] == 0
    assert metrics["win_count"] == 0
    assert metrics["loss_count"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["gross_pnl"] == 0.0
    assert metrics["total_costs"] == 0.0
    assert metrics["net_pnl"] == 0.0
    assert metrics["avg_trade_pnl"] == 0.0
    assert metrics["max_win"] == 0.0
    assert metrics["max_loss"] == 0.0
    assert metrics["profit_factor"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["max_drawdown_pct"] == 0.0
    assert metrics["avg_holding_time_min"] == 0.0
    assert metrics["avg_bars_held"] == 0.0


def test_all_winners():
    """Test with all winning trades - profit factor is unbounded."""
    base_time = datetime(2025, 2, 1, 10, 0, 0)

    trades = pl.DataFrame({
        "entry_time": [
            base_time,
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=2),
        ],
        "exit_time": [
            base_time + timedelta(minutes=30),
            base_time + timedelta(hours=1, minutes=30),
            base_time + timedelta(hours=2, minutes=30),
        ],
        "entry_price": [18000.0, 18010.0, 18020.0],
        "exit_price": [18010.0, 18020.0, 18030.0],
        "direction": [1, 1, 1],
        "size": [1, 1, 1],
    })

    metrics = compute_metrics(trades)

    assert metrics["trade_count"] == 3
    assert metrics["win_count"] == 3
    assert metrics["loss_count"] == 0
    assert metrics["win_rate"] == 1.0
    assert np.isinf(metrics["profit_factor"])
    assert metrics["net_pnl"] > 0
    assert metrics["max_loss"] == 0.0  # No losses


def test_all_losers():
    """Test with all losing trades."""
    base_time = datetime(2025, 2, 1, 10, 0, 0)

    trades = pl.DataFrame({
        "entry_time": [
            base_time,
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=2),
        ],
        "exit_time": [
            base_time + timedelta(minutes=30),
            base_time + timedelta(hours=1, minutes=30),
            base_time + timedelta(hours=2, minutes=30),
        ],
        "entry_price": [18010.0, 18020.0, 18030.0],
        "exit_price": [18000.0, 18010.0, 18020.0],
        "direction": [1, 1, 1],
        "size": [1, 1, 1],
    })

    metrics = compute_metrics(trades)

    assert metrics["trade_count"] == 3
    assert metrics["win_count"] == 0
    assert metrics["loss_count"] == 3
    assert metrics["win_rate"] == 0.0
    assert metrics["profit_factor"] == 0.0
    assert metrics["net_pnl"] < 0
    assert metrics["max_win"] == 0.0  # No wins
    # With initial_capital=100k, drawdown % is relative to peak account value
    assert metrics["max_drawdown"] > 0
    # 3 * $214.50 = $643.50 loss, peak = $100k, pct = 0.6435%
    assert metrics["max_drawdown_pct"] == pytest.approx(0.6435, abs=0.01)


def test_single_trade_sharpe():
    """Test with single trade - Sharpe should be NaN (undefined)."""
    base_time = datetime(2025, 2, 1, 10, 0, 0)

    trades = pl.DataFrame({
        "entry_time": [base_time],
        "exit_time": [base_time + timedelta(minutes=30)],
        "entry_price": [18000.0],
        "exit_price": [18010.0],
        "direction": [1],
        "size": [1],
    })

    metrics = compute_metrics(trades)

    assert metrics["trade_count"] == 1
    assert np.isnan(metrics["sharpe_ratio"])


def test_single_day_trades_sharpe_is_nan():
    """Multiple trades on one active date should produce undefined Sharpe."""
    day = datetime(2025, 2, 1, 10, 0, 0)

    trades = pl.DataFrame({
        "entry_time": [day, day + timedelta(hours=1)],
        "exit_time": [
            day + timedelta(minutes=30),
            day + timedelta(hours=1, minutes=30),
        ],
        "entry_price": [18000.0, 18020.0],
        "exit_price": [18010.0, 18010.0],
        "direction": [1, -1],
        "size": [1, 1],
    })

    metrics = compute_metrics(trades)
    assert metrics["trade_count"] == 2
    assert np.isnan(metrics["sharpe_ratio"])


def test_sharpe_zero_fills_gap_days():
    """Sharpe should zero-fill non-trading weekdays between first and last trade."""
    day1 = datetime(2025, 2, 3, 10, 0, 0)   # Monday
    day3 = datetime(2025, 2, 5, 10, 0, 0)   # Wednesday (gap: Tuesday Feb 4)

    trades = pl.DataFrame({
        "entry_time": [day1, day3],
        "exit_time": [day1 + timedelta(minutes=30), day3 + timedelta(minutes=30)],
        "entry_price": [18000.0, 18000.0],
        "exit_price": [18015.0, 18005.0],  # Net: +285.5, +85.5
        "direction": [1, 1],
        "size": [1, 1],
    })

    metrics = compute_metrics(trades)

    # 3 days (Mon, Tue=0, Wed) after zero-fill
    expected_daily = np.array([285.5, 0.0, 85.5], dtype=np.float64)
    expected_sharpe = (expected_daily.mean() / expected_daily.std(ddof=1)) * np.sqrt(252)
    assert abs(metrics["sharpe_ratio"] - float(expected_sharpe)) < 1e-9


def test_sharpe_sparse_trading_diluted():
    """Sparse trading over many weekdays should produce lower Sharpe than dense trading."""
    # Trade only on Mon and Fri of a 2-week span (4 trading days out of 10)
    trades = pl.DataFrame({
        "entry_time": [
            datetime(2025, 2, 3, 10, 0, 0),   # Mon week 1
            datetime(2025, 2, 7, 10, 0, 0),   # Fri week 1
            datetime(2025, 2, 10, 10, 0, 0),  # Mon week 2
            datetime(2025, 2, 14, 10, 0, 0),  # Fri week 2
        ],
        "exit_time": [
            datetime(2025, 2, 3, 10, 30, 0),
            datetime(2025, 2, 7, 10, 30, 0),
            datetime(2025, 2, 10, 10, 30, 0),
            datetime(2025, 2, 14, 10, 30, 0),
        ],
        "entry_price": [18000.0, 18000.0, 18000.0, 18000.0],
        "exit_price": [18010.0, 18010.0, 18010.0, 18010.0],
        "direction": [1, 1, 1, 1],
        "size": [1, 1, 1, 1],
    })

    metrics = compute_metrics(trades)

    # Net PnL per trade: 10pts = $200 gross - $14.50 = $185.50
    # 10 weekdays total (Feb 3-14), 4 with trades, 6 zero-filled
    # Daily PnLs: [185.5, 0, 0, 0, 185.5, 0, 0, 0, 0, 185.5, 0, 0, 0, 185.5]
    # wait - Feb 3 (Mon) to Feb 14 (Fri) = 10 weekdays
    pnl_per_trade = 185.5
    daily = np.zeros(10)
    daily[0] = pnl_per_trade   # Mon Feb 3
    daily[4] = pnl_per_trade   # Fri Feb 7
    daily[5] = pnl_per_trade   # Mon Feb 10
    daily[9] = pnl_per_trade   # Fri Feb 14
    expected_sharpe = (daily.mean() / daily.std(ddof=1)) * np.sqrt(252)
    assert abs(metrics["sharpe_ratio"] - expected_sharpe) < 1e-6


def test_daily_pnl_series_excludes_weekends_but_keeps_friday():
    """Zero-filled daily series should include Friday and skip Saturday/Sunday."""
    friday = datetime(2025, 2, 7, 10, 0, 0)
    monday = datetime(2025, 2, 10, 10, 0, 0)

    trades = pl.DataFrame({
        "entry_time": [friday, monday],
        "exit_time": [friday + timedelta(minutes=30), monday + timedelta(minutes=30)],
        "entry_price": [18000.0, 18000.0],
        "exit_price": [18010.0, 18010.0],
        "direction": [1, 1],
        "size": [1, 1],
    })

    daily = compute_daily_pnl_series(trades)
    dates = daily["_date"].to_list()

    assert dates == [
        datetime(2025, 2, 7).date(),
        datetime(2025, 2, 10).date(),
    ]


def test_drawdown_specific_sequence():
    """Test drawdown calculation with a specific equity sequence."""
    base_time = datetime(2025, 2, 1, 10, 0, 0)

    # Create trades with specific PnL sequence: +100, +100, -150, -100, +200
    # Equity: 0 -> 100 -> 200 -> 50 -> -50 -> 150
    # Running max: 0 -> 100 -> 200 -> 200 -> 200 -> 200
    # Drawdowns: 0 -> 0 -> 0 -> 150 -> 250 -> 50
    # Max DD = 250

    # Trade 1: +100 net (need +114.50 gross to get +100 net)
    # -> +114.50 gross = 22.9 points -> round to 23 points (18000->18023)
    # Trade 2: Same
    # Trade 3: -150 net = -135.50 gross = -27.1 points -> -27 points
    # Trade 4: -100 net = -85.50 gross = -17.1 points -> -17 points
    # Trade 5: +200 net = +214.50 gross = 42.9 points -> 43 points

    trades = pl.DataFrame({
        "entry_time": [
            base_time + timedelta(hours=i) for i in range(5)
        ],
        "exit_time": [
            base_time + timedelta(hours=i, minutes=30) for i in range(5)
        ],
        "entry_price": [18000.0, 18000.0, 18000.0, 18000.0, 18000.0],
        "exit_price": [18005.75, 18005.75, 17993.25, 17995.75, 18010.75],  # Adjusted for approx net PnL
        "direction": [1, 1, 1, 1, 1],
        "size": [1, 1, 1, 1, 1],
    })

    metrics = compute_metrics(trades)

    # Net PnLs (approx):
    # 5.75 pts = $115 gross - $14.50 = $100.50 net
    # 5.75 pts = $115 gross - $14.50 = $100.50 net
    # -6.75 pts = -$135 gross - $14.50 = -$149.50 net
    # -4.25 pts = -$85 gross - $14.50 = -$99.50 net
    # 10.75 pts = $215 gross - $14.50 = $200.50 net

    # Equity: 0 -> 100.50 -> 201 -> 51.50 -> -48 -> 152.50
    # Running max: 0 -> 100.50 -> 201 -> 201 -> 201 -> 201
    # Max DD should be around 201 - (-48) = 249

    assert metrics["trade_count"] == 5
    assert metrics["max_drawdown"] > 240  # Should be close to 249
    assert metrics["max_drawdown"] < 260
    # Max DD% = 249 / (100000 + 201) * 100 ~= 0.249%
    assert metrics["max_drawdown_pct"] > 0.20
    assert metrics["max_drawdown_pct"] < 0.30


def test_max_drawdown_pct_uses_local_peak_not_global_peak():
    """Max drawdown % must use the peak that preceded the drawdown trough."""
    base_time = datetime(2025, 2, 1, 10, 0, 0)
    # Net PnL path: +10k, -30k, +140k
    # Equity: 100k -> 110k -> 80k -> 220k
    # Max drawdown: 30k from local peak 110k => 27.27%
    trades = pl.DataFrame({
        "entry_time": [base_time + timedelta(hours=i) for i in range(3)],
        "exit_time": [base_time + timedelta(hours=i, minutes=30) for i in range(3)],
        "entry_price": [18000.0, 18000.0, 18000.0],
        "exit_price": [18500.0, 16500.0, 25000.0],  # +10k, -30k, +140k gross
        "direction": [1, 1, 1],
        "size": [1, 1, 1],
    })

    metrics = compute_metrics(trades, cost_override_col=None, initial_capital=100_000.0)

    assert metrics["max_drawdown"] == pytest.approx(30_014.5, abs=1e-6)
    local_peak = 100_000.0 + 10_000.0 - TOTAL_COST_RT
    expected_pct = (30_014.5 / local_peak) * 100.0
    assert metrics["max_drawdown_pct"] == pytest.approx(expected_pct, rel=1e-6)


def test_max_drawdown_pct_is_max_of_pointwise_percent_drawdowns():
    """Max percentage drawdown can occur at a different point than max dollar drawdown."""
    base_time = datetime(2025, 2, 1, 10, 0, 0)

    # Net PnL path (with zero override costs): +1000, -900, +100000, -1000
    # Equity: 100000 -> 101000 -> 100100 -> 200100 -> 199100
    # Dollar drawdowns: 0, 900, 0, 1000  -> max $ DD = 1000 at final point.
    # Pct drawdowns:    0, 900/101000, 0, 1000/200100
    # => max pct DD is 900/101000 (~0.8911%), not 1000/200100 (~0.4998%).
    trades = pl.DataFrame({
        "entry_time": [base_time + timedelta(hours=i) for i in range(4)],
        "exit_time": [base_time + timedelta(hours=i, minutes=30) for i in range(4)],
        "entry_price": [18000.0, 18000.0, 18000.0, 18000.0],
        "exit_price": [18050.0, 17955.0, 23000.0, 17950.0],  # +50, -45, +5000, -50 points
        "direction": [1, 1, 1, 1],
        "size": [1, 1, 1, 1],
        "cost_override": [0.0, 0.0, 0.0, 0.0],
    })

    metrics = compute_metrics(
        trades,
        cost_override_col="cost_override",
        initial_capital=100_000.0,
    )

    assert metrics["max_drawdown"] == pytest.approx(1000.0, abs=1e-9)
    expected_pct = (900.0 / 101_000.0) * 100.0
    assert metrics["max_drawdown_pct"] == pytest.approx(expected_pct, rel=1e-9)


def test_identical_trades_sharpe():
    """Test with identical daily PnLs - std = 0, Sharpe should be NaN."""
    # Spread across 5 days, 1 identical trade per day → identical daily PnLs → std=0
    trades = pl.DataFrame({
        "entry_time": [
            datetime(2025, 2, d, 10, 0, 0) for d in range(1, 6)
        ],
        "exit_time": [
            datetime(2025, 2, d, 10, 30, 0) for d in range(1, 6)
        ],
        "entry_price": [18000.0] * 5,
        "exit_price": [18010.0] * 5,
        "direction": [1] * 5,
        "size": [1] * 5,
    })

    metrics = compute_metrics(trades)

    assert metrics["trade_count"] == 5
    assert metrics["win_count"] == 5
    assert np.isnan(metrics["sharpe_ratio"])


def test_multiple_contracts():
    """Test with multiple contracts per trade (size > 1)."""
    base_time = datetime(2025, 2, 1, 10, 0, 0)

    trades = pl.DataFrame({
        "entry_time": [base_time],
        "exit_time": [base_time + timedelta(minutes=30)],
        "entry_price": [18000.0],
        "exit_price": [18010.0],
        "direction": [1],
        "size": [3],  # 3 contracts
    })

    metrics = compute_metrics(trades)

    # 10 points * 3 contracts = 30 points gross
    # Gross PnL = 30 / 0.25 * 5 = $600
    # Costs = 3 * $14.50 = $43.50
    # Net PnL = $600 - $43.50 = $556.50

    assert metrics["trade_count"] == 1
    assert abs(metrics["gross_pnl"] - 600.0) < 1e-6
    assert abs(metrics["total_costs"] - 43.50) < 1e-6
    assert abs(metrics["net_pnl"] - 556.50) < 1e-6


def test_short_trades():
    """Test short trades calculate PnL correctly."""
    base_time = datetime(2025, 2, 1, 10, 0, 0)

    trades = pl.DataFrame({
        "entry_time": [base_time, base_time + timedelta(hours=1)],
        "exit_time": [
            base_time + timedelta(minutes=30),
            base_time + timedelta(hours=1, minutes=30),
        ],
        "entry_price": [18010.0, 18020.0],
        "exit_price": [18000.0, 18030.0],
        "direction": [-1, -1],  # Both short
        "size": [1, 1],
    })

    metrics = compute_metrics(trades)

    # Trade 1: Short 18010->18000, +10 pts = $200 gross, $185.50 net (WIN)
    # Trade 2: Short 18020->18030, -10 pts = -$200 gross, -$214.50 net (LOSS)

    assert metrics["trade_count"] == 2
    assert metrics["win_count"] == 1
    assert metrics["loss_count"] == 1
    assert abs(metrics["net_pnl"] - (185.50 - 214.50)) < 1e-6
