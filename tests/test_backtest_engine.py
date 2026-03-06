"""Tests for backtesting engine."""

from datetime import datetime, timedelta, timezone
import math
import polars as pl
import pytest
from src.framework.backtest.engine import run_backtest, TRADE_SCHEMA, _points_to_dollars
from src.framework.data.constants import TICK_SIZE, TICK_VALUE

UTC = timezone.utc


def create_test_df(timestamps, closes, signals, highs=None, lows=None, opens=None):
    """Helper to create test DataFrames with UTC timestamps."""
    data = {
        "ts_event": timestamps,
        "close": closes,
        "signal": signals,
    }
    if highs is not None:
        data["high"] = highs
    if lows is not None:
        data["low"] = lows
    if opens is not None:
        data["open"] = opens
    return pl.DataFrame(data).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))


def test_points_to_dollars_rounds_half_ticks_away_from_zero():
    assert _points_to_dollars(TICK_SIZE / 2.0) == TICK_VALUE
    assert _points_to_dollars(-TICK_SIZE / 2.0) == -TICK_VALUE


def test_always_long():
    """Test always-long signal produces correct single trade."""
    # 5 bars, long from bar 0 to bar 4, exit at bar 4
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(5)]
    closes = [18000.0, 18005.0, 18010.0, 18015.0, 18020.0]
    signals = [1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df)

    assert len(trades) == 1, "Should produce 1 trade"
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == timestamps[0]
    assert trade["exit_time"] == timestamps[4]
    assert trade["entry_price"] == 18000.0
    assert trade["exit_price"] == 18020.0
    assert trade["direction"] == 1
    assert trade["size"] == 1


def test_long_short_alternating():
    """Test long-short alternating signals."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(5)]
    closes = [18000.0, 18010.0, 18015.0, 18005.0, 18000.0]
    signals = [1, 1, -1, -1, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df)

    assert len(trades) == 2, "Should produce 2 trades"

    # First trade: long from bar 0 to bar 2
    trade1 = trades.row(0, named=True)
    assert trade1["entry_time"] == timestamps[0]
    assert trade1["exit_time"] == timestamps[2]
    assert trade1["entry_price"] == 18000.0
    assert trade1["exit_price"] == 18015.0
    assert trade1["direction"] == 1

    # Second trade: short from bar 2 to bar 4
    trade2 = trades.row(1, named=True)
    assert trade2["entry_time"] == timestamps[2]
    assert trade2["exit_time"] == timestamps[4]
    assert trade2["entry_price"] == 18015.0
    assert trade2["exit_price"] == 18000.0
    assert trade2["direction"] == -1


def test_all_flat():
    """Test that all-flat signal produces no trades."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(5)]
    closes = [18000.0, 18005.0, 18010.0, 18015.0, 18020.0]
    signals = [0, 0, 0, 0, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df)

    assert len(trades) == 0, "Should produce 0 trades"
    # Check schema is correct even for empty DataFrame
    assert trades.schema == TRADE_SCHEMA


def test_end_of_day_exit():
    """Test that positions are force-closed at end of day."""
    # Day 1: long position, stays open until end of day
    day1_start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    day1_times = [day1_start + timedelta(minutes=5 * i) for i in range(3)]

    # Day 2: new day, position should have been closed at end of day 1
    day2_start = datetime(2025, 1, 2, 10, 0, 0, tzinfo=UTC)
    day2_times = [day2_start + timedelta(minutes=5 * i) for i in range(3)]

    timestamps = day1_times + day2_times
    closes = [18000.0, 18010.0, 18020.0, 18025.0, 18030.0, 18035.0]
    signals = [1, 1, 1, 0, 0, 0]  # Long on day 1, flat on day 2

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df)

    assert len(trades) == 1, "Should produce 1 trade (closed at end of day 1)"
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == day1_times[0]
    assert trade["exit_time"] == day1_times[2], "Should exit at last bar of day 1"
    assert trade["entry_price"] == 18000.0
    assert trade["exit_price"] == 18020.0
    assert trade["direction"] == 1


def test_direction_flip():
    """Test that signal flipping from long to short closes and reopens."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(4)]
    closes = [18000.0, 18010.0, 18005.0, 18015.0]
    signals = [1, 1, -1, -1]  # Long → short flip at bar 2

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df)

    assert len(trades) == 2, "Should produce 2 trades (close long, open short)"

    # First trade: long from bar 0 to bar 2
    trade1 = trades.row(0, named=True)
    assert trade1["entry_time"] == timestamps[0]
    assert trade1["exit_time"] == timestamps[2]
    assert trade1["entry_price"] == 18000.0
    assert trade1["exit_price"] == 18005.0
    assert trade1["direction"] == 1

    # Second trade: short from bar 2 to end of day
    trade2 = trades.row(1, named=True)
    assert trade2["entry_time"] == timestamps[2]
    assert trade2["exit_time"] == timestamps[3]  # Last bar of day
    assert trade2["entry_price"] == 18005.0
    assert trade2["exit_price"] == 18015.0
    assert trade2["direction"] == -1


def test_max_daily_loss():
    """Test that trading stops when daily loss exceeds limit."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(6)]

    # Set up a scenario with large losses
    # Long at 18100, exit at 18000 → 100 points = $2000 loss (exceeds $1000 limit)
    closes = [18100.0, 18000.0, 18010.0, 18020.0, 18030.0, 18040.0]
    signals = [1, 0, 1, 1, 1, 0]  # First trade loses, second signal should be ignored

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df, max_daily_loss=1000.0)

    assert len(trades) == 1, "Should produce only 1 trade (stopped after loss)"
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == timestamps[0]
    assert trade["exit_time"] == timestamps[1]
    assert trade["entry_price"] == 18100.0
    assert trade["exit_price"] == 18000.0

    # Verify PnL: -100 points * $5/tick / 0.25 tick = -$2000
    pnl_points = trade["exit_price"] - trade["entry_price"]
    pnl_dollars = (pnl_points / TICK_SIZE) * TICK_VALUE
    assert pnl_dollars == -2000.0
    assert pnl_dollars <= -1000.0, "Loss should exceed limit"


def test_max_daily_loss_intrabar_enforced_with_hilo():
    """Daily loss limit should force intrabar exit when high/low breaches it."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(6)]

    # Entry at 18100. Daily loss limit = $1000 net.
    # Engine reserves TOTAL_COST_RT for the closing trade:
    # gross-loss budget = 1000 - 14.5 = 985.5 -> 197 ticks -> 49.25 points.
    # stop level = 18100 - 49.25 = 18050.75
    closes = [18100.0, 18090.0, 18120.0, 18130.0, 18140.0, 18150.0]
    opens = [18100.0, 18100.0, 18110.0, 18120.0, 18130.0, 18140.0]
    highs = [18105.0, 18110.0, 18125.0, 18135.0, 18145.0, 18155.0]
    lows = [18095.0, 18040.0, 18100.0, 18120.0, 18130.0, 18140.0]
    signals = [1, 1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows, opens=opens)
    trades = run_backtest(df, max_daily_loss=1000.0)

    assert len(trades) == 1, "Should stop trading after intrabar daily-loss breach"
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == timestamps[0]
    assert trade["exit_time"] == timestamps[1]
    assert trade["entry_price"] == 18100.0
    assert trade["exit_price"] == 18050.75, "Exit should reserve close-cost in loss guard"


def test_max_daily_loss_close_fallback_without_hilo():
    """Daily loss limit should force close-based exit when high/low is unavailable."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(6)]

    # No high/low columns; close at bar 1 breaches daily loss threshold.
    closes = [18100.0, 18040.0, 18120.0, 18130.0, 18140.0, 18150.0]
    signals = [1, 1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df, max_daily_loss=1000.0)

    assert len(trades) == 1, "Should stop trading after close-based daily-loss breach"
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == timestamps[0]
    assert trade["exit_time"] == timestamps[1]
    assert trade["entry_price"] == 18100.0
    assert trade["exit_price"] == 18040.0


def test_max_daily_loss_close_fallback_reserves_closing_cost():
    """Close-based daily-loss guard should include close-cost in threshold check."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(4)]
    # Bar 1 unrealized gross loss = -990 dollars (49.5 points).
    # Net after close-cost would be -1004.5, which must trigger the guard.
    closes = [18100.0, 18050.50, 18080.0, 18090.0]
    signals = [1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)  # no high/low
    trades = run_backtest(df, max_daily_loss=1000.0)

    assert len(trades) == 1
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == timestamps[0]
    assert trade["exit_time"] == timestamps[1]
    assert trade["exit_price"] == 18050.50


def test_single_bar_trade():
    """Test trade that enters and exits on consecutive bars."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(3)]
    closes = [18000.0, 18010.0, 18020.0]
    signals = [0, 1, 0]  # Enter at bar 1, exit at bar 2

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df)

    assert len(trades) == 1, "Should produce 1 trade"
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == timestamps[1]
    assert trade["exit_time"] == timestamps[2]
    assert trade["entry_price"] == 18010.0
    assert trade["exit_price"] == 18020.0
    assert trade["direction"] == 1


def test_empty_dataframe():
    """Test that empty DataFrame raises assertion."""
    df = pl.DataFrame(schema={
        "ts_event": pl.Datetime("ns", "UTC"),
        "close": pl.Float64,
        "signal": pl.Int8,
    })

    with pytest.raises(AssertionError, match="Empty DataFrame"):
        run_backtest(df)


def test_missing_signal_column():
    """Test that missing signal column raises assertion."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    df = pl.DataFrame({
        "ts_event": [start],
        "close": [18000.0],
    })

    with pytest.raises(AssertionError, match="Missing signal column"):
        run_backtest(df)


def test_missing_required_columns():
    """Test that missing required columns raise assertions."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)

    # Missing ts_event
    df1 = pl.DataFrame({
        "close": [18000.0],
        "signal": [1],
    })
    with pytest.raises(AssertionError, match="Missing ts_event column"):
        run_backtest(df1)

    # Missing close
    df2 = pl.DataFrame({
        "ts_event": [start],
        "signal": [1],
    })
    with pytest.raises(AssertionError, match="Missing close column"):
        run_backtest(df2)


def test_short_trade_pnl():
    """Test that short trade PnL is calculated correctly."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(3)]
    # Short at 18020, cover at 18000 → profit 20 points
    closes = [18020.0, 18010.0, 18000.0]
    signals = [-1, -1, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df)

    assert len(trades) == 1
    trade = trades.row(0, named=True)
    assert trade["direction"] == -1
    assert trade["entry_price"] == 18020.0
    assert trade["exit_price"] == 18000.0

    # Short profit: (entry - exit) * -1 = (18020 - 18000) * -1 = 20 points
    # But direction is already -1, so: (exit - entry) * direction = (18000 - 18020) * -1 = 20
    pnl_points = (trade["exit_price"] - trade["entry_price"]) * trade["direction"]
    assert pnl_points == 20.0


def test_multiple_days_with_positions():
    """Test handling of multiple days with positions carried over and closed."""
    # Day 1: long position
    day1_start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    day1_times = [day1_start + timedelta(minutes=5 * i) for i in range(3)]

    # Day 2: short position
    day2_start = datetime(2025, 1, 2, 10, 0, 0, tzinfo=UTC)
    day2_times = [day2_start + timedelta(minutes=5 * i) for i in range(3)]

    timestamps = day1_times + day2_times
    closes = [18000.0, 18010.0, 18020.0, 18030.0, 18020.0, 18010.0]
    signals = [1, 1, 1, -1, -1, -1]  # Long day 1, short day 2

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df)

    assert len(trades) == 2, "Should produce 2 trades (1 per day)"

    # Day 1: long trade
    trade1 = trades.row(0, named=True)
    assert trade1["entry_time"].date() == day1_start.date()
    assert trade1["exit_time"] == day1_times[2]  # Closed at end of day 1
    assert trade1["direction"] == 1

    # Day 2: short trade
    trade2 = trades.row(1, named=True)
    assert trade2["entry_time"] == day2_times[0]
    assert trade2["exit_time"] == day2_times[2]  # Closed at end of day 2
    assert trade2["direction"] == -1


def test_max_daily_loss_reset_next_day():
    """Test that daily loss limit resets on next day."""
    # Day 1: exceed loss limit
    day1_start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    day1_times = [day1_start + timedelta(minutes=5 * i) for i in range(3)]

    # Day 2: should allow trading again
    day2_start = datetime(2025, 1, 2, 10, 0, 0, tzinfo=UTC)
    day2_times = [day2_start + timedelta(minutes=5 * i) for i in range(3)]

    timestamps = day1_times + day2_times
    # Day 1: lose 100 points ($2000)
    # Day 2: win 10 points ($200)
    closes = [18100.0, 18000.0, 18000.0, 18000.0, 18010.0, 18010.0]
    signals = [1, 0, 0, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df, max_daily_loss=1000.0)

    assert len(trades) == 2, "Should produce 2 trades (loss limit resets on day 2)"

    # Day 1: losing trade
    trade1 = trades.row(0, named=True)
    assert trade1["entry_time"].date() == day1_start.date()

    # Day 2: winning trade (allowed because limit reset)
    trade2 = trades.row(1, named=True)
    assert trade2["entry_time"].date() == day2_start.date()


def test_engine_metrics_integration():
    """End-to-end test: engine produces trades, metrics computes correct PnL."""
    from src.framework.backtest.metrics import compute_metrics
    from src.framework.data.constants import TOTAL_COST_RT

    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(6)]
    # Trade 1: Long, entry 18000 exit 18010 → gross +10pts = +$200, net = $200 - $14.50 = $185.50
    # Trade 2: Short, entry 18010 exit 18000 → gross +10pts = +$200, net = $200 - $14.50 = $185.50
    closes = [18000.0, 18005.0, 18010.0, 18010.0, 18005.0, 18000.0]
    signals = [1, 1, 0, -1, -1, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df)
    assert len(trades) == 2

    metrics = compute_metrics(trades)
    expected_gross = 200.0 + 200.0  # Both trades +$200 gross
    expected_costs = TOTAL_COST_RT * 2  # $14.50 * 2 = $29.00
    expected_net = expected_gross - expected_costs  # $371.00

    assert abs(metrics["gross_pnl"] - expected_gross) < 0.01, f"Gross PnL should be {expected_gross}, got {metrics['gross_pnl']}"
    assert abs(metrics["total_costs"] - expected_costs) < 0.01, f"Costs should be {expected_costs}, got {metrics['total_costs']}"
    assert abs(metrics["net_pnl"] - expected_net) < 0.01, f"Net PnL should be {expected_net}, got {metrics['net_pnl']}"
    assert metrics["trade_count"] == 2
    assert metrics["win_count"] == 2
    assert metrics["win_rate"] == 1.0
    assert math.isinf(metrics["profit_factor"])  # All winners, no losses


def test_invalid_signal_values():
    """Test that invalid signal values raise assertion."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(3)]
    closes = [18000.0, 18010.0, 18020.0]
    signals = [2, 0, -1]  # Invalid: 2 is not in {-1, 0, 1}

    df = create_test_df(timestamps, closes, signals)
    with pytest.raises(AssertionError, match="Signal must be"):
        run_backtest(df)


# --- Exit type tests ---


def test_exit_bars_closes_after_n_bars():
    """Time-based exit: close position after exit_bars bars."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(8)]
    closes = [18000.0, 18005.0, 18010.0, 18015.0, 18020.0, 18025.0, 18030.0, 18035.0]
    signals = [1, 1, 1, 1, 1, 1, 1, 0]  # Continuous long signal

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df, exit_bars=3)

    # exit_bars=3: entry bar 0, held bars 1,2,3 → exit at bar 3 (3 bars held)
    assert len(trades) >= 1
    trade1 = trades.row(0, named=True)
    assert trade1["entry_time"] == timestamps[0]
    assert trade1["exit_time"] == timestamps[3], "Should exit after 3 bars"
    assert trade1["entry_price"] == 18000.0
    assert trade1["exit_price"] == 18015.0


def test_exit_bars_reopens_if_signal_persists():
    """After time exit, re-enter if signal still active."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(8)]
    closes = [18000.0, 18005.0, 18010.0, 18015.0, 18020.0, 18025.0, 18030.0, 18035.0]
    signals = [1, 1, 1, 1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df, exit_bars=2)

    # exit_bars=2: should produce multiple trades
    assert len(trades) >= 2, "Should re-enter after time-based exit"


def test_profit_target_exit():
    """Profit target exit: close when unrealized gain exceeds target."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(6)]
    # Entry at 18000, price rises 10 pts per bar
    closes = [18000.0, 18005.0, 18010.0, 18020.0, 18025.0, 18030.0]
    signals = [1, 1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    # Profit target = 15 points. Unrealized at bar 3: (18020-18000)*1 = 20 >= 15
    trades = run_backtest(df, profit_target=15.0)

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 18000.0
    # Should exit at bar 3 where unrealized = 20 >= 15
    assert trade["exit_price"] == 18020.0


def test_stop_loss_exit():
    """Stop loss exit: close when unrealized loss exceeds stop."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(6)]
    # Entry at 18000, price drops
    closes = [18000.0, 17995.0, 17990.0, 17985.0, 17980.0, 17975.0]
    signals = [1, 1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    # Stop loss = 12 points. Unrealized at bar 3: (17985-18000)*1 = -15, abs > 12
    trades = run_backtest(df, stop_loss=12.0)

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 18000.0
    # Should exit at bar 3 where unrealized = -15 <= -12
    assert trade["exit_price"] == 17985.0


def test_stop_loss_short_trade():
    """Stop loss on short trade: close when price rises too much."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(6)]
    # Short at 18000, price rises
    closes = [18000.0, 18005.0, 18010.0, 18020.0, 18025.0, 18030.0]
    signals = [-1, -1, -1, -1, -1, 0]

    df = create_test_df(timestamps, closes, signals)
    # Stop loss = 15 pts. For short: unrealized = (close - entry) * -1
    # Bar 3: (18020-18000)*-1 = -20 <= -15 → stop
    trades = run_backtest(df, stop_loss=15.0)

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 18000.0
    assert trade["exit_price"] == 18020.0
    assert trade["direction"] == -1


def test_exit_bars_none_preserves_signal_based_exit():
    """When exit_bars=None, behavior matches original signal-based exit."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(5)]
    closes = [18000.0, 18005.0, 18010.0, 18015.0, 18020.0]
    signals = [1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df, exit_bars=None, profit_target=None, stop_loss=None)

    assert len(trades) == 1
    trade = trades.row(0, named=True)
    assert trade["exit_time"] == timestamps[4], "Should exit on signal flip"


def test_combined_exit_conditions():
    """Multiple exit conditions: whichever triggers first wins."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(10)]
    # Price rises 5 pts per bar
    closes = [18000.0 + 5.0 * i for i in range(10)]
    signals = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    # exit_bars=6, profit_target=20. Profit target at bar 4 (20pts) comes first
    trades = run_backtest(df, exit_bars=6, profit_target=20.0)

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["exit_price"] == 18020.0, "Profit target should trigger before time exit"


def test_eod_exit_overrides_exit_bars():
    """End-of-day exit still fires even if exit_bars not reached."""
    day1_start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    day1_times = [day1_start + timedelta(minutes=5 * i) for i in range(3)]
    day2_start = datetime(2025, 1, 2, 10, 0, 0, tzinfo=UTC)
    day2_times = [day2_start + timedelta(minutes=5 * i) for i in range(3)]

    timestamps = day1_times + day2_times
    closes = [18000.0, 18010.0, 18020.0, 18025.0, 18030.0, 18035.0]
    signals = [1, 1, 1, 0, 0, 0]

    df = create_test_df(timestamps, closes, signals)
    # exit_bars=10 — much longer than day, so EOD should force close
    trades = run_backtest(df, exit_bars=10)

    assert len(trades) == 1
    trade = trades.row(0, named=True)
    assert trade["exit_time"] == day1_times[2], "EOD should override exit_bars"


def test_no_reentry_on_same_bar_after_force_exit():
    """W4: After a force exit (PT/SL/exit_bars), no new position opens on the same bar."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(8)]
    # Price rises so profit target triggers at bar 2 (20pts >= 15pt target)
    closes = [18000.0, 18010.0, 18020.0, 18025.0, 18030.0, 18035.0, 18040.0, 18045.0]
    signals = [1, 1, 1, 1, 1, 1, 1, 0]  # Continuous long signal

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df, profit_target=15.0)

    # Trade 1 exits at bar 2 (20pts >= 15pt target)
    assert len(trades) >= 1
    trade1 = trades.row(0, named=True)
    assert trade1["exit_price"] == 18020.0

    # If there's a second trade, it must NOT start at bar 2 (same bar as exit)
    if len(trades) >= 2:
        trade2 = trades.row(1, named=True)
        assert trade2["entry_time"] > trade1["exit_time"], \
            "Re-entry must not happen on the same bar as force exit"


def test_combined_exit_stop_loss_fires_first():
    """W5: With exit_bars + PT + SL all set, stop loss fires before the others."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(8)]
    # Long at 18000, price drops sharply — SL should fire before exit_bars or PT
    closes = [18000.0, 17990.0, 17980.0, 17970.0, 17960.0, 17950.0, 17940.0, 17930.0]
    signals = [1, 1, 1, 1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    # exit_bars=6, profit_target=50, stop_loss=15
    # SL triggers at bar 2: unrealized = (17980-18000)*1 = -20 <= -15
    trades = run_backtest(df, exit_bars=6, profit_target=50.0, stop_loss=15.0)

    assert len(trades) >= 1
    trade1 = trades.row(0, named=True)
    assert trade1["entry_price"] == 18000.0
    assert trade1["exit_price"] == 17980.0, "Stop loss should fire at bar 2 (-20pts <= -15pt SL)"


def test_profit_target_return_per_trade():
    """Return-space PT/SL converts per-trade using entry price, not global avg."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(6)]
    # Entry at 20000, target_return=0.001 → 20 points
    closes = [20000.0, 20010.0, 20015.0, 20025.0, 20030.0, 20035.0]
    signals = [1, 1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df, profit_target_return=0.001)

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 20000.0
    # PT = 20000 * 0.001 = 20 points. Bar 3: unrealized = 25 >= 20
    assert trade["exit_price"] == 20025.0


# --- Intra-bar PT/SL tests ---


def test_intrabar_stop_loss_long():
    """Long entry, bar low hits stop but close is above entry -> exit at stop price."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(4)]
    # Entry at bar 0 close=18000. Bar 1: low=17985 hits stop, but close=18025 (profitable).
    # Without intra-bar check, engine would see close-based +25 and keep going or hit PT.
    # With intra-bar check, stop at 17990 fires.
    closes = [18000.0, 18025.0, 18030.0, 18035.0]
    highs = [18005.0, 18030.0, 18035.0, 18040.0]
    lows = [17995.0, 17985.0, 18025.0, 18030.0]
    signals = [1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows)
    # SL=10 points. Stop price for long = 18000-10 = 17990. Bar 1 low=17985 <= 17990.
    trades = run_backtest(df, stop_loss=10.0)

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 18000.0
    assert trade["exit_price"] == 17990.0, "Should exit at stop price, not bar close"
    assert trade["direction"] == 1


def test_intrabar_stop_loss_short():
    """Short entry, bar high hits stop but close is below entry -> exit at stop price."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(4)]
    # Short at bar 0 close=18000. Bar 1: high=18015 hits stop, but close=17980 (profitable).
    closes = [18000.0, 17980.0, 17975.0, 17970.0]
    highs = [18005.0, 18015.0, 17985.0, 17975.0]
    lows = [17995.0, 17975.0, 17970.0, 17965.0]
    signals = [-1, -1, -1, 0]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows)
    # SL=10 points. Stop price for short = 18000+10 = 18010. Bar 1 high=18015 >= 18010.
    trades = run_backtest(df, stop_loss=10.0)

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 18000.0
    assert trade["exit_price"] == 18010.0, "Should exit at stop price, not bar close"
    assert trade["direction"] == -1


def test_intrabar_stop_loss_long_gap_through_open():
    """Long stop should fill at bar open when market gaps through stop."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(4)]

    closes = [18000.0, 17998.0, 18005.0, 18010.0]
    opens = [17998.0, 17980.0, 18002.0, 18008.0]
    highs = [18002.0, 18001.0, 18007.0, 18012.0]
    lows = [17996.0, 17975.0, 17999.0, 18005.0]
    signals = [1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows, opens=opens)
    trades = run_backtest(df, stop_loss=10.0)  # stop=17990

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 18000.0
    assert trade["exit_price"] == 17980.0, "Gap-through stop should fill at open, not stop level"


def test_intrabar_stop_loss_short_gap_through_open():
    """Short stop should fill at bar open when market gaps through stop."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(4)]

    closes = [18000.0, 18002.0, 17998.0, 17995.0]
    opens = [18002.0, 18020.0, 18000.0, 17997.0]
    highs = [18005.0, 18025.0, 18003.0, 18000.0]
    lows = [17998.0, 18001.0, 17996.0, 17990.0]
    signals = [-1, -1, -1, 0]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows, opens=opens)
    trades = run_backtest(df, stop_loss=10.0)  # stop=18010

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 18000.0
    assert trade["exit_price"] == 18020.0, "Gap-through stop should fill at open, not stop level"


def test_intrabar_both_pt_sl_hit_worst_case():
    """Both PT and SL hit in same bar -> stop wins (worst case fill)."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(4)]
    # Long at 18000. Bar 1: wide range — low hits stop, high hits PT.
    closes = [18000.0, 18005.0, 18010.0, 18015.0]
    highs = [18005.0, 18025.0, 18015.0, 18020.0]  # PT=20, hit at 18020
    lows = [17995.0, 17985.0, 18005.0, 18010.0]   # SL=10, hit at 17990
    signals = [1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows)
    trades = run_backtest(df, profit_target=20.0, stop_loss=10.0)

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 18000.0
    # Worst case: stop wins. Exit at 18000-10 = 17990.
    assert trade["exit_price"] == 17990.0, "When both PT and SL hit, stop should win (worst case)"


def test_intrabar_pt_hit_without_sl():
    """PT hit by high, SL not hit -> exit at PT price."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(4)]
    # Long at 18000. Bar 1: high=18025 hits PT, low=17995 does NOT hit SL.
    closes = [18000.0, 18010.0, 18015.0, 18020.0]
    highs = [18005.0, 18025.0, 18020.0, 18025.0]
    lows = [17995.0, 17995.0, 18010.0, 18015.0]
    signals = [1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows)
    # PT=20 pts. PT price=18020. Bar 1 high=18025 >= 18020. SL=10, stop=17990, low=17995 > 17990.
    trades = run_backtest(df, profit_target=20.0, stop_loss=10.0)

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 18000.0
    assert trade["exit_price"] == 18020.0, "Should exit at PT price"


def test_no_high_low_fallback():
    """Without high/low columns, close-only behavior is preserved."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(6)]
    # Same scenario as test_stop_loss_exit — no high/low cols
    closes = [18000.0, 17995.0, 17990.0, 17985.0, 17980.0, 17975.0]
    signals = [1, 1, 1, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)  # No highs/lows
    assert "high" not in df.columns
    assert "low" not in df.columns

    trades = run_backtest(df, stop_loss=12.0)

    assert len(trades) >= 1
    trade = trades.row(0, named=True)
    assert trade["entry_price"] == 18000.0
    # Close-based: bar 3 unrealized = (17985-18000)*1 = -15 <= -12 → stop at close
    assert trade["exit_price"] == 17985.0, "Without high/low, should exit at bar close"


def test_entry_on_next_open_enters_at_open_price():
    """With entry_on_next_open=True, signal on T should fill at open of T+1."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(5)]
    opens = [18000.0, 18020.0, 18030.0, 18025.0, 18020.0]
    closes = [18010.0, 18025.0, 18035.0, 18020.0, 18015.0]
    highs = [18012.0, 18028.0, 18036.0, 18026.0, 18022.0]
    lows = [17998.0, 18018.0, 18028.0, 18018.0, 18012.0]
    signals = [1, 1, 0, 0, 0]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows, opens=opens)
    trades = run_backtest(df, entry_on_next_open=True)

    assert len(trades) == 1
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == timestamps[1]
    assert trade["entry_price"] == opens[1], "Entry must occur at next bar open"
    assert trade["exit_time"] == timestamps[2]
    assert trade["exit_price"] == closes[2]


def test_entry_on_next_open_carries_last_bar_signal_across_day_boundary():
    """Signal on day-end bar should execute at next day open when next bar exists."""
    day1_start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    day2_start = datetime(2025, 1, 2, 10, 0, 0, tzinfo=UTC)
    timestamps = [
        day1_start + timedelta(minutes=0),
        day1_start + timedelta(minutes=5),
        day1_start + timedelta(minutes=10),
        day2_start + timedelta(minutes=0),
        day2_start + timedelta(minutes=5),
        day2_start + timedelta(minutes=10),
    ]
    opens = [18000.0, 18005.0, 18010.0, 18100.0, 18105.0, 18110.0]
    closes = [18002.0, 18006.0, 18011.0, 18102.0, 18106.0, 18111.0]
    highs = [18003.0, 18007.0, 18012.0, 18103.0, 18107.0, 18112.0]
    lows = [17999.0, 18004.0, 18009.0, 18099.0, 18104.0, 18109.0]
    signals = [0, 0, 1, 1, 0, 0]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows, opens=opens)
    trades = run_backtest(df, entry_on_next_open=True)

    assert len(trades) == 1
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == timestamps[3]
    assert trade["entry_price"] == opens[3]
    assert trade["exit_time"] == timestamps[4]


def test_entry_on_next_open_pending_signal_expires_without_next_bar():
    """Pending signal on terminal bar should expire when there is no next bar."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(3)]
    opens = [18000.0, 18005.0, 18010.0]
    closes = [18002.0, 18006.0, 18011.0]
    highs = [18003.0, 18007.0, 18012.0]
    lows = [17999.0, 18004.0, 18009.0]
    signals = [0, 0, 1]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows, opens=opens)
    trades = run_backtest(df, entry_on_next_open=True)
    assert len(trades) == 0


def test_exit_bars_counts_fill_bar_in_next_open_mode():
    """In next-open mode, exit_bars counts bars while the trade is live."""
    start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * i) for i in range(6)]
    opens = [18000.0, 18010.0, 18020.0, 18030.0, 18040.0, 18050.0]
    closes = [18005.0, 18015.0, 18025.0, 18035.0, 18045.0, 18055.0]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    signals = [1, 1, 1, 1, 0, 0]

    df = create_test_df(timestamps, closes, signals, highs=highs, lows=lows, opens=opens)
    trades = run_backtest(df, entry_on_next_open=True, exit_bars=3)

    assert len(trades) == 1
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == timestamps[1]
    assert trade["exit_time"] == timestamps[3]


def test_default_mode_does_not_open_new_position_on_session_last_bar():
    """entry_on_next_open=False must not open on the terminal bar and carry overnight."""
    day1_start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    day2_start = datetime(2025, 1, 2, 10, 0, 0, tzinfo=UTC)
    timestamps = [
        day1_start + timedelta(minutes=0),
        day1_start + timedelta(minutes=5),
        day1_start + timedelta(minutes=10),  # session last bar
        day2_start + timedelta(minutes=0),
        day2_start + timedelta(minutes=5),
    ]
    closes = [18000.0, 18005.0, 18010.0, 18100.0, 18105.0]
    signals = [0, 0, 1, 1, 0]

    df = create_test_df(timestamps, closes, signals)
    trades = run_backtest(df, entry_on_next_open=False)

    assert len(trades) == 1
    trade = trades.row(0, named=True)
    assert trade["entry_time"] == timestamps[3]
    assert trade["entry_price"] == closes[3]
