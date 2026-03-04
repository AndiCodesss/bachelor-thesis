"""Bar-by-bar backtesting engine for NQ futures trading."""

import polars as pl
from src.framework.data.constants import TICK_SIZE, TICK_VALUE, TOTAL_COST_RT

TRADE_SCHEMA = {
    "entry_time": pl.Datetime("ns", "UTC"),
    "exit_time": pl.Datetime("ns", "UTC"),
    "entry_price": pl.Float64,
    "exit_price": pl.Float64,
    "direction": pl.Int8,
    "size": pl.Int32,
}


def _points_to_dollars(points: float) -> float:
    """Convert points to dollars using tick-quantized math."""
    ticks = int(round(points / TICK_SIZE))
    return float(ticks * TICK_VALUE)


def _trade_pnl_dollars(entry_price: float, exit_price: float, direction: int) -> float:
    points = (exit_price - entry_price) * direction
    return _points_to_dollars(points)


def run_backtest(
    df: pl.DataFrame,
    signal_col: str = "signal",
    max_daily_loss: float = 1000.0,
    exit_bars: int = None,
    profit_target: float = None,
    stop_loss: float = None,
    profit_target_return: float = None,
    stop_loss_return: float = None,
    entry_on_next_open: bool = False,
) -> pl.DataFrame:
    """Execute backtest on bar data with signal column.

    Args:
        df: DataFrame with columns:
            - ts_event (Datetime): bar timestamp
            - close (Float64): bar close price
            - {signal_col} (Int8): 1=long, -1=short, 0=flat
        signal_col: Name of signal column
        max_daily_loss: Max loss per day in dollars, stop trading if exceeded
        exit_bars: If set, close position after N bars regardless of signal.
            exit_bars=3 means the position is held for 3 full bars after
            entry and exits on the close of bar T+3 (entry at bar T).
        profit_target: If set, close when unrealized PnL (points) exceeds this.
            If DataFrame has high/low columns, checks intra-bar price action
            first. Falls back to close-based check otherwise.
        stop_loss: If set, close when unrealized loss (points) exceeds this.
            If DataFrame has high/low columns, checks intra-bar price action
            first. If both PT and SL hit in same bar, assumes stop hit first
            (worst case). Falls back to close-based check otherwise.
        profit_target_return: If set, profit target in return space (e.g. 0.003).
            Converted to points per-trade using entry price. Overrides profit_target.
        stop_loss_return: If set, stop loss in return space (e.g. 0.003).
            Converted to points per-trade using entry price. Overrides stop_loss.
        entry_on_next_open: If True, signal on bar T enters at open of bar T+1.
            Requires 'open' column. More realistic for live execution.

    Returns:
        DataFrame with trades (entry/exit times, prices, direction, size)
    """
    assert len(df) > 0, "Empty DataFrame"
    assert signal_col in df.columns, f"Missing signal column '{signal_col}'"
    assert "ts_event" in df.columns, "Missing ts_event column"
    assert "close" in df.columns, "Missing close column"
    assert df[signal_col].is_in([-1, 0, 1]).all(), "Signal must be in {-1, 0, 1}"
    if entry_on_next_open:
        assert "open" in df.columns, "entry_on_next_open requires 'open' column"

    # Sort by time to ensure correct order
    df = df.sort("ts_event")

    signals = df[signal_col].to_list()
    closes = df["close"].to_list()
    timestamps = df["ts_event"].to_list()

    # Intra-bar PT/SL: use high/low if available
    has_hilo = "high" in df.columns and "low" in df.columns
    has_open = "open" in df.columns
    if has_hilo:
        highs = df["high"].to_list()
        lows = df["low"].to_list()
    if has_open:
        opens = df["open"].to_list()

    trades = []
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_time = None
    bars_held = 0
    daily_pnl = 0.0
    current_date = None
    daily_stopped = False
    skip_reentry = False  # Mandatory 1-bar gap after force exit
    # Per-trade PT/SL in points (computed from return-space thresholds at entry)
    active_profit_target = profit_target
    active_stop_loss = stop_loss
    # Pending entry for entry_on_next_open mode
    pending_direction = 0  # signal direction waiting for next bar's open

    for i in range(len(signals)):
        bar_date = timestamps[i].date()

        # Reset daily tracking on new day
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl = 0.0
            daily_stopped = False
            pending_direction = 0  # cancel pending entry across day boundary

        # Check if this is the last bar of the day
        is_last_bar = (i == len(signals) - 1) or (timestamps[i + 1].date() != bar_date)

        # Execute ONLY the previous bar's pending signal, then expire it.
        signal_to_execute = pending_direction
        pending_direction = 0

        # Fill pending entry at this bar's open (entry_on_next_open mode)
        if signal_to_execute != 0 and position == 0 and not daily_stopped and not is_last_bar:
            position = signal_to_execute
            entry_price = opens[i]
            entry_time = timestamps[i]
            bars_held = 0
            if profit_target_return is not None:
                active_profit_target = entry_price * profit_target_return
            if stop_loss_return is not None:
                active_stop_loss = entry_price * stop_loss_return

        # If daily stopped, force signal to 0
        sig = signals[i] if not daily_stopped else 0

        # Check exit conditions for open positions
        force_exit = False
        intrabar_exit_price = None
        daily_limit_exit = False
        if position != 0:
            bars_held += 1

            # Daily loss guard: force close if current open trade breaches limit.
            remaining_loss_dollars = max_daily_loss + daily_pnl
            if remaining_loss_dollars <= 0:
                force_exit = True
                daily_limit_exit = True
            elif has_hilo:
                # Convert remaining loss budget to points from entry.
                loss_points_limit = (remaining_loss_dollars / TICK_VALUE) * TICK_SIZE
                if position == 1:
                    limit_price = entry_price - loss_points_limit
                    if lows[i] <= limit_price:
                        force_exit = True
                        daily_limit_exit = True
                        if has_open and opens[i] <= limit_price:
                            intrabar_exit_price = opens[i]
                        else:
                            intrabar_exit_price = limit_price
                else:
                    limit_price = entry_price + loss_points_limit
                    if highs[i] >= limit_price:
                        force_exit = True
                        daily_limit_exit = True
                        if has_open and opens[i] >= limit_price:
                            intrabar_exit_price = opens[i]
                        else:
                            intrabar_exit_price = limit_price

            # Intra-bar PT/SL checking (higher priority than close-based)
            if (
                not force_exit
                and has_hilo
                and (active_profit_target is not None or active_stop_loss is not None)
            ):
                if position == 1:  # Long
                    stop_hit = (active_stop_loss is not None
                                and lows[i] <= entry_price - active_stop_loss)
                    pt_hit = (active_profit_target is not None
                              and highs[i] >= entry_price + active_profit_target)
                else:  # Short
                    stop_hit = (active_stop_loss is not None
                                and highs[i] >= entry_price + active_stop_loss)
                    pt_hit = (active_profit_target is not None
                              and lows[i] <= entry_price - active_profit_target)

                if stop_hit and pt_hit:
                    # Both hit in same bar — worst case: stop wins
                    force_exit = True
                    if position == 1:
                        stop_price = entry_price - active_stop_loss
                        # Gap-through-stop: if bar opens below stop, fill at open (worse).
                        if has_open and opens[i] <= stop_price:
                            intrabar_exit_price = opens[i]
                        else:
                            intrabar_exit_price = stop_price
                    else:
                        stop_price = entry_price + active_stop_loss
                        # Gap-through-stop: if bar opens above stop, fill at open (worse).
                        if has_open and opens[i] >= stop_price:
                            intrabar_exit_price = opens[i]
                        else:
                            intrabar_exit_price = stop_price
                elif stop_hit:
                    force_exit = True
                    if position == 1:
                        stop_price = entry_price - active_stop_loss
                        # Gap-through-stop: if bar opens below stop, fill at open (worse).
                        if has_open and opens[i] <= stop_price:
                            intrabar_exit_price = opens[i]
                        else:
                            intrabar_exit_price = stop_price
                    else:
                        stop_price = entry_price + active_stop_loss
                        # Gap-through-stop: if bar opens above stop, fill at open (worse).
                        if has_open and opens[i] >= stop_price:
                            intrabar_exit_price = opens[i]
                        else:
                            intrabar_exit_price = stop_price
                elif pt_hit:
                    force_exit = True
                    if position == 1:
                        intrabar_exit_price = entry_price + active_profit_target
                    else:
                        intrabar_exit_price = entry_price - active_profit_target

            # Close-based checks (fallback when no high/low or no intra-bar hit)
            if not force_exit:
                unrealized_pnl_points = (closes[i] - entry_price) * position

                # Time-based exit
                if exit_bars is not None and bars_held >= exit_bars:
                    force_exit = True

                # Profit target exit (close-based)
                if active_profit_target is not None and unrealized_pnl_points >= active_profit_target:
                    force_exit = True

                # Stop loss exit (close-based)
                if active_stop_loss is not None and unrealized_pnl_points <= -active_stop_loss:
                    force_exit = True

                # Daily loss limit check (close fallback when no hi/low available).
                if not has_hilo:
                    unrealized_pnl_dollars = _points_to_dollars(unrealized_pnl_points)
                    if daily_pnl + unrealized_pnl_dollars <= -max_daily_loss:
                        force_exit = True
                        daily_limit_exit = True

        # Force exit at end of day or if exit condition triggered
        if (is_last_bar or force_exit) and position != 0:
            exit_price = intrabar_exit_price if intrabar_exit_price is not None else closes[i]
            exit_time = timestamps[i]
            pnl_dollars = _trade_pnl_dollars(entry_price, exit_price, position)

            trades.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": position,
                "size": 1,
            })

            daily_pnl += pnl_dollars - TOTAL_COST_RT
            position = 0
            entry_price = 0.0
            entry_time = None
            bars_held = 0

            # Check daily loss limit after close (using net daily PnL)
            if daily_limit_exit or daily_pnl <= -max_daily_loss:
                daily_stopped = True

            # After force exit (not EOD), skip re-entry on this bar
            if force_exit and not is_last_bar:
                skip_reentry = True

            continue

        # Handle signal transitions
        if sig != position:
            # Close existing position if any
            if position != 0:
                exit_price = closes[i]
                exit_time = timestamps[i]
                pnl_dollars = _trade_pnl_dollars(entry_price, exit_price, position)

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "direction": position,
                    "size": 1,
                })

                daily_pnl += pnl_dollars - TOTAL_COST_RT
                position = 0
                entry_price = 0.0
                entry_time = None
                bars_held = 0

                # Check daily loss limit after close (using net daily PnL)
                if daily_pnl <= -max_daily_loss:
                    daily_stopped = True
                    continue

            # Open new position if signal is not flat, not stopped, not last bar,
            # and not in mandatory re-entry cooldown after force exit
            if sig != 0 and not daily_stopped and not is_last_bar and not skip_reentry:
                if entry_on_next_open:
                    # Defer entry to next bar's open
                    pending_direction = sig
                else:
                    position = sig
                    entry_price = closes[i]
                    entry_time = timestamps[i]
                    bars_held = 0
                    # Compute per-trade PT/SL from return-space thresholds
                    if profit_target_return is not None:
                        active_profit_target = entry_price * profit_target_return
                    if stop_loss_return is not None:
                        active_stop_loss = entry_price * stop_loss_return

        # Clear skip_reentry at end of bar (gap enforced for this bar)
        skip_reentry = False

    # Return empty DataFrame with correct schema if no trades
    if not trades:
        return pl.DataFrame(schema=TRADE_SCHEMA)

    return pl.DataFrame(trades)
