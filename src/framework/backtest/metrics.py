"""Performance metrics for trading strategy evaluation."""

import polars as pl
import numpy as np

from src.framework.data.constants import TICK_SIZE, TICK_VALUE, TOTAL_COST_RT


def compute_metrics(trades: pl.DataFrame, bar_minutes: float = 5.0, cost_override_col: str | None = None, initial_capital: float = 100_000.0) -> dict:
    """Compute financial performance metrics from trades.

    Args:
        trades: DataFrame with columns:
            - entry_time (Datetime)
            - exit_time (Datetime)
            - entry_price (Float64)
            - exit_price (Float64)
            - direction (Int8): 1 for long, -1 for short
            - size (Int32): number of contracts

    Returns:
        dict with performance metrics (all financial metrics in dollars)
    """
    # Edge case: empty trades
    if len(trades) == 0:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0.0,
            "gross_pnl": 0.0,
            "total_costs": 0.0,
            "net_pnl": 0.0,
            "avg_trade_pnl": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_holding_time_min": 0.0,
            "avg_bars_held": 0.0,
        }

    # Determine cost expression: per-trade adaptive or flat
    use_adaptive = (
        cost_override_col is not None
        and cost_override_col in trades.columns
    )
    if use_adaptive:
        cost_expr = (pl.col(cost_override_col) * pl.col("size")).alias("costs")
    else:
        cost_expr = (pl.col("size") * TOTAL_COST_RT).alias("costs")

    # Compute PnL for each trade
    df = trades.with_columns([
        # Gross PnL in points (per contract)
        ((pl.col("exit_price") - pl.col("entry_price")) * pl.col("direction")).alias("gross_pnl_points"),
        # Transaction costs
        cost_expr,
        # Holding time
        ((pl.col("exit_time") - pl.col("entry_time")).dt.total_seconds() / 60.0).alias("holding_time_min"),
    ]).with_columns([
        # Gross PnL in dollars (multiply by size for multiple contracts)
        (pl.col("gross_pnl_points") / TICK_SIZE * TICK_VALUE * pl.col("size")).alias("gross_pnl"),
    ]).with_columns([
        # Net PnL (after costs)
        (pl.col("gross_pnl") - pl.col("costs")).alias("net_pnl"),
    ])

    # Basic counts
    trade_count = len(df)
    win_count = df.filter(pl.col("net_pnl") > 0).height
    loss_count = df.filter(pl.col("net_pnl") < 0).height
    win_rate = win_count / trade_count if trade_count > 0 else 0.0

    # PnL stats
    gross_pnl = df["gross_pnl"].sum()
    total_costs = df["costs"].sum()
    net_pnl = df["net_pnl"].sum()
    avg_trade_pnl = net_pnl / trade_count if trade_count > 0 else 0.0

    # Win/loss stats
    winners = df.filter(pl.col("net_pnl") > 0)
    losers = df.filter(pl.col("net_pnl") < 0)
    max_win = winners["net_pnl"].max() if len(winners) > 0 else 0.0
    max_loss = losers["net_pnl"].min() if len(losers) > 0 else 0.0

    # Profit factor: gross wins / abs(gross losses)
    gross_wins = df.filter(pl.col("net_pnl") > 0)["net_pnl"].sum()
    gross_losses = df.filter(pl.col("net_pnl") < 0)["net_pnl"].sum()

    if gross_losses < 0:
        profit_factor = gross_wins / abs(gross_losses)
    elif gross_wins > 0:
        # All winners, no losses -> mathematically unbounded.
        profit_factor = np.inf
    else:
        # No wins, no losses (all breakeven) → 0
        profit_factor = 0.0

    # Sharpe ratio: annualized from daily PnL aggregation
    # Aggregate trade PnLs to daily level, then use sqrt(252)
    trade_pnls = df["net_pnl"].to_numpy()
    sharpe_ratio = np.nan
    if trade_count > 1 and "exit_time" in trades.columns:
        # Group trades by exit date to get daily PnLs
        daily_pnl_df = df.with_columns(
            pl.col("exit_time").dt.date().alias("_date")
        ).group_by("_date").agg(pl.col("net_pnl").sum()).sort("_date")

        if len(daily_pnl_df) > 1:
            # Zero-fill non-trading weekdays so Sharpe is not inflated
            min_date = daily_pnl_df["_date"].min()
            max_date = daily_pnl_df["_date"].max()
            all_bdays = pl.DataFrame({
                "_date": pl.date_range(min_date, max_date, "1d", eager=True)
            }).filter(pl.col("_date").dt.weekday() <= 5)
            # Include actual trading dates (handles weekend trades in tests)
            all_dates = pl.concat([
                all_bdays.select("_date"),
                daily_pnl_df.select("_date"),
            ]).unique().sort("_date")
            daily_pnl_df = all_dates.join(daily_pnl_df, on="_date", how="left").fill_null(0)
            trading_day_pnls = daily_pnl_df["net_pnl"].to_numpy().astype(np.float64)

            if len(trading_day_pnls) > 1:
                daily_std = np.std(trading_day_pnls, ddof=1)
                if daily_std > 0:
                    sharpe_ratio = (np.mean(trading_day_pnls) / daily_std) * np.sqrt(252)

    # Drawdown calculation (relative to initial capital)
    equity_curve = np.cumsum(trade_pnls)
    equity_with_start = np.concatenate([[0.0], equity_curve])
    running_max = np.maximum.accumulate(initial_capital + equity_with_start)
    drawdowns = running_max - (initial_capital + equity_with_start)
    max_drawdown = np.max(drawdowns)

    # Max drawdown percentage: normalize by the local peak that preceded
    # the max-drawdown trough, not by the global peak reached later.
    max_dd_idx = int(np.argmax(drawdowns))
    peak_equity_at_max_dd = float(running_max[max_dd_idx])
    max_drawdown_pct = (
        (max_drawdown / peak_equity_at_max_dd) * 100.0
        if peak_equity_at_max_dd > 0
        else 0.0
    )

    # Holding time stats
    avg_holding_time_min = df["holding_time_min"].mean()
    avg_bars_held = avg_holding_time_min / bar_minutes

    return {
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "gross_pnl": float(gross_pnl),
        "total_costs": float(total_costs),
        "net_pnl": float(net_pnl),
        "avg_trade_pnl": avg_trade_pnl,
        "max_win": float(max_win),
        "max_loss": float(max_loss),
        "profit_factor": float(profit_factor),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown_pct),
        "avg_holding_time_min": float(avg_holding_time_min),
        "avg_bars_held": float(avg_bars_held),
    }
