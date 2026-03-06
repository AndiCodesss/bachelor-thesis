"""Quantitative alpha decay modeling via exponential fit on rolling Sharpe."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl

from src.framework.backtest.metrics import compute_trade_pnl_frame
from src.framework.data.constants import (
    ALPHA_DECAY_MIN_WINDOWS,
    ALPHA_DECAY_STABLE_HALFLIFE,
    ALPHA_DECAY_STEP_DAYS,
    ALPHA_DECAY_WINDOW_DAYS,
)


def _add_net_pnl(trades: pl.DataFrame) -> pl.DataFrame:
    """Compute net PnL in dollars from raw trade columns (matches metrics.py)."""
    cost_col = "adaptive_cost_rt" if "adaptive_cost_rt" in trades.columns else None
    return compute_trade_pnl_frame(
        trades,
        cost_override_col=cost_col,
    ).rename({"net_pnl": "pnl_dollars"})


def compute_rolling_sharpe(
    trades: pl.DataFrame,
    window_days: int = ALPHA_DECAY_WINDOW_DAYS,
    step_days: int = ALPHA_DECAY_STEP_DAYS,
) -> list[tuple[float, float]]:
    """Rolling Sharpe from trades grouped by exit date.

    Returns list of (day_ordinal, sharpe) where day_ordinal is the window
    midpoint relative to the first trade date.
    """
    if len(trades) == 0:
        return []

    # Compute net PnL if not already present
    if "pnl_dollars" not in trades.columns:
        trades = _add_net_pnl(trades)

    # Aggregate trade PnL to daily level
    daily = (
        trades.with_columns(pl.col("exit_time").dt.date().alias("_date"))
        .group_by("_date")
        .agg(pl.col("pnl_dollars").sum().alias("daily_pnl"))
        .sort("_date")
    )

    if len(daily) < 2:
        return []

    dates = daily["_date"].to_list()
    pnls = daily["daily_pnl"].to_list()
    pnl_by_date = dict(zip(dates, pnls))

    first_date = dates[0]
    last_date = dates[-1]
    total_days = (last_date - first_date).days + 1

    results: list[tuple[float, float]] = []
    start = 0
    while start + window_days <= total_days:
        win_start = first_date + timedelta(days=start)

        # Collect only trading-day PnLs for the window (skip non-trade days)
        win_end = win_start + timedelta(days=window_days)
        window_pnls = np.array(
            [v for d, v in pnl_by_date.items() if win_start <= d < win_end],
            dtype=np.float64,
        )
        if len(window_pnls) < 2:
            start += step_days
            continue

        std = np.std(window_pnls, ddof=1)
        if std > 0:
            sharpe = (np.mean(window_pnls) / std) * math.sqrt(252)
        else:
            sharpe = 0.0

        midpoint = start + window_days / 2.0
        results.append((midpoint, float(sharpe)))
        start += step_days

    return results


def fit_alpha_decay(
    trades: pl.DataFrame,
    window_days: int = ALPHA_DECAY_WINDOW_DAYS,
    step_days: int = ALPHA_DECAY_STEP_DAYS,
    min_windows: int = ALPHA_DECAY_MIN_WINDOWS,
) -> dict[str, Any]:
    """Fit exponential decay Sharpe(t) = a * exp(-lambda * t) via log-linear OLS.

    Returns dict with decay diagnostics and verdict.
    """
    rolling = compute_rolling_sharpe(trades, window_days, step_days)

    if len(rolling) < min_windows:
        return {"available": False, "verdict": "INSUFFICIENT_DATA"}

    # Filter to positive Sharpe values for log-linear fit
    positive = [(t, s) for t, s in rolling if s > 0]

    if len(positive) < min_windows:
        # Check if all are non-positive
        if all(s <= 0 for _, s in rolling):
            return {
                "available": True,
                "half_life_days": 0.0,
                "decay_rate": float("inf"),
                "initial_sharpe": 0.0,
                "r_squared": 0.0,
                "rolling_sharpes": rolling,
                "verdict": "DEAD",
            }
        return {"available": False, "verdict": "INSUFFICIENT_DATA"}

    t_arr = np.array([t for t, _ in positive], dtype=np.float64)
    log_sharpe = np.log(np.array([s for _, s in positive], dtype=np.float64))

    # log-linear OLS: log(Sharpe) = log(a) - lambda * t
    # np.polyfit(t, log_sharpe, 1) returns [slope, intercept]
    slope, intercept = np.polyfit(t_arr, log_sharpe, 1)
    lam = -slope  # decay rate
    a = math.exp(intercept)  # initial Sharpe estimate

    # R-squared of the log-linear fit
    predicted = slope * t_arr + intercept
    ss_res = np.sum((log_sharpe - predicted) ** 2)
    ss_tot = np.sum((log_sharpe - np.mean(log_sharpe)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Half-life in days
    if lam > 0:
        half_life = math.log(2) / lam
    else:
        half_life = float("inf")

    # Verdict logic
    if lam <= 0 or half_life > ALPHA_DECAY_STABLE_HALFLIFE:
        verdict = "STABLE"
    elif r_squared > 0.3:
        verdict = "DECAYING"
    else:
        verdict = "NOISY"

    return {
        "available": True,
        "half_life_days": float(half_life),
        "decay_rate": float(lam),
        "initial_sharpe": float(a),
        "r_squared": float(r_squared),
        "rolling_sharpes": rolling,
        "verdict": verdict,
    }
