"""Tests for alpha decay modeling (exponential fit on rolling Sharpe)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from src.framework.validation.alpha_decay import (
    compute_rolling_sharpe,
    fit_alpha_decay,
)


def _make_trades(
    daily_pnl_dollars: list[float],
    start_date: datetime | None = None,
) -> pl.DataFrame:
    """Build a trades DataFrame matching TRADE_SCHEMA with one trade per calendar day.

    Uses consecutive calendar days (no weekend gaps) so that
    compute_rolling_sharpe's calendar-day windows align perfectly with the
    synthetic PnL series.

    PnL formula: (exit - entry) * direction / 0.25 * 5.0 * size - size * 14.50
    With direction=1, size=1: (exit - entry) * 20 - 14.50 = target_pnl
    """
    if start_date is None:
        start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    rows = []
    for i, pnl in enumerate(daily_pnl_dollars):
        day = start_date + timedelta(days=i)
        entry_time = day.replace(hour=10, minute=0, second=0, microsecond=0)
        exit_time = day.replace(hour=11, minute=0, second=0, microsecond=0)
        entry_price = 21000.0
        price_diff = (pnl + 14.50) / 20.0
        exit_price = entry_price + price_diff

        rows.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": 1,
            "size": 1,
        })

    schema = {
        "entry_time": pl.Datetime("us", "UTC"),
        "exit_time": pl.Datetime("us", "UTC"),
        "entry_price": pl.Float64,
        "exit_price": pl.Float64,
        "direction": pl.Int8,
        "size": pl.Int32,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def test_known_exponential_decay():
    """Synthetic trades: signal decays exponentially against constant noise.

    Sharpe(t) ~ A*exp(-lambda*t)/sigma, so Sharpe half-life matches PnL half-life.
    With lambda=0.04, true half-life = ln(2)/0.04 ~ 17.3 days.
    """
    n_days = 120
    decay_rate = 0.04
    true_half_life = math.log(2) / decay_rate  # ~17.3 days

    rng = np.random.default_rng(42)
    daily_pnls = [
        800.0 * math.exp(-decay_rate * d) + rng.normal(0, 100)
        for d in range(n_days)
    ]

    trades = _make_trades(daily_pnls)
    result = fit_alpha_decay(trades, window_days=20, step_days=5, min_windows=5)

    assert result["available"] is True
    assert result["verdict"] == "DECAYING"
    fitted_half_life = result["half_life_days"]
    # Within 30% of true half_life (noise adds estimation error)
    assert abs(fitted_half_life - true_half_life) / true_half_life < 0.30, (
        f"fitted={fitted_half_life:.1f}, true={true_half_life:.1f}"
    )
    assert result["decay_rate"] > 0
    assert result["initial_sharpe"] > 0


def test_stable_signal():
    """Flat daily PnL (constant) should yield STABLE verdict."""
    n_days = 80
    # Constant positive PnL with tiny noise (so std > 0 within windows)
    rng = np.random.default_rng(42)
    daily_pnls = [200.0 + rng.normal(0, 5) for _ in range(n_days)]

    trades = _make_trades(daily_pnls)
    result = fit_alpha_decay(trades, window_days=15, step_days=3, min_windows=5)

    assert result["available"] is True
    assert result["verdict"] == "STABLE"
    # Half-life should be very large or infinite
    assert result["half_life_days"] > 120


def test_insufficient_data():
    """Too few trading days to form min_windows rolling windows."""
    daily_pnls = [100.0] * 10  # Only 10 days, window=20 -> 0 windows
    trades = _make_trades(daily_pnls)
    result = fit_alpha_decay(trades, window_days=20, step_days=5, min_windows=5)

    assert result["available"] is False
    assert result["verdict"] == "INSUFFICIENT_DATA"


def test_all_negative_sharpe():
    """All windows have negative Sharpe -> DEAD verdict."""
    n_days = 80
    # Consistently losing money
    rng = np.random.default_rng(42)
    daily_pnls = [-300.0 + rng.normal(0, 10) for _ in range(n_days)]

    trades = _make_trades(daily_pnls)
    result = fit_alpha_decay(trades, window_days=15, step_days=3, min_windows=5)

    assert result["available"] is True
    assert result["verdict"] == "DEAD"


def test_perfect_decay():
    """Strong exponential decay with moderate noise -> high R-squared.

    High initial amplitude (2000) with constant noise (80) gives clean
    Sharpe decay as signal crosses through the noise floor over 150 days.
    R-squared should exceed 0.80 (noise in rolling Sharpe estimation
    prevents truly perfect fit even with strong signal).
    """
    n_days = 150
    decay_rate = 0.04
    rng = np.random.default_rng(11)
    daily_pnls = [
        2000.0 * math.exp(-decay_rate * d) + rng.normal(0, 80)
        for d in range(n_days)
    ]

    trades = _make_trades(daily_pnls)
    result = fit_alpha_decay(trades, window_days=15, step_days=3, min_windows=5)

    assert result["available"] is True
    assert result["r_squared"] > 0.80, f"r_squared={result['r_squared']:.3f}"
    assert result["verdict"] == "DECAYING"
    assert result["decay_rate"] > 0


def test_rolling_sharpe_empty_trades():
    """Empty trades DataFrame returns empty list."""
    schema = {
        "entry_time": pl.Datetime("us", "UTC"),
        "exit_time": pl.Datetime("us", "UTC"),
        "entry_price": pl.Float64,
        "exit_price": pl.Float64,
        "direction": pl.Int8,
        "size": pl.Int32,
    }
    empty = pl.DataFrame(schema=schema)
    assert compute_rolling_sharpe(empty) == []


def test_rolling_sharpe_output_format():
    """Verify rolling Sharpe returns list of (float, float) tuples."""
    daily_pnls = [100.0 + i * 5 for i in range(60)]
    trades = _make_trades(daily_pnls)
    rolling = compute_rolling_sharpe(trades, window_days=15, step_days=5)

    assert len(rolling) > 0
    for t, s in rolling:
        assert isinstance(t, float)
        assert isinstance(s, float)


def test_fit_alpha_decay_returns_rolling_sharpes():
    """Result dict includes rolling_sharpes list when available."""
    n_days = 80
    rng = np.random.default_rng(42)
    daily_pnls = [300.0 * math.exp(-0.02 * d) + rng.normal(0, 10) for d in range(n_days)]

    trades = _make_trades(daily_pnls)
    result = fit_alpha_decay(trades, window_days=15, step_days=3, min_windows=5)

    assert result["available"] is True
    assert "rolling_sharpes" in result
    assert isinstance(result["rolling_sharpes"], list)
    assert len(result["rolling_sharpes"]) >= 5
