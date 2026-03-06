"""Factor attribution: decompose strategy returns into market factor exposures."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from src.framework.backtest.metrics import compute_trade_pnl_frame
from src.framework.data.constants import (
    FACTOR_MIN_DAYS,
    FACTOR_R2_THRESHOLD,
    FACTOR_SIGNIFICANCE_THRESHOLD,
)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _holm_bonferroni_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return p_values

    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = float(p_values[idx]) * (n - rank)
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def _significance_p_threshold() -> float:
    """Two-sided p-value implied by the configured t-stat significance bar."""
    return 2.0 * (1.0 - _normal_cdf(FACTOR_SIGNIFICANCE_THRESHOLD))


def _factor_verdict(
    *,
    r_squared: float,
    alpha_pvalue_adjusted: float,
    factor_pvalues_adjusted: np.ndarray,
) -> str:
    """Classify attribution outcome from corrected significance and explained variance."""
    alpha_significant = float(alpha_pvalue_adjusted) <= _significance_p_threshold()
    any_factor_significant = np.any(factor_pvalues_adjusted <= _significance_p_threshold())

    if r_squared < FACTOR_R2_THRESHOLD and alpha_significant:
        return "PURE_ALPHA"
    if r_squared >= FACTOR_R2_THRESHOLD and any_factor_significant:
        return "FACTOR_EXPOSED"
    return "INCONCLUSIVE"


def compute_factor_returns(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute daily factor returns from OHLCV bars.

    Factors: market return, volatility change, momentum (prior-day return).
    """
    assert len(bars) > 0, "empty bars DataFrame"
    for col in ("ts_event", "open", "high", "low", "close", "volume"):
        assert col in bars.columns, f"missing column: {col}"

    # Compute bar-level returns for volatility calculation
    daily = (
        bars.sort("ts_event")
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .group_by("date")
        .agg(
            pl.col("close").first().alias("first_close"),
            pl.col("close").last().alias("last_close"),
            # Std of intraday bar returns for realized volatility
            (pl.col("close").pct_change().std()).alias("realized_vol"),
        )
        .sort("date")
    )

    daily = daily.with_columns(
        # Market return: true close-to-close (includes overnight gap)
        ((pl.col("last_close") / pl.col("last_close").shift(1)) - 1.0).alias("market_return"),
    ).with_columns(
        # Volatility change: today's realized vol minus yesterday's
        (pl.col("realized_vol") - pl.col("realized_vol").shift(1)).alias("volatility_change"),
        # Momentum: prior-day close-to-close return
        pl.col("market_return").shift(1).alias("momentum"),
    )

    return daily.select("date", "market_return", "volatility_change", "momentum")


def factor_attribution(
    trades: pl.DataFrame,
    bars: pl.DataFrame,
    min_days: int = FACTOR_MIN_DAYS,
) -> dict:
    """Decompose strategy daily PnL into factor exposures via OLS.

    Returns attribution dict with alpha, betas, t-stats, r-squared, and verdict.
    """
    if len(trades) == 0:
        return {"available": False, "verdict": "INSUFFICIENT_DATA"}

    # Compute strategy daily PnL from trades.
    # Use exit date (PnL realization date). Fallback to entry_time for legacy
    # callers that do not provide exit_time.
    if "exit_time" in trades.columns:
        date_col = "exit_time"
    else:
        assert "entry_time" in trades.columns, "trades missing exit_time/entry_time"
        date_col = "entry_time"

    # Compute net PnL per trade if not already present.
    cost_col = "adaptive_cost_rt" if "adaptive_cost_rt" in trades.columns else None
    if "pnl_dollars" in trades.columns:
        pnl_col = "pnl_dollars"
    else:
        trades = compute_trade_pnl_frame(
            trades,
            cost_override_col=cost_col,
        ).with_columns(
            pl.col("net_pnl").alias("_net_pnl")
        )
        pnl_col = "_net_pnl"

    # Group by realized date to get daily strategy PnL
    strategy_daily = (
        trades.with_columns(pl.col(date_col).dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col(pnl_col).sum().alias("strategy_pnl"))
        .sort("date")
    )

    # Compute factor returns from bars
    factors = compute_factor_returns(bars)

    # Join on date
    joined = strategy_daily.join(factors, on="date", how="inner").drop_nulls()

    n_days = len(joined)
    if n_days < min_days:
        return {"available": False, "verdict": "INSUFFICIENT_DATA", "n_days": n_days}

    # Extract arrays for OLS
    y = joined["strategy_pnl"].to_numpy().astype(np.float64)
    x_mkt = joined["market_return"].to_numpy().astype(np.float64)
    x_vol = joined["volatility_change"].to_numpy().astype(np.float64)
    x_mom = joined["momentum"].to_numpy().astype(np.float64)

    # Design matrix: [intercept, market, volatility, momentum]
    X = np.column_stack([np.ones(n_days), x_mkt, x_vol, x_mom])

    # OLS via least squares
    coeffs, residuals_arr, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    alpha_daily = float(coeffs[0])
    betas = coeffs[1:]

    # Fitted values and residuals
    y_hat = X @ coeffs
    resid = y - y_hat

    # R-squared
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors: se = sqrt(diag(sigma^2 * (X'X)^-1))
    dof = max(n_days - X.shape[1], 1)
    sigma2 = ss_res / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.ones(X.shape[1])

    # t-statistics and p-values
    t_stats = np.where(se > 1e-12, coeffs / se, 0.0)
    p_values = np.array([2.0 * (1.0 - _normal_cdf(abs(float(t)))) for t in t_stats])
    p_values_adjusted = _holm_bonferroni_adjust(p_values)

    # Residual Sharpe (annualized)
    resid_std = float(np.std(resid, ddof=1)) if n_days > 1 else 0.0
    resid_mean = float(np.mean(resid))
    residual_sharpe = (resid_mean / resid_std * math.sqrt(252)) if resid_std > 1e-12 else 0.0

    # Annualized alpha
    alpha_annualized = alpha_daily * 252.0

    verdict = _factor_verdict(
        r_squared=r_squared,
        alpha_pvalue_adjusted=float(p_values_adjusted[0]),
        factor_pvalues_adjusted=p_values_adjusted[1:],
    )

    factor_names = ["market", "volatility", "momentum"]

    return {
        "available": True,
        "alpha_daily": alpha_daily,
        "alpha_annualized": alpha_annualized,
        "alpha_t_stat": float(t_stats[0]),
        "alpha_pvalue": float(p_values[0]),
        "alpha_pvalue_adjusted": float(p_values_adjusted[0]),
        "factor_betas": {name: float(betas[i]) for i, name in enumerate(factor_names)},
        "factor_t_stats": {name: float(t_stats[i + 1]) for i, name in enumerate(factor_names)},
        "factor_pvalues": {name: float(p_values[i + 1]) for i, name in enumerate(factor_names)},
        "factor_pvalues_adjusted": {
            name: float(p_values_adjusted[i + 1]) for i, name in enumerate(factor_names)
        },
        "r_squared": r_squared,
        "residual_sharpe": residual_sharpe,
        "n_days": n_days,
        "verdict": verdict,
    }
