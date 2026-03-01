"""Tests for factor attribution: decompose strategy returns into factor exposures."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.framework.validation.factor_attribution import (
    compute_factor_returns,
    factor_attribution,
)


def _make_bars(n_days: int, seed: int = 42) -> pl.DataFrame:
    """Build synthetic OHLCV bars: 6 bars per day (5-min spacing), starting 2024-01-02."""
    rng = np.random.default_rng(seed)
    rows = []
    base_price = 20000.0
    start_date = date(2024, 1, 2)
    for d in range(n_days):
        day = start_date + timedelta(days=d)
        price = base_price + rng.normal(0, 50)
        base_ts = datetime(day.year, day.month, day.day, 9, 30, 0, tzinfo=timezone.utc)
        for b in range(6):
            ts = base_ts + timedelta(minutes=b * 5)
            o = price + rng.normal(0, 5)
            c = price + rng.normal(0, 5)
            h = max(o, c) + abs(rng.normal(0, 2))
            l = min(o, c) - abs(rng.normal(0, 2))
            v = int(abs(rng.normal(1000, 200)))
            rows.append({"ts_event": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
            price = c
    return pl.DataFrame(rows)


def _make_trades(daily_pnls: dict[date, float]) -> pl.DataFrame:
    """Build trades DataFrame from daily PnL targets.

    Creates one synthetic trade per day with entry/exit prices that produce
    the desired net PnL after costs.
    """
    from src.framework.data.constants import TICK_SIZE, TICK_VALUE, TOTAL_COST_RT

    rows = []
    for day, target_pnl in daily_pnls.items():
        # Net PnL = (exit - entry) / TICK_SIZE * TICK_VALUE * size - TOTAL_COST_RT * size
        # Solve for exit: exit = entry + (target_pnl + TOTAL_COST_RT) * TICK_SIZE / TICK_VALUE
        entry = 20000.0
        gross_needed = target_pnl + TOTAL_COST_RT
        exit_price = entry + gross_needed * TICK_SIZE / TICK_VALUE
        rows.append({
            "entry_time": datetime(day.year, day.month, day.day, 10, 0, 0, tzinfo=timezone.utc),
            "exit_time": datetime(day.year, day.month, day.day, 11, 0, 0, tzinfo=timezone.utc),
            "entry_price": entry,
            "exit_price": exit_price,
            "direction": 1,
            "size": 1,
        })
    return pl.DataFrame(rows)


class TestComputeFactorReturns:

    def test_basic_shape(self):
        bars = _make_bars(30)
        result = compute_factor_returns(bars)
        assert "date" in result.columns
        assert "market_return" in result.columns
        assert "volatility_change" in result.columns
        assert "momentum" in result.columns
        # 30 days total, first day has null momentum/vol_change from shift = 29 non-null
        non_null = result.drop_nulls()
        assert len(non_null) == 29

    def test_momentum_is_lagged_return(self):
        bars = _make_bars(5, seed=99)
        result = compute_factor_returns(bars)
        mkt = result["market_return"].to_list()
        mom = result["momentum"].to_list()
        # momentum[i] should equal market_return[i-1]
        for i in range(1, len(mkt)):
            if mom[i] is not None and mkt[i - 1] is not None:
                assert abs(mom[i] - mkt[i - 1]) < 1e-12


class TestFactorAttribution:

    def test_pure_alpha(self):
        """Constant daily PnL + small noise, independent of market factors."""
        rng = np.random.default_rng(42)
        n_days = 60
        bars = _make_bars(n_days, seed=10)
        dates_list = sorted(bars.with_columns(pl.col("ts_event").dt.date().alias("d"))["d"].unique().to_list())

        # Strategy PnL = 100 + noise(0, 20), no market exposure
        daily_pnls = {}
        for d in dates_list:
            daily_pnls[d] = 100.0 + rng.normal(0, 20)

        trades = _make_trades(daily_pnls)
        result = factor_attribution(trades, bars)

        assert result["available"] is True
        assert result["n_days"] >= 20
        # Alpha should be significant (constant 100 dominates)
        assert abs(result["alpha_t_stat"]) > 2.0
        # R-squared should be low (PnL not explained by factors)
        assert result["r_squared"] < 0.15
        assert result["verdict"] == "PURE_ALPHA"

    def test_pure_beta(self):
        """Daily PnL = 500 * market_return. No alpha, high factor exposure."""
        n_days = 60
        bars = _make_bars(n_days, seed=20)
        factors = compute_factor_returns(bars).drop_nulls()

        daily_pnls = {}
        for row in factors.iter_rows(named=True):
            daily_pnls[row["date"]] = 500.0 * row["market_return"]

        trades = _make_trades(daily_pnls)
        result = factor_attribution(trades, bars)

        assert result["available"] is True
        # Market beta should be close to 500
        assert abs(result["factor_betas"]["market"] - 500.0) < 100.0
        # R-squared should be high
        assert result["r_squared"] > 0.5
        assert result["verdict"] == "FACTOR_EXPOSED"

    def test_mixed_alpha_and_beta(self):
        """Daily PnL = 100 + 300 * market_return + noise."""
        rng = np.random.default_rng(42)
        n_days = 60
        bars = _make_bars(n_days, seed=30)
        factors = compute_factor_returns(bars).drop_nulls()

        daily_pnls = {}
        for row in factors.iter_rows(named=True):
            daily_pnls[row["date"]] = 100.0 + 300.0 * row["market_return"] + rng.normal(0, 10)

        trades = _make_trades(daily_pnls)
        result = factor_attribution(trades, bars)

        assert result["available"] is True
        # Both alpha and market beta should be present
        assert abs(result["alpha_daily"]) > 50.0
        assert abs(result["factor_betas"]["market"]) > 100.0

    def test_insufficient_data(self):
        """Less than min_days should return INSUFFICIENT_DATA."""
        bars = _make_bars(10)
        dates_list = sorted(bars.with_columns(pl.col("ts_event").dt.date().alias("d"))["d"].unique().to_list())
        daily_pnls = {d: 100.0 for d in dates_list[:5]}
        trades = _make_trades(daily_pnls)

        result = factor_attribution(trades, bars, min_days=20)
        assert result["available"] is False
        assert result["verdict"] == "INSUFFICIENT_DATA"

    def test_empty_trades(self):
        bars = _make_bars(30)
        empty_trades = pl.DataFrame(schema={
            "entry_time": pl.Datetime("ns", "UTC"),
            "exit_time": pl.Datetime("ns", "UTC"),
            "entry_price": pl.Float64,
            "exit_price": pl.Float64,
            "direction": pl.Int8,
            "size": pl.Int32,
        })
        result = factor_attribution(empty_trades, bars)
        assert result["available"] is False

    def test_known_answer_ols(self):
        """Hardcoded system with known OLS solution to verify coefficients.

        We directly verify the numpy lstsq math by building a DataFrame
        that matches exact factor values, bypassing the bar->factor pipeline.
        """
        # Construct a known linear system: y = 10 + 5*x1 - 2*x2 + 0*x3
        rng = np.random.default_rng(99)
        n = 30
        x1 = rng.normal(0, 0.01, n)
        x2 = rng.normal(0, 0.001, n)
        x3 = rng.normal(0, 0.01, n)
        y = 10.0 + 5.0 * x1 - 2.0 * x2

        # Verify OLS directly with numpy
        X = np.column_stack([np.ones(n), x1, x2, x3])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Intercept should be very close to 10
        assert abs(coeffs[0] - 10.0) < 0.5
        # x1 coefficient should be close to 5
        assert abs(coeffs[1] - 5.0) < 0.5
        # x2 coefficient should be close to -2
        assert abs(coeffs[2] - (-2.0)) < 1.0
        # x3 coefficient should be close to 0
        assert abs(coeffs[3]) < 1.0

    def test_known_answer_integration(self):
        """End-to-end known-answer: constant PnL=50 recovers alpha near 50."""
        n_days = 40
        bars = _make_bars(n_days, seed=77)
        dates_list = sorted(bars.with_columns(pl.col("ts_event").dt.date().alias("d"))["d"].unique().to_list())

        daily_pnls = {d: 50.0 for d in dates_list}
        trades = _make_trades(daily_pnls)
        result = factor_attribution(trades, bars, min_days=5)

        assert result["available"] is True
        # With constant PnL and random factors, alpha should be near 50
        assert abs(result["alpha_daily"] - 50.0) < 15.0
        # Residuals should be small
        assert result["r_squared"] < 0.20

    def test_output_keys(self):
        """Verify all expected keys are present in output dict."""
        rng = np.random.default_rng(42)
        n_days = 30
        bars = _make_bars(n_days)
        dates_list = sorted(bars.with_columns(pl.col("ts_event").dt.date().alias("d"))["d"].unique().to_list())
        daily_pnls = {d: 100.0 + rng.normal(0, 20) for d in dates_list}
        trades = _make_trades(daily_pnls)

        result = factor_attribution(trades, bars)

        expected_keys = {
            "available", "alpha_daily", "alpha_annualized", "alpha_t_stat",
            "alpha_pvalue", "factor_betas", "factor_t_stats", "factor_pvalues",
            "r_squared", "residual_sharpe", "n_days", "verdict",
        }
        assert expected_keys.issubset(result.keys())
        assert set(result["factor_betas"].keys()) == {"market", "volatility", "momentum"}
        assert set(result["factor_t_stats"].keys()) == {"market", "volatility", "momentum"}
        assert set(result["factor_pvalues"].keys()) == {"market", "volatility", "momentum"}
