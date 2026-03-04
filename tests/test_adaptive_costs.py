"""Tests for adaptive transaction cost model."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.framework.backtest.costs import CostModel, compute_adaptive_costs
from src.framework.backtest.metrics import compute_metrics
from src.framework.data.constants import COMMISSION_RT, TOTAL_COST_RT


# ---------------------------------------------------------------------------
# CostModel unit tests
# ---------------------------------------------------------------------------


def test_flat_model_neutral_inputs():
    """CostModel.flat() with neutral inputs (spread=1, vol_z=0, vol_z=0) yields ~$14.50."""
    model = CostModel.flat()
    cost = model.estimate_cost_rt(
        spread_ticks=1.0, volatility_z=0.0, volume_z=0.0, session_progress=0.5
    )
    assert cost == pytest.approx(TOTAL_COST_RT, abs=0.01)


def test_wide_spread_increases_cost():
    """Wider spread (3 ticks) should cost more than narrow spread (1 tick)."""
    model = CostModel.flat()
    cost_narrow = model.estimate_cost_rt(1.0, 0.0, 0.0, 0.5)
    cost_wide = model.estimate_cost_rt(3.0, 0.0, 0.0, 0.5)
    assert cost_wide > cost_narrow


def test_high_volatility_increases_cost():
    """Higher volatility z-score should increase cost."""
    model = CostModel.flat()
    cost_calm = model.estimate_cost_rt(1.0, 0.0, 0.0, 0.5)
    cost_volatile = model.estimate_cost_rt(1.0, 2.0, 0.0, 0.5)
    assert cost_volatile > cost_calm


def test_high_volume_decreases_cost():
    """Higher volume z-score should decrease cost (more liquidity)."""
    model = CostModel.flat()
    cost_thin = model.estimate_cost_rt(1.0, 0.0, 0.0, 0.5)
    cost_liquid = model.estimate_cost_rt(1.0, 0.0, 2.0, 0.5)
    assert cost_liquid < cost_thin


def test_session_edges_increase_cost():
    """Trades at session open/close should cost more than mid-session."""
    model = CostModel.flat()
    cost_mid = model.estimate_cost_rt(1.0, 0.0, 0.0, 0.5)     # mid-session
    cost_open = model.estimate_cost_rt(1.0, 0.0, 0.0, 0.0)    # session open
    cost_close = model.estimate_cost_rt(1.0, 0.0, 0.0, 1.0)   # session close
    assert cost_open > cost_mid
    assert cost_close > cost_mid
    # Open and close should be symmetric
    assert cost_open == pytest.approx(cost_close, abs=0.01)


def test_floor_enforcement():
    """Even with extreme negative inputs, cost never drops below commission_rt."""
    model = CostModel.flat()
    # Extreme volume discount, negative vol, tight spread
    cost = model.estimate_cost_rt(
        spread_ticks=0.0, volatility_z=-10.0, volume_z=100.0, session_progress=0.5
    )
    assert cost >= COMMISSION_RT


def test_known_answer_specific_inputs():
    """Hand-computed: spread=2, vol_z=1.5, volume_z=0.5 -> $31.50."""
    model = CostModel.flat()
    cost = model.estimate_cost_rt(
        spread_ticks=2.0, volatility_z=1.5, volume_z=0.5, session_progress=0.5
    )
    # slippage_per_side = 5.00 + 0.5*1.5*5.00 + 1.0*(2-1)*5.00 - 0.1*0.5*5.00
    #                   = 5.00 + 3.75 + 5.00 - 0.25 = 13.50
    # floor check: max(2.50, 13.50) = 13.50
    # total = 4.50 + 2*13.50 = 31.50
    # final: max(4.50, 31.50) = 31.50
    assert cost == pytest.approx(31.50, abs=0.01)


def test_known_answer_volume_discount_hits_floor():
    """When volume discount pushes slippage below half-base, floor kicks in."""
    model = CostModel.flat()
    # spread=1, vol_z=0, volume_z=10 -> raw slippage = 5.00 - 0.1*10*5.00 = 5.00-5.00 = 0.00
    # floor: max(2.50, 0.00) = 2.50
    # total = 4.50 + 2*2.50 = 9.50
    cost = model.estimate_cost_rt(
        spread_ticks=1.0, volatility_z=0.0, volume_z=10.0, session_progress=0.5
    )
    assert cost == pytest.approx(9.50, abs=0.01)


# ---------------------------------------------------------------------------
# compute_adaptive_costs integration tests
# ---------------------------------------------------------------------------


def _make_bars(n: int = 50, base_time: datetime | None = None) -> pl.DataFrame:
    """Create synthetic bars with ts_event, close, volume, ask_price, bid_price."""
    if base_time is None:
        base_time = datetime(2025, 3, 1, 10, 0, 0)
    return pl.DataFrame({
        "ts_event": [base_time + timedelta(minutes=5 * i) for i in range(n)],
        "close": [18000.0 + i * 0.5 for i in range(n)],
        "volume": [100.0 + (i % 10) * 10 for i in range(n)],
        "ask_price": [18000.25 + i * 0.5 for i in range(n)],
        "bid_price": [18000.00 + i * 0.5 for i in range(n)],
    })


def _make_trades(n: int = 3, base_time: datetime | None = None) -> pl.DataFrame:
    """Create synthetic trades aligned with bar timestamps."""
    if base_time is None:
        base_time = datetime(2025, 3, 1, 10, 0, 0)
    return pl.DataFrame({
        "entry_time": [base_time + timedelta(minutes=5 * (10 + i * 5)) for i in range(n)],
        "exit_time": [base_time + timedelta(minutes=5 * (12 + i * 5)) for i in range(n)],
        "entry_price": [18005.0 + i * 2.5 for i in range(n)],
        "exit_price": [18010.0 + i * 2.5 for i in range(n)],
        "direction": [1] * n,
        "size": [1] * n,
    })


def test_compute_adaptive_costs_adds_column():
    """compute_adaptive_costs returns trades with adaptive_cost_rt column."""
    bars = _make_bars(50)
    trades = _make_trades(3)
    result = compute_adaptive_costs(trades, bars)
    assert "adaptive_cost_rt" in result.columns
    assert len(result) == 3
    # Every cost should be a positive float
    costs = result["adaptive_cost_rt"].to_list()
    for c in costs:
        assert c >= COMMISSION_RT


def test_compute_adaptive_costs_empty_trades():
    """Empty trades get adaptive_cost_rt column with no rows."""
    from src.framework.backtest.engine import TRADE_SCHEMA

    bars = _make_bars(10)
    trades = pl.DataFrame(schema=TRADE_SCHEMA)
    result = compute_adaptive_costs(trades, bars)
    assert "adaptive_cost_rt" in result.columns
    assert len(result) == 0


def test_compute_adaptive_costs_no_ask_bid():
    """Bars without ask_price/bid_price columns default spread to 1 tick."""
    bars = _make_bars(50).drop(["ask_price", "bid_price"])
    trades = _make_trades(3)
    result = compute_adaptive_costs(trades, bars)
    assert "adaptive_cost_rt" in result.columns
    assert len(result) == 3


def test_compute_adaptive_costs_preserves_trade_columns():
    """Original trade columns are preserved in output."""
    bars = _make_bars(50)
    trades = _make_trades(3)
    result = compute_adaptive_costs(trades, bars)
    for col in ["entry_time", "exit_time", "entry_price", "exit_price", "direction", "size"]:
        assert col in result.columns


def test_compute_adaptive_costs_non_rth_uses_neutral_session_progress():
    """Non-RTH trades should not get edge penalties from session_progress."""
    non_rth_bar_ts = datetime(2025, 3, 1, 8, 0, 0)   # ~03:00 ET
    mid_rth_bar_ts = datetime(2025, 3, 1, 17, 45, 0)  # ~12:45 ET (session midpoint)

    bars_non_rth = pl.DataFrame({
        "ts_event": [non_rth_bar_ts],
        "close": [18000.0],
        "volume": [100.0],
        "ask_price": [18000.25],
        "bid_price": [18000.00],
    })
    bars_mid_rth = pl.DataFrame({
        "ts_event": [mid_rth_bar_ts],
        "close": [18000.0],
        "volume": [100.0],
        "ask_price": [18000.25],
        "bid_price": [18000.00],
    })

    trades_non_rth = pl.DataFrame({
        "entry_time": [non_rth_bar_ts],
        "exit_time": [non_rth_bar_ts + timedelta(minutes=5)],
        "entry_price": [18000.0],
        "exit_price": [18001.0],
        "direction": [1],
        "size": [1],
    })
    trades_mid_rth = pl.DataFrame({
        "entry_time": [mid_rth_bar_ts],
        "exit_time": [mid_rth_bar_ts + timedelta(minutes=5)],
        "entry_price": [18000.0],
        "exit_price": [18001.0],
        "direction": [1],
        "size": [1],
    })

    cost_non_rth = compute_adaptive_costs(trades_non_rth, bars_non_rth)["adaptive_cost_rt"][0]
    cost_mid_rth = compute_adaptive_costs(trades_mid_rth, bars_mid_rth)["adaptive_cost_rt"][0]
    assert cost_non_rth == pytest.approx(cost_mid_rth, abs=1e-6)


# ---------------------------------------------------------------------------
# metrics backward compatibility
# ---------------------------------------------------------------------------


def test_metrics_backward_compatible_no_override():
    """compute_metrics without cost_override_col behaves identically to before."""
    day1 = datetime(2025, 2, 1, 10, 0, 0)
    day2 = datetime(2025, 2, 2, 10, 0, 0)

    trades = pl.DataFrame({
        "entry_time": [day1, day2],
        "exit_time": [day1 + timedelta(minutes=30), day2 + timedelta(minutes=30)],
        "entry_price": [18000.0, 18010.0],
        "exit_price": [18010.0, 18020.0],
        "direction": [1, 1],
        "size": [1, 1],
    })

    # Without override
    metrics_default = compute_metrics(trades)
    # With override set to None explicitly
    metrics_none = compute_metrics(trades, cost_override_col=None)
    # With override column name that does not exist in DataFrame
    metrics_missing = compute_metrics(trades, cost_override_col="nonexistent_col")

    assert metrics_default == metrics_none
    assert metrics_default == metrics_missing

    # Verify costs match flat model
    assert metrics_default["total_costs"] == pytest.approx(2 * TOTAL_COST_RT, abs=0.01)


def test_metrics_with_cost_override():
    """compute_metrics uses cost_override_col when column exists."""
    day1 = datetime(2025, 2, 1, 10, 0, 0)
    day2 = datetime(2025, 2, 2, 10, 0, 0)

    # Two winning trades, 10 pts each = $200 gross each
    trades = pl.DataFrame({
        "entry_time": [day1, day2],
        "exit_time": [day1 + timedelta(minutes=30), day2 + timedelta(minutes=30)],
        "entry_price": [18000.0, 18010.0],
        "exit_price": [18010.0, 18020.0],
        "direction": [1, 1],
        "size": [1, 1],
        "adaptive_cost_rt": [20.0, 30.0],  # Custom per-trade costs
    })

    metrics = compute_metrics(trades, cost_override_col="adaptive_cost_rt")

    # Total costs: 20 * 1 + 30 * 1 = $50
    assert metrics["total_costs"] == pytest.approx(50.0, abs=0.01)
    # Gross PnL: 2 * $200 = $400
    assert metrics["gross_pnl"] == pytest.approx(400.0, abs=0.01)
    # Net PnL: $400 - $50 = $350
    assert metrics["net_pnl"] == pytest.approx(350.0, abs=0.01)
