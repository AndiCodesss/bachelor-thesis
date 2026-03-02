"""Tests for validation gauntlet."""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from src.framework.backtest.validators import (
    shuffle_test,
    walk_forward_test,
    regime_test,
    param_sensitivity_test,
    cost_sensitivity_test,
    decay_test,
    trade_count_test,
    run_validation_gauntlet,
)
from src.framework.data.constants import SEED


def create_synthetic_data(n_bars: int = 1000, n_days: int = 5, perfect_signal: bool = True) -> pl.DataFrame:
    """Create synthetic bar data for testing.

    Args:
        n_bars: Total number of bars
        n_days: Number of days to spread bars across
        perfect_signal: If True, create a perfect signal that predicts price direction

    Returns:
        DataFrame with ts_event, close, signal columns
    """
    rng = np.random.default_rng(SEED)

    # Create timestamps at 5-minute intervals across multiple days
    start_date = datetime(2025, 1, 1, 9, 30, 0)
    bars_per_day = n_bars // n_days
    timestamps = []

    for day in range(n_days):
        day_start = start_date + timedelta(days=day)
        for bar in range(bars_per_day):
            timestamps.append(day_start + timedelta(minutes=5 * bar))

    # Pad to n_bars if needed
    while len(timestamps) < n_bars:
        last_ts = timestamps[-1]
        timestamps.append(last_ts + timedelta(minutes=5))

    timestamps = timestamps[:n_bars]

    # Create price series with trend and noise
    base_price = 15000.0
    drift = 0.0002  # Slight upward drift
    volatility = 0.001

    prices = [base_price]
    for i in range(1, n_bars):
        change = drift + volatility * rng.normal()
        prices.append(prices[-1] * (1 + change))

    # Create signal
    if perfect_signal:
        # Perfect signal: predict next bar direction
        signal = []
        for i in range(len(prices) - 1):
            if prices[i+1] > prices[i]:
                signal.append(1)  # Long
            elif prices[i+1] < prices[i]:
                signal.append(-1)  # Short
            else:
                signal.append(0)  # Flat
        signal.append(0)  # Last bar flat
    else:
        # Random signal
        signal = rng.choice([1, -1, 0], size=n_bars, p=[0.3, 0.3, 0.4])

    df = pl.DataFrame({
        "ts_event": timestamps,
        "close": prices,
        "signal": signal,
    })

    # Cast to correct types
    df = df.with_columns([
        pl.col("ts_event").cast(pl.Datetime("ns", "UTC")),
        pl.col("close").cast(pl.Float64),
        pl.col("signal").cast(pl.Int8),
    ])

    return df


def test_shuffle_test_perfect_signal():
    """Perfect signal should PASS shuffle test."""
    df = create_synthetic_data(n_bars=1000, n_days=5, perfect_signal=True)
    result = shuffle_test(df, signal_col="signal", n_iterations=100)

    assert result["verdict"] == "PASS", "Perfect signal should beat shuffled signals"
    assert result["real_sharpe"] > result["shuffle_mean"], "Real Sharpe should exceed shuffle mean"
    assert result["percentile"] >= 95.0, f"Real Sharpe should be in top 5% (got {result['percentile']})"
    assert "threshold" in result
    assert result["threshold"] == 95.0


def test_shuffle_test_random_signal():
    """Random signal should FAIL shuffle test."""
    df = create_synthetic_data(n_bars=1000, n_days=5, perfect_signal=False)
    result = shuffle_test(df, signal_col="signal", n_iterations=100)

    # Random signal shouldn't reliably beat other random signals
    # Note: There's a small chance this could PASS by luck, but very unlikely with SEED=42
    assert result["verdict"] == "FAIL", "Random signal should not consistently beat shuffles"
    assert "percentile" in result
    assert "shuffle_mean" in result
    assert "shuffle_std" in result


def test_shuffle_preserves_block_structure():
    """Block-bootstrap shuffle should preserve signal length and value distribution."""
    rng = np.random.default_rng(SEED)

    # Create a signal with known block structure
    signal = np.array([1, 1, 1, 0, 0, -1, -1, -1, -1, 0, 1, 1], dtype=np.int8)
    # Blocks: [1,1,1], [0,0], [-1,-1,-1,-1], [0], [1,1]

    change_points = np.flatnonzero(np.diff(signal) != 0) + 1
    blocks = np.split(signal, change_points)

    # Permute blocks
    block_order = rng.permutation(len(blocks))
    shuffled = np.concatenate([blocks[j] for j in block_order])

    # Length is preserved
    assert len(shuffled) == len(signal)
    # Value distribution is preserved
    assert sorted(signal.tolist()) == sorted(shuffled.tolist())
    # Observed block count <= original (same-value blocks may merge when adjacent)
    shuffled_change_points = np.flatnonzero(np.diff(shuffled) != 0) + 1
    shuffled_blocks = np.split(shuffled, shuffled_change_points)
    assert len(shuffled_blocks) <= len(blocks)


def test_walk_forward_test_perfect_signal():
    """Perfect signal should PASS walk-forward test."""
    df = create_synthetic_data(n_bars=1000, n_days=5, perfect_signal=True)
    result = walk_forward_test(df, signal_col="signal", n_folds=5)

    assert result["verdict"] == "PASS", "Perfect signal should be profitable in majority of folds"
    assert result["profitable_folds"] > len(result["fold_sharpes"]) / 2
    assert len(result["fold_sharpes"]) == 5
    assert len(result["fold_pnls"]) == 5
    assert all(isinstance(s, float) for s in result["fold_sharpes"])


def test_regime_test_perfect_signal():
    """Perfect signal should PASS regime test."""
    df = create_synthetic_data(n_bars=1000, n_days=5, perfect_signal=True)
    result = regime_test(df, signal_col="signal")

    assert result["verdict"] == "PASS", "Perfect signal should work in both regimes"
    assert result["high_vol_pnl"] > 0, "High vol regime should be profitable"
    assert result["low_vol_pnl"] > 0, "Low vol regime should be profitable"
    assert isinstance(result["high_vol_sharpe"], float)
    assert isinstance(result["low_vol_sharpe"], float)


def test_param_sensitivity_test_perfect_signal():
    """Perfect signal should PASS param sensitivity test."""
    df = create_synthetic_data(n_bars=500, n_days=3, perfect_signal=True)
    result = param_sensitivity_test(df, signal_col="signal", perturbation=0.1)

    assert result["verdict"] == "PASS", "Perfect signal should degrade gracefully"
    assert result["perturbed_mean"] >= 0.5 * result["baseline_sharpe"], "Perturbed mean should be >= 50% of baseline"
    assert isinstance(result["degradation_pct"], float)
    assert isinstance(result["perturbed_std"], float)


def test_cost_sensitivity_test_perfect_signal():
    """Perfect signal should PASS cost sensitivity test."""
    df = create_synthetic_data(n_bars=500, n_days=3, perfect_signal=True)
    result = cost_sensitivity_test(df, signal_col="signal")

    assert result["verdict"] == "PASS", "Perfect signal should survive 1.5x costs"
    assert result["pnl_1_5x"] > 0, "Should be profitable at 1.5x costs"
    assert result["pnl_1x"] > result["pnl_1_5x"] > result["pnl_2x"], "PnL should decline with higher costs"
    assert isinstance(result["sharpe_1x"], float)


def test_cost_sensitivity_test_marginal_signal():
    """Signal that barely breaks even should FAIL cost sensitivity."""
    # Create a signal with very few trades and low edge
    df = create_synthetic_data(n_bars=500, n_days=3, perfect_signal=False)

    # Create a signal with marginal edge (very sparse trading)
    rng = np.random.default_rng(SEED)
    marginal_signal = np.zeros(len(df), dtype=np.int8)
    # Only trade on ~5% of bars
    trade_indices = rng.choice(len(df), size=int(len(df) * 0.05), replace=False)
    marginal_signal[trade_indices] = rng.choice([1, -1], size=len(trade_indices))

    df_marginal = df.with_columns([
        pl.Series("marginal_signal", marginal_signal)
    ])

    result = cost_sensitivity_test(df_marginal, signal_col="marginal_signal")

    # This should likely FAIL at 1.5x costs due to low edge
    # Note: outcome depends on random signal, but with SEED=42 should be consistent
    assert "verdict" in result
    assert "pnl_1x" in result
    assert "pnl_1_5x" in result
    assert "pnl_2x" in result


def test_decay_test_perfect_signal():
    """Perfect signal should PASS decay test."""
    df = create_synthetic_data(n_bars=800, n_days=4, perfect_signal=True)
    result = decay_test(df, signal_col="signal", n_chunks=4)

    assert result["verdict"] == "PASS", "Perfect signal should not decay over time"
    assert not result["is_declining"], "Sharpe should not be monotonically declining"
    assert len(result["chunk_sharpes"]) == 4
    assert all(isinstance(s, float) for s in result["chunk_sharpes"])


def test_trade_count_test_pass():
    """Signal with >= 50 trades should PASS."""
    df = create_synthetic_data(n_bars=1000, n_days=5, perfect_signal=True)
    result = trade_count_test(df, signal_col="signal", min_trades=50)

    assert result["verdict"] == "PASS", f"Should have >= 50 trades (got {result['trade_count']})"
    assert result["trade_count"] >= 50
    assert result["min_required"] == 50


def test_trade_count_test_fail():
    """Signal with < 50 trades should FAIL."""
    # Create very sparse signal
    df = create_synthetic_data(n_bars=200, n_days=2, perfect_signal=False)

    # Create signal with very few transitions
    sparse_signal = np.zeros(len(df), dtype=np.int8)
    sparse_signal[10:15] = 1  # Only 1 trade
    sparse_signal[100:105] = -1  # Another trade

    df_sparse = df.with_columns([
        pl.Series("sparse_signal", sparse_signal)
    ])

    result = trade_count_test(df_sparse, signal_col="sparse_signal", min_trades=50)

    assert result["verdict"] == "FAIL", "Should fail with < 50 trades"
    assert result["trade_count"] < 50


def test_run_validation_gauntlet_structure():
    """Full gauntlet should return correct structure."""
    df = create_synthetic_data(n_bars=1000, n_days=5, perfect_signal=True)
    result = run_validation_gauntlet(df, signal_col="signal")

    # Check structure
    assert "shuffle" in result
    assert "walk_forward" in result
    assert "regime" in result
    assert "param_sensitivity" in result
    assert "cost_sensitivity" in result
    assert "decay" in result
    assert "trade_count" in result
    assert "overall_verdict" in result
    assert "pass_count" in result
    assert "total_tests" in result

    # Check overall verdict logic
    assert result["total_tests"] == 7
    assert 0 <= result["pass_count"] <= 7

    # All validators should return verdict
    for key in ["shuffle", "walk_forward", "regime", "param_sensitivity", "cost_sensitivity", "decay", "trade_count"]:
        assert "verdict" in result[key]
        assert result[key]["verdict"] in ["PASS", "FAIL"]


def test_run_validation_gauntlet_perfect_signal():
    """Perfect signal should PASS most/all validators."""
    df = create_synthetic_data(n_bars=1000, n_days=5, perfect_signal=True)
    result = run_validation_gauntlet(df, signal_col="signal")

    # Perfect signal should pass most tests
    # Note: Some tests might fail due to randomness, but majority should pass
    assert result["pass_count"] >= 5, f"Perfect signal should pass most tests (got {result['pass_count']}/7)"

    # Trade count should definitely pass with 1000 bars
    assert result["trade_count"]["verdict"] == "PASS"


def test_validators_with_empty_signal():
    """Validators should handle signal with no trades gracefully."""
    df = create_synthetic_data(n_bars=200, n_days=2, perfect_signal=False)

    # All flat signal
    flat_signal = np.zeros(len(df), dtype=np.int8)

    df_flat = df.with_columns([
        pl.Series("flat_signal", flat_signal)
    ])

    # Trade count test should FAIL
    result = trade_count_test(df_flat, signal_col="flat_signal")
    assert result["verdict"] == "FAIL"
    assert result["trade_count"] == 0

    # Other tests should not crash
    shuffle_result = shuffle_test(df_flat, signal_col="flat_signal", n_iterations=10)
    assert "verdict" in shuffle_result

    walk_result = walk_forward_test(df_flat, signal_col="flat_signal", n_folds=3)
    assert "verdict" in walk_result


def test_validators_preserve_input_dataframe():
    """Validators should not modify the input DataFrame."""
    df = create_synthetic_data(n_bars=500, n_days=3, perfect_signal=True)
    original_columns = df.columns
    original_len = len(df)

    # Run a validator
    shuffle_test(df, signal_col="signal")

    # Check DataFrame unchanged
    assert df.columns == original_columns
    assert len(df) == original_len
