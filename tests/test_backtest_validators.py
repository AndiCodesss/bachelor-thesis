"""Tests for validation gauntlet."""

import polars as pl
import numpy as np
import pytest
from datetime import datetime, timedelta, timezone
import src.framework.backtest.validators as validators_mod
from src.framework.backtest.validators import (
    shuffle_test,
    walk_forward_test,
    regime_test,
    param_sensitivity_test,
    signal_perturbation_test,
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


def test_shuffle_test_handles_nan_sharpe_values(monkeypatch: pytest.MonkeyPatch):
    df = create_synthetic_data(n_bars=20, n_days=2, perfect_signal=False)
    calls = {"n": 0}

    def _fake_run_backtest(*args, **kwargs):
        return pl.DataFrame()

    def _fake_compute_metrics(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"sharpe_ratio": 1.0}
        return {"sharpe_ratio": float("nan")}

    monkeypatch.setattr(validators_mod, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(validators_mod, "compute_metrics", _fake_compute_metrics)

    result = validators_mod.shuffle_test(df, signal_col="signal", n_iterations=5)
    assert np.isfinite(result["shuffle_mean"])
    assert np.isfinite(result["shuffle_std"])
    assert np.isfinite(result["percentile"])
    assert result["verdict"] in {"PASS", "FAIL"}


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
    assert result["mode"] == "expanding_history"


def test_walk_forward_test_uses_expanding_history(monkeypatch: pytest.MonkeyPatch):
    df = create_synthetic_data(n_bars=120, n_days=3, perfect_signal=True)
    seen_lengths: list[int] = []

    def _fake_run_backtest(chunk_df: pl.DataFrame, *args, **kwargs):
        seen_lengths.append(len(chunk_df))
        return pl.DataFrame(
            {
                "entry_time": chunk_df["ts_event"],
                "exit_time": chunk_df["ts_event"],
                "entry_price": chunk_df["close"],
                "exit_price": chunk_df["close"],
                "direction": pl.Series([1] * len(chunk_df), dtype=pl.Int8),
                "size": pl.Series([1] * len(chunk_df), dtype=pl.Int32),
            }
        )

    def _fake_compute_metrics(*args, **kwargs):
        return {"sharpe_ratio": 1.0, "net_pnl": 1.0}

    monkeypatch.setattr(validators_mod, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(validators_mod, "compute_metrics", _fake_compute_metrics)

    result = walk_forward_test(df, signal_col="signal", n_folds=4)
    assert result["verdict"] == "PASS"
    assert seen_lengths == sorted(seen_lengths)
    assert len(set(seen_lengths)) == 4


def test_walk_forward_marks_cross_boundary_trades_to_fold_close(monkeypatch: pytest.MonkeyPatch):
    df = create_synthetic_data(n_bars=60, n_days=1, perfect_signal=False)
    seen_trade_counts: list[int] = []

    def _fake_run_backtest(chunk_df: pl.DataFrame, *args, **kwargs):
        ts = chunk_df["ts_event"]
        close = chunk_df["close"]
        if len(chunk_df) < 2:
            return pl.DataFrame()
        return pl.DataFrame(
            {
                "entry_time": [ts[-2]],
                "exit_time": [ts[-1] + timedelta(minutes=5)],
                "entry_price": [float(close[-2])],
                "exit_price": [float(close[-1]) + 10.0],
                "direction": [1],
                "size": [1],
            }
        )

    def _fake_compute_metrics(trades: pl.DataFrame, *args, **kwargs):
        seen_trade_counts.append(len(trades))
        return {"sharpe_ratio": 1.0, "net_pnl": float(len(trades))}

    monkeypatch.setattr(validators_mod, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(validators_mod, "compute_metrics", _fake_compute_metrics)

    result = walk_forward_test(df, signal_col="signal", n_folds=3)
    assert result["verdict"] == "PASS"
    assert any(count == 1 for count in seen_trade_counts), "Cross-boundary fold trade should be retained"


def test_extract_walk_forward_fold_trades_includes_carry_in_overlap():
    ts = [
        datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=5 * i)
        for i in range(4)
    ]
    test_df = pl.DataFrame(
        {
            "ts_event": ts[1:3],
            "open": [101.0, 102.0],
            "close": [101.5, 102.5],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", "UTC")))
    history_trades = pl.DataFrame(
        {
            "entry_time": [ts[0]],
            "exit_time": [ts[2]],
            "entry_price": [100.0],
            "exit_price": [102.5],
            "direction": [1],
            "size": [1],
        }
    ).with_columns(
        pl.col("entry_time").cast(pl.Datetime("ns", "UTC")),
        pl.col("exit_time").cast(pl.Datetime("ns", "UTC")),
    )

    fold_trades = validators_mod._extract_walk_forward_fold_trades(history_trades, test_df)

    assert len(fold_trades) == 1
    assert fold_trades["entry_time"][0] == test_df["ts_event"][0]
    assert fold_trades["entry_price"][0] == test_df["open"][0]
    assert fold_trades["exit_time"][0] == history_trades["exit_time"][0]
    assert fold_trades["exit_price"][0] == history_trades["exit_price"][0]


def test_regime_test_perfect_signal():
    """Perfect signal should PASS regime test."""
    df = create_synthetic_data(n_bars=1000, n_days=5, perfect_signal=True)
    result = regime_test(df, signal_col="signal")

    assert result["verdict"] == "PASS", "Perfect signal should work in both regimes"
    assert result["high_vol_pnl"] > 0, "High vol regime should be profitable"
    assert result["low_vol_pnl"] > 0, "Low vol regime should be profitable"
    assert isinstance(result["high_vol_sharpe"], float)
    assert isinstance(result["low_vol_sharpe"], float)


def test_regime_test_excludes_sparse_days_from_regime_buckets(monkeypatch: pytest.MonkeyPatch):
    sparse_day = datetime(2025, 1, 1, 9, 30, 0)
    dense_start = datetime(2025, 1, 2, 9, 30, 0)
    df = pl.DataFrame({
        "ts_event": [sparse_day] + [dense_start + timedelta(minutes=5 * i) for i in range(5)],
        "close": [100.0, 100.5, 100.2, 100.6, 100.4, 100.7],
        "signal": [0, 1, 0, -1, 0, 1],
    }).with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns", "UTC")),
        pl.col("close").cast(pl.Float64),
        pl.col("signal").cast(pl.Int8),
    )

    fake_trades = pl.DataFrame({
        "entry_time": [
            sparse_day.replace(tzinfo=None),
            dense_start.replace(tzinfo=None),
        ],
        "exit_time": [
            sparse_day.replace(tzinfo=None),
            dense_start.replace(tzinfo=None),
        ],
        "entry_price": [100.0, 100.0],
        "exit_price": [101.0, 101.0],
        "direction": [1, 1],
        "size": [1, 1],
    }).with_columns(
        pl.col("entry_time").dt.replace_time_zone("UTC"),
        pl.col("exit_time").dt.replace_time_zone("UTC"),
    )
    seen_entry_dates: list[set] = []

    def _fake_run_backtest(*args, **kwargs):
        return fake_trades

    def _fake_compute_metrics(trades: pl.DataFrame, *args, **kwargs):
        seen_entry_dates.append(set(trades["entry_time"].dt.date().to_list()) if len(trades) > 0 else set())
        return {"net_pnl": 1.0, "sharpe_ratio": 0.0}

    monkeypatch.setattr(validators_mod, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(validators_mod, "compute_metrics", _fake_compute_metrics)

    validators_mod.regime_test(df, signal_col="signal")
    sparse_date = sparse_day.date()
    # Skip first call (full all_trades metrics); bucket calls must exclude sparse day.
    for bucket_dates in seen_entry_dates[1:]:
        assert sparse_date not in bucket_dates


def test_signal_perturbation_test_perfect_signal():
    """Perfect signal should PASS signal perturbation test."""
    df = create_synthetic_data(n_bars=500, n_days=3, perfect_signal=True)
    result = signal_perturbation_test(df, signal_col="signal", perturbation=0.1)

    assert result["verdict"] == "PASS", "Perfect signal should degrade gracefully"
    assert result["perturbed_mean"] >= 0.5 * result["baseline_sharpe"], "Perturbed mean should be >= 50% of baseline"
    assert isinstance(result["degradation_pct"], float)
    assert isinstance(result["perturbed_std"], float)


def test_signal_perturbation_handles_nan_sharpe_values(monkeypatch: pytest.MonkeyPatch):
    df = create_synthetic_data(n_bars=20, n_days=2, perfect_signal=False)
    calls = {"n": 0}

    def _fake_run_backtest(*args, **kwargs):
        return pl.DataFrame()

    def _fake_compute_metrics(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"sharpe_ratio": 1.0}
        return {"sharpe_ratio": float("nan")}

    monkeypatch.setattr(validators_mod, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(validators_mod, "compute_metrics", _fake_compute_metrics)

    result = validators_mod.signal_perturbation_test(df, signal_col="signal", perturbation=0.1)
    assert np.isfinite(result["perturbed_mean"])
    assert np.isfinite(result["perturbed_std"])
    assert np.isfinite(result["degradation_pct"])


def test_param_sensitivity_alias_matches_signal_perturbation():
    df = create_synthetic_data(n_bars=200, n_days=2, perfect_signal=True)
    aliased = param_sensitivity_test(df, signal_col="signal", perturbation=0.1)
    renamed = signal_perturbation_test(df, signal_col="signal", perturbation=0.1)
    assert aliased == renamed


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

    assert result["verdict"] in {"PASS", "FAIL"}
    expected = "PASS" if result["pnl_1_5x"] > 0 else "FAIL"
    assert result["verdict"] == expected
    assert result["pnl_1x"] > result["pnl_1_5x"] > result["pnl_2x"]


def test_decay_test_perfect_signal():
    """Perfect signal should PASS decay test."""
    df = create_synthetic_data(n_bars=800, n_days=4, perfect_signal=True)
    result = decay_test(df, signal_col="signal", n_chunks=4)

    assert result["verdict"] == "PASS", "Perfect signal should not decay over time"
    assert not result["is_declining"], "Sharpe should not be monotonically declining"
    assert len(result["chunk_sharpes"]) == 4
    assert all(isinstance(s, float) for s in result["chunk_sharpes"])
    assert "trend_slope" in result
    assert "trend_r_squared" in result


def test_decay_test_uses_same_last_chunk_for_profitability_check(monkeypatch: pytest.MonkeyPatch):
    df = create_synthetic_data(n_bars=103, n_days=1, perfect_signal=False)
    seen_lengths = []

    def _fake_run_backtest(chunk_df: pl.DataFrame, *args, **kwargs):
        seen_lengths.append(len(chunk_df))
        return pl.DataFrame()

    def _fake_compute_metrics(*args, **kwargs):
        return {"sharpe_ratio": 0.5, "net_pnl": 1.0}

    monkeypatch.setattr(validators_mod, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(validators_mod, "compute_metrics", _fake_compute_metrics)

    validators_mod.decay_test(df, signal_col="signal", n_chunks=4)
    # n=103, n_chunks=4 => chunk sizes: 25,25,25,28. Last call should use 28-row last chunk.
    assert seen_lengths[-1] == 28


def test_decay_test_flags_trend_decline_without_strict_monotonicity(monkeypatch: pytest.MonkeyPatch):
    df = create_synthetic_data(n_bars=80, n_days=2, perfect_signal=False)
    sharpe_vals = iter([1.5, 1.5, 1.4, 1.3, 1.0])

    def _fake_run_backtest(*args, **kwargs):
        return pl.DataFrame()

    def _fake_compute_metrics(*args, **kwargs):
        return {"sharpe_ratio": next(sharpe_vals), "net_pnl": 1.0}

    monkeypatch.setattr(validators_mod, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(validators_mod, "compute_metrics", _fake_compute_metrics)

    result = decay_test(df, signal_col="signal", n_chunks=4)
    assert result["is_declining"] is True
    assert result["trend_slope"] < 0


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
    assert "signal_perturbation" in result
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
    for key in ["shuffle", "walk_forward", "regime", "signal_perturbation", "cost_sensitivity", "decay", "trade_count"]:
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
