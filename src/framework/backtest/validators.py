"""Validation gauntlet for testing signal robustness."""

import polars as pl
import numpy as np
from src.framework.backtest.costs import compute_adaptive_costs
from src.framework.backtest.engine import run_backtest
from src.framework.backtest.metrics import compute_metrics
from src.framework.data.constants import SEED


def _sanitize_sharpe(value: float | int | None) -> float:
    """Convert non-finite Sharpe values to 0.0 for robust aggregate comparisons."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    return val if np.isfinite(val) else 0.0


def _compute_eval_metrics(trades: pl.DataFrame, bars: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Score trades using adaptive costs when bar context is available."""
    scored = trades
    if len(scored) > 0 and "ts_event" in bars.columns and "close" in bars.columns:
        cost_bars = bars
        entry_dtype = scored.schema.get("entry_time")
        ts_dtype = cost_bars.schema.get("ts_event")
        if entry_dtype is not None and ts_dtype is not None and entry_dtype != ts_dtype:
            cost_bars = cost_bars.with_columns(pl.col("ts_event").cast(entry_dtype))
        try:
            scored = compute_adaptive_costs(scored, cost_bars)
        except pl.exceptions.PolarsError:
            scored = trades
    metrics = compute_metrics(
        scored,
        cost_override_col="adaptive_cost_rt" if "adaptive_cost_rt" in scored.columns else None,
    )
    return scored, metrics


def _empty_trade_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "entry_time": pl.Datetime("ns", "UTC"),
            "exit_time": pl.Datetime("ns", "UTC"),
            "entry_price": pl.Float64,
            "exit_price": pl.Float64,
            "direction": pl.Int8,
            "size": pl.Int32,
        }
    )


def _linear_trend(values: list[float]) -> tuple[float, float]:
    """Return least-squares slope and R-squared for equally spaced observations."""
    if len(values) < 2:
        return 0.0, 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(slope), float(r_squared)


def _extract_walk_forward_fold_trades(
    history_trades: pl.DataFrame,
    test_df: pl.DataFrame,
) -> pl.DataFrame:
    """Project any fold-overlapping trades onto the test window only."""
    if len(history_trades) == 0 or len(test_df) == 0:
        return _empty_trade_frame()

    test_start_ts = test_df["ts_event"][0]
    test_end_ts = test_df["ts_event"][-1]
    fold_open = float(test_df["open"][0]) if "open" in test_df.columns else float(test_df["close"][0])
    fold_close = float(test_df["close"][-1])

    overlapping = history_trades.filter(
        (pl.col("entry_time") <= test_end_ts)
        & (pl.col("exit_time") >= test_start_ts)
    )
    if len(overlapping) == 0:
        return _empty_trade_frame()

    projected_rows: list[dict[str, object]] = []
    for row in overlapping.iter_rows(named=True):
        entry_time = row["entry_time"]
        exit_time = row["exit_time"]
        if entry_time is None or exit_time is None:
            continue

        projected_rows.append(
            {
                "entry_time": test_start_ts if entry_time < test_start_ts else entry_time,
                "exit_time": test_end_ts if exit_time > test_end_ts else exit_time,
                "entry_price": fold_open if entry_time < test_start_ts else row["entry_price"],
                "exit_price": fold_close if exit_time > test_end_ts else row["exit_price"],
                "direction": row["direction"],
                "size": row["size"],
            }
        )

    if not projected_rows:
        return _empty_trade_frame()
    return pl.DataFrame(projected_rows).cast(_empty_trade_frame().schema)


def shuffle_test(df: pl.DataFrame, signal_col: str = "signal", n_iterations: int = 100, **backtest_kwargs) -> dict:
    """Randomize signal assignment to test if performance is due to chance.

    Args:
        df: DataFrame with ts_event, close, and signal columns
        signal_col: Name of signal column
        n_iterations: Number of shuffle iterations
        **backtest_kwargs: Forwarded to run_backtest (exit_bars, profit_target, stop_loss, etc.)

    Returns:
        dict with verdict (PASS/FAIL), real_sharpe, shuffle_mean, shuffle_std, percentile
    """
    # Run backtest on real signal
    real_trades = run_backtest(df, signal_col=signal_col, **backtest_kwargs)
    _, real_metrics = _compute_eval_metrics(real_trades, df)
    real_sharpe = _sanitize_sharpe(real_metrics["sharpe_ratio"])

    # Run backtest on shuffled signals (block permutation)
    shuffle_sharpes = []
    rng = np.random.default_rng(SEED)

    # Identify contiguous blocks (runs of identical values)
    signal_arr = df[signal_col].to_numpy()
    change_points = np.flatnonzero(np.diff(signal_arr) != 0) + 1
    blocks = np.split(signal_arr, change_points)

    for i in range(n_iterations):
        # Permute block order (preserves block sizes and transition count)
        block_order = rng.permutation(len(blocks))
        shuffled_signal = np.concatenate([blocks[j] for j in block_order])

        df_shuffled = df.with_columns([
            pl.Series("_shuffled_signal", shuffled_signal)
        ])

        shuffle_trades = run_backtest(df_shuffled, signal_col="_shuffled_signal", **backtest_kwargs)
        _, shuffle_metrics = _compute_eval_metrics(shuffle_trades, df_shuffled)
        shuffle_sharpes.append(_sanitize_sharpe(shuffle_metrics["sharpe_ratio"]))

    # Compute statistics
    if len(shuffle_sharpes) == 0:
        shuffle_mean = 0.0
        shuffle_std = 0.0
    else:
        shuffle_mean = float(np.mean(shuffle_sharpes))
        shuffle_std = float(np.std(shuffle_sharpes, ddof=1)) if len(shuffle_sharpes) > 1 else 0.0

    # Compute where real_sharpe ranks
    if len(shuffle_sharpes) == 0:
        percentile = 0.0
    else:
        percentile = float(np.sum(np.array(shuffle_sharpes) <= real_sharpe) / len(shuffle_sharpes) * 100)

    verdict = "PASS" if percentile >= 95.0 else "FAIL"

    return {
        "verdict": verdict,
        "real_sharpe": real_sharpe,
        "shuffle_mean": shuffle_mean,
        "shuffle_std": shuffle_std,
        "percentile": percentile,
        "threshold": 95.0,
    }


def walk_forward_test(df: pl.DataFrame, signal_col: str = "signal", n_folds: int = 5, **backtest_kwargs) -> dict:
    """Expanding-history walk-forward evaluation over sequential out-of-sample folds.

    Args:
        df: DataFrame with ts_event, close, and signal columns
        signal_col: Name of signal column
        n_folds: Number of sequential time folds
        **backtest_kwargs: Forwarded to run_backtest

    Returns:
        dict with verdict (PASS/FAIL), fold_sharpes, fold_pnls, profitable_folds
    """
    total_rows = len(df)
    if n_folds <= 0 or total_rows < 4:
        return {
            "verdict": "FAIL",
            "fold_sharpes": [],
            "fold_pnls": [],
            "profitable_folds": 0,
            "threshold": "majority_profitable",
            "mode": "expanding_history",
        }

    # Partition into one initial training segment plus n_folds test segments.
    boundaries = np.linspace(0, total_rows, n_folds + 2, dtype=int)

    fold_sharpes = []
    fold_pnls = []
    profitable_folds = 0

    for i in range(n_folds):
        train_end = int(boundaries[i + 1])
        test_start = train_end
        test_end = int(boundaries[i + 2])
        if test_end <= test_start or train_end <= 0:
            continue

        history_df = df[:test_end]
        test_df = df[test_start:test_end]
        if len(test_df) == 0:
            continue

        history_trades = run_backtest(history_df, signal_col=signal_col, **backtest_kwargs)
        fold_trades = _extract_walk_forward_fold_trades(history_trades, test_df)
        _, fold_metrics = _compute_eval_metrics(fold_trades, test_df)

        fold_sharpes.append(_sanitize_sharpe(fold_metrics["sharpe_ratio"]))
        fold_pnls.append(float(fold_metrics["net_pnl"]))

        if fold_metrics["net_pnl"] > 0:
            profitable_folds += 1

    verdict = "PASS" if profitable_folds > len(fold_sharpes) / 2 else "FAIL"

    return {
        "verdict": verdict,
        "fold_sharpes": fold_sharpes,
        "fold_pnls": fold_pnls,
        "profitable_folds": profitable_folds,
        "threshold": "majority_profitable",
        "mode": "expanding_history",
    }


def regime_test(df: pl.DataFrame, signal_col: str = "signal", **backtest_kwargs) -> dict:
    """Test if signal works across different market regimes (high vs low volatility).

    Backtests the full contiguous series, then partitions trades by the regime
    of their entry date. This avoids discontinuous time-series artifacts.
    Regimes are classified using the full-sample daily volatility median, so this
    is a structural stress partition (not a predictive, point-in-time classifier).

    Args:
        df: DataFrame with ts_event, close, and signal columns
        signal_col: Name of signal column
        **backtest_kwargs: Forwarded to run_backtest

    Returns:
        dict with verdict (PASS/FAIL), high_vol_sharpe, low_vol_sharpe, high_vol_pnl, low_vol_pnl
    """
    # Run backtest on full contiguous data
    all_trades = run_backtest(df, signal_col=signal_col, **backtest_kwargs)
    all_trades, _ = _compute_eval_metrics(all_trades, df)
    if len(all_trades) == 0:
        return {
            "verdict": "FAIL",
            "high_vol_sharpe": 0.0,
            "low_vol_sharpe": 0.0,
            "high_vol_pnl": 0.0,
            "low_vol_pnl": 0.0,
        }

    # Classify each day as high-vol or low-vol using full-sample median.
    df_with_date = df.with_columns(pl.col("ts_event").dt.date().alias("_date"))
    daily_vol = (
        df_with_date
        .group_by("_date")
        .agg(pl.col("close").pct_change().std().alias("_daily_vol"))
        .filter(pl.col("_daily_vol").is_not_null())
    )
    if len(daily_vol) == 0:
        return {
            "verdict": "FAIL",
            "high_vol_sharpe": 0.0,
            "low_vol_sharpe": 0.0,
            "high_vol_pnl": 0.0,
            "low_vol_pnl": 0.0,
        }
    median_vol = daily_vol["_daily_vol"].median()
    valid_days = set(daily_vol["_date"].to_list())
    high_vol_days = set(daily_vol.filter(pl.col("_daily_vol") > median_vol)["_date"].to_list())

    # Partition trades by regime of entry date
    trades_with_date = all_trades.with_columns(
        pl.col("entry_time").dt.date().alias("_entry_date")
    )
    trades_with_regime = trades_with_date.filter(pl.col("_entry_date").is_in(valid_days))
    if len(trades_with_regime) == 0:
        return {
            "verdict": "FAIL",
            "high_vol_sharpe": 0.0,
            "low_vol_sharpe": 0.0,
            "high_vol_pnl": 0.0,
            "low_vol_pnl": 0.0,
        }
    high_vol_trades = trades_with_regime.filter(pl.col("_entry_date").is_in(high_vol_days)).drop("_entry_date")
    low_vol_trades = trades_with_regime.filter(~pl.col("_entry_date").is_in(high_vol_days)).drop("_entry_date")

    _, high_vol_metrics = _compute_eval_metrics(high_vol_trades, df)
    _, low_vol_metrics = _compute_eval_metrics(low_vol_trades, df)

    # PASS if BOTH regimes are profitable
    verdict = "PASS" if (high_vol_metrics["net_pnl"] > 0 and low_vol_metrics["net_pnl"] > 0) else "FAIL"

    return {
        "verdict": verdict,
        "high_vol_sharpe": high_vol_metrics["sharpe_ratio"],
        "low_vol_sharpe": low_vol_metrics["sharpe_ratio"],
        "high_vol_pnl": high_vol_metrics["net_pnl"],
        "low_vol_pnl": low_vol_metrics["net_pnl"],
    }


def signal_perturbation_test(
    df: pl.DataFrame,
    signal_col: str = "signal",
    perturbation: float = 0.1,
    **backtest_kwargs,
) -> dict:
    """Test whether small signal perturbations destroy observed edge.

    This is intentionally a signal-fragility stress test, not a parameter-rerun
    stability test. It perturbs only non-zero signal bars (the actual trades)
    to avoid injecting noise trades into sparse signals.

    Args:
        df: DataFrame with ts_event, close, and signal columns
        signal_col: Name of signal column
        perturbation: Fraction of NON-ZERO signal bars to randomly flip
        **backtest_kwargs: Forwarded to run_backtest

    Returns:
        dict with verdict (PASS/FAIL), baseline_sharpe, perturbed_mean, perturbed_std, degradation_pct
    """
    # Run baseline backtest
    baseline_trades = run_backtest(df, signal_col=signal_col, **backtest_kwargs)
    _, baseline_metrics = _compute_eval_metrics(baseline_trades, df)
    baseline_sharpe = _sanitize_sharpe(baseline_metrics["sharpe_ratio"])

    # Run backtest with perturbed signals
    n_iterations = 20
    perturbed_sharpes = []
    rng = np.random.default_rng(SEED)

    signal_arr_orig = df[signal_col].to_numpy()
    nonzero_indices = np.where(signal_arr_orig != 0)[0]

    for i in range(n_iterations):
        perturbed = signal_arr_orig.copy()

        if len(nonzero_indices) > 0:
            # Flip a fraction of non-zero signal bars only
            n_flips = max(1, int(len(nonzero_indices) * perturbation))
            flip_indices = rng.choice(nonzero_indices, size=min(n_flips, len(nonzero_indices)), replace=False)

            for idx in flip_indices:
                # Flip direction: 1 -> -1, -1 -> 1
                perturbed[idx] = -perturbed[idx]

        df_perturbed = df.with_columns([
            pl.Series("_perturbed_signal", perturbed)
        ])

        perturbed_trades = run_backtest(df_perturbed, signal_col="_perturbed_signal", **backtest_kwargs)
        _, perturbed_metrics = _compute_eval_metrics(perturbed_trades, df_perturbed)
        perturbed_sharpes.append(_sanitize_sharpe(perturbed_metrics["sharpe_ratio"]))

    # Compute statistics
    perturbed_mean = float(np.mean(perturbed_sharpes))
    perturbed_std = float(np.std(perturbed_sharpes, ddof=1)) if len(perturbed_sharpes) > 1 else 0.0

    # Compute degradation percentage
    if baseline_sharpe != 0:
        degradation_pct = ((baseline_sharpe - perturbed_mean) / abs(baseline_sharpe)) * 100.0
    else:
        degradation_pct = 0.0

    # PASS if baseline is positive and perturbed retains >= 50% of it
    if baseline_sharpe <= 0:
        verdict = "FAIL"
    else:
        verdict = "PASS" if perturbed_mean >= 0.5 * baseline_sharpe else "FAIL"

    return {
        "verdict": verdict,
        "baseline_sharpe": baseline_sharpe,
        "perturbed_mean": perturbed_mean,
        "perturbed_std": perturbed_std,
        "degradation_pct": float(degradation_pct),
    }


def param_sensitivity_test(
    df: pl.DataFrame,
    signal_col: str = "signal",
    perturbation: float = 0.1,
    **backtest_kwargs,
) -> dict:
    """Backward-compatible alias for the renamed signal perturbation test."""
    return signal_perturbation_test(
        df,
        signal_col=signal_col,
        perturbation=perturbation,
        **backtest_kwargs,
    )


def cost_sensitivity_test(df: pl.DataFrame, signal_col: str = "signal", **backtest_kwargs) -> dict:
    """Test if signal survives increased transaction costs.

    Args:
        df: DataFrame with ts_event, close, and signal columns
        signal_col: Name of signal column
        **backtest_kwargs: Forwarded to run_backtest

    Returns:
        dict with verdict (PASS/FAIL), pnl_1x, pnl_1_5x, pnl_2x, sharpe_1x
    """
    # Run baseline backtest at 1x costs
    baseline_trades = run_backtest(df, signal_col=signal_col, **backtest_kwargs)
    baseline_trades, baseline_metrics = _compute_eval_metrics(baseline_trades, df)

    pnl_1x = baseline_metrics["net_pnl"]
    sharpe_1x = baseline_metrics["sharpe_ratio"]
    gross_pnl = baseline_metrics["gross_pnl"]
    total_costs = baseline_metrics["total_costs"]

    # Reprice the observed trades under higher friction using the same per-trade
    # adaptive costs when present, or flat costs otherwise.
    pnl_1_5x = gross_pnl - (total_costs * 1.5)
    pnl_2x = gross_pnl - (total_costs * 2.0)

    # PASS if profitable at 1.5x costs
    verdict = "PASS" if pnl_1_5x > 0 else "FAIL"

    return {
        "verdict": verdict,
        "pnl_1x": pnl_1x,
        "pnl_1_5x": float(pnl_1_5x),
        "pnl_2x": float(pnl_2x),
        "sharpe_1x": sharpe_1x,
    }


def decay_test(df: pl.DataFrame, signal_col: str = "signal", n_chunks: int = 4, **backtest_kwargs) -> dict:
    """Test if signal decays over time (data mining artifact detection).

    Args:
        df: DataFrame with ts_event, close, and signal columns
        signal_col: Name of signal column
        n_chunks: Number of sequential time chunks
        **backtest_kwargs: Forwarded to run_backtest

    Returns:
        dict with verdict (PASS/FAIL), chunk_sharpes, is_declining
    """
    # Split data into n_chunks sequential time periods
    total_rows = len(df)
    chunk_size = total_rows // n_chunks

    chunk_sharpes = []
    last_chunk_df = pl.DataFrame()

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else total_rows

        chunk_df = df[start_idx:end_idx]

        if len(chunk_df) == 0:
            continue

        chunk_trades = run_backtest(chunk_df, signal_col=signal_col, **backtest_kwargs)
        _, chunk_metrics = _compute_eval_metrics(chunk_trades, chunk_df)
        chunk_sharpes.append(_sanitize_sharpe(chunk_metrics["sharpe_ratio"]))
        last_chunk_df = chunk_df

    trend_slope, trend_r_squared = _linear_trend(chunk_sharpes)
    midpoint = max(1, len(chunk_sharpes) // 2)
    early_mean = float(np.mean(chunk_sharpes[:midpoint])) if chunk_sharpes else 0.0
    late_mean = float(np.mean(chunk_sharpes[midpoint:])) if len(chunk_sharpes) > midpoint else early_mean
    is_declining = bool(
        trend_slope < 0
        and trend_r_squared >= 0.5
        and late_mean <= (early_mean * 0.9)
    )

    # Check if last chunk is profitable
    if len(last_chunk_df) == 0:
        return {
            "verdict": "FAIL",
            "chunk_sharpes": chunk_sharpes,
            "is_declining": is_declining,
            "trend_slope": trend_slope,
            "trend_r_squared": trend_r_squared,
        }
    last_chunk_trades = run_backtest(last_chunk_df, signal_col=signal_col, **backtest_kwargs)
    _, last_chunk_metrics = _compute_eval_metrics(last_chunk_trades, last_chunk_df)
    last_chunk_profitable = last_chunk_metrics["net_pnl"] > 0

    # PASS if NOT declining AND last chunk is profitable
    verdict = "PASS" if (not is_declining and last_chunk_profitable) else "FAIL"

    return {
        "verdict": verdict,
        "chunk_sharpes": chunk_sharpes,
        "is_declining": is_declining,
        "trend_slope": trend_slope,
        "trend_r_squared": trend_r_squared,
    }


def trade_count_test(df: pl.DataFrame, signal_col: str = "signal", min_trades: int = 50, **backtest_kwargs) -> dict:
    """Verify sufficient sample size for statistical significance.

    Args:
        df: DataFrame with ts_event, close, and signal columns
        signal_col: Name of signal column
        min_trades: Minimum required trade count
        **backtest_kwargs: Forwarded to run_backtest

    Returns:
        dict with verdict (PASS/FAIL), trade_count, min_required
    """
    trades = run_backtest(df, signal_col=signal_col, **backtest_kwargs)
    trade_count = len(trades)

    verdict = "PASS" if trade_count >= min_trades else "FAIL"

    return {
        "verdict": verdict,
        "trade_count": trade_count,
        "min_required": min_trades,
    }


def run_validation_gauntlet(
    df: pl.DataFrame,
    signal_col: str = "signal",
    *,
    min_trades: int = 50,
    **backtest_kwargs,
) -> dict:
    """Run all 7 validators and return combined results.

    Args:
        df: DataFrame with ts_event, close, and signal columns
        signal_col: Name of signal column
        **backtest_kwargs: Forwarded to run_backtest in all sub-tests
            (exit_bars, profit_target, stop_loss, etc.)

    Returns:
        dict with results from all validators and overall verdict
    """
    results = {
        "shuffle": shuffle_test(df, signal_col, **backtest_kwargs),
        "walk_forward": walk_forward_test(df, signal_col, **backtest_kwargs),
        "regime": regime_test(df, signal_col, **backtest_kwargs),
        "signal_perturbation": signal_perturbation_test(df, signal_col, **backtest_kwargs),
        "cost_sensitivity": cost_sensitivity_test(df, signal_col, **backtest_kwargs),
        "decay": decay_test(df, signal_col, **backtest_kwargs),
        "trade_count": trade_count_test(df, signal_col, min_trades=min_trades, **backtest_kwargs),
    }

    # Compute overall verdict
    pass_count = sum(1 for r in results.values() if isinstance(r, dict) and r.get("verdict") == "PASS")
    overall_verdict = "PASS" if pass_count == 7 else "FAIL"

    results["overall_verdict"] = overall_verdict
    results["pass_count"] = pass_count
    results["total_tests"] = 7

    return results
