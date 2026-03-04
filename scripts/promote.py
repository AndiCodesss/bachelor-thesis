#!/usr/bin/env python3
"""Candidate promotion entrypoint with walk-forward-only validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import inspect
from pathlib import Path
import re
import sys
from typing import Any, Callable

import numpy as np
import polars as pl

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.lib.atomic_io import atomic_json_write
from research.lib.candidates import load_candidate
from research.lib.promotion import verify_candidate_artifacts
from research.signals import check_signal_causality
from research.lib.trial_counter import estimate_effective_trials
from research.ml.promotion_gates import WalkForwardFold, WalkForwardValidator, evaluate_promotion_gates
from src.framework import __version__ as framework_version
from src.framework.backtest.engine import TRADE_SCHEMA, run_backtest
from src.framework.backtest.metrics import compute_metrics
from src.framework.data.constants import RESULTS_DIR, TICK_SIZE, TICK_VALUE, TOTAL_COST_RT
from src.framework.data.loader import ExecutionMode, get_parquet_files, set_execution_mode
from src.framework.features_canonical.builder import LABEL_COLUMNS, load_cached_matrix
from src.framework.security.framework_lock import verify_manifest


_MONTH_RE = re.compile(r"nq_(\d{4}-\d{2})-\d{2}\.parquet$")
_CAUSALITY_MIN_ROWS = 33


@dataclass(frozen=True)
class SignalRuntime:
    generate_fn: Callable[..., Any]
    generate_accepts_state: bool
    fit_fn: Callable[..., Any] | None
    fit_on_files_fn: Callable[..., Any] | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _resolve_path(path_raw: str, project_root: Path) -> Path:
    p = Path(path_raw)
    return p if p.is_absolute() else (project_root / p)


def _extract_month(file_path: Path) -> str:
    match = _MONTH_RE.search(file_path.name)
    if match is None:
        raise ValueError(f"Cannot parse month from parquet filename: {file_path.name}")
    return match.group(1)


def _annualized_sharpe(values: list[float], annualization: float = 252.0) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float((float(np.mean(arr)) / std) * np.sqrt(float(annualization)))


def _accepts_at_least_n_positional(fn: Callable[..., Any], n: int) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    positional = 0
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
    return positional >= n


def _load_signal_runtime(signal_path: Path) -> SignalRuntime:
    spec = importlib.util.spec_from_file_location(f"candidate_signal_{signal_path.stem}", signal_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {signal_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generate_fn = getattr(module, "generate_signal", None)
    if not callable(generate_fn):
        raise ValueError(f"Signal module must expose generate_signal(df, params): {signal_path}")

    fit_fn = getattr(module, "fit_signal", None)
    if fit_fn is not None and not callable(fit_fn):
        fit_fn = None

    fit_on_files_fn = getattr(module, "fit_signal_on_files", None)
    if fit_on_files_fn is not None and not callable(fit_on_files_fn):
        fit_on_files_fn = None

    return SignalRuntime(
        generate_fn=generate_fn,
        generate_accepts_state=_accepts_at_least_n_positional(generate_fn, 3),
        fit_fn=fit_fn,
        fit_on_files_fn=fit_on_files_fn,
    )


def _extract_candidate_params(candidate: dict[str, Any]) -> dict[str, Any]:
    for key in ("parameters", "params", "signal_params"):
        value = candidate.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _parse_bar_config(candidate: dict[str, Any]) -> dict[str, Any]:
    bar_raw = str(candidate.get("bar_type") or candidate.get("bar_config") or "time_5m").strip().lower()
    session_filter = str(candidate.get("session_filter", "rth")).strip().lower()
    if session_filter not in {"rth", "eth"}:
        session_filter = "rth"

    if bar_raw.startswith("eth_"):
        session_filter = "eth"
        bar_raw = bar_raw[4:]

    if bar_raw.startswith("tick_"):
        return {
            "bar_type": "tick",
            "bar_size": "5m",
            "bar_threshold": int(bar_raw.split("_", 1)[1]),
            "session_filter": session_filter,
        }

    if bar_raw.startswith("vol_"):
        return {
            "bar_type": "volume",
            "bar_size": "5m",
            "bar_threshold": int(bar_raw.split("_", 1)[1]),
            "session_filter": session_filter,
        }

    if bar_raw.startswith("volume_"):
        return {
            "bar_type": "volume",
            "bar_size": "5m",
            "bar_threshold": int(bar_raw.split("_", 1)[1]),
            "session_filter": session_filter,
        }

    if bar_raw.startswith("time_"):
        bar_size = bar_raw.split("_", 1)[1]
    elif bar_raw.endswith("m") or bar_raw.endswith("h"):
        bar_size = bar_raw
    elif bar_raw == "time":
        bar_size = str(candidate.get("bar_size", "5m")).strip()
    else:
        bar_size = "5m"

    return {
        "bar_type": "time",
        "bar_size": bar_size,
        "bar_threshold": None,
        "session_filter": session_filter,
    }


def _extract_backtest_kwargs(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    defaults = {
        "max_daily_loss": 1000.0,
        "entry_on_next_open": True,
    }

    candidate_bt = candidate.get("backtest") if isinstance(candidate.get("backtest"), dict) else {}
    execution_bt = candidate.get("execution_params") if isinstance(candidate.get("execution_params"), dict) else {}

    bt: dict[str, Any] = dict(defaults)
    bt.update({k: v for k, v in execution_bt.items() if v is not None})
    bt.update({k: v for k, v in candidate_bt.items() if v is not None})

    if args.entry_on_next_open is not None:
        bt["entry_on_next_open"] = bool(args.entry_on_next_open)

    if args.max_daily_loss is not None:
        bt["max_daily_loss"] = float(args.max_daily_loss)

    allowed = {
        "max_daily_loss",
        "exit_bars",
        "profit_target",
        "stop_loss",
        "profit_target_return",
        "stop_loss_return",
        "entry_on_next_open",
    }

    normalized = {k: bt[k] for k in allowed if k in bt and bt[k] is not None}
    normalized["max_daily_loss"] = float(normalized.get("max_daily_loss", 1000.0))
    normalized["entry_on_next_open"] = bool(normalized.get("entry_on_next_open", True))
    return normalized


def _collect_all_files() -> list[Path]:
    files: dict[str, Path] = {}
    for split in ("train", "validate", "test"):
        for path in get_parquet_files(split):
            files[str(path.resolve())] = path.resolve()
    return sorted(files.values())


def _split_lockbox_files(all_files: list[Path], lockbox_months: int) -> tuple[list[Path], list[Path], list[str]]:
    if lockbox_months <= 0:
        return list(all_files), [], []

    months = sorted({_extract_month(f) for f in all_files})
    if len(months) <= lockbox_months:
        raise ValueError(
            f"lockbox_months={lockbox_months} leaves no data for walk-forward (months={len(months)})",
        )

    lockbox_set = set(months[-lockbox_months:])
    wfa_files = [f for f in all_files if _extract_month(f) not in lockbox_set]
    lockbox_files = [f for f in all_files if _extract_month(f) in lockbox_set]
    if not wfa_files:
        raise ValueError("No files remain for walk-forward after lockbox split")
    return wfa_files, lockbox_files, sorted(lockbox_set)


def _daily_returns_from_trades(trades: pl.DataFrame, *, cost_multiplier: float = 1.0) -> list[float]:
    if len(trades) == 0:
        return []

    pnl_df = trades.with_columns(
        (
            ((pl.col("exit_price") - pl.col("entry_price")) * pl.col("direction") / TICK_SIZE * TICK_VALUE)
            * pl.col("size")
            - (pl.col("size") * (TOTAL_COST_RT * float(cost_multiplier)))
        ).alias("_net_pnl"),
        pl.col("exit_time").dt.date().alias("_date"),
    )

    daily = pnl_df.group_by("_date").agg(pl.col("_net_pnl").sum()).sort("_date")
    if len(daily) == 0:
        return []

    # Match backtest metrics logic: include non-trading weekdays as 0 PnL.
    min_date = daily["_date"].min()
    max_date = daily["_date"].max()
    all_bdays = pl.DataFrame({
        "_date": pl.date_range(min_date, max_date, "1d", eager=True),
    }).filter(pl.col("_date").dt.weekday() <= 5)
    # Include actual trade dates (covers weekend trades in tests/synthetic data).
    all_dates = pl.concat([
        all_bdays.select("_date"),
        daily.select("_date"),
    ]).unique().sort("_date")
    daily = all_dates.join(daily, on="_date", how="left").fill_null(0.0)

    return [float(v) for v in daily["_net_pnl"].to_list()]


def _daily_trade_counts_from_trades(trades: pl.DataFrame) -> list[int]:
    if len(trades) == 0:
        return []

    daily = (
        trades
        .with_columns(pl.col("exit_time").dt.date().alias("_date"))
        .group_by("_date")
        .agg(pl.len().alias("_n"))
        .sort("_date")
    )
    if len(daily) == 0:
        return []

    # Match _daily_returns_from_trades date grid so density metrics are consistent.
    min_date = daily["_date"].min()
    max_date = daily["_date"].max()
    all_bdays = pl.DataFrame({
        "_date": pl.date_range(min_date, max_date, "1d", eager=True),
    }).filter(pl.col("_date").dt.weekday() <= 5)
    all_dates = pl.concat([
        all_bdays.select("_date"),
        daily.select("_date"),
    ]).unique().sort("_date")
    daily = all_dates.join(daily, on="_date", how="left").fill_null(0)

    return [int(v) for v in daily["_n"].to_list()]


def _concat_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical_relaxed")


def _fit_signal_state(
    runtime: SignalRuntime,
    *,
    train_files: tuple[Path, ...] | list[Path],
    params: dict[str, Any],
    load_bars: Callable[[Path], pl.DataFrame],
) -> Any | None:
    train_file_list = [Path(p) for p in train_files]

    if runtime.fit_on_files_fn is not None:
        fn = runtime.fit_on_files_fn
        try:
            return fn(train_files=train_file_list, load_bars=load_bars, params=params)
        except TypeError:
            try:
                return fn(train_file_list, load_bars, params)
            except TypeError:
                return fn(train_file_list, params)

    if runtime.fit_fn is None:
        return None

    fn = runtime.fit_fn
    sig = inspect.signature(fn)
    params_map = sig.parameters

    frames_cache: list[pl.DataFrame] | None = None
    train_df_cache: pl.DataFrame | None = None

    def _train_frames() -> list[pl.DataFrame]:
        nonlocal frames_cache
        if frames_cache is None:
            frames_cache = []
            for fp in train_file_list:
                bars = load_bars(fp)
                if len(bars) > 0:
                    frames_cache.append(bars)
        return frames_cache

    def _train_df() -> pl.DataFrame:
        nonlocal train_df_cache
        if train_df_cache is None:
            train_df_cache = _concat_frames(_train_frames())
        return train_df_cache

    if "train_files" in params_map:
        kwargs: dict[str, Any] = {"train_files": train_file_list}
        if "load_bars" in params_map:
            kwargs["load_bars"] = load_bars
        if "params" in params_map:
            kwargs["params"] = params
        return fn(**kwargs)

    if "train_frames" in params_map:
        kwargs = {"train_frames": _train_frames()}
        if "params" in params_map:
            kwargs["params"] = params
        return fn(**kwargs)

    if "train_df" in params_map or "df" in params_map:
        kwargs = {}
        if "train_df" in params_map:
            kwargs["train_df"] = _train_df()
        if "df" in params_map:
            kwargs["df"] = _train_df()
        if "params" in params_map:
            kwargs["params"] = params
        return fn(**kwargs)

    train_df = _train_df()
    if _accepts_at_least_n_positional(fn, 2):
        return fn(train_df, params)
    if _accepts_at_least_n_positional(fn, 1):
        return fn(train_df)
    return None


def _generate_signal_array(
    runtime: SignalRuntime,
    bars: pl.DataFrame,
    params: dict[str, Any],
    model_state: Any | None,
) -> np.ndarray:
    safe_bars = _safe_signal_bars(bars)
    if runtime.generate_accepts_state:
        raw = runtime.generate_fn(safe_bars, params, model_state)
    else:
        raw = runtime.generate_fn(safe_bars, params)

    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if len(arr) != len(bars):
        raise ValueError(f"Signal length mismatch: expected {len(bars)}, got {len(arr)}")

    return np.sign(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.int8)


def _safe_signal_bars(bars: pl.DataFrame) -> pl.DataFrame:
    """Remove forward-looking label columns before calling strategy code."""
    _label_cols_present = [c for c in LABEL_COLUMNS if c in bars.columns]
    return bars.drop(_label_cols_present) if _label_cols_present else bars


def _run_signal_on_files(
    *,
    files: tuple[Path, ...] | list[Path],
    runtime: SignalRuntime,
    signal_params: dict[str, Any],
    model_state: Any | None,
    load_bars: Callable[[Path], pl.DataFrame],
    backtest_kwargs: dict[str, Any],
) -> dict[str, Any]:
    trade_frames: list[pl.DataFrame] = []
    causality_checked = False
    causality_frames: list[pl.DataFrame] = []
    causality_row_count = 0

    for file_path in files:
        bars = load_bars(Path(file_path))
        if len(bars) == 0:
            continue

        # Causality check (prefix invariance) once per evaluation run on >=33 bars.
        # Buffers short files so the check still runs for coarse bar sizes.
        if not causality_checked:
            causality_frames.append(_safe_signal_bars(bars))
            causality_row_count += len(bars)
            if causality_row_count >= _CAUSALITY_MIN_ROWS:
                causality_df = pl.concat(causality_frames).sort("ts_event")
                causality_errors = check_signal_causality(
                    generate_fn=runtime.generate_fn,
                    df=causality_df,
                    params=signal_params,
                    accepts_state=runtime.generate_accepts_state,
                    model_state=model_state,
                    mode="sign",
                )
                if causality_errors:
                    raise ValueError(f"signal causality failed: {causality_errors}")
                causality_checked = True
                causality_frames.clear()

        signal = _generate_signal_array(runtime, bars, signal_params, model_state)
        bars_with_signal = bars.with_columns(pl.Series("signal", signal).cast(pl.Int8))
        trades = run_backtest(bars_with_signal, signal_col="signal", **backtest_kwargs)
        if len(trades) > 0:
            trade_frames.append(trades)

    if causality_row_count > 0 and not causality_checked:
        raise ValueError(
            f"signal causality check requires at least "
            f"{_CAUSALITY_MIN_ROWS} bars, got {causality_row_count}"
        )

    trades_df = _concat_frames(trade_frames) if trade_frames else pl.DataFrame(schema=TRADE_SCHEMA)
    metrics = compute_metrics(
        trades_df,
        cost_override_col="adaptive_cost_rt" if "adaptive_cost_rt" in trades_df.columns else None,
    )

    daily_returns = _daily_returns_from_trades(trades_df, cost_multiplier=1.0)
    daily_returns_2x = _daily_returns_from_trades(trades_df, cost_multiplier=2.0)
    daily_trade_counts = _daily_trade_counts_from_trades(trades_df)

    return {
        "trades": trades_df,
        "trade_count": int(metrics.get("trade_count", 0)),
        "net_pnl": float(metrics.get("net_pnl", 0.0)),
        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
        "daily_returns": daily_returns,
        "daily_returns_2x": daily_returns_2x,
        "daily_trade_counts": daily_trade_counts,
        "avg_trades_per_day": float(np.mean(daily_trade_counts)) if daily_trade_counts else 0.0,
        "max_trades_per_day": int(max(daily_trade_counts)) if daily_trade_counts else 0,
        "sharpe_2x_cost": float(_annualized_sharpe(daily_returns_2x)),
    }


def _evaluate_candidate_walk_forward(
    *,
    candidate: dict[str, Any],
    candidate_path: Path,
    project_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    artifacts = candidate.get("artifacts") if isinstance(candidate.get("artifacts"), dict) else {}
    signal_path_raw = artifacts.get("signal_file") or candidate.get("signal_module")
    if not signal_path_raw:
        raise ValueError(
            "Candidate must provide signal module via artifacts.signal_file or signal_module",
        )

    signal_path = _resolve_path(str(signal_path_raw), project_root).resolve()
    if not signal_path.exists():
        raise FileNotFoundError(f"Signal file not found: {signal_path}")

    runtime = _load_signal_runtime(signal_path)
    signal_params = _extract_candidate_params(candidate)
    bar_cfg = _parse_bar_config(candidate)
    backtest_kwargs = _extract_backtest_kwargs(candidate, args)

    all_files = _collect_all_files()
    wfa_files, lockbox_files, lockbox_months = _split_lockbox_files(all_files, int(args.lockbox_months))

    def _load_bars(file_path: Path) -> pl.DataFrame:
        return load_cached_matrix(
            file_path,
            bar_size=bar_cfg["bar_size"],
            bar_type=bar_cfg["bar_type"],
            bar_threshold=bar_cfg["bar_threshold"],
            include_bar_columns=True,
            session_filter=bar_cfg["session_filter"],
        )

    validator = WalkForwardValidator(
        wfa_files,
        train_months=int(args.train_months),
        test_months=int(args.test_months),
        step_months=int(args.step_months),
        embargo_months=int(args.embargo_months),
        purge_days=int(args.purge_days),
        allow_partial_last_test=bool(args.allow_partial_last_test),
    )

    trial_stats = estimate_effective_trials(Path(args.experiments_log))
    inferred_raw = max(1, int(trial_stats.get("raw_trials", 0)))
    raw_trials = int(args.n_trials) if args.n_trials is not None else inferred_raw

    inferred_eff = int(trial_stats.get("effective_trials", 1))
    raw_from_log = max(1, int(trial_stats.get("raw_trials", 1)))
    scaled_eff = int(round(float(inferred_eff) * (float(raw_trials) / float(raw_from_log))))
    scaled_eff = max(1, min(scaled_eff, raw_trials))
    effective_trials = int(args.n_effective_trials) if args.n_effective_trials is not None else scaled_eff
    effective_trials = max(1, min(effective_trials, raw_trials))

    combined_daily_returns_2x: list[float] = []
    combined_daily_trade_counts: list[int] = []

    def _fold_eval(fold: WalkForwardFold, params: dict[str, Any]) -> dict[str, Any]:
        model_state = _fit_signal_state(
            runtime,
            train_files=fold.train_files,
            params=params,
            load_bars=_load_bars,
        )

        fold_eval = _run_signal_on_files(
            files=fold.test_files,
            runtime=runtime,
            signal_params=params,
            model_state=model_state,
            load_bars=_load_bars,
            backtest_kwargs=backtest_kwargs,
        )

        combined_daily_returns_2x.extend([float(v) for v in fold_eval["daily_returns_2x"]])
        combined_daily_trade_counts.extend([int(v) for v in fold_eval["daily_trade_counts"]])

        return {
            "trade_count": int(fold_eval["trade_count"]),
            "net_pnl": float(fold_eval["net_pnl"]),
            "sharpe_ratio": float(fold_eval["sharpe_ratio"]),
            "daily_returns": [float(v) for v in fold_eval["daily_returns"]],
        }

    wfa_result = validator.evaluate(
        _fold_eval,
        signal_params,
        raw_trial_count=raw_trials,
        effective_trial_count=effective_trials,
    )

    lockbox_metrics: dict[str, Any] | None = None
    if lockbox_files:
        lockbox_state = _fit_signal_state(
            runtime,
            train_files=tuple(wfa_files),
            params=signal_params,
            load_bars=_load_bars,
        )
        lockbox_eval = _run_signal_on_files(
            files=tuple(lockbox_files),
            runtime=runtime,
            signal_params=signal_params,
            model_state=lockbox_state,
            load_bars=_load_bars,
            backtest_kwargs=backtest_kwargs,
        )
        lockbox_metrics = {
            "months": lockbox_months,
            "trade_count": int(lockbox_eval["trade_count"]),
            "net_pnl": float(lockbox_eval["net_pnl"]),
            "sharpe": float(lockbox_eval["sharpe_ratio"]),
            "sharpe_2x_cost": float(lockbox_eval["sharpe_2x_cost"]),
            "avg_trades_per_day": float(lockbox_eval["avg_trades_per_day"]),
            "max_trades_per_day": int(lockbox_eval["max_trades_per_day"]),
        }

    thresholds = {
        "min_fold_count": int(args.min_fold_count),
        "min_positive_fold_ratio": float(args.min_positive_fold_ratio),
        "min_aggregate_sharpe": float(args.min_aggregate_sharpe),
        "min_deflated_sharpe": float(args.min_deflated_sharpe),
        "min_dsr_probability": float(args.min_dsr_probability),
        "min_lockbox_sharpe": float(args.min_lockbox_sharpe),
        "min_lockbox_net_pnl": float(args.min_lockbox_net_pnl),
        "min_lockbox_trade_count": int(args.min_lockbox_trade_count),
    }

    gates = evaluate_promotion_gates(
        wfa_result=wfa_result,
        thresholds=thresholds,
        lockbox_metrics=lockbox_metrics,
    )

    aggregate_sharpe_2x = float(_annualized_sharpe(combined_daily_returns_2x))
    avg_trades_per_day = float(np.mean(combined_daily_trade_counts)) if combined_daily_trade_counts else 0.0
    max_trades_per_day = int(max(combined_daily_trade_counts)) if combined_daily_trade_counts else 0

    gates["checks"]["min_aggregate_sharpe_2x_cost"] = {
        "passed": aggregate_sharpe_2x >= float(args.min_aggregate_sharpe_2x_cost),
        "value": aggregate_sharpe_2x,
        "min_required": float(args.min_aggregate_sharpe_2x_cost),
    }
    gates["checks"]["max_avg_trades_per_day"] = {
        "passed": avg_trades_per_day <= float(args.max_avg_trades_per_day),
        "value": avg_trades_per_day,
        "max_allowed": float(args.max_avg_trades_per_day),
    }
    gates["checks"]["max_trades_per_day"] = {
        "passed": max_trades_per_day <= int(args.max_trades_per_day),
        "value": max_trades_per_day,
        "max_allowed": int(args.max_trades_per_day),
    }

    gates["metrics"]["aggregate_sharpe_2x_cost"] = aggregate_sharpe_2x
    gates["metrics"]["avg_trades_per_day"] = avg_trades_per_day
    gates["metrics"]["max_trades_per_day"] = max_trades_per_day
    gates["passed"] = all(bool(c.get("passed", False)) for c in gates["checks"].values())

    return {
        "runner": "walk_forward",
        "candidate_path": str(candidate_path),
        "signal_file": str(signal_path),
        "signal_params": signal_params,
        "signal_contract": {
            "generate_accepts_state": bool(runtime.generate_accepts_state),
            "fit_signal_available": bool(runtime.fit_fn is not None),
            "fit_signal_on_files_available": bool(runtime.fit_on_files_fn is not None),
        },
        "bar_config": bar_cfg,
        "backtest_config": backtest_kwargs,
        "wfa_config": {
            "train_months": int(args.train_months),
            "test_months": int(args.test_months),
            "step_months": int(args.step_months),
            "embargo_months": int(args.embargo_months),
            "purge_days": int(args.purge_days),
            "allow_partial_last_test": bool(args.allow_partial_last_test),
            "lockbox_months": int(args.lockbox_months),
            "lockbox_month_labels": lockbox_months,
        },
        "trial_counts": {
            "raw_trial_count": int(raw_trials),
            "effective_trial_count": int(effective_trials),
            "estimation": trial_stats,
        },
        "thresholds": {
            **thresholds,
            "min_aggregate_sharpe_2x_cost": float(args.min_aggregate_sharpe_2x_cost),
            "max_avg_trades_per_day": float(args.max_avg_trades_per_day),
            "max_trades_per_day": int(args.max_trades_per_day),
        },
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote candidate with walk-forward validation gates.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--framework-lock-manifest", default="configs/framework_lock.json")
    parser.add_argument("--framework-lock-mode", choices=["warn", "error"], default="error")
    parser.add_argument("--out", default=None, help="Optional output path for promotion report JSON.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify candidate artifacts/provenance; skip walk-forward evaluation.",
    )

    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--step-months", type=int, default=2)
    parser.add_argument("--embargo-months", type=int, default=0)
    parser.add_argument("--purge-days", type=int, default=0)
    parser.add_argument(
        "--allow-partial-last-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lockbox-months", type=int, default=1)

    parser.add_argument("--min-fold-count", type=int, default=3)
    parser.add_argument("--min-positive-fold-ratio", type=float, default=0.60)
    parser.add_argument("--min-aggregate-sharpe", type=float, default=0.50)
    parser.add_argument("--min-deflated-sharpe", type=float, default=0.0)
    parser.add_argument("--min-dsr-probability", type=float, default=0.95)
    parser.add_argument("--min-lockbox-sharpe", type=float, default=0.0)
    parser.add_argument("--min-lockbox-net-pnl", type=float, default=0.0)
    parser.add_argument("--min-lockbox-trade-count", type=int, default=20)
    parser.add_argument("--min-aggregate-sharpe-2x-cost", type=float, default=0.0)
    parser.add_argument("--max-avg-trades-per-day", type=float, default=60.0)
    parser.add_argument("--max-trades-per-day", type=int, default=150)

    parser.add_argument("--experiments-log", type=Path, default=Path("results/logs/research_experiments.jsonl"))
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-effective-trials", type=int, default=None)

    parser.add_argument("--max-daily-loss", type=float, default=None)
    parser.add_argument(
        "--entry-on-next-open",
        dest="entry_on_next_open",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--entry-on-close",
        dest="entry_on_next_open",
        action="store_false",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    set_execution_mode(ExecutionMode.PROMOTION)

    manifest = Path(args.framework_lock_manifest)
    if not manifest.is_absolute():
        manifest = (project_root / manifest).resolve()

    lock_report = verify_manifest(manifest_path=manifest, project_root=project_root)
    if not lock_report["ok"]:
        print(
            "Framework lock verification failed: "
            f"{lock_report['verified_file_count']}/{lock_report['manifest_file_count']}",
            file=sys.stderr,
        )
        if args.framework_lock_mode == "error":
            raise SystemExit(2)

    candidate_path = args.candidate.resolve()
    candidate = load_candidate(candidate_path)
    artifact_report = verify_candidate_artifacts(
        candidate=candidate,
        project_root=project_root,
        framework_manifest_path=manifest,
    )

    evaluation: dict[str, Any] | None = None
    if not args.verify_only:
        evaluation = _evaluate_candidate_walk_forward(
            candidate=candidate,
            candidate_path=candidate_path,
            project_root=project_root,
            args=args,
        )

    strategy_id = str(candidate.get("strategy_id", "unknown_strategy"))
    run_id = f"promotion_{_utc_slug()}_{strategy_id}"
    default_out = (RESULTS_DIR / "runs" / f"{run_id}.json").resolve()
    out = Path(args.out).resolve() if args.out else default_out

    report = {
        "run_id": run_id,
        "timestamp": _utc_now(),
        "framework_version": framework_version,
        "candidate_path": str(candidate_path),
        "framework_lock_manifest": str(manifest),
        "framework_lock": lock_report,
        "artifact_verification": artifact_report,
        "evaluation": evaluation,
        "status": "VERIFIED_ONLY" if evaluation is None else (
            "PASS"
            if bool(evaluation.get("gates", {}).get("passed", False))
            else "FAIL"
        ),
    }
    atomic_json_write(out, report)
    print(f"Promotion report: {out}")

    if report["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
