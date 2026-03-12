"""Validator split evaluation and task execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from research.lib.mission_splits import ALLOWED_RESEARCH_SPLITS, resolve_research_splits
from src.framework.backtest.metrics import compute_daily_pnl_series


@dataclass(frozen=True)
class ValidatorExecutionDeps:
    atomic_json_write: Callable[[Path, Any], None]
    check_signal_causality: Callable[..., list[str]]
    compute_adaptive_costs: Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]
    compute_event_id: Callable[..., str]
    compute_metrics: Callable[..., dict[str, Any]]
    compute_strategy_id: Callable[..., str]
    deflated_sharpe_ratio: Callable[..., dict[str, Any]]
    estimate_effective_trials: Callable[[Path], dict[str, Any]]
    factor_attribution: Callable[[pl.DataFrame, pl.DataFrame], dict[str, Any]]
    filter_strategy_inputs: Callable[[pl.DataFrame, str], pl.DataFrame]
    fit_alpha_decay: Callable[[pl.DataFrame], dict[str, Any]]
    get_split_files: Callable[[str], list[Path]]
    load_cached_matrix: Callable[..., pl.DataFrame]
    load_signal_module: Callable[[str], Any]
    log_experiment: Callable[..., None]
    normalize_feature_group: Callable[..., str]
    normalize_edge_surface_config: Callable[[Any], dict[str, Any]]
    normalize_session_filter: Callable[..., str]
    parse_bar_config: Callable[[str], dict[str, Any]]
    run_backtest: Callable[..., pl.DataFrame]
    run_edge_surface: Callable[..., dict[str, Any]]
    run_validation_gauntlet: Callable[..., dict[str, Any]]
    sha256_file: Callable[[Path], str]
    task_setup_identity: Callable[[dict[str, Any]], tuple[str, str]]
    validate_signal_array: Callable[[np.ndarray, int], list[str]]
    write_candidate: Callable[..., Path]
    causality_min_prefix_bars: int
    causality_min_rows: int
    edge_surface_analysis_columns: Sequence[str]
    trade_schema: Any


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def empty_signal_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_event": pl.Series([], dtype=pl.Datetime("ns", "UTC")),
            "close": pl.Series([], dtype=pl.Float64),
            "signal": pl.Series([], dtype=pl.Int8),
        }
    )


def resolve_strategy_callable(
    *,
    task_id: str,
    strategy_name: str,
    strategy_module: Any,
) -> tuple[Callable[..., Any], bool]:
    strategy_fn = getattr(strategy_module, "generate_signal", None)
    if not callable(strategy_fn):
        strategy_fn = getattr(strategy_module, "signal", None)
    if not callable(strategy_fn):
        raise ValueError(
            f"{task_id}: strategy '{strategy_name}' has no callable "
            "generate_signal(df, params[, model_state]) or signal(...)",
        )

    sig = inspect.signature(strategy_fn)
    params = list(sig.parameters.values())
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    if len(positional) < 2 and not has_varargs:
        raise ValueError(
            f"{task_id}: strategy '{strategy_name}' must accept at least "
            "df and params positional arguments",
        )
    accepts_state = len(positional) >= 3 or has_varargs
    return strategy_fn, accepts_state


def invoke_strategy_callable(
    *,
    strategy_fn: Callable[..., Any],
    strategy_df: pl.DataFrame,
    params: dict[str, Any],
    accepts_state: bool,
    model_state: Any | None,
) -> np.ndarray:
    if accepts_state:
        return np.asarray(strategy_fn(strategy_df, params, model_state))
    return np.asarray(strategy_fn(strategy_df, params))


def task_backtest_params(task: dict[str, Any], mission: dict[str, Any]) -> dict[str, Any]:
    mission_bt = mission.get("backtest", {})
    if not isinstance(mission_bt, dict):
        mission_bt = {}

    exit_bars_raw = task.get("exit_bars", mission_bt.get("exit_bars"))
    profit_target_raw = task.get("profit_target", mission_bt.get("profit_target"))
    stop_loss_raw = task.get("stop_loss", mission_bt.get("stop_loss"))

    return {
        "entry_on_next_open": bool(
            task.get("entry_on_next_open", mission_bt.get("entry_on_next_open", True))
        ),
        "max_daily_loss": as_float(
            task.get("max_daily_loss", mission_bt.get("max_daily_loss", 1000.0)),
            1000.0,
        ),
        "exit_bars": int(exit_bars_raw) if exit_bars_raw is not None else None,
        "profit_target": float(profit_target_raw) if profit_target_raw is not None else None,
        "stop_loss": float(stop_loss_raw) if stop_loss_raw is not None else None,
    }


def task_split_plan(task: dict[str, Any], mission: dict[str, Any]) -> dict[str, str | None]:
    plan = resolve_research_splits(mission)
    search_split = str(task.get("search_split") or task.get("split") or plan["search_split"]).strip().lower()
    if search_split not in ALLOWED_RESEARCH_SPLITS:
        allowed = ", ".join(sorted(ALLOWED_RESEARCH_SPLITS))
        raise ValueError(f"unsupported split '{search_split}'. Allowed: {allowed}")

    selection_raw = task.get("selection_split", plan["selection_split"])
    selection_split = str(selection_raw).strip().lower() if selection_raw is not None else None
    if selection_split:
        if selection_split not in ALLOWED_RESEARCH_SPLITS:
            allowed = ", ".join(sorted(ALLOWED_RESEARCH_SPLITS))
            raise ValueError(f"unsupported selection split '{selection_split}'. Allowed: {allowed}")
        if selection_split == search_split:
            if task.get("selection_split") is None and task.get("split") is not None:
                selection_split = None
            else:
                raise ValueError("selection_split must differ from search_split")
    else:
        selection_split = None

    return {
        "search_split": search_split,
        "selection_split": selection_split,
        "feedback_split": search_split,
        "promotion_split": plan["promotion_split"],
    }


def selection_gate_config(
    mission: dict[str, Any],
    *,
    base_run_gauntlet: bool,
) -> dict[str, Any]:
    cfg = mission.get("selection_gate")
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "target_sharpe": float(cfg.get("target_sharpe", mission.get("target_sharpe", 0.0))),
        "min_trade_count": int(cfg.get("min_trade_count", mission.get("min_trade_count", 1))),
        "run_gauntlet": bool(cfg.get("require_gauntlet", base_run_gauntlet)),
    }


def daily_net_pnl_series(trades: pl.DataFrame) -> np.ndarray:
    if len(trades) == 0:
        return np.array([], dtype=np.float64)
    cost_col = "adaptive_cost_rt" if "adaptive_cost_rt" in trades.columns else None
    daily = compute_daily_pnl_series(trades, cost_override_col=cost_col)
    return daily["net_pnl"].to_numpy().astype(np.float64)


def evaluate_advanced_validation_gates(
    mission: dict[str, Any],
    advanced_validation: dict[str, Any],
) -> dict[str, Any]:
    cfg = mission.get("advanced_validation")
    if not isinstance(cfg, dict):
        return {"enabled": False, "passed": True, "checks": {}}

    checks: dict[str, Any] = {}

    min_dsr = cfg.get("min_dsr_probability")
    if min_dsr is not None:
        payload = advanced_validation.get("deflated_sharpe")
        available = isinstance(payload, dict) and bool(payload.get("available", False))
        value = float(payload.get("dsr", 0.0)) if available and isinstance(payload, dict) else 0.0
        checks["min_dsr_probability"] = {
            "passed": available and value >= float(min_dsr),
            "available": available,
            "value": value,
            "min_required": float(min_dsr),
        }

    allowed_decay = cfg.get("allowed_alpha_decay_verdicts")
    if isinstance(allowed_decay, list):
        payload = advanced_validation.get("alpha_decay")
        verdict = str(payload.get("verdict", "")).strip() if isinstance(payload, dict) else ""
        allowed = [str(v).strip() for v in allowed_decay if str(v).strip()]
        checks["alpha_decay_verdict"] = {
            "passed": verdict in allowed,
            "available": bool(verdict),
            "value": verdict,
            "allowed": allowed,
        }

    allowed_factor = cfg.get("allowed_factor_verdicts")
    if isinstance(allowed_factor, list):
        payload = advanced_validation.get("factor_attribution")
        verdict = str(payload.get("verdict", "")).strip() if isinstance(payload, dict) else ""
        allowed = [str(v).strip() for v in allowed_factor if str(v).strip()]
        checks["factor_verdict"] = {
            "passed": verdict in allowed,
            "available": bool(verdict),
            "value": verdict,
            "allowed": allowed,
        }

    return {
        "enabled": bool(checks),
        "passed": all(bool(check.get("passed", False)) for check in checks.values()) if checks else True,
        "checks": checks,
    }


def evaluate_strategy_split(
    *,
    deps: ValidatorExecutionDeps,
    task_id: str,
    split_label: str,
    split: str,
    bar_config: str,
    parsed_bar: dict[str, Any],
    strategy_fn: Callable[..., Any],
    strategy_accepts_state: bool,
    params: dict[str, Any],
    session_filter: str,
    feature_group: str,
    bt_kwargs: dict[str, Any],
    target_sharpe: float,
    min_trade_count: int,
    run_gauntlet: bool,
    max_files: int | None,
    mission: dict[str, Any],
    experiments_path: Path,
    task_dir: Path,
    edge_surface_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = deps.get_split_files(split)
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise ValueError(f"{task_id}: no data files for split={split}")

    split_dir = task_dir / split_label
    split_dir.mkdir(parents=True, exist_ok=True)

    all_signals: list[pl.DataFrame] = []
    all_bars: list[pl.DataFrame] = []
    execution_frames: list[pl.DataFrame] = []
    analysis_frames: list[pl.DataFrame] = []
    bars_processed = 0
    signal_count = 0
    causality_checked = False
    causality_frames: list[pl.DataFrame] = []
    causality_row_count = 0
    strategy_state: Any | None = {} if strategy_accepts_state else None

    for file_path in files:
        full_df = deps.load_cached_matrix(
            file_path,
            bar_size=parsed_bar["bar_size"],
            bar_type=parsed_bar["bar_type"],
            bar_threshold=parsed_bar["bar_threshold"],
            include_bar_columns=True,
            session_filter=session_filter,
        )
        if len(full_df) == 0:
            continue

        strategy_df = deps.filter_strategy_inputs(full_df, feature_group)

        if not causality_checked:
            causality_frames.append(strategy_df)
            causality_row_count += len(strategy_df)
            if causality_row_count >= deps.causality_min_rows:
                causality_df = pl.concat(causality_frames).sort("ts_event")
                causality_errors = deps.check_signal_causality(
                    generate_fn=strategy_fn,
                    df=causality_df,
                    params=params,
                    accepts_state=strategy_accepts_state,
                    model_state={} if strategy_accepts_state else None,
                    mode="strict",
                    min_prefix_bars=deps.causality_min_prefix_bars,
                )
                if causality_errors:
                    raise ValueError(
                        f"{task_id}: signal causality failed on {split_label} split: {causality_errors}",
                    )
                causality_checked = True
                causality_frames.clear()

        raw_signal = invoke_strategy_callable(
            strategy_fn=strategy_fn,
            strategy_df=strategy_df,
            params=params,
            accepts_state=strategy_accepts_state,
            model_state=strategy_state,
        )
        signal_errors = deps.validate_signal_array(raw_signal, len(strategy_df))
        if signal_errors:
            raise ValueError(f"{task_id}: signal contract failed on {split_label} split: {signal_errors}")

        signal_i8 = raw_signal.astype(np.int8, copy=False)
        bars_with_signal = full_df.with_columns(pl.Series("signal", signal_i8).cast(pl.Int8))
        bars_processed += len(bars_with_signal)
        signal_count += int((bars_with_signal["signal"] != 0).sum())

        signal_cols = ["ts_event", "close", "signal"]
        for col in ("open", "high", "low", "volume", "bid_price", "ask_price"):
            if col in bars_with_signal.columns:
                signal_cols.append(col)
        all_signals.append(bars_with_signal.select(signal_cols))

        analysis_cols = [
            col for col in deps.edge_surface_analysis_columns if col in bars_with_signal.columns
        ]
        analysis_frames.append(bars_with_signal.select(analysis_cols))

        eval_bar_cols = [
            col
            for col in (
                "ts_event",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "bid_price",
                "ask_price",
                "signal",
            )
            if col in bars_with_signal.columns
        ]
        execution_frame = bars_with_signal.select(eval_bar_cols)
        execution_frames.append(execution_frame)
        all_bars.append(execution_frame.drop("signal"))

    if causality_row_count > 0 and not causality_checked:
        raise ValueError(
            f"{task_id}: signal causality check on {split_label} split requires at least "
            f"{deps.causality_min_rows} bars, got {causality_row_count}",
        )

    signals_df = pl.concat(all_signals).sort("ts_event") if all_signals else empty_signal_frame()
    bars_df = pl.concat(all_bars).sort("ts_event") if all_bars else pl.DataFrame()
    edge_surface_summary = (
        deps.run_edge_surface(
            analysis_frames=analysis_frames,
            entry_on_next_open=bool(bt_kwargs.get("entry_on_next_open", False)),
            config=edge_surface_config,
        )
        if split_label == "search"
        else {
            "enabled": False,
            "passed": True,
            "status": "selection_skip",
            "has_localized_edge": False,
            "global_probe": {},
            "by_side": {},
            "conditional_slices": {},
            "best_pockets": [],
            "omitted_families": [],
        }
    )

    trades_df = pl.DataFrame(schema=deps.trade_schema)
    if bool(edge_surface_summary.get("passed", True)):
        all_trades: list[pl.DataFrame] = []
        for frame in execution_frames:
            trades = deps.run_backtest(frame, signal_col="signal", **bt_kwargs)
            if len(trades) > 0:
                trades = deps.compute_adaptive_costs(trades, frame)
                all_trades.append(trades)
        if all_trades:
            trades_df = pl.concat(all_trades)

    metrics = deps.compute_metrics(
        trades_df,
        cost_override_col="adaptive_cost_rt" if "adaptive_cost_rt" in trades_df.columns else None,
    )

    gauntlet: dict[str, Any] | None = None
    if bool(edge_surface_summary.get("passed", True)) and run_gauntlet and len(signals_df) > 0 and signal_count > 0:
        gauntlet = deps.run_validation_gauntlet(
            signals_df,
            signal_col="signal",
            min_trades=int(min_trade_count),
            **bt_kwargs,
        )

    advanced_validation: dict[str, Any] = {}
    if bool(edge_surface_summary.get("passed", True)) and len(trades_df) > 0:
        trial_stats = deps.deflated_sharpe_ratio(
            daily_net_pnl_series(trades_df),
            n_trials=max(
                1,
                int(deps.estimate_effective_trials(experiments_path).get("effective_trials", 1)),
            ),
        )
        advanced_validation["deflated_sharpe"] = trial_stats
        advanced_validation["alpha_decay"] = deps.fit_alpha_decay(trades_df)
        if len(bars_df) > 0:
            advanced_validation["factor_attribution"] = deps.factor_attribution(trades_df, bars_df)
    advanced_validation_gates = evaluate_advanced_validation_gates(mission, advanced_validation)

    meets_metric_thresholds = (
        metrics["sharpe_ratio"] >= float(target_sharpe)
        and metrics["trade_count"] >= int(min_trade_count)
    )
    meets_advanced_gates = bool(advanced_validation_gates.get("passed", True))

    verdict = "FAIL"
    failure_code: str | None = None
    if not bool(edge_surface_summary.get("passed", True)):
        failure_code = str(edge_surface_summary.get("status", "no_edge"))
    elif run_gauntlet:
        if gauntlet and gauntlet.get("overall_verdict") == "PASS" and meets_metric_thresholds and meets_advanced_gates:
            verdict = "PASS"
    elif meets_metric_thresholds and meets_advanced_gates:
        verdict = "PASS"

    summary = {
        "split_label": split_label,
        "split": split,
        "bars_processed": bars_processed,
        "signal_count": signal_count,
        "target_sharpe": float(target_sharpe),
        "min_trade_count": int(min_trade_count),
        "run_gauntlet": bool(run_gauntlet),
        "edge_surface": edge_surface_summary,
        "metrics": metrics,
        "gauntlet": gauntlet,
        "advanced_validation": advanced_validation,
        "advanced_validation_gates": advanced_validation_gates,
        "verdict": verdict,
    }
    if failure_code:
        summary["failure_code"] = failure_code
    summary_path = split_dir / "summary.json"
    signals_path = split_dir / "signals.parquet"
    trades_path = split_dir / "trades.parquet"
    deps.atomic_json_write(summary_path, summary)
    signals_df.write_parquet(signals_path)
    trades_df.write_parquet(trades_path)

    artifacts = {
        "summary": str(summary_path),
        "signals": str(signals_path),
        "trades": str(trades_path),
    }
    if edge_surface_summary.get("enabled", False):
        edge_surface_path = split_dir / "edge_surface.json"
        deps.atomic_json_write(edge_surface_path, edge_surface_summary)
        artifacts["edge_surface"] = str(edge_surface_path)
    if gauntlet is not None:
        gauntlet_path = split_dir / "gauntlet.json"
        deps.atomic_json_write(gauntlet_path, gauntlet)
        artifacts["gauntlet"] = str(gauntlet_path)

    return {**summary, "artifacts": artifacts}


def execute_claimed_task(
    *,
    deps: ValidatorExecutionDeps,
    task: dict[str, Any],
    mission: dict[str, Any],
    run_id: str,
    run_dir: Path,
    framework_lock_hash: str,
    git_commit: str | None,
    experiments_path: Path,
    experiments_lock: Path,
) -> tuple[str, dict[str, Any]]:
    task_id = str(task.get("task_id", "")).strip()
    strategy_name = str(task.get("strategy_name", "")).strip()
    bar_config = str(task.get("bar_config", "volume_2000"))
    params = task.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{task_id}: params must be an object")
    if not task_id:
        raise ValueError("task_id is required")
    if not strategy_name:
        raise ValueError(f"{task_id}: strategy_name is required")

    split_plan = task_split_plan(task, mission)
    search_split = str(split_plan["search_split"])
    selection_split = (
        str(split_plan["selection_split"])
        if split_plan["selection_split"] is not None
        else None
    )

    session_filter = deps.normalize_session_filter(
        task.get("session_filter", mission.get("session_filter", "eth")),
        default="eth",
    )
    parsed_bar = deps.parse_bar_config(bar_config)
    strategy_module = deps.load_signal_module(strategy_name)
    strategy_fn, strategy_accepts_state = resolve_strategy_callable(
        task_id=task_id,
        strategy_name=strategy_name,
        strategy_module=strategy_module,
    )
    theme_tag = str(task.get("theme_tag", "")).strip() or "other"
    setup_key, setup_label = deps.task_setup_identity(task)

    feature_group = deps.normalize_feature_group(
        task.get("feature_group", mission.get("feature_group", "all")),
        default="all",
    )
    strategy_id = deps.compute_strategy_id(
        strategy_name,
        params,
        strategy_fn,
        bar_config=bar_config,
        session_filter=session_filter,
        feature_group=feature_group,
    )
    bt_kwargs = task_backtest_params(task, mission)
    run_gauntlet = bool(task.get("run_gauntlet", mission.get("run_gauntlet", True)))
    write_candidate_flag = bool(task.get("write_candidate", mission.get("write_candidates", True)))
    max_files = task.get("max_files", mission.get("max_files_per_task"))
    max_files = int(max_files) if max_files is not None else None
    edge_surface_config = deps.normalize_edge_surface_config(mission.get("edge_surface"))
    task_dir = run_dir / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    search_result = evaluate_strategy_split(
        deps=deps,
        task_id=task_id,
        split_label="search",
        split=search_split,
        bar_config=bar_config,
        parsed_bar=parsed_bar,
        strategy_fn=strategy_fn,
        strategy_accepts_state=strategy_accepts_state,
        params=params,
        session_filter=session_filter,
        feature_group=feature_group,
        bt_kwargs=bt_kwargs,
        target_sharpe=float(mission.get("target_sharpe", 0.0)),
        min_trade_count=int(mission.get("min_trade_count", 1)),
        run_gauntlet=run_gauntlet,
        max_files=max_files,
        mission=mission,
        experiments_path=experiments_path,
        task_dir=task_dir,
        edge_surface_config=edge_surface_config,
    )

    selection_result: dict[str, Any] | None = None
    if search_result["verdict"] == "PASS" and selection_split is not None:
        selection_gate = selection_gate_config(mission, base_run_gauntlet=run_gauntlet)
        selection_result = evaluate_strategy_split(
            deps=deps,
            task_id=task_id,
            split_label="selection",
            split=selection_split,
            bar_config=bar_config,
            parsed_bar=parsed_bar,
            strategy_fn=strategy_fn,
            strategy_accepts_state=strategy_accepts_state,
            params=params,
            session_filter=session_filter,
            feature_group=feature_group,
            bt_kwargs=bt_kwargs,
            target_sharpe=float(selection_gate["target_sharpe"]),
            min_trade_count=int(selection_gate["min_trade_count"]),
            run_gauntlet=bool(selection_gate["run_gauntlet"]),
            max_files=max_files,
            mission=mission,
            experiments_path=experiments_path,
            task_dir=task_dir,
        )

    final_verdict = "FAIL"
    if search_result["verdict"] == "PASS":
        if selection_result is None or selection_result["verdict"] == "PASS":
            final_verdict = "PASS"

    candidate_status = "rejected_search"
    if search_result["verdict"] == "PASS":
        if selection_split is None:
            candidate_status = "selected"
        elif selection_result is None:
            candidate_status = "selection_not_run"
        elif selection_result["verdict"] == "PASS":
            candidate_status = "selected"
        else:
            candidate_status = "rejected_selection"

    summary = {
        "task_id": task_id,
        "run_id": run_id,
        "strategy_name": strategy_name,
        "strategy_id": strategy_id,
        "search_split": search_split,
        "selection_split": selection_split,
        "feedback_split": search_split,
        "feature_group": feature_group,
        "theme_tag": theme_tag,
        "setup_key": setup_key,
        "setup_label": setup_label,
        "bar_config": bar_config,
        "bar_params": parsed_bar,
        "params": params,
        "backtest": bt_kwargs,
        "search_result": search_result,
        "selection_result": selection_result,
        "final_verdict": final_verdict,
        "candidate_status": candidate_status,
    }
    summary_path = task_dir / "summary.json"
    deps.atomic_json_write(summary_path, summary)

    candidate_path: str | None = None
    candidate_result = selection_result or search_result
    if final_verdict == "PASS" and write_candidate_flag:
        signal_file = Path(strategy_module.__file__).resolve()
        candidate_payload = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "version": str(getattr(strategy_module, "STRATEGY_METADATA", {}).get("version", "1.0")),
            "bar_config": bar_config,
            "session_filter": session_filter,
            "feature_group": feature_group,
            "backtest": bt_kwargs,
            "parameters": params,
            "setup_key": setup_key,
            "setup_label": setup_label,
            "search_split": search_split,
            "selection_split": selection_split,
            "validation_metrics": candidate_result["metrics"],
            "gauntlet_results": candidate_result.get("gauntlet") or {},
            "edge_surface": search_result.get("edge_surface") or {},
            "advanced_validation": candidate_result.get("advanced_validation") or {},
            "advanced_validation_gates": candidate_result.get("advanced_validation_gates") or {},
            "search_result": search_result,
            "selection_result": selection_result,
            "artifacts": {
                "signal_file": str(signal_file),
                "signal_file_hash": deps.sha256_file(signal_file),
                "task_summary": str(summary_path),
                "search": dict(search_result["artifacts"]),
            },
            "provenance": {
                "run_id": run_id,
                "git_commit": git_commit,
                "framework_lock_hash": framework_lock_hash,
            },
        }
        if selection_result is not None:
            candidate_payload["artifacts"]["selection"] = dict(selection_result["artifacts"])
        candidate_out = deps.write_candidate(agent_name="validator", candidate_data=candidate_payload)
        candidate_path = str(candidate_out)
        candidate_status = "candidate_written"

    summary["candidate_status"] = candidate_status
    if candidate_path is not None:
        summary["candidate_path"] = candidate_path
    deps.atomic_json_write(summary_path, summary)

    attempt = as_int(task.get("retries"), 0) + 1
    event_id = deps.compute_event_id(
        run_id=run_id,
        task_id=task_id,
        strategy_id=strategy_id,
        stage="task",
        attempt=attempt,
    )
    deps.log_experiment(
        {
            "event_id": event_id,
            "run_id": run_id,
            "agent": "validator",
            "event": "task_result",
            "task_id": task_id,
            "strategy_name": strategy_name,
            "strategy_id": strategy_id,
            "split": search_split,
            "search_split": search_split,
            "selection_split": selection_split,
            "feature_group": feature_group,
            "theme_tag": theme_tag,
            "setup_key": setup_key,
            "setup_label": setup_label,
            "bar_config": bar_config,
            "metrics": search_result["metrics"],
            "edge_surface": search_result.get("edge_surface"),
            "gauntlet": search_result.get("gauntlet"),
            "advanced_validation": search_result.get("advanced_validation"),
            "advanced_validation_gates": search_result.get("advanced_validation_gates"),
            "verdict": search_result["verdict"],
            "artifacts": search_result["artifacts"],
            "search_result": search_result,
            "selection_result": selection_result,
            "final_verdict": final_verdict,
            "candidate_status": candidate_status,
            "candidate_path": candidate_path,
        },
        experiments_path=experiments_path,
        lock_path=experiments_lock,
    )

    details: dict[str, Any] = {
        "strategy_id": strategy_id,
        "metrics": search_result["metrics"],
        "artifacts": search_result["artifacts"],
        "gauntlet": search_result.get("gauntlet"),
        "advanced_validation": search_result.get("advanced_validation"),
        "advanced_validation_gates": search_result.get("advanced_validation_gates"),
        "feature_group": feature_group,
        "theme_tag": theme_tag,
        "setup_key": setup_key,
        "setup_label": setup_label,
        "feedback_split": search_split,
        "feedback_verdict": search_result["verdict"],
        "search_split": search_split,
        "selection_split": selection_split,
        "edge_surface": search_result.get("edge_surface"),
        "search_result": search_result,
        "selection_result": selection_result,
        "final_verdict": final_verdict,
        "candidate_status": candidate_status,
        "summary": str(summary_path),
    }
    if candidate_path:
        details["candidate_path"] = candidate_path
    return final_verdict, details


__all__ = [
    "ValidatorExecutionDeps",
    "as_float",
    "as_int",
    "daily_net_pnl_series",
    "empty_signal_frame",
    "evaluate_advanced_validation_gates",
    "evaluate_strategy_split",
    "execute_claimed_task",
    "invoke_strategy_callable",
    "resolve_strategy_callable",
    "selection_gate_config",
    "task_backtest_params",
    "task_split_plan",
]
