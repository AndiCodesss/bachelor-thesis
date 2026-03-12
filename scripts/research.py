#!/usr/bin/env python3
"""Autonomous research entrypoint for the institutional framework."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable

import numpy as np
import polars as pl

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.lib.atomic_io import atomic_json_write
from research.lib.budget import MissionBudget
from research.lib.candidates import write_candidate
from research.lib.edge_surface import (
    EDGE_SURFACE_ANALYSIS_COLUMNS,
    normalize_edge_surface_config,
    run_edge_surface,
)
from research.lib.feature_groups import filter_strategy_inputs
from research.lib.learning_scorecard import (
    empty_scorecard,
    rebuild_learning_scorecard,
    update_learning_scorecard,
)
from research.lib.script_support import (
    load_yaml_dict as _load_yaml,
    mission_state_fingerprint,
    normalize_feature_group,
    normalize_session_filter,
    parse_bar_config as _parse_bar_config,
    state_mode as _state_mode,
    utc_now_iso as _utc_now,
    validate_signal_array as _validate_signal_array,
)
from research.lib.mission_splits import resolve_research_splits
from research.lib.coordination import (
    claim_task,
    complete_task,
    compute_event_id,
    read_json_file,
    update_json_file,
    update_task_heartbeat,
    watchdog_check_timeouts,
)
from research.lib.experiments import log_experiment
from research.lib.runtime_state import (
    clear_orchestrator_state,
    ensure_shared_state,
    reset_shared_state,
    shared_state_defaults,
)
from research.lib.setup_key import task_setup_identity
from research.lib.trial_counter import estimate_effective_trials
from research.lib.validator_execution import (
    ValidatorExecutionDeps,
    as_float as _validator_as_float,
    as_int as _validator_as_int,
    daily_net_pnl_series as _validator_daily_net_pnl_series,
    empty_signal_frame as _validator_empty_signal_frame,
    evaluate_advanced_validation_gates as _validator_evaluate_advanced_validation_gates,
    evaluate_strategy_split as _validator_evaluate_strategy_split,
    execute_claimed_task as _validator_execute_claimed_task,
    invoke_strategy_callable as _validator_invoke_strategy_callable,
    resolve_strategy_callable as _validator_resolve_strategy_callable,
    selection_gate_config as _validator_selection_gate_config,
    task_backtest_params as _validator_task_backtest_params,
    task_split_plan as _validator_task_split_plan,
)
from research.signals import (
    CAUSALITY_MIN_PREFIX_BARS,
    check_signal_causality,
    compute_strategy_id,
    discover_signals,
    load_signal_module,
)
from src.framework import __version__ as framework_version
from src.framework.api import (
    ExecutionMode,
    compute_adaptive_costs,
    compute_metrics,
    deflated_sharpe_ratio,
    factor_attribution,
    fit_alpha_decay,
    get_split_files,
    load_cached_matrix,
    run_backtest,
    run_validation_gauntlet,
    set_execution_mode,
)
from src.framework.backtest.engine import TRADE_SCHEMA
from src.framework.data.constants import RESULTS_DIR
from src.framework.security.framework_lock import verify_manifest

_CAUSALITY_MIN_ROWS = CAUSALITY_MIN_PREFIX_BARS + 1
_DEFAULT_QUEUE_TERMINAL_KEEP = 2000
_MIN_QUEUE_TERMINAL_KEEP = 200

def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-").lower() or "mission"


def _run_id(mission_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{_slug(mission_name)}"

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(project_root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _run_contract_tests(project_root: Path) -> None:
    cmd = ["uv", "run", "pytest", "tests/test_signal_contract.py", "-q"]
    proc = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        raise RuntimeError("Strategy contract tests failed")


def _verify_lock(manifest: Path, mode: str) -> dict[str, Any]:
    result = verify_manifest(manifest_path=manifest, project_root=Path(__file__).resolve().parent.parent)
    if result["ok"]:
        print(
            "Framework lock PASS: "
            f"verified={result['verified_file_count']}/{result['manifest_file_count']}"
        )
        return result
    print(
        "Framework lock FAIL: "
        f"verified={result['verified_file_count']}/{result['manifest_file_count']}"
    )
    if mode == "error":
        raise SystemExit(2)
    return result


def _queue_counts(queue_path: Path, lock_path: Path) -> dict[str, int]:
    payload = read_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
    )
    tasks = list(payload.get("tasks", []))
    return {
        "pending": sum(1 for t in tasks if t.get("state") == "pending"),
        "in_progress": sum(1 for t in tasks if t.get("state") == "in_progress"),
        "completed": sum(1 for t in tasks if t.get("state") == "completed"),
        "failed": sum(1 for t in tasks if t.get("state") == "failed"),
    }


def _collect_new_terminal_task_verdicts(
    queue_path: Path,
    lock_path: Path,
    seen_task_ids: set[str],
) -> list[str]:
    payload = read_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
    )
    verdicts: list[str] = []
    for task in payload.get("tasks", []):
        state = task.get("state")
        task_id = str(task.get("task_id", ""))
        if state not in {"completed", "failed"} or not task_id:
            continue
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        verdicts.append(str(task.get("verdict", "FAIL")))
    return verdicts


def _seed_seen_terminal_task_ids(queue_path: Path, lock_path: Path, already_counted: int) -> set[str]:
    """Seed seen terminal task ids from queue for resume-safe budget accounting.

    We assume the first N terminal tasks (ordered by completed_at/created_at)
    are already reflected in MissionBudget.experiments_run, and only unseen
    terminal tasks should increment budget after resume.
    """
    payload = read_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
    )
    terminal_tasks = [
        task for task in payload.get("tasks", [])
        if task.get("state") in {"completed", "failed"} and str(task.get("task_id", "")).strip()
    ]
    terminal_tasks.sort(
        key=lambda task: (
            str(task.get("completed_at", "")),
            str(task.get("created_at", "")),
            str(task.get("task_id", "")),
        )
    )
    n_seed = max(0, min(int(already_counted), len(terminal_tasks)))
    return {
        str(task["task_id"])
        for task in terminal_tasks[:n_seed]
    }


def _prune_terminal_tasks(
    *,
    queue_path: Path,
    lock_path: Path,
    max_terminal_tasks: int,
    protected_task_ids: set[str] | None = None,
) -> int:
    """Bound queue file size by dropping oldest terminal tasks."""
    keep = max(_MIN_QUEUE_TERMINAL_KEEP, int(max_terminal_tasks))
    protected = {
        str(task_id).strip()
        for task_id in (protected_task_ids or set())
        if str(task_id).strip()
    }
    out: dict[str, int] = {"pruned": 0}

    def _update(queue: dict[str, Any]) -> dict[str, Any]:
        tasks = list(queue.get("tasks", []))
        terminal_indices = [
            idx for idx, task in enumerate(tasks)
            if task.get("state") in {"completed", "failed"}
            and str(task.get("task_id", "")).strip() not in protected
        ]
        overflow = len(terminal_indices) - keep
        if overflow <= 0:
            return queue

        terminal_indices.sort(
            key=lambda idx: (
                str(tasks[idx].get("completed_at", "")),
                str(tasks[idx].get("created_at", "")),
                str(tasks[idx].get("task_id", "")),
            )
        )
        drop = set(terminal_indices[:overflow])
        out["pruned"] = len(drop)
        queue["tasks"] = [task for idx, task in enumerate(tasks) if idx not in drop]
        return queue

    update_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
        update_fn=_update,
    )
    return int(out["pruned"])


def _pending_validation_task_ids(
    *,
    handoffs_path: Path,
    handoffs_lock: Path,
) -> set[str]:
    handoffs_payload = read_json_file(
        json_path=handoffs_path,
        lock_path=handoffs_lock,
        default_payload={"schema_version": "1.0", "pending": [], "completed": []},
    )
    task_ids: set[str] = set()
    for row in handoffs_payload.get("pending", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("handoff_type", "")) != "validation_request":
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        raw_task_ids = payload.get("task_ids")
        if not isinstance(raw_task_ids, list):
            continue
        for task_id in raw_task_ids:
            value = str(task_id).strip()
            if value:
                task_ids.add(value)
    return task_ids


def _daily_net_pnl_series(trades: pl.DataFrame) -> np.ndarray:
    return _validator_daily_net_pnl_series(trades)


def _evaluate_advanced_validation_gates(
    mission: dict[str, Any],
    advanced_validation: dict[str, Any],
) -> dict[str, Any]:
    return _validator_evaluate_advanced_validation_gates(mission, advanced_validation)


def _task_feedback_summary(task: dict[str, Any]) -> dict[str, Any]:
    details = task.get("details")
    source = task.get("source") if isinstance(task.get("source"), dict) else {}
    metrics = details.get("metrics") if isinstance(details, dict) else None
    selection_result = details.get("selection_result") if isinstance(details, dict) else None
    edge_surface = details.get("edge_surface") if isinstance(details, dict) else None
    feedback_verdict = (
        str(details.get("feedback_verdict", task.get("verdict", "")))
        if isinstance(details, dict)
        else str(task.get("verdict", ""))
    )
    feedback_split = (
        str(details.get("feedback_split", task.get("split", "")))
        if isinstance(details, dict)
        else str(task.get("split", ""))
    )
    out: dict[str, Any] = {
        "task_id": str(task.get("task_id", "")),
        "strategy_name": str(task.get("strategy_name", "")),
        "theme_tag": str(task.get("theme_tag") or source.get("theme_tag") or ""),
        "setup_key": str(task.get("setup_key") or source.get("setup_key") or ""),
        "setup_label": str(task.get("setup_label") or source.get("setup_label") or ""),
        "bar_config": str(task.get("bar_config", "")),
        "state": str(task.get("state", "")),
        "verdict": feedback_verdict,
        "split": feedback_split,
        "completed_at": str(task.get("completed_at", "")),
    }
    if isinstance(metrics, dict):
        out["sharpe_ratio"] = metrics.get("sharpe_ratio")
        out["trade_count"] = metrics.get("trade_count")
        out["net_pnl"] = metrics.get("net_pnl")
    if isinstance(details, dict):
        out["final_verdict"] = str(details.get("final_verdict", task.get("verdict", "")))
        out["candidate_status"] = str(details.get("candidate_status", ""))
        out["selection_attempted"] = isinstance(selection_result, dict)
        out["selection_verdict"] = (
            str(selection_result.get("verdict", ""))
            if isinstance(selection_result, dict)
            else ""
        )
    if isinstance(edge_surface, dict):
        out["edge_surface_status"] = str(edge_surface.get("status", ""))
        out["edge_surface_passed"] = bool(edge_surface.get("passed", False))
    error = details.get("error") if isinstance(details, dict) else None
    if error:
        out["error"] = str(error)
    return out


def _summarize_validation_request(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = 0
    fail_count = 0
    error_count = 0
    sharpe_vals: list[float] = []
    trade_counts: list[int] = []
    passing_bar_configs: list[str] = []
    failing_bar_configs: list[str] = []
    best_row: dict[str, Any] | None = None
    best_sharpe = float("-inf")

    for row in task_rows:
        verdict = str(row.get("verdict", "")).upper()
        if verdict == "PASS":
            pass_count += 1
            bar_config = str(row.get("bar_config", "")).strip()
            if bar_config:
                passing_bar_configs.append(bar_config)
        elif verdict == "ERROR":
            error_count += 1
        else:
            fail_count += 1
            bar_config = str(row.get("bar_config", "")).strip()
            if bar_config:
                failing_bar_configs.append(bar_config)

        sharpe = row.get("sharpe_ratio")
        if isinstance(sharpe, (int, float)):
            sharpe_val = float(sharpe)
            sharpe_vals.append(sharpe_val)
            if sharpe_val > best_sharpe:
                best_sharpe = sharpe_val
                best_row = row
        trades = row.get("trade_count")
        if isinstance(trades, int):
            trade_counts.append(int(trades))
        elif isinstance(trades, float):
            trade_counts.append(int(trades))

    if error_count > 0:
        overall = "ERROR"
    elif pass_count == len(task_rows) and task_rows:
        overall = "PASS"
    elif pass_count > 0:
        overall = "MIXED"
    else:
        overall = "FAIL"

    avg_sharpe = (sum(sharpe_vals) / len(sharpe_vals)) if sharpe_vals else None
    avg_trades = (sum(trade_counts) / len(trade_counts)) if trade_counts else None
    best_bar_config = (
        str(best_row.get("bar_config", "")).strip()
        if isinstance(best_row, dict)
        else None
    )

    return {
        "overall_verdict": overall,
        "task_count": len(task_rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "error_count": error_count,
        "pass_fraction": (float(pass_count) / float(len(task_rows))) if task_rows else 0.0,
        "avg_sharpe_ratio": avg_sharpe,
        "avg_trade_count": avg_trades,
        "passing_bar_configs": passing_bar_configs,
        "failing_bar_configs": failing_bar_configs,
        "best_bar_config": best_bar_config,
        "best_sharpe_ratio": (best_sharpe if best_row is not None else None),
        "tasks": task_rows,
    }


def _finalize_ready_validation_handoffs(
    *,
    queue_path: Path,
    queue_lock: Path,
    handoffs_path: Path,
    handoffs_lock: Path,
) -> list[dict[str, Any]]:
    queue_payload = read_json_file(
        json_path=queue_path,
        lock_path=queue_lock,
        default_payload={"schema_version": "1.0", "tasks": []},
    )
    tasks_by_id = {
        str(task.get("task_id", "")): task
        for task in queue_payload.get("tasks", [])
        if isinstance(task, dict) and str(task.get("task_id", "")).strip()
    }
    completed_results: list[dict[str, Any]] = []
    now_iso = _utc_now()

    def _update(handoffs: dict[str, Any]) -> dict[str, Any]:
        pending = list(handoffs.get("pending", []))
        completed = handoffs.setdefault("completed", [])
        next_pending: list[dict[str, Any]] = []

        for row in pending:
            if not isinstance(row, dict):
                continue
            if str(row.get("handoff_type", "")) != "validation_request":
                next_pending.append(row)
                continue

            payload = row.get("payload")
            if not isinstance(payload, dict):
                next_pending.append(row)
                continue
            raw_task_ids = payload.get("task_ids")
            if not isinstance(raw_task_ids, list) or not raw_task_ids:
                next_pending.append(row)
                continue
            task_ids = [str(tid).strip() for tid in raw_task_ids if str(tid).strip()]
            if not task_ids:
                next_pending.append(row)
                continue

            task_rows: list[dict[str, Any]] = []
            unresolved = False
            for tid in task_ids:
                task = tasks_by_id.get(tid)
                if task is None:
                    unresolved = True
                    break
                state = str(task.get("state", ""))
                if state not in {"completed", "failed"}:
                    unresolved = True
                    break
                task_rows.append(_task_feedback_summary(task))

            if unresolved:
                next_pending.append(row)
                continue

            enriched = dict(row)
            enriched["state"] = "completed"
            enriched["completed_at"] = now_iso
            enriched["completed_by"] = "validator"
            enriched["result"] = _summarize_validation_request(task_rows)
            completed.append(enriched)
            completed_results.append(enriched)

        handoffs["pending"] = next_pending
        return handoffs

    update_json_file(
        json_path=handoffs_path,
        lock_path=handoffs_lock,
        default_payload={"schema_version": "1.0", "pending": [], "completed": []},
        update_fn=_update,
    )
    return completed_results


def _complete_claimed_task_safely(
    *,
    state_paths: dict[str, Path],
    agent_name: str,
    claimed: dict[str, Any],
    verdict: str,
    details: dict[str, Any],
    run_id: str,
    experiments_path: Path,
    experiments_lock: Path,
) -> bool:
    task_id = str(claimed.get("task_id", "")).strip()
    strategy_name = str(claimed.get("strategy_name", "")).strip()
    try:
        complete_task(
            queue_path=state_paths["queue"],
            lock_path=state_paths["queue_lock"],
            agent_name=agent_name,
            task_id=task_id,
            verdict=verdict,
            details=details,
        )
        return True
    except (PermissionError, ValueError) as exc:
        warning = (
            f"complete_task skipped for task_id={task_id or 'unknown'} "
            f"strategy={strategy_name or 'unknown'}: {exc}"
        )
        print(f"WARN: {warning}", file=sys.stderr, flush=True)
        log_experiment(
            {
                "run_id": run_id,
                "agent": agent_name,
                "event": "task_completion_race",
                "task_id": task_id,
                "strategy_name": strategy_name,
                "attempted_verdict": verdict,
                "error": str(exc),
            },
            experiments_path=experiments_path,
            lock_path=experiments_lock,
        )
        return False


def _read_persisted_task_row(
    *,
    queue_path: Path,
    queue_lock: Path,
    task_id: str,
) -> dict[str, Any] | None:
    payload = read_json_file(
        json_path=queue_path,
        lock_path=queue_lock,
        default_payload={"schema_version": "1.0", "tasks": []},
    )
    for row in payload.get("tasks", []):
        if isinstance(row, dict) and str(row.get("task_id", "")).strip() == str(task_id).strip():
            return dict(row)
    return None


def _as_int(value: Any, default: int) -> int:
    return _validator_as_int(value, default)


def _as_float(value: Any, default: float) -> float:
    return _validator_as_float(value, default)

def _empty_signal_frame() -> pl.DataFrame:
    return _validator_empty_signal_frame()

def _resolve_strategy_callable(
    *,
    task_id: str,
    strategy_name: str,
    strategy_module: Any,
) -> tuple[Callable[..., Any], bool]:
    return _validator_resolve_strategy_callable(
        task_id=task_id,
        strategy_name=strategy_name,
        strategy_module=strategy_module,
    )


def _invoke_strategy_callable(
    *,
    strategy_fn: Callable[..., Any],
    strategy_df: pl.DataFrame,
    params: dict[str, Any],
    accepts_state: bool,
    model_state: Any | None,
) -> np.ndarray:
    return _validator_invoke_strategy_callable(
        strategy_fn=strategy_fn,
        strategy_df=strategy_df,
        params=params,
        accepts_state=accepts_state,
        model_state=model_state,
    )


def _task_bootstrap_key(
    *,
    strategy_name: str,
    split: str,
    bar_config: str,
    params: dict[str, Any],
    selection_split: str | None = None,
    session_filter: str = "",
    feature_group: str = "",
) -> str:
    params_obj = params if isinstance(params, dict) else {}
    params_blob = json.dumps(params_obj, sort_keys=True, separators=(",", ":"), default=str)
    sel = str(selection_split).strip().lower() if selection_split is not None else ""
    return (
        f"{strategy_name}|{split}|{sel}|{bar_config}|"
        f"{str(session_filter).strip().lower()}|{str(feature_group).strip().lower()}|{params_blob}"
    )


def _task_sort_key(task: dict[str, Any]) -> tuple[int, str]:
    return (_as_int(task.get("priority"), 1_000_000), str(task.get("created_at", "")))


def _claim_next_task(queue_path: Path, lock_path: Path, *, agent_name: str) -> dict[str, Any] | None:
    payload = read_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
    )
    pending = [
        task for task in payload.get("tasks", [])
        if task.get("state") == "pending" and task.get("task_id")
    ]
    pending.sort(key=_task_sort_key)

    for task in pending:
        claimed = claim_task(
            queue_path=queue_path,
            lock_path=lock_path,
            agent_name=agent_name,
            task_id=str(task["task_id"]),
            lease_minutes=_as_int(task.get("timeout_minutes"), 30),
            heartbeat_interval_seconds=_as_int(task.get("heartbeat_interval_seconds"), 300),
        )
        if claimed is not None:
            return claimed
    return None


def _bootstrap_tasks_if_empty(
    *,
    queue_path: Path,
    lock_path: Path,
    mission: dict[str, Any],
    max_new_tasks: int,
) -> int:
    if max_new_tasks <= 0:
        return 0

    signals = discover_signals()
    strategy_names = sorted(signals.keys())
    if not strategy_names:
        return 0

    split_plan = resolve_research_splits(mission)
    bar_configs_raw = mission.get("bar_configs", ["volume_2000"])
    bar_configs = [str(v) for v in bar_configs_raw] if isinstance(bar_configs_raw, list) else ["volume_2000"]
    search_split = str(split_plan["search_split"])
    selection_split = (
        str(split_plan["selection_split"])
        if split_plan["selection_split"] is not None
        else None
    )
    session_filter = normalize_session_filter(mission.get("session_filter", "eth"))
    feature_group = normalize_feature_group(mission.get("feature_group", "all"))

    max_retries = _as_int(mission.get("max_retries"), 2)
    timeout_minutes = _as_int(mission.get("task_timeout_minutes"), 30)
    heartbeat_seconds = _as_int(mission.get("heartbeat_interval_seconds"), 300)
    default_run_gauntlet = bool(mission.get("run_gauntlet", True))
    default_write_candidates = bool(mission.get("write_candidates", True))
    default_max_files = mission.get("max_files_per_task")

    created_holder = {"count": 0}

    def _update(queue: dict[str, Any]) -> dict[str, Any]:
        tasks = queue.setdefault("tasks", [])
        # Do not re-seed while work is already in-flight.
        if any(t.get("state") in {"pending", "in_progress"} for t in tasks):
            return queue

        existing_ids = {str(t.get("task_id", "")) for t in tasks}
        existing_task_keys = {
            _task_bootstrap_key(
                strategy_name=str(t.get("strategy_name", "")),
                split=str(t.get("search_split", t.get("split", ""))),
                bar_config=str(t.get("bar_config", "")),
                params=t.get("params", {}) if isinstance(t.get("params", {}), dict) else {},
                selection_split=(
                    str(t.get("selection_split")).strip().lower()
                    if t.get("selection_split") is not None
                    else None
                ),
                session_filter=str(t.get("session_filter", session_filter)),
                feature_group=str(t.get("feature_group", feature_group)),
            )
            for t in tasks
        }
        now = _utc_now()
        created = 0
        serial = len(tasks)

        for strategy_name in strategy_names:
            module = load_signal_module(strategy_name)
            default_params = getattr(module, "DEFAULT_PARAMS", {})
            if not isinstance(default_params, dict):
                default_params = {}

            for bar_cfg in bar_configs:
                if created >= max_new_tasks:
                    break
                serial += 1
                task_id = f"auto_{serial:05d}_{_slug(strategy_name)}_{_slug(bar_cfg)}"
                if task_id in existing_ids:
                    continue
                task_key = _task_bootstrap_key(
                    strategy_name=strategy_name,
                    split=search_split,
                    bar_config=bar_cfg,
                    params=default_params,
                    selection_split=selection_split,
                    session_filter=session_filter,
                    feature_group=feature_group,
                )
                if task_key in existing_task_keys:
                    continue

                task: dict[str, Any] = {
                    "task_id": task_id,
                    "state": "pending",
                    "assigned_to": None,
                    "created_at": now,
                    "strategy_name": strategy_name,
                    "split": search_split,
                    "search_split": search_split,
                    "selection_split": selection_split,
                    "bar_config": bar_cfg,
                    "session_filter": session_filter,
                    "feature_group": feature_group,
                    "params": dict(default_params),
                    "priority": serial,
                    "retries": 0,
                    "max_retries": max_retries,
                    "timeout_minutes": timeout_minutes,
                    "heartbeat_interval_seconds": heartbeat_seconds,
                    "run_gauntlet": default_run_gauntlet,
                    "write_candidate": default_write_candidates,
                }
                if default_max_files is not None:
                    task["max_files"] = int(default_max_files)
                tasks.append(task)
                existing_ids.add(task_id)
                existing_task_keys.add(task_key)
                created += 1

            if created >= max_new_tasks:
                break

        created_holder["count"] = created
        return queue

    update_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
        update_fn=_update,
    )
    return created_holder["count"]


def _task_backtest_params(task: dict[str, Any], mission: dict[str, Any]) -> dict[str, Any]:
    return _validator_task_backtest_params(task, mission)


def _task_runtime_context(task: dict[str, Any], mission: dict[str, Any]) -> dict[str, str]:
    return {
        "session_filter": normalize_session_filter(
            task.get("session_filter", mission.get("session_filter", "eth")),
            default="eth",
        ),
        "feature_group": normalize_feature_group(
            task.get("feature_group", mission.get("feature_group", "all")),
            default="all",
        ),
    }


def _task_split_plan(task: dict[str, Any], mission: dict[str, Any]) -> dict[str, str | None]:
    return _validator_task_split_plan(task, mission)


def _selection_gate_config(
    mission: dict[str, Any],
    *,
    base_run_gauntlet: bool,
) -> dict[str, Any]:
    return _validator_selection_gate_config(
        mission,
        base_run_gauntlet=base_run_gauntlet,
    )


def _validator_execution_deps() -> ValidatorExecutionDeps:
    return ValidatorExecutionDeps(
        atomic_json_write=atomic_json_write,
        check_signal_causality=check_signal_causality,
        compute_adaptive_costs=compute_adaptive_costs,
        compute_event_id=compute_event_id,
        compute_metrics=compute_metrics,
        compute_strategy_id=compute_strategy_id,
        deflated_sharpe_ratio=deflated_sharpe_ratio,
        estimate_effective_trials=estimate_effective_trials,
        factor_attribution=factor_attribution,
        filter_strategy_inputs=filter_strategy_inputs,
        fit_alpha_decay=fit_alpha_decay,
        get_split_files=get_split_files,
        load_cached_matrix=load_cached_matrix,
        load_signal_module=load_signal_module,
        log_experiment=log_experiment,
        normalize_feature_group=normalize_feature_group,
        normalize_edge_surface_config=normalize_edge_surface_config,
        normalize_session_filter=normalize_session_filter,
        parse_bar_config=_parse_bar_config,
        run_backtest=run_backtest,
        run_edge_surface=run_edge_surface,
        run_validation_gauntlet=run_validation_gauntlet,
        sha256_file=_sha256_file,
        task_setup_identity=task_setup_identity,
        validate_signal_array=_validate_signal_array,
        write_candidate=write_candidate,
        causality_min_prefix_bars=CAUSALITY_MIN_PREFIX_BARS,
        causality_min_rows=_CAUSALITY_MIN_ROWS,
        edge_surface_analysis_columns=EDGE_SURFACE_ANALYSIS_COLUMNS,
        trade_schema=TRADE_SCHEMA,
    )


def _evaluate_strategy_split(
    *,
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
    return _validator_evaluate_strategy_split(
        deps=_validator_execution_deps(),
        task_id=task_id,
        split_label=split_label,
        split=split,
        bar_config=bar_config,
        parsed_bar=parsed_bar,
        strategy_fn=strategy_fn,
        strategy_accepts_state=strategy_accepts_state,
        params=params,
        session_filter=session_filter,
        feature_group=feature_group,
        bt_kwargs=bt_kwargs,
        target_sharpe=target_sharpe,
        min_trade_count=min_trade_count,
        run_gauntlet=run_gauntlet,
        max_files=max_files,
        mission=mission,
        experiments_path=experiments_path,
        task_dir=task_dir,
        edge_surface_config=edge_surface_config,
    )


def _execute_claimed_task(
    *,
    task: dict[str, Any],
    mission: dict[str, Any],
    run_id: str,
    run_dir: Path,
    framework_lock_hash: str,
    git_commit: str | None,
    experiments_path: Path,
    experiments_lock: Path,
) -> tuple[str, dict[str, Any]]:
    return _validator_execute_claimed_task(
        deps=_validator_execution_deps(),
        task=task,
        mission=mission,
        run_id=run_id,
        run_dir=run_dir,
        framework_lock_hash=framework_lock_hash,
        git_commit=git_commit,
        experiments_path=experiments_path,
        experiments_lock=experiments_lock,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous research runner.")
    parser.add_argument("--mission", type=Path, required=True, help="Mission YAML path.")
    parser.add_argument("--max-experiments", type=int, default=None, help="Override mission max_experiments.")
    parser.add_argument("--auto-mode", action="store_true", help="Run polling loop until queue empty or budget exhausted.")
    parser.add_argument("--poll-seconds", type=int, default=10, help="Polling interval in seconds.")
    parser.add_argument("--watchdog-seconds", type=int, default=60, help="Watchdog interval in seconds.")
    parser.add_argument("--worker-agent", default="validator", help="Agent name used for task claim/complete.")
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=None,
        help="Optional runtime cap. Defaults to mission.max_runtime_hours when present.",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable auto task bootstrapping when queue is empty.",
    )
    parser.add_argument("--framework-lock-manifest", default="configs/framework_lock.json")
    parser.add_argument("--framework-lock-mode", choices=["warn", "error"], default="error")
    parser.add_argument("--skip-contract-tests", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing research/.state files (default behavior).",
    )
    parser.add_argument(
        "--fresh-state",
        action="store_true",
        help="Reset queue, handoffs, budget, and all orchestrator state before starting.",
    )
    args = parser.parse_args()
    if args.resume and args.fresh_state:
        parser.error("Use at most one of --resume or --fresh-state.")

    root = Path(__file__).resolve().parent.parent
    mission_path = args.mission.resolve()
    if not mission_path.exists():
        raise FileNotFoundError(f"Mission file not found: {mission_path}")

    set_execution_mode(ExecutionMode.RESEARCH)
    manifest = Path(args.framework_lock_manifest)
    if not manifest.is_absolute():
        manifest = (root / manifest).resolve()
    lock_result = _verify_lock(manifest=manifest, mode=args.framework_lock_mode)
    if not args.skip_contract_tests:
        _run_contract_tests(root)

    mission = _load_yaml(mission_path)
    mission_fingerprint = mission_state_fingerprint(mission)
    split_plan = resolve_research_splits(mission)
    mission_name = str(mission.get("mission_name", mission_path.stem))
    queue_terminal_keep = _as_int(
        mission.get("queue_terminal_keep", _DEFAULT_QUEUE_TERMINAL_KEEP),
        _DEFAULT_QUEUE_TERMINAL_KEEP,
    )
    run_id = _run_id(mission_name)
    run_dir = (RESULTS_DIR / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(run_dir / "mission.json", mission)

    state_mode = _state_mode(fresh_state=bool(args.fresh_state))
    if state_mode == "fresh":
        state_paths = reset_shared_state(
            root,
            mission_name=mission_name,
            mission_fingerprint=mission_fingerprint,
        )
        clear_orchestrator_state(root)
    else:
        state_paths = ensure_shared_state(
            root,
            mission_name=mission_name,
            mission_fingerprint=mission_fingerprint,
        )
    budget = MissionBudget(
        max_experiments=int(args.max_experiments if args.max_experiments is not None else mission.get("max_experiments", 100)),
        kill_criteria=dict(mission.get("kill_criteria", {"FAIL": 10, "ERROR": 5})),
        state_file=state_paths["budget"],
        lock_file=state_paths["budget_lock"],
        mission_name=mission_name,
        reset_on_mission_change=state_mode != "fresh",
    )
    seen_terminal_tasks: set[str] = (
        _seed_seen_terminal_task_ids(
            state_paths["queue"],
            state_paths["queue_lock"],
            budget.experiments_run,
        )
        if state_mode == "resume" else set()
    )

    logs_dir = root / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    experiments_path = logs_dir / "research_experiments.jsonl"
    experiments_lock = logs_dir / "research_experiments.lock"
    scorecard_path = state_paths["scorecard"]
    scorecard_lock = state_paths["scorecard_lock"]
    current_scorecard = read_json_file(
        json_path=scorecard_path,
        lock_path=scorecard_lock,
        default_payload=empty_scorecard(),
    )
    if state_mode == "fresh" or current_scorecard.get("rebuilt_at") is None:
        rebuild_learning_scorecard(
            experiments_path=experiments_path,
            handoffs_path=state_paths["handoffs"],
            handoffs_lock=state_paths["handoffs_lock"],
            scorecard_path=scorecard_path,
            scorecard_lock=scorecard_lock,
        )
    run_start_event_id = compute_event_id(
        run_id=run_id,
        task_id="run",
        strategy_id=mission_name,
        stage="run_start",
        attempt=1,
    )
    log_experiment(
        {
            "event_id": run_start_event_id,
            "run_id": run_id,
            "agent": "orchestrator",
            "event": "run_start",
            "mission_name": mission_name,
            "auto_mode": bool(args.auto_mode),
            "resume": state_mode == "resume",
            "state_mode": state_mode,
            "worker_agent": str(args.worker_agent),
        },
        experiments_path=experiments_path,
        lock_path=experiments_lock,
    )

    stop_watchdog = threading.Event()

    def _watchdog_loop() -> None:
        while not stop_watchdog.is_set():
            try:
                watchdog_check_timeouts(
                    queue_path=state_paths["queue"],
                    lock_path=state_paths["queue_lock"],
                )
            except Exception:
                print("ERROR: watchdog loop crashed; continuing.", file=sys.stderr, flush=True)
                traceback.print_exc()
            stop_watchdog.wait(timeout=max(1, int(args.watchdog_seconds)))

    watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True, name="research-watchdog")
    watchdog_thread.start()

    print(f"Run ID: {run_id}")
    print(f"Mission: {mission_name}")
    print(f"Auto mode: {bool(args.auto_mode)}")
    print(f"State mode: {state_mode}")
    print(f"Worker agent: {args.worker_agent}")
    print(
        "Discovery splits: "
        f"search={split_plan['search_split']} "
        f"selection={split_plan['selection_split'] or 'none'}",
    )
    print(f"Budget max experiments: {budget.max_experiments}")

    max_runtime_hours = (
        float(args.max_runtime_hours)
        if args.max_runtime_hours is not None
        else (
            float(mission["max_runtime_hours"])
            if mission.get("max_runtime_hours") is not None
            else None
        )
    )
    start_monotonic = time.monotonic()
    poll_seconds = max(1, int(args.poll_seconds))

    reason = "single_pass"
    try:
        while True:
            for verdict in _collect_new_terminal_task_verdicts(
                state_paths["queue"],
                state_paths["queue_lock"],
                seen_terminal_tasks,
            ):
                budget.record_experiment(verdict)

            can_continue, reason = budget.check_budget()
            if not can_continue:
                break

            if max_runtime_hours is not None:
                elapsed_hours = (time.monotonic() - start_monotonic) / 3600.0
                if elapsed_hours >= max_runtime_hours:
                    reason = "max_runtime_reached"
                    break

            counts = _queue_counts(state_paths["queue"], state_paths["queue_lock"])
            if not args.auto_mode:
                reason = "manual_single_pass"
                break

            if counts["pending"] == 0 and counts["in_progress"] == 0 and not args.no_bootstrap:
                remaining_budget = max(0, budget.max_experiments - budget.experiments_run)
                created = _bootstrap_tasks_if_empty(
                    queue_path=state_paths["queue"],
                    lock_path=state_paths["queue_lock"],
                    mission=mission,
                    max_new_tasks=remaining_budget,
                )
                if created > 0:
                    print(f"Bootstrapped tasks: {created}")
                    counts = _queue_counts(state_paths["queue"], state_paths["queue_lock"])

            claimed = _claim_next_task(
                state_paths["queue"],
                state_paths["queue_lock"],
                agent_name=str(args.worker_agent),
            )
            if claimed is not None:
                stop_heartbeat = threading.Event()

                def _heartbeat_loop() -> None:
                    interval = _as_int(claimed.get("heartbeat_interval_seconds"), 300)
                    lease_minutes = _as_int(claimed.get("timeout_minutes"), 30)
                    while not stop_heartbeat.is_set():
                        try:
                            update_task_heartbeat(
                                queue_path=state_paths["queue"],
                                lock_path=state_paths["queue_lock"],
                                task_id=str(claimed["task_id"]),
                                lease_duration_minutes=lease_minutes,
                            )
                        except Exception:
                            print(
                                f"WARN: heartbeat update failed for task {claimed.get('task_id')}",
                                file=sys.stderr,
                                flush=True,
                            )
                            traceback.print_exc()
                        stop_heartbeat.wait(timeout=max(1, interval))

                hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name="task-heartbeat")
                hb_thread.start()

                try:
                    verdict, details = _execute_claimed_task(
                        task=claimed,
                        mission=mission,
                        run_id=run_id,
                        run_dir=run_dir,
                        framework_lock_hash=_sha256_file(manifest),
                        git_commit=_git_commit(root),
                        experiments_path=experiments_path,
                        experiments_lock=experiments_lock,
                    )
                except Exception as exc:
                    verdict = "ERROR"
                    details = {"error": f"{type(exc).__name__}: {exc}"}
                    event_id = compute_event_id(
                        run_id=run_id,
                        task_id=str(claimed.get("task_id", "unknown")),
                        strategy_id=str(claimed.get("strategy_name", "unknown")),
                        stage="task_error",
                        attempt=_as_int(claimed.get("retries"), 0) + 1,
                    )
                    log_experiment(
                        {
                            "event_id": event_id,
                            "run_id": run_id,
                            "agent": str(args.worker_agent),
                            "event": "task_error",
                            "task_id": str(claimed.get("task_id", "")),
                            "strategy_name": str(claimed.get("strategy_name", "")),
                            "error": details["error"],
                            "verdict": "ERROR",
                        },
                        experiments_path=experiments_path,
                        lock_path=experiments_lock,
                    )
                finally:
                    stop_heartbeat.set()
                    hb_thread.join(timeout=5)

                task_completed = _complete_claimed_task_safely(
                    state_paths=state_paths,
                    agent_name=str(args.worker_agent),
                    claimed=claimed,
                    verdict=verdict,
                    details=details,
                    run_id=run_id,
                    experiments_path=experiments_path,
                    experiments_lock=experiments_lock,
                )
                if task_completed:
                    persisted_task = _read_persisted_task_row(
                        queue_path=state_paths["queue"],
                        queue_lock=state_paths["queue_lock"],
                        task_id=str(claimed.get("task_id", "")),
                    )
                    if persisted_task is not None:
                        update_learning_scorecard(
                            scorecard_path=state_paths["scorecard"],
                            scorecard_lock=state_paths["scorecard_lock"],
                            task=persisted_task,
                        )
                resolved_handoffs = _finalize_ready_validation_handoffs(
                    queue_path=state_paths["queue"],
                    queue_lock=state_paths["queue_lock"],
                    handoffs_path=state_paths["handoffs"],
                    handoffs_lock=state_paths["handoffs_lock"],
                )
                for handoff in resolved_handoffs:
                    update_learning_scorecard(
                        scorecard_path=state_paths["scorecard"],
                        scorecard_lock=state_paths["scorecard_lock"],
                        handoff=handoff,
                    )
                    payload = handoff.get("payload") if isinstance(handoff.get("payload"), dict) else {}
                    result = handoff.get("result") if isinstance(handoff.get("result"), dict) else {}
                    task_rows = result.get("tasks") if isinstance(result.get("tasks"), list) else []
                    selection_attempted = any(
                        bool(task_row.get("selection_attempted", False))
                        for task_row in task_rows
                        if isinstance(task_row, dict)
                    )
                    selection_passed = any(
                        str(task_row.get("selection_verdict", "")).upper() == "PASS"
                        for task_row in task_rows
                        if isinstance(task_row, dict)
                    )
                    log_experiment(
                        {
                            "run_id": run_id,
                            "agent": "validator",
                            "event": "validation_handoff_completed",
                            "handoff_id": str(handoff.get("handoff_id", "")),
                            "strategy_name": str(payload.get("strategy_name", "")),
                            "hypothesis_id": str(payload.get("hypothesis_id", "")),
                            "theme_tag": str(payload.get("theme_tag", "")),
                            "setup_key": str(payload.get("setup_key", "")),
                            "setup_label": str(payload.get("setup_label", "")),
                            "task_count": result.get("task_count"),
                            "overall_verdict": result.get("overall_verdict"),
                            "pass_count": result.get("pass_count"),
                            "fail_count": result.get("fail_count"),
                            "error_count": result.get("error_count"),
                            "selection_attempted": selection_attempted,
                            "selection_passed": selection_passed,
                        },
                        experiments_path=experiments_path,
                        lock_path=experiments_lock,
                    )
                protected_task_ids = _pending_validation_task_ids(
                    handoffs_path=state_paths["handoffs"],
                    handoffs_lock=state_paths["handoffs_lock"],
                )
                pruned = _prune_terminal_tasks(
                    queue_path=state_paths["queue"],
                    lock_path=state_paths["queue_lock"],
                    max_terminal_tasks=queue_terminal_keep,
                    protected_task_ids=protected_task_ids,
                )
                if pruned > 0:
                    print(f"Queue compaction: pruned {pruned} terminal tasks.")
                continue

            time.sleep(poll_seconds)
    finally:
        stop_watchdog.set()
        watchdog_thread.join(timeout=5)

    provenance = {
        "framework_version": framework_version,
        "git_commit": _git_commit(root),
        "framework_lock_hash": _sha256_file(manifest),
        "python_version": sys.version.split()[0],
    }
    atomic_json_write(run_dir / "provenance.json", provenance)
    atomic_json_write(
        run_dir / "queue_final.json",
        read_json_file(
            json_path=state_paths["queue"],
            lock_path=state_paths["queue_lock"],
            default_payload={"schema_version": "1.0", "tasks": []},
        ),
    )
    runtime_defaults = shared_state_defaults(mission_name)
    atomic_json_write(
        run_dir / "handoffs_final.json",
        read_json_file(
            json_path=state_paths["handoffs"],
            lock_path=state_paths["handoffs_lock"],
            default_payload=runtime_defaults["handoffs"],
        ),
    )
    atomic_json_write(
        run_dir / "mission_budget_final.json",
        read_json_file(
            json_path=state_paths["budget"],
            lock_path=state_paths["budget_lock"],
            default_payload=runtime_defaults["budget"],
        ),
    )

    summary = {
        "run_id": run_id,
        "timestamp": _utc_now(),
        "framework_version": framework_version,
        "mission_path": str(mission_path),
        "mission_name": mission_name,
        "framework_lock_manifest": str(manifest),
        "framework_lock": lock_result,
        "provenance": provenance,
        "budget": budget.snapshot(),
        "queue_counts": _queue_counts(state_paths["queue"], state_paths["queue_lock"]),
        "stop_reason": reason,
    }
    atomic_json_write(run_dir / "summary.json", summary)

    run_end_event_id = compute_event_id(
        run_id=run_id,
        task_id="run",
        strategy_id=mission_name,
        stage="run_end",
        attempt=1,
    )
    log_experiment(
        {
            "event_id": run_end_event_id,
            "run_id": run_id,
            "agent": "orchestrator",
            "event": "run_end",
            "mission_name": mission_name,
            "stop_reason": reason,
            "budget": budget.snapshot(),
            "summary_path": str(run_dir / "summary.json"),
        },
        experiments_path=experiments_path,
        lock_path=experiments_lock,
    )
    print(f"Run summary: {run_dir / 'summary.json'}")
    print(f"Stop reason: {reason}")


if __name__ == "__main__":
    main()
