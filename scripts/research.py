#!/usr/bin/env python3
"""Autonomous research entrypoint for the institutional framework."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
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
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.lib.atomic_io import atomic_json_write
from research.lib.budget import MissionBudget
from research.lib.candidates import write_candidate
from research.lib.edge_probe import normalize_edge_probe_config, run_edge_probe
from research.lib.feature_groups import filter_strategy_inputs
from research.lib.learning_scorecard import (
    empty_scorecard,
    rebuild_learning_scorecard,
    update_learning_scorecard,
)
from research.lib.mission_splits import ALLOWED_RESEARCH_SPLITS, resolve_research_splits
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
from src.framework.backtest.metrics import compute_daily_pnl_series
from src.framework.data.constants import RESULTS_DIR
from src.framework.security.framework_lock import verify_manifest

_CAUSALITY_MIN_ROWS = CAUSALITY_MIN_PREFIX_BARS + 1
_DEFAULT_QUEUE_TERMINAL_KEEP = 2000
_MIN_QUEUE_TERMINAL_KEEP = 200


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-").lower() or "mission"


def _run_id(mission_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{_slug(mission_name)}"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Mission file must be a YAML object: {path}")
    return payload


def _state_mode(*, fresh_state: bool) -> str:
    return "fresh" if fresh_state else "resume"


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
    """Daily net PnL series used for DSR diagnostics."""
    if len(trades) == 0:
        return np.array([], dtype=np.float64)
    cost_col = "adaptive_cost_rt" if "adaptive_cost_rt" in trades.columns else None
    daily = compute_daily_pnl_series(trades, cost_override_col=cost_col)
    return daily["net_pnl"].to_numpy().astype(np.float64)


def _evaluate_advanced_validation_gates(
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


def _task_feedback_summary(task: dict[str, Any]) -> dict[str, Any]:
    details = task.get("details")
    source = task.get("source") if isinstance(task.get("source"), dict) else {}
    metrics = details.get("metrics") if isinstance(details, dict) else None
    selection_result = details.get("selection_result") if isinstance(details, dict) else None
    edge_probe = details.get("edge_probe") if isinstance(details, dict) else None
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
    if isinstance(edge_probe, dict):
        out["edge_probe_status"] = str(edge_probe.get("status", ""))
        out["edge_probe_passed"] = bool(edge_probe.get("passed", False))
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
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_bar_config(bar_config: str) -> dict[str, Any]:
    raw = str(bar_config).strip().lower()
    if raw.startswith("tick_"):
        return {
            "bar_type": "tick",
            "bar_size": "5m",
            "bar_threshold": int(raw.split("_", 1)[1]),
        }
    if raw.startswith("volume_"):
        return {
            "bar_type": "volume",
            "bar_size": "5m",
            "bar_threshold": int(raw.split("_", 1)[1]),
        }
    if raw.startswith("vol_"):
        return {
            "bar_type": "volume",
            "bar_size": "5m",
            "bar_threshold": int(raw.split("_", 1)[1]),
        }
    if raw.startswith("time_"):
        return {
            "bar_type": "time",
            "bar_size": raw.split("_", 1)[1],
            "bar_threshold": None,
        }
    raise ValueError(f"Unsupported bar_config '{bar_config}'")


def _empty_signal_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_event": pl.Series([], dtype=pl.Datetime("ns", "UTC")),
            "close": pl.Series([], dtype=pl.Float64),
            "signal": pl.Series([], dtype=pl.Int8),
        }
    )


def _validate_signal_array(signal: np.ndarray, expected_len: int) -> list[str]:
    errors: list[str] = []
    if signal.ndim != 1:
        errors.append(f"signal must be 1D, got ndim={signal.ndim}")
        return errors
    if len(signal) != expected_len:
        errors.append(f"signal length {len(signal)} != expected {expected_len}")
        return errors
    if np.isnan(signal).any():
        errors.append("signal contains NaN")
    uniq = set(np.unique(signal).tolist())
    if not uniq.issubset({-1, 0, 1}):
        errors.append(f"signal contains invalid values: {sorted(uniq)}")
    return errors


def _resolve_strategy_callable(
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
        p for p in params
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


def _invoke_strategy_callable(
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


def _task_bootstrap_key(
    *,
    strategy_name: str,
    split: str,
    bar_config: str,
    params: dict[str, Any],
    selection_split: str | None = None,
) -> str:
    params_obj = params if isinstance(params, dict) else {}
    params_blob = json.dumps(params_obj, sort_keys=True, separators=(",", ":"), default=str)
    sel = str(selection_split).strip().lower() if selection_split is not None else ""
    return f"{strategy_name}|{split}|{sel}|{bar_config}|{params_blob}"


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
    mission_bt = mission.get("backtest", {})
    if not isinstance(mission_bt, dict):
        mission_bt = {}

    exit_bars_raw = task.get("exit_bars", mission_bt.get("exit_bars"))
    profit_target_raw = task.get("profit_target", mission_bt.get("profit_target"))
    stop_loss_raw = task.get("stop_loss", mission_bt.get("stop_loss"))

    return {
        "entry_on_next_open": bool(task.get("entry_on_next_open", mission_bt.get("entry_on_next_open", True))),
        "max_daily_loss": _as_float(task.get("max_daily_loss", mission_bt.get("max_daily_loss", 1000.0)), 1000.0),
        "exit_bars": int(exit_bars_raw) if exit_bars_raw is not None else None,
        "profit_target": float(profit_target_raw) if profit_target_raw is not None else None,
        "stop_loss": float(stop_loss_raw) if stop_loss_raw is not None else None,
    }


def _task_split_plan(task: dict[str, Any], mission: dict[str, Any]) -> dict[str, str | None]:
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
            # Resume compatibility: legacy queued tasks may only carry `split`.
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


def _selection_gate_config(
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
    edge_probe_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = get_split_files(split)
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise ValueError(f"{task_id}: no data files for split={split}")

    split_dir = task_dir / split_label
    split_dir.mkdir(parents=True, exist_ok=True)

    all_signals: list[pl.DataFrame] = []
    all_bars: list[pl.DataFrame] = []
    execution_frames: list[pl.DataFrame] = []
    bars_processed = 0
    signal_count = 0
    causality_checked = False
    causality_frames: list[pl.DataFrame] = []
    causality_row_count = 0
    strategy_state: Any | None = {} if strategy_accepts_state else None

    for file_path in files:
        full_df = load_cached_matrix(
            file_path,
            bar_size=parsed_bar["bar_size"],
            bar_type=parsed_bar["bar_type"],
            bar_threshold=parsed_bar["bar_threshold"],
            include_bar_columns=True,
            session_filter=session_filter,
        )
        if len(full_df) == 0:
            continue

        strategy_df = filter_strategy_inputs(full_df, feature_group)

        if not causality_checked:
            causality_frames.append(strategy_df)
            causality_row_count += len(strategy_df)
            if causality_row_count >= _CAUSALITY_MIN_ROWS:
                causality_df = pl.concat(causality_frames).sort("ts_event")
                causality_errors = check_signal_causality(
                    generate_fn=strategy_fn,
                    df=causality_df,
                    params=params,
                    accepts_state=strategy_accepts_state,
                    model_state={} if strategy_accepts_state else None,
                    mode="strict",
                    min_prefix_bars=CAUSALITY_MIN_PREFIX_BARS,
                )
                if causality_errors:
                    raise ValueError(
                        f"{task_id}: signal causality failed on {split_label} split: {causality_errors}",
                    )
                causality_checked = True
                causality_frames.clear()

        raw_signal = _invoke_strategy_callable(
            strategy_fn=strategy_fn,
            strategy_df=strategy_df,
            params=params,
            accepts_state=strategy_accepts_state,
            model_state=strategy_state,
        )
        signal_errors = _validate_signal_array(raw_signal, len(strategy_df))
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

        eval_bar_cols = [
            col for col in ("ts_event", "open", "high", "low", "close", "volume", "bid_price", "ask_price", "signal")
            if col in bars_with_signal.columns
        ]
        execution_frame = bars_with_signal.select(eval_bar_cols)
        execution_frames.append(execution_frame)
        all_bars.append(execution_frame.drop("signal"))

    if causality_row_count > 0 and not causality_checked:
        raise ValueError(
            f"{task_id}: signal causality check on {split_label} split requires at least "
            f"{_CAUSALITY_MIN_ROWS} bars, got {causality_row_count}",
        )

    signals_df = pl.concat(all_signals).sort("ts_event") if all_signals else _empty_signal_frame()
    bars_df = pl.concat(all_bars).sort("ts_event") if all_bars else pl.DataFrame()
    edge_probe_summary = (
        run_edge_probe(
            execution_frames=execution_frames,
            entry_on_next_open=bool(bt_kwargs.get("entry_on_next_open", False)),
            config=edge_probe_config,
        )
        if split_label == "search"
        else {"enabled": False, "passed": True, "status": "selection_skip", "events": 0, "horizon_results": []}
    )

    trades_df = pl.DataFrame(schema=TRADE_SCHEMA)
    if bool(edge_probe_summary.get("passed", True)):
        all_trades: list[pl.DataFrame] = []
        for frame in execution_frames:
            trades = run_backtest(frame, signal_col="signal", **bt_kwargs)
            if len(trades) > 0:
                trades = compute_adaptive_costs(trades, frame)
                all_trades.append(trades)
        if all_trades:
            trades_df = pl.concat(all_trades)

    metrics = compute_metrics(
        trades_df,
        cost_override_col="adaptive_cost_rt" if "adaptive_cost_rt" in trades_df.columns else None,
    )

    gauntlet: dict[str, Any] | None = None
    if (
        bool(edge_probe_summary.get("passed", True))
        and run_gauntlet
        and len(signals_df) > 0
        and signal_count > 0
    ):
        gauntlet = run_validation_gauntlet(
            signals_df,
            signal_col="signal",
            min_trades=int(min_trade_count),
            **bt_kwargs,
        )

    advanced_validation: dict[str, Any] = {}
    if bool(edge_probe_summary.get("passed", True)) and len(trades_df) > 0:
        trial_stats = deflated_sharpe_ratio(
            _daily_net_pnl_series(trades_df),
            n_trials=max(1, int(estimate_effective_trials(experiments_path).get("effective_trials", 1))),
        )
        advanced_validation["deflated_sharpe"] = trial_stats
        advanced_validation["alpha_decay"] = fit_alpha_decay(trades_df)
        if len(bars_df) > 0:
            advanced_validation["factor_attribution"] = factor_attribution(trades_df, bars_df)
    advanced_validation_gates = _evaluate_advanced_validation_gates(mission, advanced_validation)

    meets_metric_thresholds = (
        metrics["sharpe_ratio"] >= float(target_sharpe)
        and metrics["trade_count"] >= int(min_trade_count)
    )
    meets_advanced_gates = bool(advanced_validation_gates.get("passed", True))

    verdict = "FAIL"
    failure_code: str | None = None
    if not bool(edge_probe_summary.get("passed", True)):
        failure_code = str(edge_probe_summary.get("status", "no_raw_edge"))
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
        "edge_probe": edge_probe_summary,
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
    atomic_json_write(summary_path, summary)
    signals_df.write_parquet(signals_path)
    trades_df.write_parquet(trades_path)

    artifacts = {
        "summary": str(summary_path),
        "signals": str(signals_path),
        "trades": str(trades_path),
    }
    if edge_probe_summary.get("enabled", False):
        edge_probe_path = split_dir / "edge_probe.json"
        atomic_json_write(edge_probe_path, edge_probe_summary)
        artifacts["edge_probe"] = str(edge_probe_path)
    if gauntlet is not None:
        gauntlet_path = split_dir / "gauntlet.json"
        atomic_json_write(gauntlet_path, gauntlet)
        artifacts["gauntlet"] = str(gauntlet_path)

    return {
        **summary,
        "artifacts": artifacts,
    }


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

    split_plan = _task_split_plan(task, mission)
    search_split = str(split_plan["search_split"])
    selection_split = (
        str(split_plan["selection_split"])
        if split_plan["selection_split"] is not None
        else None
    )

    # Session filter from mission config (default: eth for extended hours)
    session_filter = str(mission.get("session_filter", "eth")).lower()

    parsed_bar = _parse_bar_config(bar_config)
    strategy_module = load_signal_module(strategy_name)
    strategy_fn, strategy_accepts_state = _resolve_strategy_callable(
        task_id=task_id,
        strategy_name=strategy_name,
        strategy_module=strategy_module,
    )
    theme_tag = str(task.get("theme_tag", "")).strip() or "other"
    setup_key, setup_label = task_setup_identity(task)

    strategy_id = compute_strategy_id(
        strategy_name, params, strategy_fn,
        bar_config=bar_config, session_filter=session_filter,
    )
    bt_kwargs = _task_backtest_params(task, mission)
    run_gauntlet = bool(task.get("run_gauntlet", mission.get("run_gauntlet", True)))
    write_candidate_flag = bool(task.get("write_candidate", mission.get("write_candidates", True)))
    max_files = task.get("max_files", mission.get("max_files_per_task"))
    max_files = int(max_files) if max_files is not None else None
    edge_probe_config = normalize_edge_probe_config(mission.get("edge_probe"))

    # A/B experiment: restrict features visible to the strategy
    feature_group = str(mission.get("feature_group", "all")).lower()

    task_dir = run_dir / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    search_result = _evaluate_strategy_split(
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
        edge_probe_config=edge_probe_config,
    )

    selection_result: dict[str, Any] | None = None
    if search_result["verdict"] == "PASS" and selection_split is not None:
        selection_gate = _selection_gate_config(
            mission,
            base_run_gauntlet=run_gauntlet,
        )
        selection_result = _evaluate_strategy_split(
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
    atomic_json_write(summary_path, summary)

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
            "edge_probe": search_result.get("edge_probe") or {},
            "advanced_validation": candidate_result.get("advanced_validation") or {},
            "advanced_validation_gates": candidate_result.get("advanced_validation_gates") or {},
            "search_result": search_result,
            "selection_result": selection_result,
            "artifacts": {
                "signal_file": str(signal_file),
                "signal_file_hash": _sha256_file(signal_file),
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
        candidate_out = write_candidate(agent_name="validator", candidate_data=candidate_payload)
        candidate_path = str(candidate_out)
        candidate_status = "candidate_written"

    summary["candidate_status"] = candidate_status
    if candidate_path is not None:
        summary["candidate_path"] = candidate_path
    atomic_json_write(summary_path, summary)

    attempt = _as_int(task.get("retries"), 0) + 1
    event_id = compute_event_id(
        run_id=run_id,
        task_id=task_id,
        strategy_id=strategy_id,
        stage="task",
        attempt=attempt,
    )
    log_experiment(
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
            "edge_probe": search_result.get("edge_probe"),
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
        "edge_probe": search_result.get("edge_probe"),
        "search_result": search_result,
        "selection_result": selection_result,
        "final_verdict": final_verdict,
        "candidate_status": candidate_status,
        "summary": str(summary_path),
    }
    if candidate_path:
        details["candidate_path"] = candidate_path
    return final_verdict, details


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
        state_paths = reset_shared_state(root, mission_name=mission_name)
        clear_orchestrator_state(root)
    else:
        state_paths = ensure_shared_state(root, mission_name=mission_name)
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
