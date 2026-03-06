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
import portalocker
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.lib.atomic_io import atomic_json_write
from research.lib.budget import MissionBudget
from research.lib.candidates import write_candidate
from research.lib.feature_groups import filter_strategy_inputs
from research.lib.coordination import (
    claim_task,
    complete_task,
    compute_event_id,
    update_json_file,
    update_task_heartbeat,
    watchdog_check_timeouts,
)
from research.lib.experiments import log_experiment
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

_ALLOWED_RESEARCH_SPLITS = {"train", "validate"}
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


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _read_json_locked(path: Path, lock_path: Path, default_payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(lock_path, mode="a", timeout=30):
        if not path.exists():
            atomic_json_write(path, default_payload)
        return _read_json(path)


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


def _ensure_runtime_state(root: Path, *, resume: bool, mission_name: str) -> dict[str, Path]:
    state_dir = root / "research" / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "queue": state_dir / "experiment_queue.json",
        "queue_lock": state_dir / "experiment_queue.lock",
        "handoffs": state_dir / "handoffs.json",
        "handoffs_lock": state_dir / "handoffs.lock",
        "budget": state_dir / "mission_budget.json",
    }
    defaults = {
        "queue": {"schema_version": "1.0", "tasks": []},
        "handoffs": {"schema_version": "1.0", "pending": [], "completed": []},
        "budget": {
            "schema_version": "1.0",
            "mission_name": mission_name,
            "experiments_run": 0,
            "failures_by_type": {},
            "started_at": None,
            "last_updated": None,
        },
    }
    for key in ("queue", "handoffs", "budget"):
        if (not resume) or (not paths[key].exists()):
            atomic_json_write(paths[key], defaults[key])
    return paths


def _queue_counts(queue_path: Path, lock_path: Path) -> dict[str, int]:
    payload = _read_json_locked(
        queue_path,
        lock_path,
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
    payload = _read_json_locked(
        queue_path,
        lock_path,
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
    payload = _read_json_locked(
        queue_path,
        lock_path,
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
) -> int:
    """Bound queue file size by dropping oldest terminal tasks."""
    keep = max(_MIN_QUEUE_TERMINAL_KEEP, int(max_terminal_tasks))
    out: dict[str, int] = {"pruned": 0}

    def _update(queue: dict[str, Any]) -> dict[str, Any]:
        tasks = list(queue.get("tasks", []))
        terminal_indices = [
            idx for idx, task in enumerate(tasks)
            if task.get("state") in {"completed", "failed"}
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
    metrics = details.get("metrics") if isinstance(details, dict) else None
    out: dict[str, Any] = {
        "task_id": str(task.get("task_id", "")),
        "strategy_name": str(task.get("strategy_name", "")),
        "bar_config": str(task.get("bar_config", "")),
        "state": str(task.get("state", "")),
        "verdict": str(task.get("verdict", "")),
        "completed_at": str(task.get("completed_at", "")),
    }
    if isinstance(metrics, dict):
        out["sharpe_ratio"] = metrics.get("sharpe_ratio")
        out["trade_count"] = metrics.get("trade_count")
        out["net_pnl"] = metrics.get("net_pnl")
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

    for row in task_rows:
        verdict = str(row.get("verdict", "")).upper()
        if verdict == "PASS":
            pass_count += 1
        elif verdict == "ERROR":
            error_count += 1
        else:
            fail_count += 1

        sharpe = row.get("sharpe_ratio")
        if isinstance(sharpe, (int, float)):
            sharpe_vals.append(float(sharpe))
        trades = row.get("trade_count")
        if isinstance(trades, int):
            trade_counts.append(int(trades))
        elif isinstance(trades, float):
            trade_counts.append(int(trades))

    if error_count > 0:
        overall = "ERROR"
    elif pass_count > 0:
        overall = "PASS"
    else:
        overall = "FAIL"

    avg_sharpe = (sum(sharpe_vals) / len(sharpe_vals)) if sharpe_vals else None
    avg_trades = (sum(trade_counts) / len(trade_counts)) if trade_counts else None

    return {
        "overall_verdict": overall,
        "task_count": len(task_rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "error_count": error_count,
        "avg_sharpe_ratio": avg_sharpe,
        "avg_trade_count": avg_trades,
        "tasks": task_rows,
    }


def _finalize_ready_validation_handoffs(
    *,
    queue_path: Path,
    queue_lock: Path,
    handoffs_path: Path,
    handoffs_lock: Path,
) -> list[dict[str, Any]]:
    queue_payload = _read_json_locked(
        queue_path,
        queue_lock,
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


def _task_bootstrap_key(*, strategy_name: str, split: str, bar_config: str, params: dict[str, Any]) -> str:
    params_obj = params if isinstance(params, dict) else {}
    params_blob = json.dumps(params_obj, sort_keys=True, separators=(",", ":"), default=str)
    return f"{strategy_name}|{split}|{bar_config}|{params_blob}"


def _task_sort_key(task: dict[str, Any]) -> tuple[int, str]:
    return (_as_int(task.get("priority"), 1_000_000), str(task.get("created_at", "")))


def _claim_next_task(queue_path: Path, lock_path: Path, *, agent_name: str) -> dict[str, Any] | None:
    payload = _read_json(queue_path)
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

    bar_configs_raw = mission.get("bar_configs", ["volume_2000"])
    bar_configs = [str(v) for v in bar_configs_raw] if isinstance(bar_configs_raw, list) else ["volume_2000"]
    split = str((mission.get("splits_allowed") or ["validate"])[0]).lower()

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
                split=str(t.get("split", "")),
                bar_config=str(t.get("bar_config", "")),
                params=t.get("params", {}) if isinstance(t.get("params", {}), dict) else {},
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
                    split=split,
                    bar_config=bar_cfg,
                    params=default_params,
                )
                if task_key in existing_task_keys:
                    continue

                task: dict[str, Any] = {
                    "task_id": task_id,
                    "state": "pending",
                    "assigned_to": None,
                    "created_at": now,
                    "strategy_name": strategy_name,
                    "split": split,
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
    split = str(task.get("split", "validate")).strip().lower()
    bar_config = str(task.get("bar_config", "volume_2000"))
    params = task.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{task_id}: params must be an object")
    if not task_id:
        raise ValueError("task_id is required")
    if not strategy_name:
        raise ValueError(f"{task_id}: strategy_name is required")
    if split == "test":
        raise PermissionError(f"{task_id}: test split is forbidden in research mode")
    if split not in _ALLOWED_RESEARCH_SPLITS:
        allowed = ", ".join(sorted(_ALLOWED_RESEARCH_SPLITS))
        raise ValueError(f"{task_id}: unsupported split '{split}'. Allowed: {allowed}")

    # Session filter from mission config (default: eth for extended hours)
    session_filter = str(mission.get("session_filter", "eth")).lower()

    parsed_bar = _parse_bar_config(bar_config)
    strategy_module = load_signal_module(strategy_name)
    strategy_fn, strategy_accepts_state = _resolve_strategy_callable(
        task_id=task_id,
        strategy_name=strategy_name,
        strategy_module=strategy_module,
    )
    strategy_state: Any | None = {} if strategy_accepts_state else None

    strategy_id = compute_strategy_id(
        strategy_name, params, strategy_fn,
        bar_config=bar_config, session_filter=session_filter,
    )
    bt_kwargs = _task_backtest_params(task, mission)
    max_files = task.get("max_files", mission.get("max_files_per_task"))
    max_files = int(max_files) if max_files is not None else None
    run_gauntlet = bool(task.get("run_gauntlet", mission.get("run_gauntlet", True)))
    write_candidate_flag = bool(task.get("write_candidate", mission.get("write_candidates", True)))

    files = get_split_files(split)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise ValueError(f"{task_id}: no data files for split={split}")

    # A/B experiment: restrict features visible to the strategy
    feature_group = str(mission.get("feature_group", "all")).lower()

    task_dir = run_dir / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    all_signals: list[pl.DataFrame] = []
    all_trades: list[pl.DataFrame] = []
    all_bars: list[pl.DataFrame] = []
    bars_processed = 0
    signal_count = 0
    causality_checked = False
    causality_frames: list[pl.DataFrame] = []
    causality_row_count = 0

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

        # Causality check (prefix invariance) once per task on >=33 bars.
        # Buffers short files so the check still runs for coarser bar sizes.
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
                    raise ValueError(f"{task_id}: signal causality failed: {causality_errors}")
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
            raise ValueError(f"{task_id}: signal contract failed: {signal_errors}")

        signal_i8 = raw_signal.astype(np.int8, copy=False)
        bars_with_signal = full_df.with_columns(pl.Series("signal", signal_i8).cast(pl.Int8))
        bars_processed += len(bars_with_signal)
        signal_count += int((bars_with_signal["signal"] != 0).sum())
        # Keep the evaluator-visible bar surface for the gauntlet and adaptive costs.
        _sig_cols = ["ts_event", "close", "signal"]
        for _c in ("open", "high", "low", "volume", "bid_price", "ask_price"):
            if _c in bars_with_signal.columns:
                _sig_cols.append(_c)
        all_signals.append(bars_with_signal.select(_sig_cols))
        eval_bar_cols = [
            c for c in ("ts_event", "open", "high", "low", "close", "volume", "bid_price", "ask_price")
            if c in bars_with_signal.columns
        ]
        all_bars.append(bars_with_signal.select(eval_bar_cols))

        trades = run_backtest(bars_with_signal, signal_col="signal", **bt_kwargs)
        if len(trades) > 0:
            trades = compute_adaptive_costs(trades, bars_with_signal)
            all_trades.append(trades)

    if causality_row_count > 0 and not causality_checked:
        raise ValueError(
            f"{task_id}: signal causality check requires at least "
            f"{_CAUSALITY_MIN_ROWS} bars, got {causality_row_count}"
        )

    signals_df = pl.concat(all_signals).sort("ts_event") if all_signals else _empty_signal_frame()
    bars_df = pl.concat(all_bars).sort("ts_event") if all_bars else pl.DataFrame()
    trades_df = pl.concat(all_trades) if all_trades else pl.DataFrame(schema=TRADE_SCHEMA)
    metrics = compute_metrics(
        trades_df,
        cost_override_col="adaptive_cost_rt" if "adaptive_cost_rt" in trades_df.columns else None,
    )

    gauntlet: dict[str, Any] | None = None
    if run_gauntlet and len(signals_df) > 0 and signal_count > 0:
        gauntlet = run_validation_gauntlet(
            signals_df,
            signal_col="signal",
            min_trades=int(mission.get("min_trade_count", 50)),
            **bt_kwargs,
        )

    advanced_validation: dict[str, Any] = {}
    if len(trades_df) > 0:
        trial_stats = deflated_sharpe_ratio(
            _daily_net_pnl_series(trades_df),
            n_trials=max(1, int(estimate_effective_trials(experiments_path).get("effective_trials", 1))),
        )
        advanced_validation["deflated_sharpe"] = trial_stats
        advanced_validation["alpha_decay"] = fit_alpha_decay(trades_df)
        if len(bars_df) > 0:
            advanced_validation["factor_attribution"] = factor_attribution(trades_df, bars_df)
    advanced_validation_gates = _evaluate_advanced_validation_gates(mission, advanced_validation)

    target_sharpe = float(mission.get("target_sharpe", 0.0))
    min_trade_count = int(mission.get("min_trade_count", 1))
    meets_metric_thresholds = (
        metrics["sharpe_ratio"] >= target_sharpe
        and metrics["trade_count"] >= min_trade_count
    )
    meets_advanced_gates = bool(advanced_validation_gates.get("passed", True))

    verdict = "FAIL"
    if run_gauntlet:
        if gauntlet and gauntlet.get("overall_verdict") == "PASS" and meets_metric_thresholds and meets_advanced_gates:
            verdict = "PASS"
    else:
        if meets_metric_thresholds and meets_advanced_gates:
            verdict = "PASS"

    summary = {
        "task_id": task_id,
        "run_id": run_id,
        "strategy_name": strategy_name,
        "strategy_id": strategy_id,
        "split": split,
        "feature_group": feature_group,
        "bar_config": bar_config,
        "bar_params": parsed_bar,
        "params": params,
        "backtest": bt_kwargs,
        "bars_processed": bars_processed,
        "signal_count": signal_count,
        "metrics": metrics,
        "gauntlet": gauntlet,
        "advanced_validation": advanced_validation,
        "advanced_validation_gates": advanced_validation_gates,
        "verdict": verdict,
    }
    summary_path = task_dir / "summary.json"
    signals_path = task_dir / "signals.parquet"
    trades_path = task_dir / "trades.parquet"
    atomic_json_write(summary_path, summary)
    signals_df.write_parquet(signals_path)
    trades_df.write_parquet(trades_path)

    artifacts = {
        "summary": str(summary_path),
        "signals": str(signals_path),
        "trades": str(trades_path),
    }
    if gauntlet is not None:
        gauntlet_path = task_dir / "gauntlet.json"
        atomic_json_write(gauntlet_path, gauntlet)
        artifacts["gauntlet"] = str(gauntlet_path)

    candidate_path: str | None = None
    if verdict == "PASS" and write_candidate_flag:
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
            "validation_metrics": metrics,
            "gauntlet_results": gauntlet or {},
            "advanced_validation": advanced_validation,
            "advanced_validation_gates": advanced_validation_gates,
            "artifacts": {
                "signal_file": str(signal_file),
                "signal_file_hash": _sha256_file(signal_file),
                "summary": artifacts["summary"],
                "signals": artifacts["signals"],
                "trades": artifacts["trades"],
            },
            "provenance": {
                "run_id": run_id,
                "git_commit": git_commit,
                "framework_lock_hash": framework_lock_hash,
            },
        }
        if "gauntlet" in artifacts:
            candidate_payload["artifacts"]["gauntlet"] = artifacts["gauntlet"]
        candidate_out = write_candidate(agent_name="validator", candidate_data=candidate_payload)
        candidate_path = str(candidate_out)

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
            "split": split,
            "feature_group": feature_group,
            "bar_config": bar_config,
            "metrics": metrics,
            "gauntlet": gauntlet,
            "advanced_validation": advanced_validation,
            "advanced_validation_gates": advanced_validation_gates,
            "verdict": verdict,
            "artifacts": artifacts,
            "candidate_path": candidate_path,
        },
        experiments_path=experiments_path,
        lock_path=experiments_lock,
    )

    details: dict[str, Any] = {
        "strategy_id": strategy_id,
        "metrics": metrics,
        "artifacts": artifacts,
        "gauntlet": gauntlet,
        "advanced_validation": advanced_validation,
        "advanced_validation_gates": advanced_validation_gates,
        "feature_group": feature_group,
    }
    if candidate_path:
        details["candidate_path"] = candidate_path
    return verdict, details


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
        help="Resume existing research/.state files instead of resetting state for this run.",
    )
    args = parser.parse_args()

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
    splits_allowed = [str(s).lower() for s in mission.get("splits_allowed", ["validate"])]
    if "test" in splits_allowed:
        raise ValueError("Mission cannot include test split in research mode")
    mission_name = str(mission.get("mission_name", mission_path.stem))
    queue_terminal_keep = _as_int(
        mission.get("queue_terminal_keep", _DEFAULT_QUEUE_TERMINAL_KEEP),
        _DEFAULT_QUEUE_TERMINAL_KEEP,
    )
    run_id = _run_id(mission_name)
    run_dir = (RESULTS_DIR / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(run_dir / "mission.json", mission)

    state_paths = _ensure_runtime_state(root, resume=bool(args.resume), mission_name=mission_name)
    budget = MissionBudget(
        max_experiments=int(args.max_experiments if args.max_experiments is not None else mission.get("max_experiments", 100)),
        kill_criteria=dict(mission.get("kill_criteria", {"FAIL": 10, "ERROR": 5})),
        state_file=state_paths["budget"],
        mission_name=mission_name,
        reset_on_mission_change=True,
    )
    seen_terminal_tasks: set[str] = (
        _seed_seen_terminal_task_ids(
            state_paths["queue"],
            state_paths["queue_lock"],
            budget.experiments_run,
        )
        if args.resume else set()
    )

    logs_dir = root / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    experiments_path = logs_dir / "research_experiments.jsonl"
    experiments_lock = logs_dir / "research_experiments.lock"
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
            "resume": bool(args.resume),
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
    print(f"Worker agent: {args.worker_agent}")
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

                complete_task(
                    queue_path=state_paths["queue"],
                    lock_path=state_paths["queue_lock"],
                    agent_name=str(args.worker_agent),
                    task_id=str(claimed["task_id"]),
                    verdict=verdict,
                    details=details,
                )
                resolved_handoffs = _finalize_ready_validation_handoffs(
                    queue_path=state_paths["queue"],
                    queue_lock=state_paths["queue_lock"],
                    handoffs_path=state_paths["handoffs"],
                    handoffs_lock=state_paths["handoffs_lock"],
                )
                for handoff in resolved_handoffs:
                    payload = handoff.get("payload") if isinstance(handoff.get("payload"), dict) else {}
                    result = handoff.get("result") if isinstance(handoff.get("result"), dict) else {}
                    log_experiment(
                        {
                            "run_id": run_id,
                            "agent": "validator",
                            "event": "validation_handoff_completed",
                            "handoff_id": str(handoff.get("handoff_id", "")),
                            "strategy_name": str(payload.get("strategy_name", "")),
                            "hypothesis_id": str(payload.get("hypothesis_id", "")),
                            "task_count": result.get("task_count"),
                            "overall_verdict": result.get("overall_verdict"),
                            "pass_count": result.get("pass_count"),
                            "fail_count": result.get("fail_count"),
                            "error_count": result.get("error_count"),
                        },
                        experiments_path=experiments_path,
                        lock_path=experiments_lock,
                    )
                pruned = _prune_terminal_tasks(
                    queue_path=state_paths["queue"],
                    lock_path=state_paths["queue_lock"],
                    max_terminal_tasks=queue_terminal_keep,
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
        _read_json_locked(
            state_paths["queue"],
            state_paths["queue_lock"],
            default_payload={"schema_version": "1.0", "tasks": []},
        ),
    )
    atomic_json_write(run_dir / "handoffs_final.json", _read_json(state_paths["handoffs"]))
    atomic_json_write(run_dir / "mission_budget_final.json", _read_json(state_paths["budget"]))

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
