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
from typing import Any

import numpy as np
import polars as pl
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.lib.atomic_io import atomic_json_write
from research.lib.budget import MissionBudget
from research.lib.candidates import write_candidate
from research.lib.feature_groups import filter_feature_group
from research.lib.coordination import (
    claim_task,
    complete_task,
    compute_event_id,
    update_json_file,
    update_task_heartbeat,
    watchdog_check_timeouts,
)
from research.lib.experiments import log_experiment
from research.signals import compute_strategy_id, discover_signals, load_signal_module
from src.framework import __version__ as framework_version
from src.framework.api import (
    ExecutionMode,
    compute_metrics,
    get_split_files,
    load_cached_matrix,
    run_backtest,
    run_validation_gauntlet,
    set_execution_mode,
)
from src.framework.backtest.engine import TRADE_SCHEMA
from src.framework.data.constants import RESULTS_DIR
from src.framework.features_canonical.builder import LABEL_COLUMNS
from src.framework.security.framework_lock import verify_manifest


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
    cmd = ["uv", "run", "pytest", "research/signals/tests", "-q"]
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


def _queue_counts(queue_path: Path) -> dict[str, int]:
    with open(queue_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    tasks = list(payload.get("tasks", []))
    return {
        "pending": sum(1 for t in tasks if t.get("state") == "pending"),
        "in_progress": sum(1 for t in tasks if t.get("state") == "in_progress"),
        "completed": sum(1 for t in tasks if t.get("state") == "completed"),
        "failed": sum(1 for t in tasks if t.get("state") == "failed"),
    }


def _collect_new_terminal_task_verdicts(queue_path: Path, seen_task_ids: set[str]) -> list[str]:
    with open(queue_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
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
    if split == "test":
        split = "validate"

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
    split = str(task.get("split", "validate")).lower()
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

    # Session filter from mission config (default: eth for extended hours)
    session_filter = str(mission.get("session_filter", "eth")).lower()

    parsed_bar = _parse_bar_config(bar_config)
    strategy_module = load_signal_module(strategy_name)
    strategy_fn = getattr(strategy_module, "generate_signal", None)
    if not callable(strategy_fn):
        raise ValueError(f"{task_id}: strategy '{strategy_name}' has no generate_signal(df, params)")

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
    bars_processed = 0
    signal_count = 0

    for file_path in files:
        df = load_cached_matrix(
            file_path,
            bar_size=parsed_bar["bar_size"],
            bar_type=parsed_bar["bar_type"],
            bar_threshold=parsed_bar["bar_threshold"],
            include_bar_columns=True,
            session_filter=session_filter,
        )
        if len(df) == 0:
            continue

        df = filter_feature_group(df, feature_group)
        # Strip label columns so strategy code cannot access forward returns
        _label_cols_present = [c for c in LABEL_COLUMNS if c in df.columns]
        strategy_df = df.drop(_label_cols_present) if _label_cols_present else df
        raw_signal = np.asarray(strategy_fn(strategy_df, params))
        signal_errors = _validate_signal_array(raw_signal, len(df))
        if signal_errors:
            raise ValueError(f"{task_id}: signal contract failed: {signal_errors}")

        signal_i8 = raw_signal.astype(np.int8, copy=False)
        df_signal = df.with_columns(pl.Series("signal", signal_i8).cast(pl.Int8))
        bars_processed += len(df_signal)
        signal_count += int((df_signal["signal"] != 0).sum())
        # Keep OHLC columns needed by engine (open for entry_on_next_open, high/low for PT/SL)
        _sig_cols = ["ts_event", "close", "signal"]
        for _c in ("open", "high", "low"):
            if _c in df_signal.columns:
                _sig_cols.append(_c)
        all_signals.append(df_signal.select(_sig_cols))

        trades = run_backtest(df_signal, signal_col="signal", **bt_kwargs)
        if len(trades) > 0:
            all_trades.append(trades)

    signals_df = pl.concat(all_signals).sort("ts_event") if all_signals else _empty_signal_frame()
    trades_df = pl.concat(all_trades) if all_trades else pl.DataFrame(schema=TRADE_SCHEMA)
    metrics = compute_metrics(trades_df)

    gauntlet: dict[str, Any] | None = None
    if run_gauntlet and len(signals_df) > 0 and signal_count > 0:
        gauntlet = run_validation_gauntlet(signals_df, signal_col="signal", **bt_kwargs)

    verdict = "FAIL"
    if run_gauntlet:
        if gauntlet and gauntlet.get("overall_verdict") == "PASS":
            verdict = "PASS"
    else:
        target_sharpe = float(mission.get("target_sharpe", 0.0))
        min_trade_count = int(mission.get("min_trade_count", 1))
        if metrics["sharpe_ratio"] >= target_sharpe and metrics["trade_count"] >= min_trade_count:
            verdict = "PASS"

    summary = {
        "task_id": task_id,
        "run_id": run_id,
        "strategy_name": strategy_name,
        "strategy_id": strategy_id,
        "split": split,
        "bar_config": bar_config,
        "bar_params": parsed_bar,
        "params": params,
        "backtest": bt_kwargs,
        "bars_processed": bars_processed,
        "signal_count": signal_count,
        "metrics": metrics,
        "gauntlet": gauntlet,
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
            "backtest": bt_kwargs,
            "parameters": params,
            "validation_metrics": metrics,
            "gauntlet_results": gauntlet or {},
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
            "bar_config": bar_config,
            "metrics": metrics,
            "gauntlet": gauntlet,
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
    seen_terminal_tasks: set[str] = set()

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
            watchdog_check_timeouts(
                queue_path=state_paths["queue"],
                lock_path=state_paths["queue_lock"],
            )
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

    reason = "single_pass"
    try:
        while True:
            for verdict in _collect_new_terminal_task_verdicts(state_paths["queue"], seen_terminal_tasks):
                budget.record_experiment(verdict)

            can_continue, reason = budget.check_budget()
            if not can_continue:
                break

            if max_runtime_hours is not None:
                elapsed_hours = (time.monotonic() - start_monotonic) / 3600.0
                if elapsed_hours >= max_runtime_hours:
                    reason = "max_runtime_reached"
                    break

            counts = _queue_counts(state_paths["queue"])
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
                    counts = _queue_counts(state_paths["queue"])

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
                            # Best effort heartbeat; watchdog handles true failures.
                            pass
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
                continue

            if counts["pending"] == 0 and counts["in_progress"] == 0:
                reason = "queue_empty"
                break

            time.sleep(max(1, int(args.poll_seconds)))
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
    atomic_json_write(run_dir / "queue_final.json", _read_json(state_paths["queue"]))
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
        "queue_counts": _queue_counts(state_paths["queue"]),
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
