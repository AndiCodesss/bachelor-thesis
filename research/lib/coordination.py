"""Queue, handoff, and watchdog coordination primitives for autonomous agents."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import portalocker

from research.lib.atomic_io import atomic_json_write


VALID_VERDICTS = {"PASS", "FAIL", "ERROR", "NEEDS_WORK", "ABANDON"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    val = str(raw).strip()
    if not val:
        return None
    if val.endswith("Z"):
        val = val[:-1] + "+00:00"
    return datetime.fromisoformat(val)


def compute_event_id(
    *,
    run_id: str,
    task_id: str,
    strategy_id: str,
    stage: str,
    attempt: int,
) -> str:
    """Deterministic idempotency key for experiment/handoff events."""
    material = f"{run_id}:{task_id}:{strategy_id}:{stage}:{int(attempt)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _ensure_json(path: Path, default_payload: dict[str, Any]) -> None:
    if path.exists():
        return
    atomic_json_write(path, default_payload)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload at {path}")
    return payload


def update_json_file(
    *,
    json_path: Path,
    lock_path: Path,
    default_payload: dict[str, Any],
    update_fn: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any]:
    """Lock sidecar file, apply update_fn, atomically persist JSON."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_json(json_path, default_payload)
    with portalocker.Lock(lock_path, mode="a", timeout=5):
        payload = _read_json(json_path)
        updated = update_fn(payload)
        if updated is None:
            updated = payload
        atomic_json_write(json_path, updated)
        return updated


def claim_task(
    *,
    queue_path: Path,
    lock_path: Path,
    agent_name: str,
    task_id: str,
    lease_minutes: int = 30,
    heartbeat_interval_seconds: int = 300,
) -> dict[str, Any] | None:
    """Atomically claim pending task (pending -> in_progress)."""
    result_holder: dict[str, Any] = {"task": None}

    def _update(queue: dict[str, Any]) -> dict[str, Any]:
        tasks = queue.setdefault("tasks", [])
        now = _utc_now()
        for task in tasks:
            if task.get("task_id") != task_id:
                continue
            if task.get("state") != "pending":
                return queue
            task["state"] = "in_progress"
            task["assigned_to"] = agent_name
            task["claimed_at"] = now.isoformat()
            task["last_heartbeat"] = now.isoformat()
            task["heartbeat_interval_seconds"] = int(heartbeat_interval_seconds)
            task["lease_expires_at"] = (now + timedelta(minutes=int(lease_minutes))).isoformat()
            result_holder["task"] = dict(task)
            return queue
        return queue

    update_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
        update_fn=_update,
    )
    claimed = result_holder["task"]
    return None if claimed is None else dict(claimed)


def complete_task(
    *,
    queue_path: Path,
    lock_path: Path,
    agent_name: str,
    task_id: str,
    verdict: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Complete claimed task (in_progress -> completed/failed)."""
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict}")

    completed: dict[str, Any] | None = None

    def _update(queue: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        tasks = queue.setdefault("tasks", [])
        for task in tasks:
            if task.get("task_id") != task_id:
                continue
            if task.get("state") != "in_progress":
                raise ValueError(f"Task {task_id} is not in progress")
            if task.get("assigned_to") != agent_name:
                raise PermissionError(f"Task {task_id} not assigned to {agent_name}")
            task["state"] = "completed" if verdict == "PASS" else "failed"
            task["verdict"] = verdict
            task["completed_at"] = _utc_now().isoformat()
            if details:
                task["details"] = details
            completed = dict(task)
            return queue
        raise ValueError(f"Task not found: {task_id}")

    update_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
        update_fn=_update,
    )
    assert completed is not None
    return completed


def update_task_heartbeat(
    *,
    queue_path: Path,
    lock_path: Path,
    task_id: str,
    lease_duration_minutes: int = 30,
) -> None:
    """Refresh last_heartbeat and renew lease for healthy long task."""

    def _update(queue: dict[str, Any]) -> dict[str, Any]:
        tasks = queue.setdefault("tasks", [])
        now = _utc_now()
        for task in tasks:
            if task.get("task_id") != task_id:
                continue
            if task.get("state") != "in_progress":
                return queue
            task["last_heartbeat"] = now.isoformat()
            task["lease_expires_at"] = (now + timedelta(minutes=int(lease_duration_minutes))).isoformat()
            return queue
        return queue

    update_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
        update_fn=_update,
    )


def watchdog_check_timeouts(
    *,
    queue_path: Path,
    lock_path: Path,
    now: datetime | None = None,
) -> dict[str, int]:
    """Reclaim timed-out tasks based on lease expiry and heartbeat misses."""
    current = now or _utc_now()
    out = {"requeued": 0, "failed": 0}

    def _update(queue: dict[str, Any]) -> dict[str, Any]:
        tasks = queue.setdefault("tasks", [])
        for task in tasks:
            if task.get("state") != "in_progress":
                continue

            owner = task.get("assigned_to")
            retries = int(task.get("retries", 0))
            max_retries = int(task.get("max_retries", 2))
            heartbeat_seconds = int(task.get("heartbeat_interval_seconds", 300))
            lease_ts = _parse_ts(task.get("lease_expires_at"))
            hb_ts = _parse_ts(task.get("last_heartbeat"))

            lease_expired = lease_ts is not None and current > lease_ts
            missed_heartbeat = False
            if hb_ts is not None:
                missed_heartbeat = current - hb_ts > timedelta(seconds=heartbeat_seconds * 2)

            if not lease_expired and not missed_heartbeat:
                continue

            reason = (
                f"Agent {owner} lease expired"
                if lease_expired
                else f"Agent {owner} missed heartbeat"
            )

            if retries < max_retries:
                task["state"] = "pending"
                task["assigned_to"] = None
                task["retries"] = retries + 1
                task["timeout_reason"] = reason
                task["last_requeued_at"] = current.isoformat()
                out["requeued"] += 1
            else:
                task["state"] = "failed"
                task["verdict"] = "ERROR"
                task["failure_reason"] = f"{reason}; max retries ({max_retries}) exceeded"
                task["completed_at"] = current.isoformat()
                out["failed"] += 1
        return queue

    update_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
        update_fn=_update,
    )
    return out


def append_handoff(
    *,
    handoffs_path: Path,
    lock_path: Path,
    handoff: dict[str, Any],
) -> dict[str, Any]:
    """Append structured handoff to pending queue."""
    if "timestamp" not in handoff:
        handoff["timestamp"] = _utc_now().isoformat()

    def _update(payload: dict[str, Any]) -> dict[str, Any]:
        pending = payload.setdefault("pending", [])
        pending.append(handoff)
        return payload

    update_json_file(
        json_path=handoffs_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "pending": [], "completed": []},
        update_fn=_update,
    )
    return handoff
