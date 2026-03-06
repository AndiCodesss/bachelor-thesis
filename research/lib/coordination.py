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
DEFAULT_LOCK_TIMEOUT_SECONDS = 30


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
    lock_timeout_seconds: int = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Lock sidecar file, apply update_fn, atomically persist JSON."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(lock_path, mode="a", timeout=max(1, int(lock_timeout_seconds))):
        _ensure_json(json_path, default_payload)
        payload = _read_json(json_path)
        updated = update_fn(payload)
        if updated is None:
            updated = payload
        atomic_json_write(json_path, updated)
        return updated


def read_json_file(
    *,
    json_path: Path,
    lock_path: Path,
    default_payload: dict[str, Any],
    lock_timeout_seconds: int = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Lock sidecar file, ensure JSON exists, and return the current payload."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(lock_path, mode="a", timeout=max(1, int(lock_timeout_seconds))):
        _ensure_json(json_path, default_payload)
        return _read_json(json_path)


def enqueue_task(
    *,
    queue_path: Path,
    lock_path: Path,
    task: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Atomically append one pending task if task_id is not already present."""
    if not isinstance(task, dict):
        raise TypeError("task must be a dict")
    task_id = str(task.get("task_id", "")).strip()
    if not task_id:
        raise ValueError("task_id is required")

    out: dict[str, Any] = {"inserted": False, "task": None}

    def _update(queue: dict[str, Any]) -> dict[str, Any]:
        tasks = queue.setdefault("tasks", [])

        for existing in tasks:
            if str(existing.get("task_id", "")) == task_id:
                out["inserted"] = False
                out["task"] = dict(existing)
                return queue

        now = _utc_now().isoformat()
        max_priority = 0
        for existing in tasks:
            try:
                max_priority = max(max_priority, int(existing.get("priority", 0)))
            except (TypeError, ValueError):
                continue

        row = dict(task)
        row.setdefault("state", "pending")
        row.setdefault("assigned_to", None)
        row.setdefault("created_at", now)
        row.setdefault("retries", 0)
        row.setdefault("max_retries", 2)
        row.setdefault("timeout_minutes", 30)
        row.setdefault("heartbeat_interval_seconds", 300)
        row.setdefault("priority", max_priority + 1)

        tasks.append(row)
        out["inserted"] = True
        out["task"] = dict(row)
        return queue

    update_json_file(
        json_path=queue_path,
        lock_path=lock_path,
        default_payload={"schema_version": "1.0", "tasks": []},
        update_fn=_update,
    )
    assert isinstance(out["inserted"], bool)
    assert isinstance(out["task"], dict)
    return bool(out["inserted"]), dict(out["task"])


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
    """Complete claimed task (in_progress -> completed/failed or requeued)."""
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
            now = _utc_now().isoformat()
            if verdict == "PASS":
                task["state"] = "completed"
                task["completed_at"] = now
            elif verdict == "NEEDS_WORK":
                retries = int(task.get("retries", 0))
                max_retries = int(task.get("max_retries", 2))
                if retries < max_retries:
                    task["state"] = "pending"
                    task["assigned_to"] = None
                    task["retries"] = retries + 1
                    task["last_requeued_at"] = now
                    task.pop("claimed_at", None)
                    task.pop("last_heartbeat", None)
                    task.pop("lease_expires_at", None)
                else:
                    task["state"] = "failed"
                    task["completed_at"] = now
                    task["failure_reason"] = (
                        f"NEEDS_WORK retries exhausted ({max_retries})"
                    )
            else:
                task["state"] = "failed"
                task["completed_at"] = now
            task["verdict"] = verdict
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
