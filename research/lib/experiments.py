"""Experiment event logging with idempotency and cross-process safety."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any

import portalocker


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if limit <= 0:
        return lines
    return lines[-limit:]


def read_recent_event_ids(
    experiments_path: Path | str,
    *,
    limit: int = 2000,
) -> set[str]:
    """Read most recent event ids from JSONL log."""
    path = Path(experiments_path)
    out: set[str] = set()
    for raw in _tail_lines(path, limit):
        try:
            row = json.loads(raw)
        except Exception:
            continue
        event_id = row.get("event_id") if isinstance(row, dict) else None
        if event_id:
            out.add(str(event_id))
    return out


def _default_event_id(record: dict[str, Any]) -> str:
    stable = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def log_experiment(
    record: dict[str, Any],
    *,
    experiments_path: Path | str = Path("results/logs/research_experiments.jsonl"),
    lock_path: Path | str = Path("results/logs/research_experiments.lock"),
    dedupe_window: int = 2000,
) -> bool:
    """Append one experiment row; return False if idempotent duplicate."""
    if not isinstance(record, dict):
        raise TypeError("record must be a dict")
    path = Path(experiments_path)
    lock = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)

    row = dict(record)
    row.setdefault("schema_version", "1.0")
    row.setdefault("timestamp", _utc_now())
    row.setdefault("event_id", _default_event_id(row) if row else str(uuid.uuid4()))
    event_id = str(row["event_id"])

    with portalocker.Lock(lock, mode="a", timeout=5):
        recent = read_recent_event_ids(path, limit=dedupe_window)
        if event_id in recent:
            return False

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return True


__all__ = ["log_experiment", "read_recent_event_ids"]
