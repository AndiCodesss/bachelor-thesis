"""Cumulative hypothesis trial counter for deflated Sharpe ratio correction."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import portalocker

from research.lib.atomic_io import atomic_json_write


_DEFAULT_STATE_PATH = Path("research/.state/trial_count.json")
_DEFAULT_LOCK_PATH = Path("research/.state/trial_count.lock")
_DEFAULT_EXPERIMENTS_PATH = Path("results/logs/research_experiments.jsonl")

_SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "cumulative_trials": 1,
        "last_updated": _utc_now(),
        "last_synced_from_jsonl": None,
    }


def _read_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return _empty_state()
    with open(state_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # Floor at 1 — trial count must never be 0
    payload["cumulative_trials"] = max(int(payload.get("cumulative_trials", 1)), 1)
    return payload


def get_trial_count(
    state_path: Path | str = _DEFAULT_STATE_PATH,
) -> int:
    """Read current cumulative trial count. Returns 1 if no state file."""
    path = Path(state_path)
    if not path.exists():
        return 1
    state = _read_state(path)
    return max(int(state["cumulative_trials"]), 1)


def increment_trial(
    state_path: Path | str = _DEFAULT_STATE_PATH,
    lock_path: Path | str = _DEFAULT_LOCK_PATH,
    count: int = 1,
) -> int:
    """Atomically increment trial count. Returns new total."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    sp = Path(state_path)
    lp = Path(lock_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    lp.parent.mkdir(parents=True, exist_ok=True)

    with portalocker.Lock(lp, mode="a", timeout=5):
        state = _read_state(sp)
        state["cumulative_trials"] = int(state["cumulative_trials"]) + int(count)
        state["last_updated"] = _utc_now()
        atomic_json_write(sp, state)
        return int(state["cumulative_trials"])


def count_trials(
    experiments_path: Path | str = _DEFAULT_EXPERIMENTS_PATH,
) -> int:
    """Read-only count of every backtest execution from experiments JSONL.

    Counts all valid rows, not just unique strategy_ids. This correctly
    penalizes parameter sweeps on the same strategy in the DSR calculation.
    """
    path = Path(experiments_path)
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                count += 1
    return count


def sync_trial_count(
    experiments_path: Path | str = _DEFAULT_EXPERIMENTS_PATH,
    state_path: Path | str = _DEFAULT_STATE_PATH,
    lock_path: Path | str = _DEFAULT_LOCK_PATH,
) -> int:
    """Sync state to max(current, unique strategy_ids in JSONL). Monotonic."""
    sp = Path(state_path)
    lp = Path(lock_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    lp.parent.mkdir(parents=True, exist_ok=True)

    jsonl_count = count_trials(experiments_path)

    with portalocker.Lock(lp, mode="a", timeout=5):
        state = _read_state(sp)
        current = int(state["cumulative_trials"])
        # Monotonic: only increase, never decrease
        new_count = max(current, jsonl_count, 1)
        state["cumulative_trials"] = new_count
        state["last_updated"] = _utc_now()
        state["last_synced_from_jsonl"] = _utc_now()
        atomic_json_write(sp, state)
        return new_count


__all__ = [
    "get_trial_count",
    "increment_trial",
    "count_trials",
    "sync_trial_count",
]
