"""Cumulative hypothesis trial counter for deflated Sharpe ratio correction."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
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


def _strategy_family(row: dict[str, Any]) -> str | None:
    """Best-effort strategy family label for trial correlation grouping."""
    name = row.get("strategy_name") or row.get("strategy")
    if name:
        return str(name).strip() or None
    sid = row.get("strategy_id")
    if sid:
        return str(sid).strip() or None
    return None


def estimate_effective_trials(
    experiments_path: Path | str = _DEFAULT_EXPERIMENTS_PATH,
) -> dict[str, Any]:
    """Estimate correlation-adjusted effective trial count.

    Method:
    - raw_trials: all valid JSON object rows
    - unique_strategies: unique strategy_id or (strategy_name, params) keys
    - effective_trials: sum(sqrt(n_family)) across strategy families

    The sqrt-family aggregation is a conservative correlation adjustment:
    large parameter sweeps in one family contribute sublinearly.
    """
    path = Path(experiments_path)
    if not path.exists():
        return {
            "raw_trials": 0,
            "unique_strategies": 0,
            "family_counts": {},
            "family_count": 0,
            "effective_trials": 1,
            "method": "sqrt_family",
        }

    raw_trials = 0
    families: Counter[str] = Counter()
    unique: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row_raw = line.strip()
            if not row_raw:
                continue
            try:
                row = json.loads(row_raw)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue

            raw_trials += 1

            strategy_id = row.get("strategy_id")
            if strategy_id:
                unique.add(str(strategy_id))
            else:
                strategy_name = row.get("strategy_name") or row.get("strategy")
                if strategy_name:
                    params = row.get("params", {})
                    key = (
                        f"{strategy_name}:"
                        f"{json.dumps(params, sort_keys=True, separators=(',', ':'), default=str)}"
                    )
                    unique.add(key)

            fam = _strategy_family(row)
            if fam:
                families[fam] += 1

    if raw_trials <= 0:
        eff = 1
    elif families:
        eff = int(round(sum(math.sqrt(float(n)) for n in families.values())))
        eff = max(1, min(eff, raw_trials))
    else:
        eff = 1

    return {
        "raw_trials": int(raw_trials),
        "unique_strategies": int(len(unique)),
        "family_counts": dict(families),
        "family_count": int(len(families)),
        "effective_trials": int(eff),
        "method": "sqrt_family",
    }


def sync_trial_count(
    experiments_path: Path | str = _DEFAULT_EXPERIMENTS_PATH,
    state_path: Path | str = _DEFAULT_STATE_PATH,
    lock_path: Path | str = _DEFAULT_LOCK_PATH,
) -> int:
    """Sync state to max(current, JSONL row count). Monotonic."""
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
    "estimate_effective_trials",
    "sync_trial_count",
]
