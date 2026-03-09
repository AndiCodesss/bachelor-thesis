"""Helpers for autonomy runtime state paths, defaults, and reset semantics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from research.lib.atomic_io import atomic_json_write
from research.lib.learning_scorecard import empty_scorecard
import portalocker


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_dir(root: Path) -> Path:
    return Path(root) / "research" / ".state"


def shared_state_paths(root: Path) -> dict[str, Path]:
    root = Path(root)
    state_root = state_dir(root)
    return {
        "state_dir": state_root,
        "queue": state_root / "experiment_queue.json",
        "queue_lock": state_root / "experiment_queue.lock",
        "handoffs": state_root / "handoffs.json",
        "handoffs_lock": state_root / "handoffs.lock",
        "budget": state_root / "mission_budget.json",
        "budget_lock": state_root / "mission_budget.lock",
        "scorecard": state_root / "learning_scorecard.json",
        "scorecard_lock": state_root / "learning_scorecard.lock",
    }


def shared_state_defaults(mission_name: str) -> dict[str, dict[str, Any]]:
    return {
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
        "scorecard": empty_scorecard(),
    }


def ensure_shared_state(root: Path, *, mission_name: str) -> dict[str, Path]:
    paths = shared_state_paths(root)
    paths["state_dir"].mkdir(parents=True, exist_ok=True)
    defaults = shared_state_defaults(mission_name)
    for key, payload in defaults.items():
        if not paths[key].exists():
            atomic_json_write(paths[key], payload)
    return paths


def reset_shared_state(root: Path, *, mission_name: str) -> dict[str, Path]:
    paths = shared_state_paths(root)
    paths["state_dir"].mkdir(parents=True, exist_ok=True)
    defaults = shared_state_defaults(mission_name)
    for key, payload in defaults.items():
        atomic_json_write(paths[key], payload)
    return paths


def orchestrator_state_path(root: Path, lane_id: str | None = None) -> Path:
    filename = f"llm_orchestrator_{lane_id}.json" if lane_id else "llm_orchestrator.json"
    return state_dir(root) / filename


def orchestrator_state_lock_path(root: Path, lane_id: str | None = None) -> Path:
    return orchestrator_state_path(root, lane_id=lane_id).with_suffix(".lock")


def orchestrator_state_lock_path_for(state_path: Path) -> Path:
    return Path(state_path).with_suffix(".lock")


def orchestrator_state_default(mission_name: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "mission_name": mission_name,
        "iterations_completed": 0,
        "total_tasks_enqueued": 0,
        "generated_modules": [],
        "last_updated": _utc_now(),
    }


def ensure_orchestrator_state(root: Path, *, mission_name: str, lane_id: str | None = None) -> Path:
    path = orchestrator_state_path(root, lane_id=lane_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_orchestrator_state(path, orchestrator_state_default(mission_name))
    return path


def reset_orchestrator_state(root: Path, *, mission_name: str, lane_id: str | None = None) -> Path:
    path = orchestrator_state_path(root, lane_id=lane_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_orchestrator_state(path, orchestrator_state_default(mission_name))
    return path


def list_orchestrator_state_files(root: Path) -> list[Path]:
    state_root = state_dir(root)
    if not state_root.exists():
        return []
    return sorted(state_root.glob("llm_orchestrator*.json"))


def clear_orchestrator_state(root: Path) -> list[Path]:
    removed: list[Path] = []
    for path in list_orchestrator_state_files(root):
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def read_orchestrator_state(path: Path) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    lock_path = orchestrator_state_lock_path_for(state_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(lock_path, mode="a", timeout=5):
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def write_orchestrator_state(path: Path, payload: dict[str, Any]) -> None:
    state_path = Path(path)
    lock_path = orchestrator_state_lock_path_for(state_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(lock_path, mode="a", timeout=5):
        atomic_json_write(state_path, payload)
