"""Helpers for autonomy runtime state paths, defaults, and reset semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.lib.atomic_io import atomic_json_write
from research.lib.learning_scorecard import empty_scorecard
from research.lib.script_support import utc_now_iso
import portalocker


def state_dir(root: Path) -> Path:
    return Path(root) / "research" / ".state"


def shared_state_paths(root: Path) -> dict[str, Path]:
    root = Path(root)
    state_root = state_dir(root)
    return {
        "state_dir": state_root,
        "meta": state_root / "state_meta.json",
        "meta_lock": state_root / "state_meta.lock",
        "queue": state_root / "experiment_queue.json",
        "queue_lock": state_root / "experiment_queue.lock",
        "handoffs": state_root / "handoffs.json",
        "handoffs_lock": state_root / "handoffs.lock",
        "budget": state_root / "mission_budget.json",
        "budget_lock": state_root / "mission_budget.lock",
        "scorecard": state_root / "learning_scorecard.json",
        "scorecard_lock": state_root / "learning_scorecard.lock",
    }


def state_meta_default(mission_name: str, mission_fingerprint: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "mission_name": mission_name,
        "mission_fingerprint": mission_fingerprint,
        "created_at": utc_now_iso(),
        "last_updated": utc_now_iso(),
    }


def shared_state_defaults(
    mission_name: str,
    mission_fingerprint: str | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "meta": state_meta_default(mission_name, mission_fingerprint),
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


def _shared_state_mismatch_error(*, saved_name: str, requested_name: str) -> ValueError:
    return ValueError(
        "Existing shared runtime state belongs to a different mission "
        f"('{saved_name}' != '{requested_name}'). Use --fresh-state or run cleanup.",
    )


def _shared_state_fingerprint_error() -> ValueError:
    return ValueError(
        "Existing shared runtime state was created for a different mission execution context. "
        "Use --fresh-state or run cleanup.",
    )


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _ensure_state_meta(
    paths: dict[str, Path],
    *,
    mission_name: str,
    mission_fingerprint: str | None,
) -> None:
    meta_path = paths["meta"]
    lock_path = paths["meta_lock"]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(lock_path, mode="a", timeout=5):
        if meta_path.exists():
            payload = _read_json_if_exists(meta_path)
            saved_name = str(payload.get("mission_name", "")).strip()
            saved_fingerprint = str(payload.get("mission_fingerprint", "")).strip() or None
            if saved_name and saved_name != mission_name:
                raise _shared_state_mismatch_error(
                    saved_name=saved_name,
                    requested_name=mission_name,
                )
            if mission_fingerprint and saved_fingerprint and saved_fingerprint != mission_fingerprint:
                raise _shared_state_fingerprint_error()

            updated = False
            if not saved_name:
                payload["mission_name"] = mission_name
                updated = True
            if mission_fingerprint and saved_fingerprint != mission_fingerprint:
                payload["mission_fingerprint"] = mission_fingerprint
                updated = True
            if updated:
                payload.setdefault("schema_version", "1.0")
                payload["last_updated"] = utc_now_iso()
                atomic_json_write(meta_path, payload)
            return

        legacy_budget = _read_json_if_exists(paths["budget"])
        saved_name = str(legacy_budget.get("mission_name", "")).strip()
        if saved_name and saved_name != mission_name:
            raise _shared_state_mismatch_error(
                saved_name=saved_name,
                requested_name=mission_name,
            )
        atomic_json_write(meta_path, state_meta_default(mission_name, mission_fingerprint))


def ensure_shared_state(
    root: Path,
    *,
    mission_name: str,
    mission_fingerprint: str | None = None,
) -> dict[str, Path]:
    paths = shared_state_paths(root)
    paths["state_dir"].mkdir(parents=True, exist_ok=True)
    _ensure_state_meta(
        paths,
        mission_name=mission_name,
        mission_fingerprint=mission_fingerprint,
    )
    defaults = shared_state_defaults(mission_name, mission_fingerprint)
    for key, payload in defaults.items():
        if not paths[key].exists():
            atomic_json_write(paths[key], payload)
    return paths


def reset_shared_state(
    root: Path,
    *,
    mission_name: str,
    mission_fingerprint: str | None = None,
) -> dict[str, Path]:
    paths = shared_state_paths(root)
    paths["state_dir"].mkdir(parents=True, exist_ok=True)
    defaults = shared_state_defaults(mission_name, mission_fingerprint)
    for key, payload in defaults.items():
        atomic_json_write(paths[key], payload)
    return paths


def orchestrator_state_path(root: Path, lane_id: str | None = None) -> Path:
    filename = f"llm_orchestrator_{lane_id}.json" if lane_id else "llm_orchestrator.json"
    return state_dir(root) / filename


def thinker_memory_path(root: Path, lane_id: str | None = None) -> Path:
    filename = f"thinker_memory_{lane_id}.json" if lane_id else "thinker_memory.json"
    return state_dir(root) / filename


def orchestrator_state_lock_path(root: Path, lane_id: str | None = None) -> Path:
    return orchestrator_state_path(root, lane_id=lane_id).with_suffix(".lock")


def thinker_memory_lock_path(root: Path, lane_id: str | None = None) -> Path:
    return thinker_memory_path(root, lane_id=lane_id).with_suffix(".lock")


def orchestrator_state_lock_path_for(state_path: Path) -> Path:
    return Path(state_path).with_suffix(".lock")


def orchestrator_state_default(
    mission_name: str,
    mission_fingerprint: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "mission_name": mission_name,
        "mission_fingerprint": mission_fingerprint,
        "iterations_completed": 0,
        "total_tasks_enqueued": 0,
        "generated_modules": [],
        "last_updated": utc_now_iso(),
    }


def ensure_orchestrator_state(
    root: Path,
    *,
    mission_name: str,
    mission_fingerprint: str | None = None,
    lane_id: str | None = None,
) -> Path:
    path = orchestrator_state_path(root, lane_id=lane_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_orchestrator_state(
            path,
            orchestrator_state_default(mission_name, mission_fingerprint),
        )
        return path

    payload = read_orchestrator_state(path)
    saved_name = str(payload.get("mission_name", "")).strip()
    saved_fingerprint = str(payload.get("mission_fingerprint", "")).strip() or None
    if saved_name and saved_name != mission_name:
        raise _shared_state_mismatch_error(
            saved_name=saved_name,
            requested_name=mission_name,
        )
    if mission_fingerprint and saved_fingerprint and saved_fingerprint != mission_fingerprint:
        raise _shared_state_fingerprint_error()
    if (not saved_name) or (mission_fingerprint and saved_fingerprint != mission_fingerprint):
        merged = dict(payload)
        merged.setdefault("schema_version", "1.0")
        merged["mission_name"] = mission_name
        if mission_fingerprint:
            merged["mission_fingerprint"] = mission_fingerprint
        merged["last_updated"] = utc_now_iso()
        write_orchestrator_state(path, merged)
    return path


def reset_orchestrator_state(
    root: Path,
    *,
    mission_name: str,
    mission_fingerprint: str | None = None,
    lane_id: str | None = None,
) -> Path:
    path = orchestrator_state_path(root, lane_id=lane_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_orchestrator_state(
        path,
        orchestrator_state_default(mission_name, mission_fingerprint),
    )
    return path


def list_orchestrator_state_files(root: Path) -> list[Path]:
    state_root = state_dir(root)
    if not state_root.exists():
        return []
    return sorted(state_root.glob("llm_orchestrator*.json"))


def list_thinker_memory_files(root: Path) -> list[Path]:
    state_root = state_dir(root)
    if not state_root.exists():
        return []
    return sorted(state_root.glob("thinker_memory*.json"))


def clear_orchestrator_state(root: Path) -> list[Path]:
    removed: list[Path] = []
    for path in list_orchestrator_state_files(root):
        path.unlink(missing_ok=True)
        removed.append(path)
        lock_path = path.with_suffix(".lock")
        if lock_path.exists():
            lock_path.unlink(missing_ok=True)
            removed.append(lock_path)
    for path in list_thinker_memory_files(root):
        path.unlink(missing_ok=True)
        removed.append(path)
        lock_path = path.with_suffix(".lock")
        if lock_path.exists():
            lock_path.unlink(missing_ok=True)
            removed.append(lock_path)
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
