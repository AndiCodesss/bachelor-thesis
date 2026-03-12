from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.lib.runtime_state import (
    clear_orchestrator_state,
    ensure_orchestrator_state,
    ensure_shared_state,
    list_orchestrator_state_files,
    thinker_memory_lock_path,
    thinker_memory_path,
    reset_shared_state,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ensure_shared_state_is_resume_safe(tmp_path: Path):
    paths = ensure_shared_state(tmp_path, mission_name="mission_a")
    assert _read_json(paths["scorecard"])["schema_version"] == "1.0"
    assert _read_json(paths["meta"])["mission_name"] == "mission_a"
    paths["queue"].write_text(
        json.dumps({"schema_version": "1.0", "tasks": [{"task_id": "t1", "state": "pending"}]}),
        encoding="utf-8",
    )

    ensure_shared_state(tmp_path, mission_name="mission_a")

    payload = _read_json(paths["queue"])
    assert payload["tasks"] == [{"task_id": "t1", "state": "pending"}]


def test_ensure_shared_state_rejects_mission_name_mismatch(tmp_path: Path):
    ensure_shared_state(tmp_path, mission_name="mission_a")
    with pytest.raises(ValueError, match="different mission"):
        ensure_shared_state(tmp_path, mission_name="mission_b")


def test_ensure_shared_state_rejects_fingerprint_mismatch(tmp_path: Path):
    ensure_shared_state(tmp_path, mission_name="mission_a", mission_fingerprint="fp_a")
    with pytest.raises(ValueError, match="different mission execution context"):
        ensure_shared_state(tmp_path, mission_name="mission_a", mission_fingerprint="fp_b")


def test_reset_shared_state_resets_queue_and_budget(tmp_path: Path):
    paths = ensure_shared_state(tmp_path, mission_name="mission_a")
    paths["queue"].write_text(
        json.dumps({"schema_version": "1.0", "tasks": [{"task_id": "t1", "state": "completed"}]}),
        encoding="utf-8",
    )
    paths["budget"].write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "mission_name": "mission_a",
                "experiments_run": 17,
                "failures_by_type": {"FAIL": 3},
                "started_at": "2026-01-01T00:00:00+00:00",
                "last_updated": "2026-01-01T00:01:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    reset_paths = reset_shared_state(tmp_path, mission_name="mission_b")

    assert _read_json(reset_paths["queue"]) == {"schema_version": "1.0", "tasks": []}
    assert _read_json(reset_paths["meta"])["mission_name"] == "mission_b"
    budget = _read_json(reset_paths["budget"])
    assert budget["mission_name"] == "mission_b"
    assert budget["experiments_run"] == 0
    assert budget["failures_by_type"] == {}
    scorecard = _read_json(reset_paths["scorecard"])
    assert scorecard["theme_stats"] == {}
    assert scorecard["bar_config_affinity"] == {}


def test_clear_orchestrator_state_removes_legacy_and_lane_files(tmp_path: Path):
    ensure_orchestrator_state(tmp_path, mission_name="mission_a")
    ensure_orchestrator_state(tmp_path, mission_name="mission_a", lane_id="A")
    ensure_orchestrator_state(tmp_path, mission_name="mission_a", lane_id="B")
    thinker_memory_path(tmp_path, lane_id="A").parent.mkdir(parents=True, exist_ok=True)
    thinker_memory_path(tmp_path, lane_id="A").write_text("{}", encoding="utf-8")
    thinker_memory_lock_path(tmp_path, lane_id="A").write_text("", encoding="utf-8")
    thinker_memory_path(tmp_path, lane_id="B").write_text("{}", encoding="utf-8")
    thinker_memory_lock_path(tmp_path, lane_id="B").write_text("", encoding="utf-8")

    names_before = [path.name for path in list_orchestrator_state_files(tmp_path)]
    assert names_before == [
        "llm_orchestrator.json",
        "llm_orchestrator_A.json",
        "llm_orchestrator_B.json",
    ]

    removed = clear_orchestrator_state(tmp_path)

    removed_names = sorted(path.name for path in removed)
    assert removed_names == sorted(
        names_before
        + [
            "llm_orchestrator.lock",
            "llm_orchestrator_A.lock",
            "llm_orchestrator_B.lock",
            "thinker_memory_A.json",
            "thinker_memory_A.lock",
            "thinker_memory_B.json",
            "thinker_memory_B.lock",
        ]
    )
    assert list_orchestrator_state_files(tmp_path) == []


def test_ensure_orchestrator_state_rejects_mission_name_mismatch(tmp_path: Path):
    ensure_orchestrator_state(tmp_path, mission_name="mission_a", lane_id="A")
    with pytest.raises(ValueError, match="different mission"):
        ensure_orchestrator_state(tmp_path, mission_name="mission_b", lane_id="A")
