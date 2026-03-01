from __future__ import annotations

from pathlib import Path

from research.lib.experiments import log_experiment, read_recent_event_ids


def test_log_experiment_is_idempotent_for_same_event_id(tmp_path: Path) -> None:
    events = tmp_path / "experiments.jsonl"
    lock = tmp_path / "experiments.lock"
    row = {"event_id": "evt-1", "agent": "validator", "verdict": "PASS"}

    first = log_experiment(row, experiments_path=events, lock_path=lock)
    second = log_experiment(row, experiments_path=events, lock_path=lock)

    assert first is True
    assert second is False
    lines = events.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_read_recent_event_ids_returns_last_ids(tmp_path: Path) -> None:
    events = tmp_path / "experiments.jsonl"
    lock = tmp_path / "experiments.lock"
    for i in range(5):
        log_experiment(
            {"event_id": f"evt-{i}", "agent": "orchestrator"},
            experiments_path=events,
            lock_path=lock,
        )
    ids = read_recent_event_ids(events, limit=3)
    assert "evt-4" in ids
    assert "evt-3" in ids
    assert "evt-2" in ids

