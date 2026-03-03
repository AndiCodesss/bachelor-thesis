from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.lib.coordination import (
    append_handoff,
    claim_task,
    complete_task,
    compute_event_id,
    enqueue_task,
    watchdog_check_timeouts,
)


def _read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_compute_event_id_is_deterministic():
    a = compute_event_id(
        run_id="run1",
        task_id="task1",
        strategy_id="s1",
        stage="smoke_test",
        attempt=1,
    )
    b = compute_event_id(
        run_id="run1",
        task_id="task1",
        strategy_id="s1",
        stage="smoke_test",
        attempt=1,
    )
    c = compute_event_id(
        run_id="run1",
        task_id="task1",
        strategy_id="s1",
        stage="smoke_test",
        attempt=2,
    )
    assert a == b
    assert a != c


def test_claim_and_complete_task(tmp_path: Path):
    queue = tmp_path / "queue.json"
    lock = tmp_path / "queue.lock"
    queue.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tasks": [
                    {
                        "task_id": "t1",
                        "state": "pending",
                        "assigned_to": None,
                        "retries": 0,
                        "max_retries": 2,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    claimed = claim_task(
        queue_path=queue,
        lock_path=lock,
        agent_name="creative-researcher",
        task_id="t1",
    )
    assert claimed is not None
    payload = _read(queue)
    task = payload["tasks"][0]
    assert task["state"] == "in_progress"
    assert task["assigned_to"] == "creative-researcher"
    assert task["claimed_at"]
    assert task["lease_expires_at"]

    out = complete_task(
        queue_path=queue,
        lock_path=lock,
        agent_name="creative-researcher",
        task_id="t1",
        verdict="PASS",
    )
    assert out["state"] == "completed"
    payload = _read(queue)
    assert payload["tasks"][0]["state"] == "completed"
    assert payload["tasks"][0]["verdict"] == "PASS"


def test_watchdog_requeues_then_fails_on_max_retries(tmp_path: Path):
    queue = tmp_path / "queue.json"
    lock = tmp_path / "queue.lock"
    now = datetime.now(timezone.utc)
    expired = (now - timedelta(minutes=1)).isoformat()
    old_hb = (now - timedelta(minutes=20)).isoformat()
    queue.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tasks": [
                    {
                        "task_id": "t1",
                        "state": "in_progress",
                        "assigned_to": "agent-a",
                        "retries": 0,
                        "max_retries": 1,
                        "lease_expires_at": expired,
                        "last_heartbeat": old_hb,
                        "heartbeat_interval_seconds": 300,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    out1 = watchdog_check_timeouts(queue_path=queue, lock_path=lock, now=now)
    assert out1["requeued"] == 1
    task = _read(queue)["tasks"][0]
    assert task["state"] == "pending"
    assert task["assigned_to"] is None
    assert task["retries"] == 1
    assert "agent-a" in task["timeout_reason"]

    # Simulate claimed again and expired with max retries reached.
    task["state"] = "in_progress"
    task["assigned_to"] = "agent-a"
    task["lease_expires_at"] = expired
    task["last_heartbeat"] = old_hb
    queue.write_text(json.dumps({"schema_version": "1.0", "tasks": [task]}), encoding="utf-8")

    out2 = watchdog_check_timeouts(queue_path=queue, lock_path=lock, now=now)
    assert out2["failed"] == 1
    task2 = _read(queue)["tasks"][0]
    assert task2["state"] == "failed"
    assert task2["verdict"] == "ERROR"


def test_append_handoff(tmp_path: Path):
    handoffs = tmp_path / "handoffs.json"
    lock = tmp_path / "handoffs.lock"
    row = append_handoff(
        handoffs_path=handoffs,
        lock_path=lock,
        handoff={
            "handoff_type": "validation_request",
            "from_agent": "creative-researcher",
            "to_agent": "validator",
            "payload": {"strategy_id": "s1"},
        },
    )
    assert row["handoff_type"] == "validation_request"
    payload = _read(handoffs)
    assert len(payload["pending"]) == 1


def test_enqueue_task_assigns_defaults_and_priority(tmp_path: Path):
    queue = tmp_path / "queue.json"
    lock = tmp_path / "queue.lock"
    queue.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tasks": [
                    {
                        "task_id": "existing_1",
                        "state": "pending",
                        "priority": 3,
                    },
                    {
                        "task_id": "existing_2",
                        "state": "completed",
                        "priority": 9,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    inserted, row = enqueue_task(
        queue_path=queue,
        lock_path=lock,
        task={
            "task_id": "new_1",
            "strategy_name": "s1",
            "split": "validate",
            "bar_config": "tick_610",
            "params": {},
        },
    )
    assert inserted is True
    assert row["task_id"] == "new_1"
    assert row["state"] == "pending"
    assert row["priority"] == 10
    assert row["assigned_to"] is None
    assert row["created_at"]


def test_enqueue_task_is_idempotent_for_same_task_id(tmp_path: Path):
    queue = tmp_path / "queue.json"
    lock = tmp_path / "queue.lock"
    queue.write_text(json.dumps({"schema_version": "1.0", "tasks": []}), encoding="utf-8")

    first_inserted, first_row = enqueue_task(
        queue_path=queue,
        lock_path=lock,
        task={"task_id": "dup_1", "strategy_name": "s1"},
    )
    second_inserted, second_row = enqueue_task(
        queue_path=queue,
        lock_path=lock,
        task={"task_id": "dup_1", "strategy_name": "s2"},
    )

    assert first_inserted is True
    assert second_inserted is False
    assert second_row["task_id"] == "dup_1"
    assert second_row["strategy_name"] == "s1"

    payload = _read(queue)
    assert len(payload["tasks"]) == 1
