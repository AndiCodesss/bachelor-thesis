from __future__ import annotations

from pathlib import Path

from research.lib.budget import MissionBudget


def test_mission_budget_persists_state(tmp_path: Path):
    state_file = tmp_path / "mission_budget.json"
    budget = MissionBudget(
        max_experiments=5,
        kill_criteria={"FAIL": 3, "ERROR": 2},
        state_file=state_file,
    )
    budget.record_experiment("PASS")
    budget.record_experiment("FAIL")
    budget.record_experiment("ERROR")

    resumed = MissionBudget(
        max_experiments=5,
        kill_criteria={"FAIL": 3, "ERROR": 2},
        state_file=state_file,
    )
    snap = resumed.snapshot()
    assert snap["experiments_run"] == 3
    assert snap["failures_by_type"]["FAIL"] == 1
    assert snap["failures_by_type"]["ERROR"] == 1
    assert state_file.with_suffix(".lock").exists()


def test_mission_budget_limits(tmp_path: Path):
    state_file = tmp_path / "mission_budget.json"
    budget = MissionBudget(
        max_experiments=2,
        kill_criteria={"FAIL": 2},
        state_file=state_file,
    )
    ok, _ = budget.check_budget()
    assert ok

    budget.record_experiment("PASS")
    budget.record_experiment("PASS")
    ok2, reason2 = budget.check_budget()
    assert ok2 is False
    assert "max_experiments_reached" in reason2


def test_mission_budget_resets_on_mission_change(tmp_path: Path):
    state_file = tmp_path / "mission_budget.json"
    budget_a = MissionBudget(
        max_experiments=10,
        kill_criteria={"FAIL": 5},
        state_file=state_file,
        mission_name="mission_a",
    )
    budget_a.record_experiment("FAIL")
    assert budget_a.snapshot()["experiments_run"] == 1

    budget_b = MissionBudget(
        max_experiments=10,
        kill_criteria={"FAIL": 5},
        state_file=state_file,
        mission_name="mission_b",
        reset_on_mission_change=True,
    )
    snap = budget_b.snapshot()
    assert snap["mission_name"] == "mission_b"
    assert snap["experiments_run"] == 0
    assert snap["failures_by_type"] == {}
