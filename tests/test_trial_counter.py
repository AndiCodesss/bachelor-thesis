from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.lib.trial_counter import (
    count_trials,
    get_trial_count,
    increment_trial,
    sync_trial_count,
)
from src.framework.validation.robustness import deflated_sharpe_ratio


def test_fresh_start_returns_one(tmp_path: Path) -> None:
    """No state file on disk => get_trial_count returns 1."""
    state = tmp_path / "trial_count.json"
    assert not state.exists()
    assert get_trial_count(state) == 1


def test_increment_five_times(tmp_path: Path) -> None:
    """1 (initial) + 5 increments = 6."""
    state = tmp_path / "trial_count.json"
    lock = tmp_path / "trial_count.lock"
    for _ in range(5):
        increment_trial(state, lock)
    assert get_trial_count(state) == 6


def test_batch_increment(tmp_path: Path) -> None:
    """Single call with count=10 adds 10."""
    state = tmp_path / "trial_count.json"
    lock = tmp_path / "trial_count.lock"
    result = increment_trial(state, lock, count=10)
    assert result == 11  # 1 initial + 10
    assert get_trial_count(state) == 11


def test_sync_from_experiments_jsonl(tmp_path: Path) -> None:
    """4 rows in JSONL => count >= 4."""
    experiments = tmp_path / "experiments.jsonl"
    state = tmp_path / "trial_count.json"
    lock = tmp_path / "trial_count.lock"

    rows = [
        {"strategy_id": "strat_a", "verdict": "PASS"},
        {"strategy_id": "strat_b", "verdict": "FAIL"},
        {"strategy_id": "strat_c", "verdict": "PASS"},
        {"strategy_id": "strat_a", "verdict": "FAIL"},  # duplicate
    ]
    with open(experiments, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    result = sync_trial_count(experiments, state, lock)
    assert result >= 3
    assert get_trial_count(state) == result


def test_monotonic_after_sync_with_fewer_ids(tmp_path: Path) -> None:
    """Count never decreases even when JSONL has fewer strategy_ids."""
    state = tmp_path / "trial_count.json"
    lock = tmp_path / "trial_count.lock"
    experiments = tmp_path / "experiments.jsonl"

    # Set count high via increments
    for _ in range(20):
        increment_trial(state, lock)
    assert get_trial_count(state) == 21  # 1 + 20

    # JSONL has only 2 unique strategy_ids
    with open(experiments, "w", encoding="utf-8") as f:
        f.write(json.dumps({"strategy_id": "x"}) + "\n")
        f.write(json.dumps({"strategy_id": "y"}) + "\n")

    result = sync_trial_count(experiments, state, lock)
    assert result == 21  # must not decrease


def test_dsr_decreases_with_more_trials() -> None:
    """Integration: more trials => higher expected max Sharpe => lower DSR."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, size=500)

    dsr_1 = deflated_sharpe_ratio(returns, n_trials=1)
    dsr_10 = deflated_sharpe_ratio(returns, n_trials=10)
    dsr_100 = deflated_sharpe_ratio(returns, n_trials=100)

    assert dsr_1["available"]
    assert dsr_10["available"]
    assert dsr_100["available"]

    # More trials => stricter bar => lower DSR probability
    assert dsr_1["dsr"] > dsr_10["dsr"]
    assert dsr_10["dsr"] > dsr_100["dsr"]


def test_count_trials_read_only(tmp_path: Path) -> None:
    """count_trials counts every valid row (not just unique strategy_ids)."""
    experiments = tmp_path / "experiments.jsonl"
    with open(experiments, "w", encoding="utf-8") as f:
        for sid in ["alpha", "beta", "gamma", "alpha"]:
            f.write(json.dumps({"strategy_id": sid}) + "\n")

    assert count_trials(experiments) == 4  # all rows, duplicates count


def test_count_trials_missing_file(tmp_path: Path) -> None:
    """count_trials returns 0 when JSONL file does not exist."""
    assert count_trials(tmp_path / "nonexistent.jsonl") == 0


def test_count_trials_counts_all_valid_dicts(tmp_path: Path) -> None:
    """Every valid JSON dict row is counted; invalid JSON is skipped."""
    experiments = tmp_path / "experiments.jsonl"
    with open(experiments, "w", encoding="utf-8") as f:
        f.write(json.dumps({"strategy_id": "a"}) + "\n")
        f.write(json.dumps({"event": "run_start"}) + "\n")  # valid dict, counts
        f.write(json.dumps({"strategy_id": "b"}) + "\n")
        f.write("not json at all\n")  # invalid, skipped
    assert count_trials(experiments) == 3


def test_state_file_schema(tmp_path: Path) -> None:
    """State file matches expected schema after write."""
    state = tmp_path / "trial_count.json"
    lock = tmp_path / "trial_count.lock"
    increment_trial(state, lock, count=5)

    with open(state, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["schema_version"] == "1.0"
    assert payload["cumulative_trials"] == 6
    assert "last_updated" in payload
    assert "last_synced_from_jsonl" in payload


def test_increment_rejects_zero_count(tmp_path: Path) -> None:
    """count < 1 raises ValueError."""
    state = tmp_path / "trial_count.json"
    lock = tmp_path / "trial_count.lock"
    try:
        increment_trial(state, lock, count=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
