from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import types

import numpy as np
import polars as pl
import pytest


def _load_runner_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "research.py"
    spec = importlib.util.spec_from_file_location("research_runner_module", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_bar_config_variants():
    mod = _load_runner_module()

    assert mod._parse_bar_config("tick_610") == {
        "bar_type": "tick",
        "bar_size": "5m",
        "bar_threshold": 610,
    }
    assert mod._parse_bar_config("volume_2000") == {
        "bar_type": "volume",
        "bar_size": "5m",
        "bar_threshold": 2000,
    }
    assert mod._parse_bar_config("vol_5000") == {
        "bar_type": "volume",
        "bar_size": "5m",
        "bar_threshold": 5000,
    }
    assert mod._parse_bar_config("time_1m") == {
        "bar_type": "time",
        "bar_size": "1m",
        "bar_threshold": None,
    }

    with pytest.raises(ValueError):
        mod._parse_bar_config("range_123")


def test_validate_signal_array_contract():
    mod = _load_runner_module()

    ok = np.array([0, 1, -1, 0], dtype=np.int8)
    assert mod._validate_signal_array(ok, 4) == []

    bad_vals = np.array([0, 2, -1], dtype=np.int8)
    errors = mod._validate_signal_array(bad_vals, 3)
    assert errors and "invalid values" in errors[0]

    wrong_len = np.array([0, 1], dtype=np.int8)
    errors = mod._validate_signal_array(wrong_len, 3)
    assert errors and "length" in errors[0]


def test_bootstrap_tasks_and_claim_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_runner_module()
    queue = tmp_path / "queue.json"
    lock = tmp_path / "queue.lock"
    queue.write_text(json.dumps({"schema_version": "1.0", "tasks": []}), encoding="utf-8")

    def _dummy_signal(_df, _params):
        return np.array([0], dtype=np.int8)

    dummy_module = types.SimpleNamespace(DEFAULT_PARAMS={"lookback": 5}, generate_signal=_dummy_signal)
    monkeypatch.setattr(mod, "discover_signals", lambda: {"alpha_a": _dummy_signal})
    monkeypatch.setattr(mod, "load_signal_module", lambda _name: dummy_module)

    mission = {
        "bar_configs": ["tick_610", "volume_2000"],
        "splits_allowed": ["validate"],
        "run_gauntlet": True,
        "write_candidates": True,
    }
    created = mod._bootstrap_tasks_if_empty(
        queue_path=queue,
        lock_path=lock,
        mission=mission,
        max_new_tasks=10,
    )
    assert created == 2

    payload = json.loads(queue.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    assert len(tasks) == 2
    assert all(t["state"] == "pending" for t in tasks)
    assert {t["bar_config"] for t in tasks} == {"tick_610", "volume_2000"}
    assert all(t["params"] == {"lookback": 5} for t in tasks)

    # Reorder priorities to verify claim order is priority-ascending.
    tasks[0]["priority"] = 50
    tasks[1]["priority"] = 10
    queue.write_text(json.dumps(payload), encoding="utf-8")

    claimed = mod._claim_next_task(queue, lock, agent_name="validator")
    assert claimed is not None
    assert claimed["priority"] == 10
    assert claimed["state"] == "in_progress"
    assert claimed["assigned_to"] == "validator"


def test_execute_claimed_task_passes_session_filter_to_strategy_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression: session_filter must be defined before compute_strategy_id call."""
    mod = _load_runner_module()

    captured: dict[str, str] = {}

    def _fake_compute_strategy_id(_name, _params, _fn, bar_config="", session_filter=""):
        captured["bar_config"] = bar_config
        captured["session_filter"] = session_filter
        raise RuntimeError("stop_after_strategy_id")

    def _dummy_signal(_df, _params):
        return np.array([0], dtype=np.int8)

    dummy_module = types.SimpleNamespace(
        generate_signal=_dummy_signal,
        __file__=str(tmp_path / "dummy_signal.py"),
        STRATEGY_METADATA={"version": "1.0"},
    )

    monkeypatch.setattr(mod, "compute_strategy_id", _fake_compute_strategy_id)
    monkeypatch.setattr(mod, "load_signal_module", lambda _name: dummy_module)

    with pytest.raises(RuntimeError, match="stop_after_strategy_id"):
        mod._execute_claimed_task(
            task={
                "task_id": "t1",
                "strategy_name": "dummy",
                "split": "validate",
                "bar_config": "tick_610",
                "params": {},
            },
            mission={"session_filter": "rth"},
            run_id="run_x",
            run_dir=tmp_path,
            framework_lock_hash="abc123",
            git_commit=None,
            experiments_path=tmp_path / "experiments.jsonl",
            experiments_lock=tmp_path / "experiments.lock",
        )

    assert captured["bar_config"] == "tick_610"
    assert captured["session_filter"] == "rth"


def test_seed_seen_terminal_task_ids_prevents_resume_double_count(tmp_path: Path):
    mod = _load_runner_module()
    queue_path = tmp_path / "queue.json"
    queue = {
        "schema_version": "1.0",
        "tasks": [
            {
                "task_id": "t1",
                "state": "completed",
                "verdict": "PASS",
                "created_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:01:00+00:00",
            },
            {
                "task_id": "t2",
                "state": "failed",
                "verdict": "FAIL",
                "created_at": "2026-01-01T00:02:00+00:00",
                "completed_at": "2026-01-01T00:03:00+00:00",
            },
            {
                "task_id": "t3",
                "state": "completed",
                "verdict": "PASS",
                "created_at": "2026-01-01T00:04:00+00:00",
                "completed_at": "2026-01-01T00:05:00+00:00",
            },
        ],
    }
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    seen = mod._seed_seen_terminal_task_ids(queue_path, already_counted=2)
    assert seen == {"t1", "t2"}

    # After seeding, only truly new terminal tasks should be counted.
    new_verdicts = mod._collect_new_terminal_task_verdicts(queue_path, seen)
    assert new_verdicts == ["PASS"]


def test_execute_claimed_task_rejects_non_causal_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_runner_module()

    dummy_module = types.SimpleNamespace(
        generate_signal=lambda _df, _params: np.zeros(64, dtype=np.int8),
        __file__=str(tmp_path / "dummy_signal.py"),
        STRATEGY_METADATA={"version": "1.0"},
    )
    monkeypatch.setattr(mod, "load_signal_module", lambda _name: dummy_module)
    monkeypatch.setattr(mod, "compute_strategy_id", lambda *args, **kwargs: "sid_1")

    fake_file = tmp_path / "nq_2024-01-02.parquet"
    fake_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "get_split_files", lambda _split: [fake_file])

    start = np.datetime64("2024-01-02T14:30:00")
    bars = pl.DataFrame(
        {
            "ts_event": pl.datetime_range(start, start + np.timedelta64(63, "m"), interval="1m", eager=True),
            "open": np.linspace(100.0, 101.0, 64),
            "high": np.linspace(100.5, 101.5, 64),
            "low": np.linspace(99.5, 100.5, 64),
            "close": np.linspace(100.0, 101.0, 64),
        }
    ).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
    monkeypatch.setattr(mod, "load_cached_matrix", lambda *args, **kwargs: bars)
    monkeypatch.setattr(mod, "check_signal_causality", lambda **kwargs: ["non-causal signal"])

    with pytest.raises(ValueError, match="signal causality failed"):
        mod._execute_claimed_task(
            task={
                "task_id": "t_causal_guard",
                "strategy_name": "dummy",
                "split": "validate",
                "bar_config": "time_1m",
                "params": {},
                "run_gauntlet": False,
            },
            mission={
                "run_gauntlet": False,
                "target_sharpe": 10.0,
                "min_trade_count": 100,
                "session_filter": "rth",
            },
            run_id="run_x",
            run_dir=tmp_path,
            framework_lock_hash="abc123",
            git_commit=None,
            experiments_path=tmp_path / "experiments.jsonl",
            experiments_lock=tmp_path / "experiments.lock",
        )
