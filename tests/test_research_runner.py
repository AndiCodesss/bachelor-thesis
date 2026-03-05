from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import types
from typing import Any

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


def test_bootstrap_tasks_deduplicates_existing_combinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_runner_module()
    queue = tmp_path / "queue.json"
    lock = tmp_path / "queue.lock"
    queue.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tasks": [
                    {
                        "task_id": "done_1",
                        "state": "completed",
                        "strategy_name": "alpha_a",
                        "split": "validate",
                        "bar_config": "tick_610",
                        "params": {"lookback": 5},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _dummy_signal(_df, _params):
        return np.array([0], dtype=np.int8)

    dummy_module = types.SimpleNamespace(DEFAULT_PARAMS={"lookback": 5}, generate_signal=_dummy_signal)
    monkeypatch.setattr(mod, "discover_signals", lambda: {"alpha_a": _dummy_signal})
    monkeypatch.setattr(mod, "load_signal_module", lambda _name: dummy_module)

    created = mod._bootstrap_tasks_if_empty(
        queue_path=queue,
        lock_path=lock,
        mission={
            "bar_configs": ["tick_610"],
            "splits_allowed": ["validate"],
        },
        max_new_tasks=10,
    )
    assert created == 0


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

    seen = mod._seed_seen_terminal_task_ids(queue_path, queue_path.with_suffix(".lock"), already_counted=2)
    assert seen == {"t1", "t2"}

    # After seeding, only truly new terminal tasks should be counted.
    new_verdicts = mod._collect_new_terminal_task_verdicts(queue_path, queue_path.with_suffix(".lock"), seen)
    assert new_verdicts == ["PASS"]


def test_prune_terminal_tasks_keeps_recent_terminal_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_runner_module()
    monkeypatch.setattr(mod, "_MIN_QUEUE_TERMINAL_KEEP", 1)

    queue_path = tmp_path / "queue.json"
    lock_path = tmp_path / "queue.lock"
    queue_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tasks": [
                    {"task_id": "p1", "state": "pending"},
                    {"task_id": "c1", "state": "completed", "created_at": "2026-01-01T00:00:00+00:00", "completed_at": "2026-01-01T00:01:00+00:00"},
                    {"task_id": "c2", "state": "failed", "created_at": "2026-01-01T00:02:00+00:00", "completed_at": "2026-01-01T00:03:00+00:00"},
                    {"task_id": "c3", "state": "completed", "created_at": "2026-01-01T00:04:00+00:00", "completed_at": "2026-01-01T00:05:00+00:00"},
                    {"task_id": "w1", "state": "in_progress"},
                ],
            }
        ),
        encoding="utf-8",
    )

    pruned = mod._prune_terminal_tasks(
        queue_path=queue_path,
        lock_path=lock_path,
        max_terminal_tasks=2,
    )
    assert pruned == 1

    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    remaining_ids = [str(t.get("task_id")) for t in payload.get("tasks", [])]
    assert "p1" in remaining_ids
    assert "w1" in remaining_ids
    assert "c1" not in remaining_ids
    assert "c2" in remaining_ids
    assert "c3" in remaining_ids


def test_finalize_ready_validation_handoffs_moves_terminal_request(
    tmp_path: Path,
):
    mod = _load_runner_module()
    queue_path = tmp_path / "queue.json"
    queue_lock = tmp_path / "queue.lock"
    handoffs_path = tmp_path / "handoffs.json"
    handoffs_lock = tmp_path / "handoffs.lock"

    queue_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tasks": [
                    {
                        "task_id": "t1",
                        "state": "failed",
                        "verdict": "FAIL",
                        "strategy_name": "alpha_x",
                        "bar_config": "tick_610",
                        "completed_at": "2026-01-01T00:01:00+00:00",
                        "details": {
                            "metrics": {
                                "sharpe_ratio": -0.5,
                                "trade_count": 10,
                                "net_pnl": -120.0,
                            }
                        },
                    },
                    {
                        "task_id": "t2",
                        "state": "failed",
                        "verdict": "FAIL",
                        "strategy_name": "alpha_x",
                        "bar_config": "volume_2000",
                        "completed_at": "2026-01-01T00:02:00+00:00",
                        "details": {
                            "metrics": {
                                "sharpe_ratio": -0.2,
                                "trade_count": 14,
                                "net_pnl": -80.0,
                            }
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    handoffs_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "pending": [
                    {
                        "handoff_id": "h1",
                        "handoff_type": "validation_request",
                        "from_agent": "llm_orchestrator",
                        "to_agent": "validator",
                        "state": "pending",
                        "payload": {
                            "strategy_name": "alpha_x",
                            "hypothesis_id": "h_001",
                            "task_ids": ["t1", "t2"],
                        },
                    }
                ],
                "completed": [],
            }
        ),
        encoding="utf-8",
    )

    resolved = mod._finalize_ready_validation_handoffs(
        queue_path=queue_path,
        queue_lock=queue_lock,
        handoffs_path=handoffs_path,
        handoffs_lock=handoffs_lock,
    )
    assert len(resolved) == 1
    assert resolved[0]["state"] == "completed"
    assert resolved[0]["result"]["overall_verdict"] == "FAIL"
    assert resolved[0]["result"]["task_count"] == 2
    assert resolved[0]["result"]["fail_count"] == 2

    payload = json.loads(handoffs_path.read_text(encoding="utf-8"))
    assert payload["pending"] == []
    assert len(payload["completed"]) == 1


def test_finalize_ready_validation_handoffs_keeps_active_request_pending(
    tmp_path: Path,
):
    mod = _load_runner_module()
    queue_path = tmp_path / "queue.json"
    queue_lock = tmp_path / "queue.lock"
    handoffs_path = tmp_path / "handoffs.json"
    handoffs_lock = tmp_path / "handoffs.lock"

    queue_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tasks": [
                    {
                        "task_id": "t1",
                        "state": "in_progress",
                        "strategy_name": "alpha_x",
                        "bar_config": "tick_610",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    handoffs_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "pending": [
                    {
                        "handoff_id": "h2",
                        "handoff_type": "validation_request",
                        "payload": {
                            "strategy_name": "alpha_x",
                            "hypothesis_id": "h_002",
                            "task_ids": ["t1"],
                        },
                    }
                ],
                "completed": [],
            }
        ),
        encoding="utf-8",
    )

    resolved = mod._finalize_ready_validation_handoffs(
        queue_path=queue_path,
        queue_lock=queue_lock,
        handoffs_path=handoffs_path,
        handoffs_lock=handoffs_lock,
    )
    assert resolved == []
    payload = json.loads(handoffs_path.read_text(encoding="utf-8"))
    assert len(payload["pending"]) == 1
    assert payload["completed"] == []


def test_execute_claimed_task_gauntlet_respects_metric_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_runner_module()

    def _signal_fn(df, _params):
        arr = np.zeros(len(df), dtype=np.int8)
        arr[::2] = 1
        arr[1::2] = -1
        return arr

    dummy_module = types.SimpleNamespace(
        generate_signal=_signal_fn,
        __file__=str(tmp_path / "dummy_signal.py"),
        STRATEGY_METADATA={"version": "1.0"},
    )
    monkeypatch.setattr(mod, "load_signal_module", lambda _name: dummy_module)
    monkeypatch.setattr(mod, "compute_strategy_id", lambda *args, **kwargs: "sid_thresholds")
    monkeypatch.setattr(mod, "check_signal_causality", lambda **kwargs: [])
    monkeypatch.setattr(mod, "run_validation_gauntlet", lambda *_args, **_kwargs: {"overall_verdict": "PASS"})
    monkeypatch.setattr(
        mod,
        "compute_metrics",
        lambda _trades, **_kwargs: {
            "trade_count": 5,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0.0,
            "gross_pnl": 0.0,
            "total_costs": 0.0,
            "net_pnl": 0.0,
            "avg_trade_pnl": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_holding_time_min": 0.0,
            "avg_bars_held": 0.0,
        },
    )
    monkeypatch.setattr(mod, "run_backtest", lambda *_args, **_kwargs: pl.DataFrame(schema=mod.TRADE_SCHEMA))

    fake_file = tmp_path / "nq_2024-01-02.parquet"
    fake_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "get_split_files", lambda _split: [fake_file])

    start = np.datetime64("2024-01-02T14:30:00")
    bars = pl.DataFrame(
        {
            "ts_event": pl.datetime_range(start, start + np.timedelta64(39, "m"), interval="1m", eager=True),
            "open": np.linspace(100.0, 101.0, 40),
            "high": np.linspace(100.5, 101.5, 40),
            "low": np.linspace(99.5, 100.5, 40),
            "close": np.linspace(100.0, 101.0, 40),
        }
    ).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
    monkeypatch.setattr(mod, "load_cached_matrix", lambda *args, **kwargs: bars)

    verdict, _details = mod._execute_claimed_task(
        task={
            "task_id": "t_gauntlet_thresholds",
            "strategy_name": "dummy",
            "split": "validate",
            "bar_config": "time_1m",
            "params": {},
            "run_gauntlet": True,
            "write_candidate": False,
        },
        mission={
            "run_gauntlet": True,
            "target_sharpe": 1.0,
            "min_trade_count": 10,
            "session_filter": "rth",
            "write_candidates": False,
        },
        run_id="run_x",
        run_dir=tmp_path,
        framework_lock_hash="abc123",
        git_commit=None,
        experiments_path=tmp_path / "experiments.jsonl",
        experiments_lock=tmp_path / "experiments.lock",
    )

    assert verdict == "FAIL"


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


def test_execute_claimed_task_rejects_unknown_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_runner_module()

    dummy_module = types.SimpleNamespace(
        generate_signal=lambda _df, _params: np.zeros(8, dtype=np.int8),
        __file__=str(tmp_path / "dummy_signal.py"),
        STRATEGY_METADATA={"version": "1.0"},
    )
    monkeypatch.setattr(mod, "load_signal_module", lambda _name: dummy_module)

    with pytest.raises(ValueError, match="unsupported split"):
        mod._execute_claimed_task(
            task={
                "task_id": "t_bad_split",
                "strategy_name": "dummy",
                "split": "paper",
                "bar_config": "time_1m",
                "params": {},
                "run_gauntlet": False,
            },
            mission={
                "run_gauntlet": False,
                "target_sharpe": 0.0,
                "min_trade_count": 0,
                "session_filter": "rth",
            },
            run_id="run_x",
            run_dir=tmp_path,
            framework_lock_hash="abc123",
            git_commit=None,
            experiments_path=tmp_path / "experiments.jsonl",
            experiments_lock=tmp_path / "experiments.lock",
        )


def test_execute_claimed_task_requires_min_bars_for_causality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_runner_module()

    dummy_module = types.SimpleNamespace(
        generate_signal=lambda _df, _params: np.zeros(20, dtype=np.int8),
        __file__=str(tmp_path / "dummy_signal.py"),
        STRATEGY_METADATA={"version": "1.0"},
    )
    monkeypatch.setattr(mod, "load_signal_module", lambda _name: dummy_module)
    monkeypatch.setattr(mod, "compute_strategy_id", lambda *args, **kwargs: "sid_short")

    fake_file = tmp_path / "nq_2024-01-02.parquet"
    fake_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "get_split_files", lambda _split: [fake_file])

    start = np.datetime64("2024-01-02T14:30:00")
    bars = pl.DataFrame(
        {
            "ts_event": pl.datetime_range(start, start + np.timedelta64(19, "m"), interval="1m", eager=True),
            "open": np.linspace(100.0, 101.0, 20),
            "high": np.linspace(100.5, 101.5, 20),
            "low": np.linspace(99.5, 100.5, 20),
            "close": np.linspace(100.0, 101.0, 20),
        }
    ).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
    monkeypatch.setattr(mod, "load_cached_matrix", lambda *args, **kwargs: bars)

    causality_calls = {"count": 0}

    def _record_causality(**_kwargs):
        causality_calls["count"] += 1
        return []

    monkeypatch.setattr(mod, "check_signal_causality", _record_causality)

    with pytest.raises(ValueError, match="requires at least 33 bars"):
        mod._execute_claimed_task(
            task={
                "task_id": "t_short_causal_guard",
                "strategy_name": "dummy",
                "split": "validate",
                "bar_config": "time_1m",
                "params": {},
                "run_gauntlet": False,
            },
            mission={
                "run_gauntlet": False,
                "target_sharpe": 0.0,
                "min_trade_count": 0,
                "session_filter": "rth",
            },
            run_id="run_x",
            run_dir=tmp_path,
            framework_lock_hash="abc123",
            git_commit=None,
            experiments_path=tmp_path / "experiments.jsonl",
            experiments_lock=tmp_path / "experiments.lock",
        )

    assert causality_calls["count"] == 0


def test_execute_claimed_task_accepts_legacy_signal_function_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_runner_module()

    def _legacy_signal(df, _params):
        return np.zeros(len(df), dtype=np.int8)

    dummy_module = types.SimpleNamespace(
        signal=_legacy_signal,
        __file__=str(tmp_path / "dummy_signal.py"),
        STRATEGY_METADATA={"version": "1.0"},
    )
    monkeypatch.setattr(mod, "load_signal_module", lambda _name: dummy_module)
    monkeypatch.setattr(mod, "compute_strategy_id", lambda *args, **kwargs: "sid_legacy")
    monkeypatch.setattr(mod, "check_signal_causality", lambda **kwargs: [])

    fake_file = tmp_path / "nq_2024-01-02.parquet"
    fake_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "get_split_files", lambda _split: [fake_file])

    start = np.datetime64("2024-01-02T14:30:00")
    bars = pl.DataFrame(
        {
            "ts_event": pl.datetime_range(start, start + np.timedelta64(39, "m"), interval="1m", eager=True),
            "open": np.linspace(100.0, 101.0, 40),
            "high": np.linspace(100.5, 101.5, 40),
            "low": np.linspace(99.5, 100.5, 40),
            "close": np.linspace(100.0, 101.0, 40),
        }
    ).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
    monkeypatch.setattr(mod, "load_cached_matrix", lambda *args, **kwargs: bars)

    verdict, _details = mod._execute_claimed_task(
        task={
            "task_id": "t_legacy_fn",
            "strategy_name": "dummy",
            "split": "validate",
            "bar_config": "time_1m",
            "params": {},
            "run_gauntlet": False,
            "write_candidate": False,
        },
        mission={
            "run_gauntlet": False,
            "target_sharpe": -1.0,
            "min_trade_count": 0,
            "session_filter": "rth",
            "write_candidates": False,
        },
        run_id="run_x",
        run_dir=tmp_path,
        framework_lock_hash="abc123",
        git_commit=None,
        experiments_path=tmp_path / "experiments.jsonl",
        experiments_lock=tmp_path / "experiments.lock",
    )
    assert verdict in {"PASS", "FAIL"}


def test_execute_claimed_task_supports_stateful_signal_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_runner_module()
    captures: dict[str, Any] = {"accepts_state": None, "model_state_type": None, "last_state_calls": 0}

    def _stateful_signal(df, _params, model_state):
        assert isinstance(model_state, dict)
        model_state["calls"] = int(model_state.get("calls", 0)) + 1
        captures["last_state_calls"] = model_state["calls"]
        return np.zeros(len(df), dtype=np.int8)

    dummy_module = types.SimpleNamespace(
        generate_signal=_stateful_signal,
        __file__=str(tmp_path / "dummy_signal.py"),
        STRATEGY_METADATA={"version": "1.0"},
    )
    monkeypatch.setattr(mod, "load_signal_module", lambda _name: dummy_module)
    monkeypatch.setattr(mod, "compute_strategy_id", lambda *args, **kwargs: "sid_stateful")

    def _fake_causality(**kwargs):
        captures["accepts_state"] = kwargs.get("accepts_state")
        captures["model_state_type"] = type(kwargs.get("model_state")).__name__
        return []

    monkeypatch.setattr(mod, "check_signal_causality", _fake_causality)

    fake_1 = tmp_path / "nq_2024-01-02.parquet"
    fake_2 = tmp_path / "nq_2024-01-03.parquet"
    fake_1.write_text("", encoding="utf-8")
    fake_2.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "get_split_files", lambda _split: [fake_1, fake_2])

    start = np.datetime64("2024-01-02T14:30:00")
    bars = pl.DataFrame(
        {
            "ts_event": pl.datetime_range(start, start + np.timedelta64(39, "m"), interval="1m", eager=True),
            "open": np.linspace(100.0, 101.0, 40),
            "high": np.linspace(100.5, 101.5, 40),
            "low": np.linspace(99.5, 100.5, 40),
            "close": np.linspace(100.0, 101.0, 40),
        }
    ).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
    monkeypatch.setattr(mod, "load_cached_matrix", lambda *args, **kwargs: bars)

    verdict, _details = mod._execute_claimed_task(
        task={
            "task_id": "t_stateful",
            "strategy_name": "dummy",
            "split": "validate",
            "bar_config": "time_1m",
            "params": {},
            "run_gauntlet": False,
            "write_candidate": False,
        },
        mission={
            "run_gauntlet": False,
            "target_sharpe": -1.0,
            "min_trade_count": 0,
            "session_filter": "rth",
            "write_candidates": False,
        },
        run_id="run_x",
        run_dir=tmp_path,
        framework_lock_hash="abc123",
        git_commit=None,
        experiments_path=tmp_path / "experiments.jsonl",
        experiments_lock=tmp_path / "experiments.lock",
    )

    assert verdict in {"PASS", "FAIL"}
    assert captures["accepts_state"] is True
    assert captures["model_state_type"] == "dict"
    assert captures["last_state_calls"] == 2
