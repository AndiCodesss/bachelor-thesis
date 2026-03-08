from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_cleanup.py"
    spec = importlib.util.spec_from_file_location("run_cleanup_module", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_state_files_includes_runtime_scorecard_and_trial_state(tmp_path: Path, monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    state_dir = tmp_path / "research" / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    for rel in (
        "experiment_queue.json",
        "handoffs.json",
        "learning_scorecard.json",
        "learning_scorecard.lock",
        "mission_budget.lock",
        "trial_count.json",
        "trial_count.lock",
        "llm_orchestrator.json",
        "llm_orchestrator_A.json",
    ):
        (state_dir / rel).write_text("", encoding="utf-8")

    files = [path.relative_to(tmp_path).as_posix() for path in mod.collect_state_files()]
    assert "research/.state/learning_scorecard.json" in files
    assert "research/.state/learning_scorecard.lock" in files
    assert "research/.state/mission_budget.lock" in files
    assert "research/.state/trial_count.json" in files
    assert "research/.state/trial_count.lock" in files
    assert "research/.state/llm_orchestrator.json" in files
    assert "research/.state/llm_orchestrator_A.json" in files


def test_collect_log_files_includes_lane_logs(tmp_path: Path, monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    logs_dir = tmp_path / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for rel in (
        "llm_orchestrator.jsonl",
        "llm_orchestrator_A.jsonl",
        "llm_orchestrator_A.lock",
        "llm_orchestrator_A.out",
        "notebook_queries.jsonl",
        "notebook_queries.lock",
    ):
        (logs_dir / rel).write_text("", encoding="utf-8")

    files = [path.relative_to(tmp_path).as_posix() for path in mod.collect_log_files()]
    assert "results/logs/llm_orchestrator.jsonl" in files
    assert "results/logs/llm_orchestrator_A.jsonl" in files
    assert "results/logs/llm_orchestrator_A.lock" in files
    assert "results/logs/llm_orchestrator_A.out" in files
    assert "results/logs/notebook_queries.jsonl" in files
    assert "results/logs/notebook_queries.lock" in files
