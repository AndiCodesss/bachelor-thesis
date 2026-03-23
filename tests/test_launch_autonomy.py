from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "launch_autonomy.py"
    spec = importlib.util.spec_from_file_location("launch_autonomy_module", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_orchestrator_workers_single_lane_preserves_legacy_names(tmp_path: Path):
    mod = _load_module()

    workers = mod._orchestrator_workers(session_prefix="alpha", logs_dir=tmp_path, lane_count=1)

    assert len(workers) == 1
    assert workers[0].lane_id is None
    assert workers[0].session_name == "alpha_orchestrator"
    assert workers[0].out_path.name == "llm_orchestrator.out"
    assert workers[0].log_path.name == "llm_orchestrator.jsonl"


def test_orchestrator_workers_multi_lane_use_namespaced_sessions_and_logs(tmp_path: Path):
    mod = _load_module()

    workers = mod._orchestrator_workers(session_prefix="alpha", logs_dir=tmp_path, lane_count=3)

    assert [worker.lane_id for worker in workers] == ["A", "B", "C"]
    assert [worker.session_name for worker in workers] == [
        "alpha_orchestrator_a",
        "alpha_orchestrator_b",
        "alpha_orchestrator_c",
    ]
    assert [worker.out_path.name for worker in workers] == [
        "llm_orchestrator_A.out",
        "llm_orchestrator_B.out",
        "llm_orchestrator_C.out",
    ]


def test_last_json_event_across_uses_latest_timestamp(tmp_path: Path):
    mod = _load_module()
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    log_a.write_text(
        json.dumps({"event": "generation_enqueued", "timestamp": "2026-01-01T00:00:01+00:00", "strategy_name": "older"}) + "\n",
        encoding="utf-8",
    )
    log_b.write_text(
        json.dumps({"event": "generation_enqueued", "timestamp": "2026-01-01T00:00:02+00:00", "strategy_name": "newer"}) + "\n",
        encoding="utf-8",
    )

    latest = mod._last_json_event_across([log_a, log_b], {"generation_enqueued"})

    assert latest is not None
    assert latest["strategy_name"] == "newer"


def test_orchestrator_cli_command_defaults_to_claude(tmp_path: Path):
    mod = _load_module()
    agent_cfg = tmp_path / "agent.yaml"
    agent_cfg.write_text("provider: claude_cli\nclaude_cli:\n  binary: claude-custom\n", encoding="utf-8")

    out = mod._orchestrator_cli_command(agent_cfg)

    assert out == "claude-custom"


def test_orchestrator_cli_command_supports_codex(tmp_path: Path):
    mod = _load_module()
    agent_cfg = tmp_path / "agent.yaml"
    agent_cfg.write_text("provider: codex_cli\ncodex_cli:\n  binary: codex-custom\n", encoding="utf-8")

    out = mod._orchestrator_cli_command(agent_cfg)

    assert out == "codex-custom"
