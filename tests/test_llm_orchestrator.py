from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "llm_orchestrator.py"
    spec = importlib.util.spec_from_file_location("llm_orchestrator_module", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normalize_generation_payload_filters_bar_configs_and_code():
    mod = _load_module()
    thinker_payload = {
        "hypothesis_id": "OFI Bounce 01",
        "strategy_name_hint": "OFI Bounce",
        "bar_configs": ["foo", "tick_610", "time_1m"],
        "params_template": {"lookback": 12},
        "thesis": "test thesis",
        "entry_logic": "entry",
        "exit_logic": "exit",
        "risk_controls": ["r1"],
        "anti_lookahead_checks": ["no shift(-1)"],
        "validation_focus": ["sharpe"],
    }
    thinker = mod._normalize_thinker_brief(
        thinker_payload,
        mission_bar_configs=["tick_610", "volume_2000"],
    )
    assert thinker["hypothesis_id"] == "ofi_bounce_01"
    assert thinker["strategy_name_hint"] == "ofi_bounce"
    assert thinker["bar_configs"] == ["tick_610"]
    assert thinker["params_template"] == {"lookback": 12}

    coder_payload = {
        "strategy_name": " OFI Reversal v1 ",
        "bar_configs": ["foo", "tick_610", "time_1m"],
        "params": {"lookback": 12},
        "code": "```python\nimport numpy as np\nimport polars as pl\n"
        "def generate_signal(df, params):\n    return np.zeros(len(df), dtype=np.int8)\n```",
    }
    out = mod._normalize_coder_payload(
        coder_payload,
        mission_bar_configs=["tick_610", "volume_2000"],
        thinker_brief=thinker,
    )
    assert out["strategy_name"] == "ofi_reversal_v1"
    assert out["bar_configs"] == ["tick_610"]
    assert out["params"] == {"lookback": 12}
    assert out["code"].startswith("import numpy as np")
    assert out["code"].endswith("\n")


def test_normalize_feedback_digest_defaults():
    mod = _load_module()
    out = mod._normalize_feedback_digest({})
    assert out["strengths"]
    assert out["weaknesses"]
    assert out["guardrails"]
    assert out["next_focus"]


def test_should_wait_for_validation_requires_no_active_queue_work():
    mod = _load_module()
    assert mod._should_wait_for_validation({"pending": 0, "in_progress": 0}) is False
    assert mod._should_wait_for_validation({"pending": 1, "in_progress": 0}) is True
    assert mod._should_wait_for_validation({"pending": 0, "in_progress": 1}) is True


def test_collect_feedback_items_from_handoffs_reads_completed_results(tmp_path: Path):
    mod = _load_module()
    handoffs = tmp_path / "handoffs.json"
    handoffs.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "pending": [],
                "completed": [
                    {
                        "handoff_type": "validation_request",
                        "payload": {
                            "strategy_name": "alpha_x",
                            "hypothesis_id": "h_001",
                        },
                        "result": {
                            "overall_verdict": "FAIL",
                            "task_count": 2,
                            "pass_count": 0,
                            "fail_count": 2,
                            "error_count": 0,
                            "avg_sharpe_ratio": -0.6,
                            "avg_trade_count": 42,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out = mod._collect_feedback_items_from_handoffs(handoffs, max_items=4)
    assert len(out) == 1
    assert out[0]["event"] == "validation_result"
    assert out[0]["strategy_name"] == "alpha_x"
    assert out[0]["overall_verdict"] == "FAIL"


def test_choose_module_path_versions_when_content_differs(tmp_path: Path):
    mod = _load_module()
    signals_dir = tmp_path
    (signals_dir / "alpha_x.py").write_text("print('old')\n", encoding="utf-8")

    path1, is_new1 = mod._choose_module_path(
        signals_dir,
        strategy_name="alpha_x",
        module_code="print('new')\n",
    )
    assert path1.name == "alpha_x_02.py"
    assert is_new1 is True

    (signals_dir / "alpha_x_02.py").write_text("print('new')\n", encoding="utf-8")
    path2, is_new2 = mod._choose_module_path(
        signals_dir,
        strategy_name="alpha_x",
        module_code="print('new')\n",
    )
    assert path2.name == "alpha_x_02.py"
    assert is_new2 is False


def test_choose_module_path_whitespace_only_difference_versions_file(tmp_path: Path):
    mod = _load_module()
    signals_dir = tmp_path
    (signals_dir / "alpha_x.py").write_text("print('same')\n", encoding="utf-8")

    path, is_new = mod._choose_module_path(
        signals_dir,
        strategy_name="alpha_x",
        module_code="print('same') \n",
    )
    assert path.name == "alpha_x_02.py"
    assert is_new is True


def test_maybe_write_can_block_unexpected_overwrite(tmp_path: Path):
    mod = _load_module()
    path = tmp_path / "alpha.py"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        mod._maybe_write(path, "new\n", allow_overwrite=False)
    assert path.read_text(encoding="utf-8") == "old\n"


def test_task_id_is_stable_for_same_inputs():
    mod = _load_module()
    params = {"a": 1, "b": 2}
    t1 = mod._task_id("alpha_x", "tick_610", params, "abcd1234")
    t2 = mod._task_id("alpha_x", "tick_610", {"b": 2, "a": 1}, "abcd1234")
    assert t1 == t2


def test_extract_retry_after_seconds():
    mod = _load_module()
    assert mod._extract_retry_after_seconds("Please retry in 18.817401823s.") == 18.817401823
    assert abs(mod._extract_retry_after_seconds("Please retry in 925.654664ms.") - 0.925654664) < 1e-9
    assert mod._extract_retry_after_seconds("no retry hint") is None


def test_parse_bar_config_rejects_malformed_thresholds():
    mod = _load_module()
    with pytest.raises(ValueError):
        mod._parse_bar_config("tick_")
    with pytest.raises(ValueError):
        mod._parse_bar_config("volume_bad")
    with pytest.raises(ValueError):
        mod._parse_bar_config("time_0m")


def test_call_stage_json_repairs_invalid_json():
    mod = _load_module()

    class _FakeClient:
        def __init__(self):
            self.calls = []
            self._responses = [
                "not-json",
                '{"strengths":["x"],"weaknesses":["y"],"error_patterns":[],"guardrails":["g"],"next_focus":["n"]}',
            ]

        def generate_raw(self, **kwargs):
            self.calls.append(dict(kwargs))
            text = self._responses.pop(0)
            return SimpleNamespace(
                raw_text=text,
                model="fake-model",
                response_id="resp1",
                usage={"total_tokens": 1},
            )

    fake = _FakeClient()
    out = mod._call_stage_json(
        stage_name="feedback_analyst",
        schema_hint="feedback schema",
        client=fake,
        system_prompt="s",
        user_prompt="u",
        temperature=0.1,
        max_output_tokens=500,
        max_attempts=2,
        json_repair_attempts=1,
        stage_backoff_seconds=0.01,
        quota_backoff_seconds=0.01,
        max_backoff_seconds=0.05,
    )
    assert out.repaired is True
    assert out.payload["strengths"] == ["x"]
    assert len(fake.calls) == 2


def test_normalize_coder_payload_accepts_alternate_code_key():
    mod = _load_module()
    thinker = {
        "strategy_name_hint": "hint_name",
        "bar_configs": ["tick_610"],
    }
    payload = {
        "name": "alt_name",
        "bar_config": "tick_610",
        "params": {"x": 1},
        "python_code": "import numpy as np\nimport polars as pl\n"
        "def generate_signal(df, params):\n    return np.zeros(len(df), dtype=np.int8)\n",
    }
    out = mod._normalize_coder_payload(
        payload,
        mission_bar_configs=["tick_610"],
        thinker_brief=thinker,
    )
    assert out["strategy_name"] == "alt_name"
    assert out["bar_configs"] == ["tick_610"]
    assert "generate_signal" in out["code"]


def test_normalize_coder_payload_falls_back_to_thinker_params_template():
    mod = _load_module()
    thinker = {
        "strategy_name_hint": "hint_name",
        "bar_configs": ["tick_610"],
        "params_template": {"lookback": 21, "threshold": 0.3},
    }
    payload = {
        "strategy_name": "with_empty_params",
        "bar_configs": ["tick_610"],
        "params": {},
        "code": "import numpy as np\nimport polars as pl\n"
        "def generate_signal(df, params):\n    return np.zeros(len(df), dtype=np.int8)\n",
    }
    out = mod._normalize_coder_payload(
        payload,
        mission_bar_configs=["tick_610"],
        thinker_brief=thinker,
    )
    assert out["params"] == {"lookback": 21, "threshold": 0.3}


def test_coder_handoff_prompt_contains_structured_thinker_payload():
    mod = _load_module()
    thinker = {
        "hypothesis_id": "h_01",
        "strategy_name_hint": "alpha_hint",
        "bar_configs": ["tick_610"],
        "params_template": {"lookback": 10},
        "thesis": "x",
        "entry_logic": "y",
        "exit_logic": "z",
        "risk_controls": ["r"],
        "anti_lookahead_checks": ["a"],
        "validation_focus": ["v"],
    }
    mission = {
        "bar_configs": ["tick_610", "volume_2000"],
        "session_filter": "eth",
        "feature_group": "all",
    }
    handoff = mod._build_coder_handoff(
        thinker_brief=thinker,
        mission=mission,
        thinker_payload_hash="abc123",
    )
    assert handoff["source_role"] == "quant_thinker"
    assert handoff["payload_hash"] == "abc123"
    prompt = mod._build_coder_user_prompt(thinker_handoff=handoff)
    assert "THINKER_HANDOFF_JSON_BEGIN" in prompt
    assert '"hypothesis_id": "h_01"' in prompt
    assert "THINKER_HANDOFF_JSON_END" in prompt


def test_build_feature_knowledge_from_cached_samples():
    mod = _load_module()
    sample_cache = {
        "tick_610": pl.DataFrame({"close": [1.0], "ema_ratio_8": [0.01], "delta": [2.0]}),
        "volume_2000": pl.DataFrame({"close": [1.0], "ema_ratio_8": [0.02], "atr_14": [3.0]}),
    }
    out = mod._build_feature_knowledge(
        mission_bar_configs=["tick_610", "volume_2000"],
        split="validate",
        session_filter="eth",
        feature_group="all",
        sample_cache=sample_cache,
    )
    assert out["common_columns"] == ["close", "ema_ratio_8"]
    assert out["per_bar_extra_columns"]["tick_610"] == ["delta"]
    assert out["per_bar_extra_columns"]["volume_2000"] == ["atr_14"]
    assert out["errors"] == {}


def test_prompts_include_feature_knowledge_markers():
    mod = _load_module()
    feature_knowledge = {
        "schema_version": "1.0",
        "common_columns": ["close", "ema_ratio_8"],
        "bar_configs": {"tick_610": {"total_columns": 2, "extra_columns": 0}},
        "per_bar_extra_columns": {"tick_610": []},
        "computation_notes": ["ema_ratio_N = close / EMA - 1"],
        "errors": {},
    }
    thinker_prompt = mod._build_thinker_user_prompt(
        mission={"objective": "x", "bar_configs": ["tick_610"], "session_filter": "eth", "feature_group": "all"},
        existing_strategies=[],
        feedback_digest={"strengths": [], "weaknesses": [], "error_patterns": [], "guardrails": [], "next_focus": []},
        feature_knowledge=feature_knowledge,
    )
    assert "AVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN" in thinker_prompt
    assert '"common_columns"' in thinker_prompt

    coder_prompt = mod._build_coder_user_prompt(
        thinker_handoff={
            "handoff_version": "1.0",
            "source_role": "quant_thinker",
            "payload_hash": "h1",
            "mission_constraints": {},
            "hypothesis": {"hypothesis_id": "h1", "strategy_name_hint": "s1"},
        },
        feature_knowledge=feature_knowledge,
    )
    assert "AVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN" in coder_prompt
    assert "Prefer these precomputed features directly" in coder_prompt


def test_thinker_system_prompt_requires_internal_brainstorm():
    mod = _load_module()
    prompt = mod._build_thinker_system_prompt()
    assert "internally brainstorm" in prompt


def test_build_llm_client_rejects_non_claude_provider(tmp_path: Path):
    mod = _load_module()
    with pytest.raises(ValueError):
        mod._build_llm_client(
            provider="unsupported_provider",
            model="sonnet",
            agent_cfg={},
            root=tmp_path,
        )


def test_collect_orchestrator_feedback_items_reads_generation_rejected(tmp_path: Path):
    mod = _load_module()
    log = tmp_path / "llm_orchestrator.jsonl"
    log.write_text(
        json.dumps({
            "event": "generation_rejected",
            "strategy_name": "bad_strat",
            "errors": ["bad_strat: generate_signal failed for tick_610: KeyError: 'ema_ratio_8'"],
            "hypothesis_id": "h001",
        }) + "\n" +
        json.dumps({
            "event": "generation_error",
            "error": "LLMClientError: timeout",
            "iteration": 1,
        }) + "\n" +
        json.dumps({
            "event": "generation_enqueued",
            "strategy_name": "good_strat",
        }) + "\n",
        encoding="utf-8",
    )
    out = mod._collect_orchestrator_feedback_items(log, max_items=10)
    assert len(out) == 2
    events = {x["event"] for x in out}
    assert "generation_rejected" in events
    assert "generation_error" in events
    # generation_enqueued must NOT be included
    assert all(x["event"] != "generation_enqueued" for x in out)


def test_collect_orchestrator_feedback_items_empty_when_no_log(tmp_path: Path):
    mod = _load_module()
    out = mod._collect_orchestrator_feedback_items(tmp_path / "missing.jsonl", max_items=10)
    assert out == []


def test_collect_orchestrator_feedback_items_respects_max_items(tmp_path: Path):
    mod = _load_module()
    log = tmp_path / "llm_orchestrator.jsonl"
    lines = "\n".join(
        json.dumps({"event": "generation_rejected", "strategy_name": f"s{i}", "errors": ["e"]})
        for i in range(10)
    ) + "\n"
    log.write_text(lines, encoding="utf-8")
    out = mod._collect_orchestrator_feedback_items(log, max_items=3)
    assert len(out) == 3


def test_merged_feedback_includes_all_sources(tmp_path: Path):
    """_build_merged_feedback_items merges handoffs + research log + orchestrator log."""
    mod = _load_module()

    handoffs = tmp_path / "handoffs.json"
    handoffs.write_text(json.dumps({
        "schema_version": "1.0", "pending": [],
        "completed": [{
            "handoff_type": "validation_request",
            "payload": {"strategy_name": "s_handoff", "hypothesis_id": "h1"},
            "result": {"overall_verdict": "FAIL", "task_count": 1,
                       "pass_count": 0, "fail_count": 1, "error_count": 0,
                       "avg_sharpe_ratio": -0.5, "avg_trade_count": 10},
        }],
    }), encoding="utf-8")

    research_log = tmp_path / "research.jsonl"
    research_log.write_text(
        json.dumps({"event": "task_error", "strategy_name": "s_error",
                    "error": "ValueError: oops"}) + "\n",
        encoding="utf-8",
    )

    orch_log = tmp_path / "orch.jsonl"
    orch_log.write_text(
        json.dumps({"event": "generation_rejected", "strategy_name": "s_rejected",
                    "errors": ["causality failed"]}) + "\n",
        encoding="utf-8",
    )

    out = mod._build_merged_feedback_items(
        handoffs_path=handoffs,
        research_log_path=research_log,
        orchestrator_log_path=orch_log,
        max_items=40,
    )
    events = [x["event"] for x in out]
    assert "validation_result" in events
    assert "task_error" in events
    assert "generation_rejected" in events


def test_merged_feedback_works_when_some_sources_empty(tmp_path: Path):
    mod = _load_module()
    # Only research log has content
    research_log = tmp_path / "research.jsonl"
    research_log.write_text(
        json.dumps({"event": "task_result", "strategy_name": "s1", "verdict": "FAIL",
                    "bar_config": "tick_610", "metrics": {"sharpe_ratio": -1.0,
                    "trade_count": 5}}) + "\n",
        encoding="utf-8",
    )
    out = mod._build_merged_feedback_items(
        handoffs_path=tmp_path / "missing_handoffs.json",
        research_log_path=research_log,
        orchestrator_log_path=tmp_path / "missing_orch.jsonl",
        max_items=40,
    )
    assert len(out) == 1
    assert out[0]["event"] == "task_result"


def test_build_coder_repair_user_prompt_includes_errors_and_code():
    mod = _load_module()
    thinker_handoff = {"hypothesis": {"hypothesis_id": "h001", "thesis": "test thesis"}}
    previous_code = "def generate_signal(df, params):\n    return df['bad_col'].to_numpy()\n"
    validation_errors = [
        "my_strat: generate_signal failed for tick_610: KeyError: 'bad_col'",
        "my_strat: contract failed for volume_2000: signal contains NaN",
    ]
    prompt = mod._build_coder_repair_user_prompt(
        thinker_handoff=thinker_handoff,
        previous_code=previous_code,
        validation_errors=validation_errors,
        common_columns=["close", "ema_ratio_8", "cvd_price_divergence_6"],
    )
    assert "bad_col" in prompt
    assert "KeyError" in prompt
    assert "NaN" in prompt
    assert "generate_signal" in prompt
    assert "ema_ratio_8" in prompt


def test_build_coder_repair_user_prompt_truncates_long_code():
    mod = _load_module()
    long_code = "x = 1\n" * 1000  # very long
    prompt = mod._build_coder_repair_user_prompt(
        thinker_handoff={},
        previous_code=long_code,
        validation_errors=["error"],
        common_columns=[],
    )
    # Should be truncated — prompt must be shorter than long_code alone
    assert len(prompt) < len(long_code) + 500
