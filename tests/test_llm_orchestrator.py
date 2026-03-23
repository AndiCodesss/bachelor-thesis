from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def _research_brief(**overrides):
    brief = {
        "event": "Aggressive participation expands after a quiet rotation and the next few bars should follow through.",
        "mechanism": "short-term participation imbalance after quiet-session compression",
        "expected_side": "long",
        "expected_horizon_bars": 5,
        "market_regime": "ETH conditions with rising activity and relative volatility already moving above baseline.",
        "structural_location": "Price is interacting with prior value from below and is close enough to the structural level that acceptance or rejection matters now.",
        "micro_trigger": "order_flow_imbalance turns positive while absorption or CVD divergence confirms buyers are taking control at the level.",
        "post_cost_rationale": "The event is sparse and directional enough that a short burst can clear one-turn costs.",
        "falsification": "If volume_ratio expands but trade_intensity does not stay elevated on entry bars, the imbalance thesis is wrong.",
        "novelty_vs_recent_failures": "This is not a generic OFI threshold stack; it requires quiet-session compression first and then expansion.",
    }
    brief.update(overrides)
    return brief


def test_normalize_generation_payload_filters_bar_configs_and_code():
    mod = _load_module()
    thinker_payload = {
        "hypothesis_id": "OFI Bounce 01",
        "theme_tag": "amt_value_area",
        "strategy_name_hint": "OFI Bounce",
        "research_brief": _research_brief(),
        "bar_configs": ["foo", "tick_610", "time_1m"],
        "params_template": {"lookback": 12, "vol_ratio_min": 1.1},
        "entry_conditions": [
            {
                "feature": "volume_ratio",
                "op": ">",
                "param_key": "vol_ratio_min",
                "role": "primary",
            }
        ],
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
        allowed_horizons=[1, 3, 5, 10],
    )
    assert thinker["hypothesis_id"] == "ofi_bounce_01"
    assert thinker["theme_tag"] == "amt_value_area"
    assert thinker["strategy_name_hint"] == "ofi_bounce"
    assert thinker["research_brief"]["mechanism_key"] == "short_term_participation_imbalance_after_quiet_session_compression"
    assert thinker["bar_configs"] == ["tick_610"]
    assert thinker["params_template"] == {"lookback": 12, "vol_ratio_min": 1.1}
    assert thinker["entry_conditions"][0]["feature"] == "volume_ratio"

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


def test_normalize_coder_payload_preserves_single_thinker_bar_config():
    mod = _load_module()
    out = mod._normalize_coder_payload(
        {
            "strategy_name": "bar_spread_case",
            "bar_configs": ["volume_2000", "tick_610"],
            "params": {"lookback": 12},
            "code": "import numpy as np\nimport polars as pl\n\ndef generate_signal(df, params):\n    return np.zeros(len(df), dtype=np.int8)\n",
        },
        mission_bar_configs=["tick_610", "volume_2000"],
        thinker_brief={"bar_configs": ["tick_610"]},
    )
    assert out["bar_configs"] == ["tick_610"]


def test_normalize_thinker_brief_requires_research_brief_and_enriches_error():
    mod = _load_module()
    with pytest.raises(mod.ThinkerResearchContractError) as exc_info:
        mod._normalize_thinker_brief(
            {
                "hypothesis_id": "missing_brief",
                "theme_tag": "amt_value_area",
                "strategy_name_hint": "missing_brief",
                "bar_configs": ["tick_610"],
                "params_template": {"vol_ratio_min": 1.1},
                "entry_conditions": [
                    {"feature": "volume_ratio", "op": ">", "param_key": "vol_ratio_min", "role": "primary"}
                ],
                "thesis": "x",
                "entry_logic": "y",
                "exit_logic": "z",
            },
            mission_bar_configs=["tick_610"],
            allowed_horizons=[1, 3, 5, 10],
        )

    assert "research_brief must be an object" in str(exc_info.value)
    assert exc_info.value.brief["theme_tag"] == "amt_value_area"
    assert exc_info.value.brief["entry_conditions"][0]["feature"] == "volume_ratio"


def test_normalize_thinker_brief_uses_live_bar_risk_floor_hints():
    mod = _load_module()
    with pytest.raises(ValueError, match="remove volume_2000 from bar_configs"):
        mod._normalize_thinker_brief(
            {
                "hypothesis_id": "risk_floor_case",
                "theme_tag": "amt_value_area",
                "strategy_name_hint": "risk_floor_case",
                "research_brief": _research_brief(),
                "bar_configs": ["volume_2000"],
                "params_template": {"pt_ticks": 60, "sl_ticks": 22},
                "entry_conditions": [
                    {"feature": "volume_ratio", "op": ">", "param_key": "sl_ticks", "role": "primary"}
                ],
                "thesis": "x",
                "entry_logic": "y",
                "exit_logic": "z",
            },
            mission_bar_configs=["volume_2000"],
            allowed_horizons=[1, 3, 5, 10],
            sample_bar_context={"volume_2000": {"range_ticks": {"median": 59.0}}},
        )


def test_normalize_thinker_brief_prunes_legacy_multi_bar_risk_floor_payload():
    mod = _load_module()
    out = mod._normalize_thinker_brief(
        {
            "hypothesis_id": "risk_floor_prune_case",
            "theme_tag": "amt_value_area",
            "strategy_name_hint": "risk_floor_prune_case",
            "research_brief": _research_brief(),
            "bar_configs": ["tick_610", "volume_2000"],
            "params_template": {"pt_ticks": 60, "sl_ticks": 20},
            "entry_conditions": [
                {"feature": "volume_ratio", "op": ">", "param_key": "sl_ticks", "role": "primary"}
            ],
            "thesis": "x",
            "entry_logic": "y",
            "exit_logic": "z",
        },
        mission_bar_configs=["tick_610", "volume_2000"],
        allowed_horizons=[1, 3, 5, 10],
        sample_bar_context={
            "tick_610": {"range_ticks": {"median": 40.0}},
            "volume_2000": {"range_ticks": {"median": 59.0}},
        },
    )
    assert out["bar_configs"] == ["tick_610"]


def test_format_results_table_sorts_by_sharpe_and_marks_near_misses():
    mod = _load_module()
    items = [
        {"event": "task_result", "strategy_name": "bad_strat", "bar_config": "tick_610",
         "verdict": "FAIL", "sharpe_ratio": -2.1, "trade_count": 80},
        {"event": "task_result", "strategy_name": "near_miss_strat", "bar_config": "time_1m",
         "verdict": "FAIL", "sharpe_ratio": 0.42, "trade_count": 25,
         "failed_checks": ["trade_count", "shuffle"], "dsr": 0.31, "alpha_decay_verdict": "DECAYING"},
        {"event": "generation_rejected", "strategy_name": "broken_gen", "error": "KeyError: ema_ratio"},
    ]
    table = mod._format_results_table(items)
    # near_miss_strat should appear before bad_strat (sorted by sharpe desc)
    assert table.index("near_miss_strat") < table.index("bad_strat")
    # near-miss annotation present
    assert "NEAR-MISS" in table
    assert "research/signals/near_miss_strat.py" in table
    # error block present
    assert "broken_gen" in table
    assert "KeyError" in table
    assert "fails=trade_count,shuffle" in table
    assert "dsr=0.31" in table
    assert "decay=DECAYING" in table


def test_format_results_table_empty_items():
    mod = _load_module()
    table = mod._format_results_table([])
    assert "iteration 1" in table


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

    out = mod._collect_feedback_items_from_handoffs(
        handoffs,
        handoffs.with_suffix(".lock"),
        max_items=4,
    )
    assert len(out) == 1
    assert out[0]["event"] == "validation_result"
    assert out[0]["strategy_name"] == "alpha_x"
    assert out[0]["overall_verdict"] == "FAIL"


def test_collect_feedback_items_from_handoffs_preserves_family_summary_fields(tmp_path: Path):
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
                            "strategy_name": "alpha_mix",
                            "hypothesis_id": "h_mix",
                        },
                        "result": {
                            "overall_verdict": "MIXED",
                            "task_count": 3,
                            "pass_count": 1,
                            "fail_count": 2,
                            "error_count": 0,
                            "pass_fraction": 1 / 3,
                            "avg_sharpe_ratio": 0.4,
                            "avg_trade_count": 18,
                            "passing_bar_configs": ["tick_610"],
                            "failing_bar_configs": ["time_1m", "volume_2000"],
                            "best_bar_config": "tick_610",
                            "best_sharpe_ratio": 1.2,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out = mod._collect_feedback_items_from_handoffs(
        handoffs,
        handoffs.with_suffix(".lock"),
        max_items=4,
    )
    assert out[0]["overall_verdict"] == "MIXED"
    assert out[0]["pass_count"] == 1
    assert out[0]["task_count"] == 3
    assert out[0]["best_bar_config"] == "tick_610"


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


def test_task_id_differs_by_execution_context():
    mod = _load_module()
    params = {"a": 1}
    base = mod._task_id(
        "alpha_x",
        "tick_610",
        params,
        "abcd1234",
        search_split="train",
        selection_split="validate",
        session_filter="eth",
        feature_group="all",
    )
    other_arm = mod._task_id(
        "alpha_x",
        "tick_610",
        params,
        "abcd1234",
        search_split="train",
        selection_split="validate",
        session_filter="eth",
        feature_group="ohlcv",
    )
    other_session = mod._task_id(
        "alpha_x",
        "tick_610",
        params,
        "abcd1234",
        search_split="train",
        selection_split="validate",
        session_filter="rth",
        feature_group="all",
    )
    assert base != other_arm
    assert base != other_session


def test_build_task_persists_runtime_context():
    mod = _load_module()
    task = mod._build_task(
        strategy_name="alpha_x",
        search_split="train",
        selection_split="validate",
        bar_config="tick_610",
        params={"lookback": 5},
        mission={"session_filter": "eth", "feature_group": "ohlcv"},
        code_hash="abcd1234",
        iteration=1,
        hypothesis_id="hyp_1",
        theme_tag="reversion",
        research_brief={},
        setup_key="setup_1",
        setup_label="setup 1",
    )
    assert task["session_filter"] == "eth"
    assert task["feature_group"] == "ohlcv"


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
        "theme_tag": "amt_value_area",
        "strategy_name_hint": "alpha_hint",
        "research_brief": _research_brief(),
        "bar_configs": ["tick_610"],
        "params_template": {"lookback": 10},
        "entry_conditions": [
            {
                "feature": "volume_ratio",
                "op": ">",
                "param_key": "lookback",
                "role": "primary",
            }
        ],
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
    assert handoff["hypothesis"]["theme_tag"] == "amt_value_area"
    assert handoff["hypothesis"]["research_brief"]["mechanism"] == _research_brief()["mechanism"]
    assert handoff["hypothesis"]["entry_conditions"][0]["feature"] == "volume_ratio"
    prompt = mod._build_coder_user_prompt(thinker_handoff=handoff)
    assert "THINKER_HANDOFF_JSON_BEGIN" in prompt
    assert '"hypothesis_id": "h_01"' in prompt
    assert '"entry_conditions"' in prompt
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


def test_compact_feature_knowledge_prioritizes_regime_macro_micro_features():
    mod = _load_module()
    compact = mod._compact_feature_knowledge_for_thinker(
        {
            "schema_version": "1.0",
            "session_filter": "eth",
            "bar_configs": {"tick_610": {"total_columns": 8, "extra_columns": 0}},
            "common_columns": [
                "close",
                "vol_zscore",
                "trade_intensity",
                "prev_day_va_position",
                "dist_prev_vah",
                "failed_auction_score",
                "cvd_price_divergence_3",
                "absorption_signal",
                "order_flow_imbalance",
            ],
            "feature_catalog": [
                "vol_zscore | x | regime",
                "prev_day_va_position | x | macro",
                "dist_prev_vah | x | macro",
                "failed_auction_score | x | macro",
                "cvd_price_divergence_3 | x | micro",
                "absorption_signal | x | micro",
            ],
            "computation_notes": [],
            "errors": {},
        },
        selected_bar_configs=["tick_610"],
    )
    assert compact["priority_feature_families"]["regime"] == ["vol_zscore", "trade_intensity"]
    assert "prev_day_va_position" in compact["priority_feature_families"]["macro_structure"]
    assert "cvd_price_divergence_3" in compact["priority_feature_families"]["micro_execution"]
    assert compact["common_columns_sample"][:3] == ["vol_zscore", "trade_intensity", "prev_day_va_position"]
    assert "prev_day_va_position | x | macro" in compact["feature_catalog_highlights"]
    assert "Active mission is ETH." in compact["session_structure_guidance"]


def test_prompts_include_feature_knowledge_markers():
    mod = _load_module()
    feature_knowledge = {
        "schema_version": "1.0",
        "session_filter": "eth",
        "common_columns": [
            "close",
            "ema_ratio_8",
            "vol_zscore",
            "prev_day_va_position",
            "dist_prev_vah",
            "cvd_price_divergence_3",
            "absorption_signal",
            "order_flow_imbalance",
        ],
        "bar_configs": {"tick_610": {"total_columns": 8, "extra_columns": 0}},
        "per_bar_extra_columns": {"tick_610": []},
        "computation_notes": ["ema_ratio_N = close / EMA - 1"],
        "feature_catalog": [
            "vol_zscore | x | regime",
            "prev_day_va_position | x | macro",
            "dist_prev_vah | x | macro",
            "cvd_price_divergence_3 | x | micro",
            "absorption_signal | x | micro",
            "order_flow_imbalance | x | micro",
        ],
        "errors": {},
    }
    thinker_feature_knowledge = mod._compact_feature_knowledge_for_thinker(
        feature_knowledge,
        selected_bar_configs=["tick_610"],
    )
    feature_surface_context = (
        "FEATURE_SURFACE_INTELLIGENCE:\n"
        "- tick_610: 1200 sampled rows across 3 file(s)\n"
        "  Dead features: squeeze_score (0.00% non-zero)\n"
        "  Sparse features: failed_auction_bull (0.08% non-zero)"
    )
    thinker_prompt = mod._build_thinker_user_prompt(
        mission={
            "objective": "x",
            "bar_configs": ["tick_610"],
            "current_focus": [
                "AMT value area rejection and rotation (VAH/VAL/POC mean-reversion)",
                "Orderflow divergence reversals (fade delta extremes, not follow them)",
            ],
            "session_filter": "eth",
            "feature_group": "all",
            "notebooklm_notebook_url": "https://notebooklm.google.com/notebook/test-id",
            "notebook_research_guidance": "Use high-quality trusted sources.",
            "lane_notebook_query_budget": {
                "max_total_queries": 3,
                "max_research_queries": 1,
                "max_deep_research_queries": 0,
            },
        },
        existing_strategies=[],
        feedback_items=[],
        learning_context="LEARNING_SCORECARD:\nLow-sample themes: amt_value_area",
        feature_surface_context=feature_surface_context,
        param_feasibility_context=(
            "PARAM_FEASIBILITY_HINTS:\n"
            "- tick_610:\n"
            "  bb_bandwidth_20: p10=0.0016 p50=0.0025 p90=0.0042"
        ),
        runtime_context={
            "split": "validate",
            "min_trade_count": 30,
            "allowed_edge_horizons": [1, 3, 5, 10],
            "sample_bar_context": {
                "tick_610": {"sample_rows": 1200, "range_ticks": {"median": 40.0}},
                "volume_2000": {"sample_rows": 300, "range_ticks": {"median": 59.0}},
                "time_1m": {"sample_rows": 700, "range_ticks": {"median": 33.0}},
            },
        },
        feature_knowledge=thinker_feature_knowledge,
    )
    assert "AVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN" in thinker_prompt
    assert "RUNTIME_MISSION_CONTEXT_JSON_BEGIN" in thinker_prompt
    assert "KNOWLEDGE_BASE_COMMAND" in thinker_prompt
    assert "https://notebooklm.google.com/notebook/test-id" in thinker_prompt
    assert "query_notebook.py" in thinker_prompt
    assert "--research" in thinker_prompt
    assert '--deep-research "question"' not in thinker_prompt
    assert "choose the research direction yourself" in thinker_prompt
    assert "Hard runtime budget this iteration: at most 1 --research query and 3 total notebook queries." in thinker_prompt
    assert "--deep-research is disabled for the autonomy thinker loop" in thinker_prompt
    assert "Use high-quality trusted sources." in thinker_prompt
    assert "BAR_CONFIG_RISK_FLOORS:" in thinker_prompt
    assert "volume_2000: sl_ticks >= 40" in thinker_prompt
    assert '"common_column_count"' in thinker_prompt
    assert '"common_columns_sample"' in thinker_prompt
    assert "Current focus anchors" in thinker_prompt
    assert "- amt_value_area" in thinker_prompt
    assert "Return exactly one concise `theme_tag` in snake_case." in thinker_prompt
    assert "AUCTION_MARKET_THEORY_CLARITY:" in thinker_prompt
    assert "Use Auction Market Theory as the primary interpretive lens" in thinker_prompt
    assert "THINKING_SEQUENCE:" in thinker_prompt
    assert "Only after the theory is clear, map it onto exact columns and minimal params." in thinker_prompt
    assert "A proper NQ hypothesis must follow this funnel: Market regime -> Structural location -> Micro trigger." in thinker_prompt
    assert "You must return a required `research_brief` object before `entry_conditions`" in thinker_prompt
    assert "The `research_brief` should read like an auction narrative" in thinker_prompt
    assert "- `structural_location`: the structural location where the setup matters" in thinker_prompt
    assert "- `micro_trigger`: the L1/orderflow confirmation that says fire now instead of wait" in thinker_prompt
    assert "- `expected_horizon_bars`: one of [1, 3, 5, 10]" in thinker_prompt
    assert "THREE_LAYER_HYPOTHESIS_ARCHITECTURE:" in thinker_prompt
    assert "Active mission is ETH." in thinker_prompt
    assert "Market regime fields: vol_zscore" in thinker_prompt
    assert "Structural location fields: prev_day_va_position" in thinker_prompt
    assert "Micro execution fields: cvd_price_divergence_3" in thinker_prompt
    assert "The `entry_conditions` must be directly traceable to the `research_brief.market_regime`" in thinker_prompt
    assert "Return exactly one selected `bar_config` inside `bar_configs`" in thinker_prompt
    assert "LEARNING_SCORECARD:" in thinker_prompt
    assert "TRANSLATION_DISCIPLINE:" in thinker_prompt
    assert "TRANSLATION_AIDS:" in thinker_prompt
    assert "FEATURE_SURFACE_INTELLIGENCE:" in thinker_prompt
    assert "Dead features: squeeze_score" in thinker_prompt
    assert "PARAM_FEASIBILITY_HINTS:" in thinker_prompt

    coder_prompt = mod._build_coder_user_prompt(
        thinker_handoff={
            "handoff_version": "1.0",
            "source_role": "quant_thinker",
            "payload_hash": "h1",
            "mission_constraints": {},
            "hypothesis": {"hypothesis_id": "h1", "strategy_name_hint": "s1"},
        },
        feature_knowledge=feature_knowledge,
        feature_surface_warning=(
            "REFERENCED_FEATURE_SURFACE_WARNINGS:\n"
            "- tick_610: squeeze_score (0.00% non-zero)"
        ),
    )
    assert "AVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN" in coder_prompt
    assert "Prefer precomputed features directly" in coder_prompt
    assert "REFERENCED_FEATURE_SURFACE_WARNINGS:" in coder_prompt
    assert "Keep the handoff's single selected `bar_config` unchanged." in coder_prompt
    assert "translate that auction idea into the smallest faithful set of feature checks" in coder_prompt


def test_coder_system_prompt_requires_safe_column_helpers():
    mod = _load_module()
    prompt = mod._build_coder_system_prompt()
    assert "nq-signal-coder" in prompt
    assert "Signal Coding Contract Skill" in prompt
    assert "Return only the required JSON object" in prompt
    assert "leanest faithful signal implementation" in prompt
    assert "Preserve the handoff's single selected bar_config" in prompt


def test_thinker_system_prompt_requires_internal_brainstorm():
    mod = _load_module()
    prompt = mod._build_thinker_system_prompt(notebooklm_enabled=True)
    assert "quant-thinker" in prompt
    assert "Work theory-first." in prompt
    assert "Notebook Alpha Research Skill" in prompt
    assert "runtime mission context" in prompt


def test_thinker_system_prompt_uses_runtime_context_not_stale_session_claims():
    mod = _load_module()
    prompt = mod._build_thinker_system_prompt(notebooklm_enabled=False)
    assert "prior session volume. VAH = va_high" not in prompt
    assert "source of truth" in prompt
    assert "Notebook Alpha Research Skill" not in prompt


def test_thinker_user_prompt_omits_notebook_command_when_no_url():
    mod = _load_module()
    prompt = mod._build_thinker_user_prompt(
        mission={"objective": "x", "bar_configs": ["tick_610"]},
        existing_strategies=[],
        feedback_items=[],
    )
    assert "KNOWLEDGE_BASE_COMMAND" not in prompt
    assert "query_notebook.py" not in prompt


def test_disable_notebooklm_runtime_mission_removes_notebook_context():
    mod = _load_module()
    runtime_mission = mod._disable_notebooklm_runtime_mission(
        {
            "objective": "x",
            "bar_configs": ["tick_610"],
            "notebooklm": {"mode": "lane_fresh"},
            "notebooklm_notebook_url": "https://notebooklm.google.com/notebook/test-id",
            "notebook_research_guidance": "Use trusted sources.",
            "lane_notebook_query_budget": {
                "max_total_queries": 3,
                "max_research_queries": 1,
                "max_deep_research_queries": 0,
            },
        }
    )
    assert "notebooklm" not in runtime_mission
    assert runtime_mission["notebooklm_notebook_url"] == ""
    assert runtime_mission["notebook_research_guidance"] == ""
    assert runtime_mission["lane_notebook_query_budget"] == {
        "max_total_queries": 0,
        "max_research_queries": 0,
        "max_deep_research_queries": 0,
    }
    prompt = mod._build_thinker_user_prompt(
        mission=runtime_mission,
        existing_strategies=[],
        feedback_items=[],
    )
    assert "KNOWLEDGE_BASE_COMMAND" not in prompt
    assert "NotebookLM is disabled for this run." in prompt


def test_attempt_research_fields_canonicalizes_legacy_names():
    mod = _load_module()
    out = mod._attempt_research_fields(
        {
            "mechanism": "failed auction fade",
            "mechanism_key": "failed_auction_fade",
            "expected_regime": "ETH with rising volatility.",
            "macro_location": "Price is probing prior value from below.",
            "micro_trigger": "Order flow flips positive at the level.",
            "expected_horizon_bars": 5,
        }
    )
    assert out["market_regime"] == "ETH with rising volatility."
    assert out["structural_location"] == "Price is probing prior value from below."


def test_resolve_notebook_query_budget_defaults_to_bounded_runtime_policy():
    mod = _load_module()
    assert mod._resolve_notebook_query_budget({}) == {
        "max_total_queries": 3,
        "max_research_queries": 1,
        "max_deep_research_queries": 0,
    }


def test_build_learning_context_reads_scorecard(tmp_path: Path):
    mod = _load_module()
    scorecard_path = tmp_path / "learning_scorecard.json"
    scorecard_lock = tmp_path / "learning_scorecard.lock"
    scorecard_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "rebuilt_at": "2026-03-07T14:00:00+00:00",
                "setup_stats": {
                    "setup_123": {
                        "label": "tick_610 | volume_ratio > (primary) | exit=sl_ticks,pt_ticks",
                        "attempts": 2,
                        "edge_passes": 1,
                        "search_passes": 1,
                        "selection_attempts": 0,
                        "selection_passes": 0,
                        "fail_counts": {"no_edge": 1},
                        "recent_outcomes": [
                            {
                                "strategy_name": "alpha_setup_a",
                                "bar_config": "tick_610",
                                "edge_status": "global_edge",
                                "search_verdict": "PASS",
                                "final_verdict": "FAIL",
                                "failure_codes": ["walk_forward"],
                                "completed_at": "2026-03-07T14:01:00+00:00",
                            }
                        ],
                    }
                },
                "theme_stats": {
                    "amt_value_area": {
                        "attempts": 4,
                        "search_passes": 2,
                        "search_rate": 0.50,
                        "selection_attempts": 1,
                        "selection_passes": 0,
                        "selection_rate": 0.33,
                        "fail_counts": {"alpha_decay": 2},
                    },
                    "orderflow_divergence": {
                        "attempts": 1,
                        "search_passes": 0,
                        "search_rate": 0.33,
                        "selection_attempts": 0,
                        "selection_passes": 0,
                        "selection_rate": 0.50,
                        "fail_counts": {},
                    }
                },
                "bar_config_affinity": {},
                "near_misses": [],
                "low_sample_themes": [],
            }
        ),
        encoding="utf-8",
    )

    context = mod._build_learning_context(
        scorecard_path=scorecard_path,
        scorecard_lock=scorecard_lock,
    )
    assert "LEARNING_SCORECARD:" in context
    assert "Recent concrete setup outcomes:" in context
    assert "tick_610 | volume_ratio > (primary)" in context
    assert "Repeated setup failure modes:" in context
    assert "no_edge (1)" in context
    assert "edge 1/2 | search 1/2" in context
    assert "amt_value_area: search 2/4 | selection 0/1" in context
    assert "Low-sample themes: orderflow_divergence" in context


def test_persist_notebook_progress_marks_lane_seeded_after_imports(tmp_path: Path):
    mod = _load_module()
    notebook_meta = {
        "configured": True,
        "fresh": True,
        "seeded": False,
        "imported_sources": 0,
    }
    summary = {
        "query_count": 1,
        "modes_used": ["research"],
        "imported_sources": 9,
    }
    orchestrator_state = {"schema_version": "1.0"}
    orchestrator_path = tmp_path / "orchestrator.json"

    mod._persist_notebook_progress(
        notebook_meta=notebook_meta,
        notebook_summary=summary,
        orchestrator_state=orchestrator_state,
        state_paths={"orchestrator": orchestrator_path},
    )

    assert notebook_meta["seeded"] is True
    assert notebook_meta["fresh"] is False
    assert notebook_meta["imported_sources"] == 9
    assert notebook_meta["seed_query_count"] == 1
    assert notebook_meta["seed_modes_used"] == ["research"]

    persisted = json.loads(orchestrator_path.read_text(encoding="utf-8"))
    assert persisted["notebooklm"]["seeded"] is True
    assert persisted["notebooklm"]["imported_sources"] == 9


def test_persist_notebook_progress_preserves_partial_imports(tmp_path: Path):
    mod = _load_module()
    notebook_meta = {
        "configured": True,
        "fresh": True,
        "seeded": False,
        "imported_sources": 0,
    }
    summary = {
        "query_count": 1,
        "modes_used": ["plain"],
        "imported_sources": 4,
    }
    orchestrator_state = {"schema_version": "1.0"}
    orchestrator_path = tmp_path / "orchestrator.json"

    mod._persist_notebook_progress(
        notebook_meta=notebook_meta,
        notebook_summary=summary,
        orchestrator_state=orchestrator_state,
        state_paths={"orchestrator": orchestrator_path},
    )

    assert notebook_meta["seeded"] is True
    assert notebook_meta["fresh"] is False
    assert notebook_meta["imported_sources"] == 4
    assert notebook_meta["seed_query_count"] == 1
    assert notebook_meta["seed_modes_used"] == ["plain"]

    persisted = json.loads(orchestrator_path.read_text(encoding="utf-8"))
    assert persisted["notebooklm"]["seeded"] is True
    assert persisted["notebooklm"]["imported_sources"] == 4


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
            "result": {"overall_verdict": "FAIL", "task_count": 2,
                       "pass_count": 0, "fail_count": 2, "error_count": 0,
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
        handoffs_lock_path=handoffs.with_suffix(".lock"),
        research_log_path=research_log,
        orchestrator_log_path=orch_log,
        max_items=40,
    )
    events = [x["event"] for x in out]
    assert "validation_result" in events
    assert "task_error" in events
    assert "generation_rejected" in events


def test_default_notebook_summary_marks_configuration_state():
    mod = _load_module()
    configured = mod._default_notebook_summary(configured=True)
    unconfigured = mod._default_notebook_summary(configured=False)
    assert configured["configured"] is True
    assert configured["used"] is False
    assert configured["query_count"] == 0
    assert unconfigured["configured"] is False


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
        handoffs_lock_path=tmp_path / "missing_handoffs.lock",
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
        feature_surface_warning="REFERENCED_FEATURE_SURFACE_WARNINGS:\n- tick_610: bad_col (0.00% non-zero)",
    )
    assert "bad_col" in prompt
    assert "KeyError" in prompt
    assert "NaN" in prompt
    assert "generate_signal" in prompt
    assert "ema_ratio_8" in prompt
    assert "REFERENCED_FEATURE_SURFACE_WARNINGS:" in prompt


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


def test_coder_system_prompt_requires_int8():
    """Coder prompt must explicitly mandate dtype=np.int8."""
    mod = _load_module()
    prompt = mod._build_coder_system_prompt()
    assert "int8" in prompt
    assert "astype" in prompt


def test_coder_system_prompt_forbids_negative_shift():
    """Coder prompt must warn against shift(-1) lookahead."""
    mod = _load_module()
    prompt = mod._build_coder_system_prompt()
    assert "shift(-1)" in prompt


def test_diagnose_zero_signal_shows_bool_firing_rate():
    mod = _load_module()
    df = pl.DataFrame({
        "absorption_signal": [True, False, False, False, False],  # 20%
        "close": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    code = 'df["absorption_signal"] & (df["close"] > 2.0)'
    result = mod._diagnose_zero_signal(df, code)
    assert "absorption_signal" in result
    assert "20.00%" in result


def test_diagnose_zero_signal_shows_float_percentiles():
    mod = _load_module()
    df = pl.DataFrame({
        "cvd_price_divergence_3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    })
    code = 'df["cvd_price_divergence_3"] > 0.8'
    result = mod._diagnose_zero_signal(df, code)
    assert "cvd_price_divergence_3" in result
    assert "p50" in result


def test_diagnose_zero_signal_skips_unreferenced_columns():
    mod = _load_module()
    df = pl.DataFrame({
        "close": [1.0, 2.0],
        "volume": [100.0, 200.0],
        "absorption_signal": [True, False],
    })
    code = 'df["close"] > 1.5'
    result = mod._diagnose_zero_signal(df, code)
    assert "close" in result
    assert "volume" not in result
    assert "absorption_signal" not in result


def test_diagnose_zero_signal_handles_no_referenced_columns():
    mod = _load_module()
    df = pl.DataFrame({"close": [1.0, 2.0]})
    code = "some_var > threshold"  # no quoted column names
    result = mod._diagnose_zero_signal(df, code)
    assert "no referenced columns" in result


def test_validate_generated_strategy_errors_on_zero_signal_rate(tmp_path):
    mod = _load_module()

    code = (
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import numpy as np\n"
        "import polars as pl\n"
        "from research.signals import safe_f64_col\n"
        'DEFAULT_PARAMS: dict = {"absorption_threshold": 0.9999}\n'
        "def generate_signal(df: pl.DataFrame, params: dict) -> np.ndarray:\n"
        '    cfg = dict(DEFAULT_PARAMS); cfg.update(params or {})\n'
        '    _ = safe_f64_col(df, "absorption_signal", fill=0.0)\n'
        "    return np.zeros(len(df), dtype=np.int8)\n"
        'STRATEGY_METADATA = {"name": "zero_strat", "version": "1.0",'
        ' "features_required": ["close"], "description": "test"}\n'
    )
    (tmp_path / "zero_strat.py").write_text(code, encoding="utf-8")

    sample_df = pl.DataFrame({
        "close": [float(i) for i in range(100)],
        "absorption_signal": [i % 20 == 0 for i in range(100)],
    })

    errors, report = mod._validate_generated_strategy(
        strategy_name="zero_strat",
        signals_dir=tmp_path,
        params={},
        bar_configs=["tick_610"],
        split="validate",
        session_filter="eth",
        feature_group="all",
        sample_cache={"tick_610": sample_df},
        code=code,
    )
    assert len(errors) == 1
    assert "signal_rate=0.0%" in errors[0]
    assert "absorption_signal" in errors[0]
    assert "ACTION REQUIRED" in errors[0]
    assert report["bar_results"][0]["status"] == "zero_signal"


def test_validate_generated_strategy_skips_context_unavailable_sample(tmp_path):
    mod = _load_module()

    code = (
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import numpy as np\n"
        "import polars as pl\n"
        "from research.signals import safe_f64_col\n"
        'DEFAULT_PARAMS: dict = {"va_pos_min": 1.0}\n'
        "def generate_signal(df: pl.DataFrame, params: dict) -> np.ndarray:\n"
        '    cfg = dict(DEFAULT_PARAMS); cfg.update(params or {})\n'
        '    va_pos = safe_f64_col(df, "prev_day_va_position", fill=np.nan)\n'
        '    return np.where(va_pos > cfg["va_pos_min"], 1, 0).astype(np.int8)\n'
        'STRATEGY_METADATA = {"name": "context_sensitive", "version": "1.0",'
        ' "features_required": ["prev_day_va_position"], "description": "test"}\n'
    )
    (tmp_path / "context_sensitive.py").write_text(code, encoding="utf-8")

    errors, report = mod._validate_generated_strategy(
        strategy_name="context_sensitive",
        signals_dir=tmp_path,
        params={},
        bar_configs=["tick_610"],
        split="validate",
        session_filter="eth",
        feature_group="all",
        validation_sample_cache={
            "tick_610": [
                ("nq_2022-10-03", pl.DataFrame({"prev_day_va_position": [None, None, None]})),
                ("nq_2022-10-04", pl.DataFrame({"prev_day_va_position": [0.8] * 98 + [1.2, 1.3]})),
            ]
        },
        code=code,
    )

    assert errors == []
    assert [row["status"] for row in report["bar_results"]] == ["context_unavailable", "ok"]
    assert report["bar_results"][0]["context_columns"] == ["prev_day_va_position"]


def test_normalize_and_assess_thinker_brief_rejects_dead_primary_feature():
    mod = _load_module()
    with pytest.raises(mod.ThinkerFeasibilityError, match="DEAD"):
        mod._normalize_and_assess_thinker_brief(
            {
                "hypothesis_id": "dead_feature_case",
                "theme_tag": "amt_value_area",
                "strategy_name_hint": "dead_feature_case",
                "research_brief": _research_brief(
                    falsification=(
                        "If stale_anchor is not consistently elevated on entry bars, the structural anchor "
                        "premise is wrong and the setup should not fire."
                    ),
                ),
                "bar_configs": ["tick_610"],
                "params_template": {"anchor_min": 1.05, "pt_ticks": 40, "sl_ticks": 20},
                "entry_conditions": [
                    {
                        "feature": "stale_anchor",
                        "op": ">",
                        "param_key": "anchor_min",
                        "role": "primary",
                    }
                ],
                "thesis": "x",
                "entry_logic": "y",
                "exit_logic": "z",
            },
            mission_bar_configs=["tick_610"],
            allowed_horizons=[1, 3, 5, 10],
            sample_bar_context={"tick_610": {"range_ticks": {"median": 12.0}}},
            validation_sample_cache={
                "tick_610": [
                (
                    "sample_day",
                    pl.DataFrame(
                        {
                            "stale_anchor": [None, None, None],
                            "volume_ratio": [1.0, 1.1, 1.2],
                        }
                    ),
                    )
                ]
            },
        )


def test_normalize_and_assess_thinker_brief_accepts_context_unavailable_edge_day_when_other_sample_is_usable():
    mod = _load_module()
    out = mod._normalize_and_assess_thinker_brief(
        {
            "hypothesis_id": "context_edge_case",
            "theme_tag": "amt_value_area",
            "strategy_name_hint": "context_edge_case",
            "research_brief": _research_brief(
                structural_location="re-enter prior value from below",
                falsification=(
                    "If prev_day_va_position is unavailable or fails to recover above the prior-value threshold "
                    "on usable entry bars, the structural re-entry premise is wrong."
                ),
            ),
            "bar_configs": ["tick_610"],
            "params_template": {"va_pos_min": 1.0, "pt_ticks": 40, "sl_ticks": 20},
            "entry_conditions": [
                {
                    "feature": "prev_day_va_position",
                    "op": ">",
                    "param_key": "va_pos_min",
                    "role": "primary",
                }
            ],
            "thesis": "x",
            "entry_logic": "y",
            "exit_logic": "z",
        },
        mission_bar_configs=["tick_610"],
        allowed_horizons=[1, 3, 5, 10],
        sample_bar_context={"tick_610": {"range_ticks": {"median": 12.0}}},
        validation_sample_cache={
            "tick_610": [
                ("nq_2022-10-03", pl.DataFrame({"prev_day_va_position": [None, None, None]})),
                ("nq_2022-10-04", pl.DataFrame({"prev_day_va_position": [0.8] * 98 + [1.2, 1.3]})),
            ]
        },
    )
    assert out["entry_conditions"][0]["feature"] == "prev_day_va_position"


def test_normalize_and_assess_thinker_brief_accepts_feasible_conditions():
    mod = _load_module()
    out = mod._normalize_and_assess_thinker_brief(
        {
            "hypothesis_id": "feasible_case",
            "theme_tag": "amt_value_area",
            "strategy_name_hint": "feasible_case",
            "research_brief": _research_brief(),
            "bar_configs": ["tick_610"],
            "params_template": {"vol_ratio_min": 1.05, "pt_ticks": 40, "sl_ticks": 20},
            "entry_conditions": [
                {
                    "feature": "volume_ratio",
                    "op": ">",
                    "param_key": "vol_ratio_min",
                    "role": "primary",
                }
            ],
            "thesis": "x",
            "entry_logic": "y",
            "exit_logic": "z",
        },
        mission_bar_configs=["tick_610"],
        allowed_horizons=[1, 3, 5, 10],
        sample_bar_context={"tick_610": {"range_ticks": {"median": 12.0}}},
        validation_sample_cache={
            "tick_610": [
                (
                    "sample_day",
                    pl.DataFrame(
                        {
                            "volume_ratio": [1.0] * 98 + [1.2, 1.3],
                        }
                    ),
                )
            ]
        },
    )
    assert out["entry_conditions"][0]["feature"] == "volume_ratio"


def test_normalize_and_assess_thinker_brief_auto_repairs_zero_signal_threshold():
    mod = _load_module()
    out = mod._normalize_and_assess_thinker_brief(
        {
            "hypothesis_id": "auto_repair_zero_signal",
            "theme_tag": "amt_value_area",
            "strategy_name_hint": "auto_repair_zero_signal",
            "research_brief": _research_brief(),
            "bar_configs": ["tick_610"],
            "params_template": {"vol_ratio_min": 2.0, "pt_ticks": 40, "sl_ticks": 20},
            "entry_conditions": [
                {
                    "feature": "volume_ratio",
                    "op": ">",
                    "param_key": "vol_ratio_min",
                    "role": "primary",
                }
            ],
            "thesis": "x",
            "entry_logic": "y",
            "exit_logic": "z",
        },
        mission_bar_configs=["tick_610"],
        allowed_horizons=[1, 3, 5, 10],
        sample_bar_context={"tick_610": {"range_ticks": {"median": 12.0}}},
        validation_sample_cache={
            "tick_610": [
                (
                    "sample_day",
                    pl.DataFrame({"volume_ratio": [1.0] * 98 + [1.2, 1.3]}),
                )
            ]
        },
    )
    assert out["params_template"]["vol_ratio_min"] < 2.0


def test_normalize_and_assess_thinker_brief_auto_repairs_over_signal_threshold():
    mod = _load_module()
    out = mod._normalize_and_assess_thinker_brief(
        {
            "hypothesis_id": "auto_repair_over_signal",
            "theme_tag": "amt_value_area",
            "strategy_name_hint": "auto_repair_over_signal",
            "research_brief": _research_brief(
                mechanism="fade one-sided delta extremes after participation spikes",
                falsification="If delta_heat stays elevated but volume_ratio does not expand on entry bars, the exhaustion thesis is wrong.",
            ),
            "bar_configs": ["tick_610"],
            "params_template": {"delta_min": 0.0, "pt_ticks": 40, "sl_ticks": 20},
            "entry_conditions": [
                {
                    "feature": "delta_heat",
                    "op": ">",
                    "param_key": "delta_min",
                    "role": "primary",
                }
            ],
            "thesis": "x",
            "entry_logic": "y",
            "exit_logic": "z",
        },
        mission_bar_configs=["tick_610"],
        allowed_horizons=[1, 3, 5, 10],
        sample_bar_context={"tick_610": {"range_ticks": {"median": 12.0}}},
        validation_sample_cache={
            "tick_610": [
                (
                    "sample_day",
                    pl.DataFrame({"delta_heat": [float(i) for i in range(100)]}),
                )
            ]
        },
    )
    assert out["params_template"]["delta_min"] > 90.0


def test_normalize_and_assess_thinker_brief_prunes_to_single_best_bar_config():
    mod = _load_module()
    out = mod._normalize_and_assess_thinker_brief(
        {
            "hypothesis_id": "bar_prune_case",
            "theme_tag": "orderflow_divergence",
            "strategy_name_hint": "bar_prune_case",
            "research_brief": _research_brief(),
            "bar_configs": ["tick_610", "volume_2000"],
            "params_template": {"vol_ratio_min": 1.25, "pt_ticks": 60, "sl_ticks": 40},
            "entry_conditions": [
                {
                    "feature": "volume_ratio",
                    "op": ">",
                    "param_key": "vol_ratio_min",
                    "role": "primary",
                }
            ],
            "thesis": "x",
            "entry_logic": "y",
            "exit_logic": "z",
        },
        mission_bar_configs=["tick_610", "volume_2000"],
        allowed_horizons=[1, 3, 5, 10],
        sample_bar_context={
            "tick_610": {"range_ticks": {"median": 40.0}},
            "volume_2000": {"range_ticks": {"median": 59.0}},
        },
        validation_sample_cache={
            "tick_610": [
                (
                    "tick_day",
                    pl.DataFrame({"volume_ratio": [1.0] * 98 + [1.3, 1.4]}),
                )
            ],
            "volume_2000": [
                (
                    "volume_day",
                    pl.DataFrame({"volume_ratio": [1.0] * 120}),
                )
            ],
        },
    )
    assert out["bar_configs"] == ["tick_610"]


def test_validate_generated_strategy_errors_on_high_signal_rate(tmp_path):
    mod = _load_module()

    code = (
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import numpy as np\n"
        "import polars as pl\n"
        "DEFAULT_PARAMS: dict = {}\n"
        "def generate_signal(df: pl.DataFrame, params: dict) -> np.ndarray:\n"
        "    return np.ones(len(df), dtype=np.int8)\n"
        'STRATEGY_METADATA = {"name": "always_long", "version": "1.0",'
        ' "features_required": [], "description": "test"}\n'
    )
    (tmp_path / "always_long.py").write_text(code, encoding="utf-8")

    sample_df = pl.DataFrame({"close": [float(i) for i in range(100)]})

    errors, report = mod._validate_generated_strategy(
        strategy_name="always_long",
        signals_dir=tmp_path,
        params={},
        bar_configs=["tick_610"],
        split="validate",
        session_filter="eth",
        feature_group="all",
        sample_cache={"tick_610": sample_df},
        code=code,
    )
    assert len(errors) == 1
    assert "signal_rate=" in errors[0]
    assert "cost destruction" in errors[0]
    assert report["bar_results"][0]["status"] == "over_signal"


def test_validate_generated_strategy_passes_for_healthy_signal_rate(tmp_path):
    mod = _load_module()

    code = (
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import numpy as np\n"
        "import polars as pl\n"
        "DEFAULT_PARAMS: dict = {}\n"
        "def generate_signal(df: pl.DataFrame, params: dict) -> np.ndarray:\n"
        "    out = np.zeros(len(df), dtype=np.int8)\n"
        "    if len(df) > 10:\n"
        "        out[5] = 1\n"
        "    return out\n"
        'STRATEGY_METADATA = {"name": "sparse_strat", "version": "1.0",'
        ' "features_required": [], "description": "test"}\n'
    )
    (tmp_path / "sparse_strat.py").write_text(code, encoding="utf-8")

    sample_df = pl.DataFrame({"close": [float(i) for i in range(100)]})

    errors, report = mod._validate_generated_strategy(
        strategy_name="sparse_strat",
        signals_dir=tmp_path,
        params={},
        bar_configs=["tick_610"],
        split="validate",
        session_filter="eth",
        feature_group="all",
        sample_cache={"tick_610": sample_df},
        code=code,
    )
    assert errors == []
    assert report["bar_results"][0]["status"] == "ok"


def test_validate_generated_strategy_returns_module_validation_errors(tmp_path):
    mod = _load_module()

    code = (
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "DEFAULT_PARAMS = {}\n"
        "def generate_signal(df, params):\n"
        '    values = df["close"].to_numpy()\n'
        "    return np.sign(values).astype(np.int8)\n"
        'STRATEGY_METADATA = {"name": "bad_numpy", "version": "1.0", "features_required": ["close"], "description": "test"}\n'
    )
    (tmp_path / "bad_numpy.py").write_text(code, encoding="utf-8")

    sample_df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})

    errors, report = mod._validate_generated_strategy(
        strategy_name="bad_numpy",
        signals_dir=tmp_path,
        params={},
        bar_configs=["tick_610"],
        split="validate",
        session_filter="eth",
        feature_group="all",
        sample_cache={"tick_610": sample_df},
        code=code,
    )
    assert len(errors) == 1
    assert "module validation failed" in errors[0]
    assert "to_numpy()" in errors[0]
    assert report["bar_results"][0]["status"] == "module_validation_failed"


def test_build_validation_attempt_record_captures_blocking_conditions_and_params():
    mod = _load_module()
    record = mod._build_validation_attempt_record(
        iteration=4,
        hypothesis_id="h004",
        theme_tag="vol_compression",
        research_brief=_research_brief(mechanism="fade failed compression after volatility expansion"),
        strategy_name="compression_ignition",
        bar_configs=["tick_610"],
        params={"squeeze_threshold": 0.5, "volume_ratio_min": 1.8},
        validation_report={
            "bar_results": [
                {
                    "status": "zero_signal",
                    "bar_config": "tick_610",
                    "sample_label": "nq_2023-03-06",
                    "nonzero": 0,
                    "total": 523,
                    "signal_rate_pct": 0.0,
                    "condition_rows": [
                        {
                            "column": "squeeze_score",
                            "operator": ">",
                            "param_key": "squeeze_threshold",
                            "threshold": 0.5,
                            "pass_rate_pct": 0.0,
                            "severity": "blocks_all",
                        },
                        {
                            "column": "volume_ratio",
                            "operator": ">",
                            "param_key": "volume_ratio_min",
                            "threshold": 1.8,
                            "pass_rate_pct": 2.0,
                            "severity": "normal",
                        },
                    ],
                }
            ]
        },
    )
    assert record is not None
    assert record["failure_type"] == "zero_signal"
    assert record["summary"] == "0/523 bars on tick_610 (nq_2023-03-06)."
    assert record["conditions_label"] == "Blocking"
    assert record["highlighted_conditions"][0]["column"] == "squeeze_score"
    assert record["offending_params"] == {"squeeze_threshold": 0.5}
    assert record["mechanism"] == "fade failed compression after volatility expansion"


def test_build_validation_attempt_record_strips_internal_condition_payload():
    mod = _load_module()
    record = mod._build_validation_attempt_record(
        iteration=5,
        hypothesis_id="h005",
        theme_tag="amt_value_area",
        research_brief=_research_brief(),
        strategy_name="va_rejection",
        bar_configs=["tick_610"],
        params={"vol_ratio_min": 1.2},
        validation_report={
            "bar_results": [
                {
                    "status": "zero_signal",
                    "bar_config": "tick_610",
                    "sample_label": "sample_day",
                    "nonzero": 0,
                    "total": 100,
                    "signal_rate_pct": 0.0,
                    "condition_rows": [
                        {
                            "column": "volume_ratio",
                            "operator": ">=",
                            "param_key": "vol_ratio_min",
                            "threshold": np.float64(1.2),
                            "pass_rate_pct": np.float64(0.0),
                            "severity": "blocks_all",
                            "_mask": np.array([False, False, False]),
                        }
                    ],
                }
            ]
        },
    )
    assert record is not None
    assert record["highlighted_conditions"][0]["threshold"] == 1.2
    assert record["highlighted_conditions"][0]["pass_rate_pct"] == 0.0
    assert "_mask" not in record["highlighted_conditions"][0]


def test_build_exception_attempt_record_captures_invalid_risk_floor_params():
    mod = _load_module()
    record = mod._build_exception_attempt_record(
        iteration=2,
        hypothesis_id="h002",
        theme_tag="va_rejection",
        research_brief=_research_brief(),
        bar_configs=["tick_610", "volume_2000"],
        params={"sl_ticks": 22, "pt_ticks": 60},
        exc=ValueError(
            "Invalid pt_ticks/sl_ticks in params_template: "
            "sl_ticks=22 too tight for volume_2000 (median bar range ~94t) — minimum sl_ticks=40 required to survive entry bar noise"
        ),
    )
    assert record["failure_type"] == "invalid_risk_floor"
    assert record["bar_config"] == "volume_2000"
    assert record["offending_params"] == {"sl_ticks": 22}


def test_build_exception_attempt_record_uses_feasibility_brief_metadata():
    mod = _load_module()
    exc = mod.ThinkerFeasibilityError(
        "THINKER_FEASIBILITY on tick_610 (sample_day): primary feature stale_anchor has no finite values",
        report={
            "bar_results": [
                {
                    "status": "dead_feature_primary",
                    "bar_config": "tick_610",
                    "sample_label": "sample_day",
                    "error": "primary feature stale_anchor has no finite values",
                    "condition_rows": [
                        {
                            "column": "stale_anchor",
                            "operator": ">",
                            "param_key": "anchor_min",
                            "threshold": np.float64(1.2),
                            "severity": "dead_feature",
                            "pass_rate_pct": np.float64(0.0),
                            "_mask": np.array([False, False, False]),
                        }
                    ],
                }
            ]
        },
        brief={
            "hypothesis_id": "h_dead",
            "theme_tag": "amt_value_area",
            "bar_configs": ["tick_610"],
            "params_template": {"anchor_min": 1.2},
        },
    )
    record = mod._build_exception_attempt_record(
        iteration=7,
        hypothesis_id="iter_7",
        theme_tag="unknown",
        research_brief=None,
        bar_configs=[],
        params={},
        exc=exc,
    )
    assert record["failure_type"] == "dead_feature_primary"
    assert record["bar_config"] == "tick_610"
    assert record["offending_params"] == {"anchor_min": 1.2}
    assert "_mask" not in record["highlighted_conditions"][0]


def test_build_validation_attempt_record_includes_cross_sample_conflicts():
    mod = _load_module()
    record = mod._build_validation_attempt_record(
        iteration=3,
        hypothesis_id="h_conflict",
        theme_tag="amt_value_area",
        research_brief=_research_brief(),
        strategy_name="thinker_hypothesis",
        bar_configs=["tick_610"],
        params={"va_position_min": 1.385153},
        validation_report={
            "bar_results": [
                {
                    "status": "zero_signal",
                    "bar_config": "tick_610",
                    "sample_label": "nq_2023-08-08",
                    "nonzero": 0,
                    "total": 576,
                    "signal_rate_pct": 0.0,
                    "condition_rows": [
                        {
                            "column": "prev_day_va_position",
                            "operator": ">=",
                            "role": "primary",
                            "param_key": "va_position_min",
                            "threshold": np.float64(1.385153),
                            "p10": np.float64(-2.4855),
                            "p50": np.float64(-1.6756),
                            "p90": np.float64(-0.3554),
                            "pass_rate_pct": np.float64(0.0),
                            "_mask": np.array([False, False, False]),
                        }
                    ],
                },
                {
                    "status": "over_signal",
                    "bar_config": "tick_610",
                    "sample_label": "nq_2023-03-06",
                    "nonzero": 100,
                    "total": 523,
                    "signal_rate_pct": 19.12,
                    "condition_rows": [
                        {
                            "column": "prev_day_va_position",
                            "operator": ">=",
                            "role": "primary",
                            "param_key": "va_position_min",
                            "threshold": np.float64(1.385153),
                            "p10": np.float64(1.3261),
                            "p50": np.float64(1.5651),
                            "p90": np.float64(1.9812),
                            "pass_rate_pct": np.float64(19.12),
                            "_mask": np.array([True, True, True]),
                        }
                    ],
                },
            ]
        },
    )

    assert record is not None
    assert record["cross_sample_conflicts"]
    assert "sample p10 range -2.4855..1.3261" in record["cross_sample_conflicts"][0]


def test_normalize_with_semantic_retry_uses_repaired_feasibility_brief(monkeypatch: pytest.MonkeyPatch):
    mod = _load_module()
    prompts: list[str] = []

    def fake_call_stage_json(**kwargs):
        prompts.append(kwargs["user_prompt"])
        return mod.StageJSONResult(
            payload={"hypothesis_id": "retry_payload", "theme_tag": "amt_value_area"},
            model="test-model",
            response_id="resp-2",
            usage={},
            raw_text="{}",
            attempts=1,
            repaired=False,
        )

    attempt = {"count": 0}

    def normalize_fn(payload):
        if attempt["count"] == 0:
            attempt["count"] += 1
            raise mod.ThinkerFeasibilityError(
                "THINKER_FEASIBILITY on tick_610 (sample_day): bad thresholds",
                report={"bar_results": []},
                brief={"hypothesis_id": "repaired_payload", "theme_tag": "amt_value_area"},
            )
        return payload

    monkeypatch.setattr(mod, "_call_stage_json", fake_call_stage_json)

    out, stage = mod._normalize_with_semantic_retry(
        stage_name="quant_thinker",
        stage_result=mod.StageJSONResult(
            payload={"hypothesis_id": "original_payload", "theme_tag": "amt_value_area"},
            model="test-model",
            response_id="resp-1",
            usage={},
            raw_text="{}",
            attempts=1,
            repaired=False,
        ),
        normalize_fn=normalize_fn,
        client=SimpleNamespace(),
        system_prompt="system",
        base_user_prompt="user",
        temperature=0.2,
        max_output_tokens=1000,
        max_semantic_retries=1,
        max_attempts=1,
        json_repair_attempts=0,
        stage_backoff_seconds=0.0,
        quota_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
        schema_hint="schema",
    )

    assert out["hypothesis_id"] == "retry_payload"
    assert stage.payload["hypothesis_id"] == "retry_payload"
    assert '"hypothesis_id": "repaired_payload"' in prompts[0]


def test_normalize_with_semantic_retry_compacts_large_base_prompt(monkeypatch: pytest.MonkeyPatch):
    mod = _load_module()
    prompts: list[str] = []
    base_user_prompt = (
        "Mission objective:\nFind one valid setup.\n\n"
        "RUNTIME_MISSION_CONTEXT_JSON_BEGIN\n"
        + ("x" * 5000)
        + "\nRUNTIME_MISSION_CONTEXT_JSON_END\n\n"
        "AVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN\n"
        + ("y" * 5000)
        + "\nAVAILABLE_PRECOMPUTED_FEATURES_JSON_END\n"
    )

    def fake_call_stage_json(**kwargs):
        prompts.append(kwargs["user_prompt"])
        return mod.StageJSONResult(
            payload={"hypothesis_id": "retry_payload", "theme_tag": "amt_value_area"},
            model="test-model",
            response_id="resp-2",
            usage={},
            raw_text="{}",
            attempts=1,
            repaired=False,
        )

    attempt = {"count": 0}

    def normalize_fn(payload):
        if attempt["count"] == 0:
            attempt["count"] += 1
            raise mod.ThinkerFeasibilityError(
                "THINKER_FEASIBILITY on tick_610 (sample_day): bad thresholds",
                report={"bar_results": []},
                brief={"hypothesis_id": "retry_payload", "theme_tag": "amt_value_area"},
            )
        return payload

    monkeypatch.setattr(mod, "_call_stage_json", fake_call_stage_json)

    mod._normalize_with_semantic_retry(
        stage_name="quant_thinker",
        stage_result=mod.StageJSONResult(
            payload={"hypothesis_id": "original_payload", "theme_tag": "amt_value_area"},
            model="test-model",
            response_id="resp-1",
            usage={},
            raw_text="{}",
            attempts=1,
            repaired=False,
        ),
        normalize_fn=normalize_fn,
        client=SimpleNamespace(),
        system_prompt="system",
        base_user_prompt=base_user_prompt,
        temperature=0.2,
        max_output_tokens=1000,
        max_semantic_retries=1,
        max_attempts=1,
        json_repair_attempts=0,
        stage_backoff_seconds=0.0,
        quota_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
        schema_hint="schema",
    )

    assert prompts
    assert "Retained task context:" in prompts[0]
    assert "[omitted for semantic retry]" in prompts[0]
    assert "RUNTIME_MISSION_CONTEXT_JSON_BEGIN" in prompts[0]
    assert "AVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN" in prompts[0]
    assert "x" * 200 not in prompts[0]
    assert "y" * 200 not in prompts[0]
    assert len(prompts[0]) < len(base_user_prompt)


def test_normalize_and_assess_thinker_brief_detects_repair_cycle(monkeypatch: pytest.MonkeyPatch):
    mod = _load_module()
    base_brief = {
        "hypothesis_id": "h_cycle",
        "theme_tag": "amt_value_area",
        "bar_configs": ["tick_610"],
        "params_template": {"position_in_va_min": 0.724548},
        "entry_conditions": [
            {"feature": "position_in_va", "op": ">=", "role": "primary", "param_key": "position_in_va_min"}
        ],
    }

    def fake_normalize_thinker_brief(*args, **kwargs):
        return dict(base_brief)

    def fake_assess_entry_condition_feasibility(*, params_template, **kwargs):
        threshold = float(params_template["position_in_va_min"])
        if threshold < 1.0:
            return {
                "bar_results": [
                    {
                        "status": "over_signal",
                        "bar_config": "tick_610",
                        "sample_label": "nq_2023-03-06",
                        "nonzero": 55,
                        "total": 523,
                        "signal_rate_pct": 10.52,
                        "condition_rows": [
                            {
                                "column": "position_in_va",
                                "operator": ">=",
                                "role": "primary",
                                "param_key": "position_in_va_min",
                                "threshold": threshold,
                                "p10": 0.0924,
                                "p50": 0.5726,
                                "p90": 1.4852,
                                "pass_rate_pct": 42.8,
                            }
                        ],
                    }
                ]
            }
        return {
            "bar_results": [
                {
                    "status": "zero_signal",
                    "bar_config": "tick_610",
                    "sample_label": "nq_2024-06-13",
                    "nonzero": 0,
                    "total": 539,
                    "signal_rate_pct": 0.0,
                    "condition_rows": [
                        {
                            "column": "position_in_va",
                            "operator": ">=",
                            "role": "primary",
                            "param_key": "position_in_va_min",
                            "threshold": threshold,
                            "p10": -0.6538,
                            "p50": 0.0944,
                            "p90": 0.7245,
                            "pass_rate_pct": 0.0,
                        }
                    ],
                }
            ]
        }

    def fake_repair_thinker_brief_for_feasibility(brief, report):
        current = float(brief["params_template"]["position_in_va_min"])
        repaired = dict(brief)
        repaired["params_template"] = dict(brief["params_template"])
        repaired["params_template"]["position_in_va_min"] = 1.684913 if current < 1.0 else 0.724548
        return repaired, ["toggle threshold"]

    monkeypatch.setattr(mod, "_normalize_thinker_brief", fake_normalize_thinker_brief)
    monkeypatch.setattr(mod, "assess_entry_condition_feasibility", fake_assess_entry_condition_feasibility)
    monkeypatch.setattr(mod, "repair_thinker_brief_for_feasibility", fake_repair_thinker_brief_for_feasibility)

    with pytest.raises(mod.ThinkerFeasibilityError) as exc_info:
        mod._normalize_and_assess_thinker_brief(
            payload={},
            mission_bar_configs=["tick_610"],
            allowed_horizons=[1, 3, 5],
            sample_bar_context=None,
            validation_sample_cache={},
        )

    assert "Auto-repair oscillated between incompatible thresholds" in str(exc_info.value)


def test_normalize_and_assess_thinker_brief_rejects_cross_sample_repair_conflicts(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_module()
    base_brief = {
        "hypothesis_id": "h_conflict",
        "theme_tag": "amt_value_area",
        "bar_configs": ["tick_610"],
        "params_template": {"position_in_va_min": 0.724548},
        "entry_conditions": [
            {"feature": "position_in_va", "op": ">=", "role": "primary", "param_key": "position_in_va_min"}
        ],
    }

    def fake_normalize_thinker_brief(*args, **kwargs):
        return dict(base_brief)

    def fake_assess_entry_condition_feasibility(**kwargs):
        return {
            "bar_results": [
                {
                    "status": "over_signal",
                    "bar_config": "tick_610",
                    "sample_label": "nq_2023-03-06",
                    "nonzero": 55,
                    "total": 523,
                    "signal_rate_pct": 10.52,
                    "condition_rows": [
                        {
                            "column": "position_in_va",
                            "operator": ">=",
                            "role": "primary",
                            "param_key": "position_in_va_min",
                            "threshold": 0.724548,
                            "p10": 0.0924,
                            "p50": 0.5726,
                            "p90": 1.4852,
                            "pass_rate_pct": 42.8,
                        }
                    ],
                },
                {
                    "status": "zero_signal",
                    "bar_config": "tick_610",
                    "sample_label": "nq_2024-06-13",
                    "nonzero": 0,
                    "total": 539,
                    "signal_rate_pct": 0.0,
                    "condition_rows": [
                        {
                            "column": "position_in_va",
                            "operator": ">=",
                            "role": "primary",
                            "param_key": "position_in_va_min",
                            "threshold": 0.724548,
                            "p10": -0.6538,
                            "p50": 0.0944,
                            "p90": 0.7245,
                            "pass_rate_pct": 0.0,
                        }
                    ],
                },
            ]
        }

    def fail_repair(*args, **kwargs):
        raise AssertionError("repair should not be attempted when cross-sample conflicts already exist")

    monkeypatch.setattr(mod, "_normalize_thinker_brief", fake_normalize_thinker_brief)
    monkeypatch.setattr(mod, "assess_entry_condition_feasibility", fake_assess_entry_condition_feasibility)
    monkeypatch.setattr(mod, "repair_thinker_brief_for_feasibility", fail_repair)

    with pytest.raises(mod.ThinkerFeasibilityError) as exc_info:
        mod._normalize_and_assess_thinker_brief(
            payload={},
            mission_bar_configs=["tick_610"],
            allowed_horizons=[1, 3, 5],
            sample_bar_context=None,
            validation_sample_cache={},
        )

    assert "Cross-sample feasibility constraints conflict across validation samples" in str(exc_info.value)


def test_should_attempt_coder_repair_skips_hypothesis_level_failures():
    mod = _load_module()
    assert (
        mod._should_attempt_coder_repair(
            validation_report={
                "bar_results": [
                    {"status": "zero_signal"},
                    {"status": "over_signal"},
                ]
            }
        )
        is False
    )
    assert (
        mod._should_attempt_coder_repair(
            validation_report={
                "bar_results": [
                    {"status": "dead_feature_primary"},
                    {"status": "context_unavailable"},
                    {"status": "ok"},
                ]
            }
        )
        is False
    )
    assert (
        mod._should_attempt_coder_repair(
            validation_report={
                "bar_results": [
                    {"status": "zero_signal"},
                    {"status": "ok"},
                ]
            }
        )
        is False
    )
    assert (
        mod._should_attempt_coder_repair(
            validation_report={
                "bar_results": [
                    {"status": "over_signal"},
                    {"status": "empty_sample"},
                ]
            }
        )
        is False
    )


def test_should_attempt_coder_repair_keeps_runtime_and_contract_failures():
    mod = _load_module()
    assert (
        mod._should_attempt_coder_repair(
            validation_report={
                "bar_results": [
                    {"status": "runtime_error"},
                ]
            }
        )
        is True
    )
    assert (
        mod._should_attempt_coder_repair(
            validation_report={
                "bar_results": [
                    {"status": "contract_failed"},
                ]
            }
        )
        is True
    )


def test_build_thinker_prompt_includes_recent_attempt_memory():
    mod = _load_module()
    prompt = mod._build_thinker_user_prompt(
        mission={"bar_configs": ["tick_610"], "session_filter": "eth", "feature_group": "all"},
        existing_strategies=[],
        feedback_items=[],
        thinker_memory_context=(
            "RECENT_ATTEMPT_MEMORY (last 3 iterations, this lane):\n"
            "[iter 4] vol_compression -> REJECTED (zero_signal): 0/523 bars on tick_610.\n"
            "  Blocking: squeeze_score > 0.5 (0.0% pass)\n"
            "  Params: squeeze_threshold=0.5"
        ),
    )
    assert "RECENT_ATTEMPT_MEMORY" in prompt
    assert "vol_compression" in prompt


def test_repair_prompt_includes_signal_rate_guidance_on_zero_rate():
    mod = _load_module()
    thinker_handoff = {"hypothesis": {"hypothesis_id": "h001", "thesis": "test"}}
    previous_code = (
        'def generate_signal(df, params):\n'
        '    cond = df["absorption_signal"] & (df["poc_distance"] > 0.5)\n'
        '    return np.where(cond, 1, 0).astype(np.int8)\n'
    )
    validation_errors = [
        "my_strat: signal_rate=0.0% for tick_610 (0/1200 bars non-zero — target 0.05–0.3%). "
        "Column statistics for referenced features:\n"
        "  absorption_signal (bool): True=0.08% (1/1200 bars)\n"
        "  poc_distance: p10=0.0100 p25=0.0500 p50=0.1200 p75=0.2800 p90=0.4500\n"
        "ACTION REQUIRED: Relax the most restrictive threshold(s) above "
        "to produce at least 1 signal in this 1200-bar sample."
    ]
    prompt = mod._build_coder_repair_user_prompt(
        thinker_handoff=thinker_handoff,
        previous_code=previous_code,
        validation_errors=validation_errors,
        common_columns=["close", "absorption_signal", "poc_distance"],
    )
    assert "ZERO signals" in prompt
    assert "RELAX" in prompt
    assert "Do NOT add new conditions" in prompt


def test_repair_prompt_uses_standard_instruction_for_non_rate_errors():
    mod = _load_module()
    validation_errors = [
        "my_strat: non-causal for tick_610: signal at bar 5 changed when prefix window shrunk"
    ]
    prompt = mod._build_coder_repair_user_prompt(
        thinker_handoff={},
        previous_code="code",
        validation_errors=validation_errors,
        common_columns=[],
    )
    assert "Fix ONLY the validation errors" in prompt
    assert "ZERO signals" not in prompt


def test_repair_prompt_targets_helper_contract_errors():
    mod = _load_module()
    prompt = mod._build_coder_repair_user_prompt(
        thinker_handoff={},
        previous_code='def generate_signal(df, params):\n    return df["close"].to_numpy()\n',
        validation_errors=[
            "bad_numpy: module validation failed: ValueError: Strategy module /tmp/bad_numpy.py may not call to_numpy(); use safe_f64_col(...) instead",
        ],
        common_columns=["close"],
    )
    assert "safe_f64_col" in prompt
    assert "to_numpy()" in prompt
    assert "Keep the strategy logic the same" in prompt
