from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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
