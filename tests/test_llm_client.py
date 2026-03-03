from __future__ import annotations

import json

import pytest

from research.lib import llm_client


def test_extract_json_object_from_fenced_block():
    raw = """```json
{"strategy_name":"s1","params":{"a":1}}
```"""
    out = llm_client._extract_json_object(raw)
    assert out["strategy_name"] == "s1"
    assert out["params"]["a"] == 1


def test_generate_json_parses_chat_completion(monkeypatch: pytest.MonkeyPatch):
    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self) -> bytes:
            payload = {
                "id": "chatcmpl_123",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "strategy_name": "alpha_s1",
                                    "params": {"lookback": 20},
                                    "bar_configs": ["tick_610"],
                                    "code": "import numpy as np\\nimport polars as pl\\n"
                                            "def generate_signal(df, params):\\n"
                                            "    return np.zeros(len(df), dtype=np.int8)\\n",
                                }
                            ),
                        },
                    },
                ],
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(llm_client, "urlopen", lambda *_args, **_kwargs: _FakeResp())
    client = llm_client.OpenAIChatJSONClient(model="gpt-5-mini", api_key="test-key")
    out = client.generate_json(system_prompt="sys", user_prompt="usr")
    assert out.response_id == "chatcmpl_123"
    assert out.payload["strategy_name"] == "alpha_s1"
    assert out.usage["total_tokens"] == 30


def test_generate_raw_returns_text_without_json_parse(monkeypatch: pytest.MonkeyPatch):
    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self) -> bytes:
            payload = {
                "id": "chatcmpl_raw_1",
                "usage": {"total_tokens": 12},
                "choices": [{"message": {"content": "plain text response"}}],
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(llm_client, "urlopen", lambda *_args, **_kwargs: _FakeResp())
    client = llm_client.OpenAIChatJSONClient(model="gpt-5-mini", api_key="test-key")
    out = client.generate_raw(system_prompt="sys", user_prompt="usr", force_json_object=False)
    assert out.response_id == "chatcmpl_raw_1"
    assert out.raw_text == "plain text response"
