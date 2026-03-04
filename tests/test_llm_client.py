from __future__ import annotations

import pytest

from research.lib import llm_client


def test_extract_json_object_from_fenced_block():
    raw = """```json
{"strategy_name":"s1","params":{"a":1}}
```"""
    out = llm_client._extract_json_object(raw)
    assert out["strategy_name"] == "s1"
    assert out["params"]["a"] == 1


def test_claude_cli_generate_raw(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client.shutil, "which", lambda _name: "/usr/bin/claude")

    class _FakeProc:
        returncode = 0
        stdout = '{"ok": true}'
        stderr = ""

    def _fake_run(cmd, **kwargs):
        assert cmd[0] == "/usr/bin/claude"
        assert "-p" in cmd
        assert "--model" in cmd
        assert "--system-prompt" in cmd
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return _FakeProc()

    monkeypatch.setattr(llm_client.subprocess, "run", _fake_run)
    client = llm_client.ClaudeCodeCLIClient(model="sonnet", cli_binary="claude")
    out = client.generate_raw(system_prompt="sys", user_prompt="usr", force_json_object=True)
    assert out.model == "sonnet"
    assert out.raw_text == '{"ok": true}'
    assert out.response_id is not None
    assert out.usage == {}


def test_claude_cli_missing_binary_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client.shutil, "which", lambda _name: None)
    with pytest.raises(llm_client.LLMClientError):
        llm_client.ClaudeCodeCLIClient(model="sonnet", cli_binary="claude")


def test_claude_cli_nonzero_exit_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client.shutil, "which", lambda _name: "/usr/bin/claude")

    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "rate limit"

    monkeypatch.setattr(llm_client.subprocess, "run", lambda *_args, **_kwargs: _FakeProc())
    client = llm_client.ClaudeCodeCLIClient(model="sonnet", cli_binary="claude")
    with pytest.raises(llm_client.LLMClientError):
        client.generate_raw(system_prompt="sys", user_prompt="usr")


def test_claude_cli_retry_backoff_is_exponential_with_cap(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client.shutil, "which", lambda _name: "/usr/bin/claude")

    class _FailProc:
        returncode = 1
        stdout = ""
        stderr = "temporary failure"

    class _OkProc:
        returncode = 0
        stdout = '{"ok": true}'
        stderr = ""

    attempts = {"count": 0}

    def _fake_run(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] <= 3:
            return _FailProc()
        return _OkProc()

    sleeps: list[float] = []
    monkeypatch.setattr(llm_client.subprocess, "run", _fake_run)
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: sleeps.append(float(seconds)))

    client = llm_client.ClaudeCodeCLIClient(
        model="sonnet",
        cli_binary="claude",
        max_retries=3,
        retry_backoff_seconds=1.0,
        retry_max_backoff_seconds=2.0,
    )
    out = client.generate_raw(system_prompt="sys", user_prompt="usr")

    assert out.raw_text == '{"ok": true}'
    assert sleeps == [1.0, 2.0, 2.0]
