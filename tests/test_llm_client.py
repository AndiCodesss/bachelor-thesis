from __future__ import annotations

from pathlib import Path

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
        assert "--agent" not in cmd
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


def test_claude_cli_generate_raw_passes_agent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client.shutil, "which", lambda _name: "/usr/bin/claude")

    class _FakeProc:
        returncode = 0
        stdout = '{"ok": true}'
        stderr = ""

    def _fake_run(cmd, **kwargs):
        assert "--agent" in cmd
        assert cmd[cmd.index("--agent") + 1] == "quant-thinker"
        assert "--system-prompt" not in cmd
        assert "--append-system-prompt" in cmd
        return _FakeProc()

    monkeypatch.setattr(llm_client.subprocess, "run", _fake_run)
    client = llm_client.ClaudeCodeCLIClient(
        model="sonnet",
        cli_binary="claude",
        agent_name="quant-thinker",
    )
    out = client.generate_raw(system_prompt="sys", user_prompt="usr", force_json_object=True)
    assert out.raw_text == '{"ok": true}'



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


def test_codex_cli_generate_raw_uses_output_file_and_agent_prompt(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client.shutil, "which", lambda _name: "/usr/bin/codex")

    class _FakeProc:
        returncode = 0
        stdout = "banner"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        assert cmd[0] == "/usr/bin/codex"
        assert cmd[1] == "exec"
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "gpt-5.4"
        assert "--output-schema" not in cmd
        assert "-o" in cmd
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.write_text('{"ok": true}', encoding="utf-8")
        prompt = kwargs["input"]
        assert "SYSTEM INSTRUCTIONS:" in prompt
        assert "PROJECT AGENT PROFILE (quant-thinker):" in prompt
        assert "Use NotebookLM carefully." in prompt
        assert "CRITICAL OUTPUT RULE:" in prompt
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return _FakeProc()

    monkeypatch.setattr(llm_client.subprocess, "run", _fake_run)
    client = llm_client.CodexCLIClient(
        model="gpt-5.4",
        cli_binary="codex",
        agent_name="quant-thinker",
        agent_prompt="Use NotebookLM carefully.",
    )
    out = client.generate_raw(system_prompt="sys", user_prompt="usr", force_json_object=True)
    assert out.model == "gpt-5.4"
    assert out.raw_text == '{"ok": true}'
    assert out.response_id is not None
    assert out.usage == {}


def test_codex_cli_missing_binary_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client.shutil, "which", lambda _name: None)
    with pytest.raises(llm_client.LLMClientError):
        llm_client.CodexCLIClient(model="gpt-5.4", cli_binary="codex")


def test_codex_cli_nonzero_exit_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client.shutil, "which", lambda _name: "/usr/bin/codex")

    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "rate limit"

    monkeypatch.setattr(llm_client.subprocess, "run", lambda *_args, **_kwargs: _FakeProc())
    client = llm_client.CodexCLIClient(model="gpt-5.4", cli_binary="codex")
    with pytest.raises(llm_client.LLMClientError):
        client.generate_raw(system_prompt="sys", user_prompt="usr")


def test_codex_cli_failure_uses_error_tail(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_client.shutil, "which", lambda _name: "/usr/bin/codex")

    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "header\n" + ("x" * 5000) + "\nactual error"

    monkeypatch.setattr(llm_client.subprocess, "run", lambda *_args, **_kwargs: _FakeProc())
    client = llm_client.CodexCLIClient(model="gpt-5.4", cli_binary="codex")

    with pytest.raises(llm_client.LLMClientError) as exc_info:
        client.generate_raw(system_prompt="sys", user_prompt="usr")

    assert "actual error" in str(exc_info.value)
