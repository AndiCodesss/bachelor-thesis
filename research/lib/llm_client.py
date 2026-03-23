"""Minimal LLM API client helpers for autonomous research orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Protocol


class LLMClientError(RuntimeError):
    """Raised when an LLM API call fails or returns invalid output."""


@dataclass(frozen=True)
class LLMRawGeneration:
    raw_text: str
    model: str
    response_id: str | None
    usage: dict[str, Any]


class LLMRawClient(Protocol):
    @property
    def model(self) -> str: ...

    def generate_raw(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2500,
        force_json_object: bool = True,
    ) -> LLMRawGeneration: ...


def _resolve_cli_binary(
    *,
    cli_binary: str,
    default_binary: str,
    provider_label: str,
) -> str:
    binary = str(cli_binary).strip() or str(default_binary).strip()
    if not binary:
        raise LLMClientError(f"{provider_label} CLI binary is required")
    # If user passed a bare command name, resolve through PATH early for clearer errors.
    if "/" not in binary and "\\" not in binary and not (len(binary) >= 2 and binary[1] == ":"):
        resolved = shutil.which(binary)
        if not resolved:
            raise LLMClientError(f"{provider_label} CLI binary not found on PATH: {binary}")
        return resolved
    path = Path(binary)
    if not path.exists():
        raise LLMClientError(f"{provider_label} CLI binary not found: {binary}")
    return str(path)


def _strip_code_fences(text: str) -> str:
    raw = str(text).strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return raw


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = _strip_code_fences(text)
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise LLMClientError("LLM response did not contain a JSON object")
    snippet = raw[start : end + 1]
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"Invalid JSON in LLM response: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMClientError("LLM response JSON must be an object")
    return payload


class ClaudeCodeCLIClient:
    """Claude Code CLI client (uses local `claude -p` execution)."""

    def __init__(
        self,
        *,
        model: str,
        cli_binary: str = "claude",
        agent_name: str | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        retry_max_backoff_seconds: float = 30.0,
        workdir: str | Path | None = None,
        extra_args: list[str] | None = None,
        disable_slash_commands: bool = True,
    ) -> None:
        self._model = str(model).strip()
        if not self._model:
            raise LLMClientError("model is required")
        self._cli_binary = _resolve_cli_binary(
            cli_binary=cli_binary,
            default_binary="claude",
            provider_label="Claude",
        )

        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_seconds = max(0.1, float(retry_backoff_seconds))
        self._retry_max_backoff_seconds = max(
            self._retry_backoff_seconds,
            float(retry_max_backoff_seconds),
        )
        self._agent_name = None if agent_name is None else (str(agent_name).strip() or None)
        self._workdir = str(workdir) if workdir is not None else None
        self._extra_args = [str(v) for v in (extra_args or []) if str(v).strip()]
        self._disable_slash_commands = bool(disable_slash_commands)

    @property
    def model(self) -> str:
        return self._model

    def _run_once(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        force_json_object: bool,
    ) -> LLMRawGeneration:
        prompt = str(user_prompt)
        if force_json_object:
            prompt = (
                f"{prompt}\n\n"
                "CRITICAL OUTPUT RULE: Return ONLY one valid JSON object. "
                "No markdown fences and no extra prose."
            )
        cmd = [
            self._cli_binary,
            "-p",
            prompt,
            "--output-format",
            "text",
            "--model",
            self._model,
        ]
        if self._agent_name:
            cmd.extend(["--agent", self._agent_name])
            cmd.extend(["--append-system-prompt", str(system_prompt)])
        else:
            cmd.extend(["--system-prompt", str(system_prompt)])
        cmd.extend(self._extra_args)
        # Keep orchestrator requests lean and deterministic: disable tool/runtime discovery overhead.
        if "--strict-mcp-config" not in cmd:
            cmd.append("--strict-mcp-config")
        if self._disable_slash_commands and "--disable-slash-commands" not in cmd:
            cmd.append("--disable-slash-commands")
        if "--tools" not in cmd and "--allowedTools" not in cmd:
            cmd.extend(["--tools", ""])
        # Claude CLI does not expose strict max-token control for local subscription usage.
        if max_output_tokens > 0:
            cmd.extend(
                [
                    "--append-system-prompt",
                    (
                        "Keep responses concise and respect an approximate output budget "
                        f"of {int(max_output_tokens)} tokens."
                    ),
                ],
            )

        # Strip CLAUDECODE so the claude CLI doesn't refuse to run inside a Claude Code session.
        child_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        try:
            proc = subprocess.run(
                cmd,
                cwd=self._workdir,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
                env=child_env,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMClientError(f"Claude CLI request timed out after {self._timeout_seconds}s") from exc
        except OSError as exc:
            raise LLMClientError(f"Failed to execute Claude CLI: {exc}") from exc

        stdout = str(proc.stdout or "").strip()
        stderr = str(proc.stderr or "").strip()
        if proc.returncode != 0:
            detail = stderr or stdout or f"exit_code={proc.returncode}"
            raise LLMClientError(f"Claude CLI failed: {detail[:500]}")
        if not stdout:
            raise LLMClientError("Claude CLI returned empty output")

        response_id = f"claude_cli_{int(time.time() * 1000)}"
        return LLMRawGeneration(
            raw_text=stdout,
            model=self._model,
            response_id=response_id,
            usage={},
        )

    def generate_raw(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2500,
        force_json_object: bool = True,
    ) -> LLMRawGeneration:
        _ = float(temperature)  # not currently configurable in Claude CLI
        attempt = 0
        while True:
            try:
                return self._run_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                    force_json_object=force_json_object,
                )
            except LLMClientError:
                if attempt >= self._max_retries:
                    raise
            attempt += 1
            sleep_seconds = self._retry_backoff_seconds
            for _ in range(max(0, attempt - 1)):
                sleep_seconds = min(self._retry_max_backoff_seconds, sleep_seconds * 2.0)
                if sleep_seconds >= self._retry_max_backoff_seconds:
                    break
            time.sleep(sleep_seconds)


class CodexCLIClient:
    """Codex CLI client (uses local `codex exec` execution)."""

    _VALID_SANDBOXES = {"read-only", "workspace-write", "danger-full-access"}

    def __init__(
        self,
        *,
        model: str,
        cli_binary: str = "codex",
        agent_name: str | None = None,
        agent_prompt: str | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        retry_max_backoff_seconds: float = 30.0,
        workdir: str | Path | None = None,
        extra_args: list[str] | None = None,
        sandbox: str = "read-only",
    ) -> None:
        self._model = str(model).strip()
        if not self._model:
            raise LLMClientError("model is required")
        self._cli_binary = _resolve_cli_binary(
            cli_binary=cli_binary,
            default_binary="codex",
            provider_label="Codex",
        )

        sandbox_value = str(sandbox).strip() or "read-only"
        if sandbox_value not in self._VALID_SANDBOXES:
            raise LLMClientError(
                "Codex sandbox must be one of "
                f"{sorted(self._VALID_SANDBOXES)}, got: {sandbox_value!r}"
            )

        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_seconds = max(0.1, float(retry_backoff_seconds))
        self._retry_max_backoff_seconds = max(
            self._retry_backoff_seconds,
            float(retry_max_backoff_seconds),
        )
        self._agent_name = None if agent_name is None else (str(agent_name).strip() or None)
        self._agent_prompt = None if agent_prompt is None else (str(agent_prompt).strip() or None)
        self._workdir = str(workdir) if workdir is not None else None
        self._extra_args = [str(v) for v in (extra_args or []) if str(v).strip()]
        self._sandbox = sandbox_value

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _failure_detail(*, stdout: str, stderr: str, returncode: int) -> str:
        detail = stderr or stdout or f"exit_code={returncode}"
        compact = detail.strip()
        if len(compact) <= 4000:
            return compact
        return f"...{compact[-4000:]}"

    def _build_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        force_json_object: bool,
    ) -> str:
        sections: list[str] = []
        if system_prompt.strip():
            sections.append(f"SYSTEM INSTRUCTIONS:\n{system_prompt.strip()}")
        if self._agent_prompt:
            agent_header = self._agent_name or "project-agent"
            sections.append(f"PROJECT AGENT PROFILE ({agent_header}):\n{self._agent_prompt}")
        sections.append(f"USER TASK:\n{str(user_prompt).strip()}")
        if max_output_tokens > 0:
            sections.append(
                "OUTPUT BUDGET:\n"
                "Keep responses concise and respect an approximate output budget "
                f"of {int(max_output_tokens)} tokens."
            )
        if force_json_object:
            sections.append(
                "CRITICAL OUTPUT RULE:\n"
                "Return ONLY one valid JSON object. No markdown fences and no extra prose."
            )
        return "\n\n".join(section for section in sections if section.strip())

    def _run_once(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        force_json_object: bool,
    ) -> LLMRawGeneration:
        prompt = self._build_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            force_json_object=force_json_object,
        )

        with tempfile.TemporaryDirectory(prefix="codex_cli_") as tmpdir_raw:
            tmpdir = Path(tmpdir_raw)
            output_path = tmpdir / "final_message.txt"
            cmd = [
                self._cli_binary,
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                self._sandbox,
                "--color",
                "never",
                "--model",
                self._model,
                "-o",
                str(output_path),
            ]
            cmd.extend(self._extra_args)
            cmd.append("-")

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self._workdir,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    check=False,
                    input=prompt,
                    env=os.environ.copy(),
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMClientError(f"Codex CLI request timed out after {self._timeout_seconds}s") from exc
            except OSError as exc:
                raise LLMClientError(f"Failed to execute Codex CLI: {exc}") from exc

            stdout = str(proc.stdout or "").strip()
            stderr = str(proc.stderr or "").strip()
            if proc.returncode != 0:
                detail = self._failure_detail(
                    stdout=stdout,
                    stderr=stderr,
                    returncode=proc.returncode,
                )
                raise LLMClientError(f"Codex CLI failed: {detail}")

            try:
                raw_text = output_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise LLMClientError(
                    "Codex CLI completed without writing the final response file"
                ) from exc
            if not raw_text:
                raise LLMClientError("Codex CLI returned empty output")

        response_id = f"codex_cli_{int(time.time() * 1000)}"
        return LLMRawGeneration(
            raw_text=raw_text,
            model=self._model,
            response_id=response_id,
            usage={},
        )

    def generate_raw(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2500,
        force_json_object: bool = True,
    ) -> LLMRawGeneration:
        _ = float(temperature)  # not currently configurable in Codex CLI
        attempt = 0
        while True:
            try:
                return self._run_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                    force_json_object=force_json_object,
                )
            except LLMClientError:
                if attempt >= self._max_retries:
                    raise
            attempt += 1
            sleep_seconds = self._retry_backoff_seconds
            for _ in range(max(0, attempt - 1)):
                sleep_seconds = min(self._retry_max_backoff_seconds, sleep_seconds * 2.0)
                if sleep_seconds >= self._retry_max_backoff_seconds:
                    break
            time.sleep(sleep_seconds)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from model text output."""
    return _extract_json_object(text)


__all__ = [
    "LLMClientError",
    "LLMRawClient",
    "LLMRawGeneration",
    "ClaudeCodeCLIClient",
    "CodexCLIClient",
    "extract_json_object",
]
