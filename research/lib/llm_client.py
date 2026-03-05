"""Minimal LLM API client helpers for autonomous research orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
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
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        retry_max_backoff_seconds: float = 30.0,
        workdir: str | Path | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self._model = str(model).strip()
        if not self._model:
            raise LLMClientError("model is required")

        binary = str(cli_binary).strip() or "claude"
        # If user passed a bare command name, resolve through PATH early for clearer errors.
        if "/" not in binary:
            resolved = shutil.which(binary)
            if not resolved:
                raise LLMClientError(f"Claude CLI binary not found on PATH: {binary}")
            self._cli_binary = resolved
        else:
            path = Path(binary)
            if not path.exists():
                raise LLMClientError(f"Claude CLI binary not found: {binary}")
            self._cli_binary = str(path)

        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_seconds = max(0.1, float(retry_backoff_seconds))
        self._retry_max_backoff_seconds = max(
            self._retry_backoff_seconds,
            float(retry_max_backoff_seconds),
        )
        self._workdir = str(workdir) if workdir is not None else None
        self._extra_args = [str(v) for v in (extra_args or []) if str(v).strip()]

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
            "--system-prompt",
            str(system_prompt),
        ]
        cmd.extend(self._extra_args)
        # Keep orchestrator requests lean and deterministic: disable tool/runtime discovery overhead.
        if "--strict-mcp-config" not in cmd:
            cmd.append("--strict-mcp-config")
        if "--disable-slash-commands" not in cmd:
            cmd.append("--disable-slash-commands")
        if "--tools" not in cmd:
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


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from model text output."""
    return _extract_json_object(text)


__all__ = [
    "LLMClientError",
    "LLMRawClient",
    "LLMRawGeneration",
    "ClaudeCodeCLIClient",
    "extract_json_object",
]
