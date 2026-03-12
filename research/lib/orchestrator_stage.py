"""Shared stage-retry and Claude client helpers for the LLM orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

from research.lib.llm_client import (
    ClaudeCodeCLIClient,
    LLMClientError,
    LLMRawClient,
    extract_json_object,
)
from research.lib.thinker_feasibility import ThinkerFeasibilityError
from research.lib.thinker_research_contract import ThinkerResearchContractError


@dataclass(frozen=True)
class StageJSONResult:
    payload: dict[str, Any]
    model: str
    response_id: str | None
    usage: dict[str, Any]
    raw_text: str
    attempts: int
    repaired: bool


def extract_retry_after_seconds(error_text: str) -> float | None:
    raw = str(error_text)
    match = re.search(
        r"retry in\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|secs|second|seconds)",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "ms":
        return max(0.1, value / 1000.0)
    return max(0.1, value)


def repair_json_payload(
    *,
    stage_name: str,
    schema_hint: str,
    raw_text: str,
    client: LLMRawClient,
    max_output_tokens: int,
) -> StageJSONResult:
    system_prompt = (
        "You are a strict JSON repair engine. "
        "Return ONLY one valid JSON object and no markdown, no prose."
    )
    user_prompt = (
        f"Stage: {stage_name}\n"
        f"Required schema summary:\n{schema_hint}\n\n"
        "Fix this output into valid JSON matching the schema:\n"
        f"{raw_text}"
    )
    repaired = client.generate_raw(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_output_tokens=max(300, min(int(max_output_tokens), 1800)),
        force_json_object=True,
    )
    payload = extract_json_object(repaired.raw_text)
    return StageJSONResult(
        payload=payload,
        model=repaired.model,
        response_id=repaired.response_id,
        usage=repaired.usage,
        raw_text=repaired.raw_text,
        attempts=1,
        repaired=True,
    )


def call_stage_json(
    *,
    stage_name: str,
    schema_hint: str,
    client: LLMRawClient,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    max_attempts: int,
    json_repair_attempts: int,
    stage_backoff_seconds: float,
    quota_backoff_seconds: float,
    max_backoff_seconds: float,
) -> StageJSONResult:
    attempts = max(1, int(max_attempts))
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            raw = client.generate_raw(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
                force_json_object=True,
            )
            try:
                payload = extract_json_object(raw.raw_text)
                return StageJSONResult(
                    payload=payload,
                    model=raw.model,
                    response_id=raw.response_id,
                    usage=raw.usage,
                    raw_text=raw.raw_text,
                    attempts=attempt,
                    repaired=False,
                )
            except LLMClientError as parse_exc:
                last_exc = parse_exc
                repairs = max(0, int(json_repair_attempts))
                for _ in range(repairs):
                    try:
                        repaired = repair_json_payload(
                            stage_name=stage_name,
                            schema_hint=schema_hint,
                            raw_text=raw.raw_text,
                            client=client,
                            max_output_tokens=max_output_tokens,
                        )
                        return StageJSONResult(
                            payload=repaired.payload,
                            model=repaired.model,
                            response_id=repaired.response_id,
                            usage=repaired.usage,
                            raw_text=repaired.raw_text,
                            attempts=attempt,
                            repaired=True,
                        )
                    except Exception as repair_exc:  # pragma: no cover - best-effort repair
                        last_exc = repair_exc
        except Exception as exc:
            last_exc = exc

        if attempt >= attempts:
            break

        delay = float(stage_backoff_seconds) * float(attempt)
        if last_exc is not None:
            err = str(last_exc)
            if "HTTP 429" in err or "RESOURCE_EXHAUSTED" in err:
                retry_after = extract_retry_after_seconds(err)
                delay = max(float(quota_backoff_seconds), (retry_after or 0.0) + 1.0)
            elif "HTTP 503" in err or "UNAVAILABLE" in err:
                delay = max(delay, float(stage_backoff_seconds) * 2.0)
        time.sleep(min(float(max_backoff_seconds), max(0.1, delay)))

    if last_exc is None:
        raise RuntimeError(f"{stage_name}: failed without explicit error")
    raise last_exc


def normalize_with_semantic_retry(
    *,
    stage_name: str,
    stage_result: StageJSONResult,
    normalize_fn: Callable[[dict[str, Any]], Any],
    client: LLMRawClient,
    system_prompt: str,
    base_user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    max_semantic_retries: int,
    max_attempts: int,
    json_repair_attempts: int,
    stage_backoff_seconds: float,
    quota_backoff_seconds: float,
    max_backoff_seconds: float,
    schema_hint: str,
    call_stage_json_fn: Callable[..., StageJSONResult] | None = None,
) -> tuple[Any, StageJSONResult]:
    stage_json = call_stage_json_fn or call_stage_json
    current = stage_result
    retries = max(0, int(max_semantic_retries))
    for semantic_try in range(retries + 1):
        try:
            return normalize_fn(current.payload), current
        except ValueError as exc:
            if semantic_try >= retries:
                raise
            previous_payload: dict[str, Any] = current.payload
            if (
                isinstance(exc, (ThinkerFeasibilityError, ThinkerResearchContractError))
                and isinstance(exc.brief, dict)
                and exc.brief
            ):
                previous_payload = exc.brief
            repair_prompt = (
                f"{base_user_prompt}\n\n"
                f"Validation error: {exc}\n"
                f"Previous JSON:\n{json.dumps(previous_payload, indent=2, default=str)}\n\n"
                "Return corrected JSON only."
            )
            current = stage_json(
                stage_name=stage_name,
                schema_hint=schema_hint,
                client=client,
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_attempts=max_attempts,
                json_repair_attempts=json_repair_attempts,
                stage_backoff_seconds=stage_backoff_seconds,
                quota_backoff_seconds=quota_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
    raise RuntimeError(f"{stage_name}: normalization retry failed unexpectedly")


def cfg_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def build_llm_client(
    *,
    provider: str,
    model: str,
    agent_cfg: dict[str, Any],
    root: Path,
    role_agent_name: str | None = None,
    role_extra_args: list[str] | None = None,
    role_disable_slash_commands: bool | None = None,
    role_timeout_seconds: float | None = None,
) -> LLMRawClient:
    raw_provider = str(provider).strip().lower()
    if raw_provider not in {"claude", "claude_cli", "claude_code"}:
        raise ValueError(
            "Unsupported provider. This orchestrator is configured for Claude Code only "
            "(set `provider: claude_cli`)."
        )

    cli_cfg = cfg_dict(agent_cfg.get("claude_cli"))
    cli_binary = str(
        cli_cfg.get(
            "binary",
            agent_cfg.get("claude_binary", os.getenv("CLAUDE_CODE_BIN", "claude")),
        ),
    ).strip() or "claude"
    timeout_seconds = float(cli_cfg.get("timeout_seconds", agent_cfg.get("timeout_seconds", 180)))
    if role_timeout_seconds is not None:
        timeout_seconds = float(role_timeout_seconds)
    retries = int(cli_cfg.get("retries", agent_cfg.get("retries", 2)))
    retry_backoff_seconds = float(
        cli_cfg.get("retry_backoff_seconds", agent_cfg.get("retry_backoff_seconds", 1.5))
    )
    disable_slash_commands_raw = cli_cfg.get(
        "disable_slash_commands",
        agent_cfg.get("disable_slash_commands", True),
    )
    disable_slash_commands = bool(disable_slash_commands_raw)
    if role_disable_slash_commands is not None:
        disable_slash_commands = bool(role_disable_slash_commands)
    workdir_raw = str(cli_cfg.get("workdir", "")).strip()
    if not workdir_raw:
        workdir = root
    else:
        workdir_path = Path(workdir_raw)
        workdir = workdir_path if workdir_path.is_absolute() else (root / workdir_path)
    global_extra_args_raw = cli_cfg.get("extra_args", [])
    global_extra_args = (
        [str(v) for v in global_extra_args_raw]
        if isinstance(global_extra_args_raw, list)
        else []
    )
    extra_args = global_extra_args + (role_extra_args or [])

    return ClaudeCodeCLIClient(
        model=model,
        cli_binary=cli_binary,
        agent_name=role_agent_name,
        timeout_seconds=timeout_seconds,
        max_retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        workdir=workdir,
        extra_args=extra_args,
        disable_slash_commands=disable_slash_commands,
    )


def resolve_role_cfg(
    *,
    agent_cfg: dict[str, Any],
    role: str,
    default_temperature: float,
    default_max_output_tokens: int,
) -> dict[str, Any]:
    role_cfg = cfg_dict(agent_cfg.get(role))
    model = str(role_cfg.get("model", agent_cfg.get("model", ""))).strip()
    if not model:
        raise ValueError(f"agent config requires `{role}.model` (or legacy `model`)")

    role_cli_cfg = cfg_dict(role_cfg.get("claude_cli"))
    role_extra_args_raw = role_cli_cfg.get("extra_args", [])
    role_extra_args = [str(v) for v in role_extra_args_raw] if isinstance(role_extra_args_raw, list) else []
    disable_slash_commands = role_cli_cfg.get("disable_slash_commands")
    disable_slash_commands_out = (
        bool(disable_slash_commands)
        if isinstance(disable_slash_commands, bool)
        else None
    )

    role_timeout_raw = role_cfg.get("timeout_seconds")
    role_timeout_out = float(role_timeout_raw) if role_timeout_raw is not None else None
    agent_name = str(role_cfg.get("agent_name", "")).strip() or None

    return {
        "agent_name": agent_name,
        "model": model,
        "temperature": float(
            role_cfg.get(
                "temperature",
                agent_cfg.get("temperature", default_temperature),
            ),
        ),
        "max_output_tokens": int(
            role_cfg.get(
                "max_output_tokens",
                agent_cfg.get("max_output_tokens", default_max_output_tokens),
            ),
        ),
        "extra_args": role_extra_args,
        "disable_slash_commands": disable_slash_commands_out,
        "timeout_seconds": role_timeout_out,
    }


__all__ = [
    "StageJSONResult",
    "build_llm_client",
    "call_stage_json",
    "cfg_dict",
    "extract_retry_after_seconds",
    "normalize_with_semantic_retry",
    "repair_json_payload",
    "resolve_role_cfg",
]
