"""Shared stage-retry and provider client helpers for the LLM orchestrator."""

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
    CodexCLIClient,
    LLMClientError,
    LLMRawClient,
    extract_json_object,
)
from research.lib.thinker_feasibility import ThinkerFeasibilityError
from research.lib.thinker_policy import ThinkerPolicyError
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


_SEMANTIC_RETRY_CONTEXT_LIMIT = 4000
_SEMANTIC_RETRY_OMITTED_SECTIONS = (
    ("RUNTIME_MISSION_CONTEXT_JSON_BEGIN", "RUNTIME_MISSION_CONTEXT_JSON_END"),
    ("AVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN", "AVAILABLE_PRECOMPUTED_FEATURES_JSON_END"),
)
_PROVIDER_ALIASES = {
    "claude": "claude_cli",
    "claude_cli": "claude_cli",
    "claude_code": "claude_cli",
    "codex": "codex_cli",
    "codex_cli": "codex_cli",
}
_PROVIDER_CONFIG = {
    "claude_cli": {
        "config_key": "claude_cli",
        "binary_env_var": "CLAUDE_CODE_BIN",
        "default_binary": "claude",
        "legacy_binary_keys": ("claude_binary",),
        "supports_project_agents": True,
    },
    "codex_cli": {
        "config_key": "codex_cli",
        "binary_env_var": "CODEX_CLI_BIN",
        "default_binary": "codex",
        "legacy_binary_keys": ("codex_binary",),
        "supports_project_agents": False,
    },
}


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


def _compact_semantic_retry_context(base_user_prompt: str, *, limit: int = _SEMANTIC_RETRY_CONTEXT_LIMIT) -> str:
    compact = str(base_user_prompt or "").strip()
    if not compact:
        return ""

    for begin_marker, end_marker in _SEMANTIC_RETRY_OMITTED_SECTIONS:
        pattern = re.compile(
            rf"{re.escape(begin_marker)}.*?{re.escape(end_marker)}",
            re.DOTALL,
        )
        compact = pattern.sub(f"{begin_marker}\n[omitted for semantic retry]\n{end_marker}", compact)

    compact = re.sub(r"\n{3,}", "\n\n", compact).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 30].rstrip() + "\n...[truncated retry context]..."


def _build_semantic_retry_prompt(
    *,
    stage_name: str,
    base_user_prompt: str,
    validation_error: Exception,
    previous_payload: dict[str, Any],
) -> str:
    prompt_parts = [
        f"Stage: {stage_name}",
        "Repair the previous JSON object to satisfy the validation error.",
        "Preserve fields that are already valid and make the smallest changes necessary.",
    ]
    compact_context = _compact_semantic_retry_context(base_user_prompt)
    if compact_context:
        prompt_parts.append(f"Retained task context:\n{compact_context}")
    contract_guidance = _semantic_retry_guidance(
        validation_error=validation_error,
        previous_payload=previous_payload,
    )
    if contract_guidance:
        prompt_parts.append(contract_guidance)
    prompt_parts.append(f"Validation error: {validation_error}")
    prompt_parts.append(f"Previous JSON:\n{json.dumps(previous_payload, indent=2, default=str)}")
    prompt_parts.append("Return corrected JSON only.")
    return "\n\n".join(prompt_parts)


def _semantic_retry_guidance(
    *,
    validation_error: Exception,
    previous_payload: dict[str, Any],
) -> str:
    error_text = str(validation_error)
    if isinstance(validation_error, ThinkerPolicyError):
        lines = [
            "Thinker policy repair requirements:",
            "- Do not reuse recently blocked primary feature families from the previous failed attempts.",
            "- Keep the market event if it still makes sense, but choose a materially different structural anchor feature.",
            "- Avoid making a cosmetic threshold tweak on the same blocked primary feature.",
        ]
        if "PRIMARY_FEATURE_COOLDOWN:" in error_text:
            lines.append("- Blocked families from the last attempts: " + error_text.split("PRIMARY_FEATURE_COOLDOWN:", 1)[1].strip())
        return "\n".join(lines)
    if isinstance(validation_error, ThinkerResearchContractError):
        entry_conditions = previous_payload.get("entry_conditions")
        features = []
        if isinstance(entry_conditions, list):
            for row in entry_conditions:
                if not isinstance(row, dict):
                    continue
                feature = str(row.get("feature", "")).strip()
                if feature:
                    features.append(feature)
        lines = [
            "Research-brief repair requirements:",
            "- Preserve the existing market event, mechanism, and structural story unless the validation error requires changing them.",
            "- Keep the repair minimal and focused on the invalid `research_brief` fields.",
            "- `research_brief.falsification` must be an observable entry-time disconfirmation, not a PnL/Sharpe/backtest statement.",
            "- Keep `research_brief.post_cost_rationale` and `research_brief.novelty_vs_recent_failures` as full explanatory sentences, not short labels.",
        ]
        if features:
            lines.append(
                "- If you edit `research_brief.falsification`, name at least one entry-condition feature exactly as written here: "
                + ", ".join(features[:6])
            )
        return "\n".join(lines)
    if "entry_conditions" in error_text:
        lines = [
            "Entry-condition repair requirements:",
            "- Allowed roles are only `primary` and `confirmation`.",
            "- Keep at most 3 conditions total: 2 primary and 1 confirmation.",
            "- If you have extra filters, drop the weakest extras instead of inventing new role labels.",
            "- Use `primary` for structural/location gates and `confirmation` for the micro trigger.",
        ]
        return "\n".join(lines)
    return ""


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
                isinstance(exc, (ThinkerFeasibilityError, ThinkerResearchContractError, ThinkerPolicyError))
                and isinstance(exc.brief, dict)
                and exc.brief
            ):
                previous_payload = exc.brief
            repair_prompt = _build_semantic_retry_prompt(
                stage_name=stage_name,
                base_user_prompt=base_user_prompt,
                validation_error=exc,
                previous_payload=previous_payload,
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


def normalize_provider(provider: str | None) -> str:
    raw = str(provider or "").strip().lower()
    normalized = _PROVIDER_ALIASES.get(raw)
    if normalized:
        return normalized
    supported = ", ".join(sorted(_PROVIDER_CONFIG))
    raise ValueError(f"Unsupported provider {provider!r}. Supported providers: {supported}")


def provider_config_key(provider: str | None) -> str:
    normalized = normalize_provider(provider)
    return str(_PROVIDER_CONFIG[normalized]["config_key"])


def resolve_provider_cli_binary(provider: str | None, agent_cfg: dict[str, Any]) -> str:
    normalized = normalize_provider(provider)
    meta = _PROVIDER_CONFIG[normalized]
    cli_cfg = cfg_dict(agent_cfg.get(meta["config_key"]))
    for legacy_key in meta["legacy_binary_keys"]:
        legacy_value = str(agent_cfg.get(legacy_key, "")).strip()
        if legacy_value:
            legacy_binary = legacy_value
            break
    else:
        legacy_binary = ""
    return str(
        cli_cfg.get(
            "binary",
            legacy_binary or os.getenv(str(meta["binary_env_var"]), str(meta["default_binary"])),
        ),
    ).strip() or str(meta["default_binary"])


def _provider_supports_project_agents(provider: str | None) -> bool:
    normalized = normalize_provider(provider)
    return bool(_PROVIDER_CONFIG[normalized]["supports_project_agents"])


def _load_project_agent_prompt(root: Path, agent_name: str | None) -> str:
    name = str(agent_name or "").strip()
    if not name:
        return ""
    path = root / ".claude" / "agents" / f"{name}.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :].lstrip("\n")
    return text.strip()


def _resolve_role_model(
    *,
    provider: str,
    agent_cfg: dict[str, Any],
    role_cfg: dict[str, Any],
    role: str,
) -> str:
    normalized = normalize_provider(provider)
    role_provider_models = cfg_dict(role_cfg.get("provider_models"))
    agent_provider_models = cfg_dict(agent_cfg.get("provider_models"))
    model = str(
        role_provider_models.get(
            normalized,
            agent_provider_models.get(
                normalized,
                role_cfg.get("model", agent_cfg.get("model", "")),
            ),
        ),
    ).strip()
    if not model:
        raise ValueError(
            f"agent config requires `{role}.model` or `{role}.provider_models.{normalized}`"
        )
    return model


def build_llm_client(
    *,
    provider: str,
    model: str,
    agent_cfg: dict[str, Any],
    root: Path,
    role_agent_name: str | None = None,
    role_extra_args: list[str] | None = None,
    role_disable_slash_commands: bool | None = None,
    role_sandbox: str | None = None,
    role_timeout_seconds: float | None = None,
) -> LLMRawClient:
    normalized_provider = normalize_provider(provider)
    provider_key = provider_config_key(normalized_provider)
    cli_cfg = cfg_dict(agent_cfg.get(provider_key))
    cli_binary = resolve_provider_cli_binary(normalized_provider, agent_cfg)
    timeout_seconds = float(cli_cfg.get("timeout_seconds", agent_cfg.get("timeout_seconds", 180)))
    if role_timeout_seconds is not None:
        timeout_seconds = float(role_timeout_seconds)
    retries = int(cli_cfg.get("retries", agent_cfg.get("retries", 2)))
    retry_backoff_seconds = float(
        cli_cfg.get("retry_backoff_seconds", agent_cfg.get("retry_backoff_seconds", 1.5))
    )
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
    agent_prompt = ""
    if role_agent_name and not _provider_supports_project_agents(normalized_provider):
        agent_prompt = _load_project_agent_prompt(root, role_agent_name)

    if normalized_provider == "claude_cli":
        disable_slash_commands_raw = cli_cfg.get(
            "disable_slash_commands",
            agent_cfg.get("disable_slash_commands", True),
        )
        disable_slash_commands = bool(disable_slash_commands_raw)
        if role_disable_slash_commands is not None:
            disable_slash_commands = bool(role_disable_slash_commands)
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

    sandbox = str(cli_cfg.get("sandbox", agent_cfg.get("sandbox", "read-only"))).strip() or "read-only"
    if role_sandbox is not None:
        sandbox = str(role_sandbox).strip() or sandbox
    return CodexCLIClient(
        model=model,
        cli_binary=cli_binary,
        agent_name=role_agent_name,
        agent_prompt=agent_prompt,
        timeout_seconds=timeout_seconds,
        max_retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        workdir=workdir,
        extra_args=extra_args,
        sandbox=sandbox,
    )


def resolve_role_cfg(
    *,
    agent_cfg: dict[str, Any],
    provider: str,
    role: str,
    default_temperature: float,
    default_max_output_tokens: int,
) -> dict[str, Any]:
    normalized_provider = normalize_provider(provider)
    role_cfg = cfg_dict(agent_cfg.get(role))
    model = _resolve_role_model(
        provider=normalized_provider,
        agent_cfg=agent_cfg,
        role_cfg=role_cfg,
        role=role,
    )

    role_cli_cfg = cfg_dict(role_cfg.get(provider_config_key(normalized_provider)))
    role_extra_args_raw = role_cli_cfg.get("extra_args", [])
    role_extra_args = [str(v) for v in role_extra_args_raw] if isinstance(role_extra_args_raw, list) else []
    disable_slash_commands = role_cli_cfg.get("disable_slash_commands")
    disable_slash_commands_out = (
        bool(disable_slash_commands)
        if isinstance(disable_slash_commands, bool)
        else None
    )
    sandbox_raw = str(role_cli_cfg.get("sandbox", "")).strip()
    sandbox_out = sandbox_raw or None

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
        "sandbox": sandbox_out,
        "timeout_seconds": role_timeout_out,
    }


__all__ = [
    "StageJSONResult",
    "build_llm_client",
    "call_stage_json",
    "cfg_dict",
    "extract_retry_after_seconds",
    "normalize_with_semantic_retry",
    "normalize_provider",
    "provider_config_key",
    "repair_json_payload",
    "resolve_provider_cli_binary",
    "resolve_role_cfg",
]
