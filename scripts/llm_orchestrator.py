#!/usr/bin/env python3
"""LLM-driven signal generator that enqueues research tasks autonomously."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
from typing import Any, Callable

import numpy as np
import polars as pl
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.lib.atomic_io import atomic_json_write
from research.lib.coordination import append_handoff, enqueue_task
from research.lib.experiments import log_experiment
from research.lib.feature_groups import filter_feature_group
from research.lib.llm_client import (
    ClaudeCodeCLIClient,
    LLMClientError,
    LLMRawClient,
    extract_json_object,
)
from research.signals import check_signal_causality, load_signal_module
from src.framework.api import (
    ExecutionMode,
    get_split_files,
    load_cached_matrix,
    set_execution_mode,
)
from src.framework.features_canonical.builder import LABEL_COLUMNS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_").lower() or "strategy"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StageJSONResult:
    def __init__(
        self,
        *,
        payload: dict[str, Any],
        model: str,
        response_id: str | None,
        usage: dict[str, Any],
        raw_text: str,
        attempts: int,
        repaired: bool,
    ) -> None:
        self.payload = payload
        self.model = model
        self.response_id = response_id
        self.usage = usage
        self.raw_text = raw_text
        self.attempts = attempts
        self.repaired = repaired


def _extract_retry_after_seconds(error_text: str) -> float | None:
    raw = str(error_text)
    match = re.search(r"retry in\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec|secs|second|seconds)", raw, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "ms":
        return max(0.1, value / 1000.0)
    return max(0.1, value)


def _repair_json_payload(
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


def _call_stage_json(
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
                        repaired = _repair_json_payload(
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
                retry_after = _extract_retry_after_seconds(err)
                delay = max(float(quota_backoff_seconds), (retry_after or 0.0) + 1.0)
            elif "HTTP 503" in err or "UNAVAILABLE" in err:
                delay = max(delay, float(stage_backoff_seconds) * 2.0)
        time.sleep(min(float(max_backoff_seconds), max(0.1, delay)))

    if last_exc is None:
        raise RuntimeError(f"{stage_name}: failed without explicit error")
    if isinstance(last_exc, Exception):
        raise last_exc
    raise RuntimeError(f"{stage_name}: {last_exc}")


def _normalize_with_semantic_retry(
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
) -> tuple[Any, StageJSONResult]:
    current = stage_result
    retries = max(0, int(max_semantic_retries))
    for semantic_try in range(retries + 1):
        try:
            return normalize_fn(current.payload), current
        except ValueError as exc:
            if semantic_try >= retries:
                raise
            repair_prompt = (
                f"{base_user_prompt}\n\n"
                f"Validation error: {exc}\n"
                f"Previous JSON:\n{json.dumps(current.payload, indent=2, default=str)}\n\n"
                "Return corrected JSON only."
            )
            current = _call_stage_json(
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


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML object expected: {path}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        rows = f.readlines()
    return rows[-limit:] if limit > 0 else rows


def _strip_code_fences(text: str) -> str:
    raw = str(text).strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return raw


def _parse_bar_config(bar_config: str) -> dict[str, Any]:
    raw = str(bar_config).strip().lower()
    match = re.fullmatch(r"(tick|volume|vol|time)_(.+)", raw)
    if not match:
        raise ValueError(f"Unsupported bar_config '{bar_config}'")
    kind, suffix = match.groups()

    if kind == "time":
        if not re.fullmatch(r"[1-9][0-9]*[mh]", suffix):
            raise ValueError(f"Invalid time bar size in '{bar_config}'")
        return {"bar_type": "time", "bar_size": suffix, "bar_threshold": None}

    if not suffix.isdigit() or int(suffix) <= 0:
        raise ValueError(f"Invalid bar threshold in '{bar_config}'")

    bar_type = "volume" if kind in {"volume", "vol"} else "tick"
    return {"bar_type": bar_type, "bar_size": "5m", "bar_threshold": int(suffix)}


def _validate_signal_array(signal: np.ndarray, expected_len: int) -> list[str]:
    errors: list[str] = []
    if signal.ndim != 1:
        return [f"signal must be 1D, got ndim={signal.ndim}"]
    if len(signal) != expected_len:
        return [f"signal length {len(signal)} != expected {expected_len}"]
    if np.isnan(signal).any():
        errors.append("signal contains NaN")
    uniq = set(np.unique(signal).tolist())
    if not uniq.issubset({-1, 0, 1}):
        errors.append(f"signal contains invalid values: {sorted(uniq)}")
    return errors


def _ensure_runtime_state(root: Path, *, resume: bool, mission_name: str) -> dict[str, Path]:
    state_dir = root / "research" / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "queue": state_dir / "experiment_queue.json",
        "queue_lock": state_dir / "experiment_queue.lock",
        "handoffs": state_dir / "handoffs.json",
        "handoffs_lock": state_dir / "handoffs.lock",
        "orchestrator": state_dir / "llm_orchestrator.json",
    }

    defaults = {
        "queue": {"schema_version": "1.0", "tasks": []},
        "handoffs": {"schema_version": "1.0", "pending": [], "completed": []},
        "orchestrator": {
            "schema_version": "1.0",
            "mission_name": mission_name,
            "iterations_completed": 0,
            "total_tasks_enqueued": 0,
            "generated_modules": [],
            "last_updated": _utc_now(),
        },
    }

    for key in ("queue", "handoffs"):
        if not paths[key].exists():
            atomic_json_write(paths[key], defaults[key])

    if (not resume) or (not paths["orchestrator"].exists()):
        atomic_json_write(paths["orchestrator"], defaults["orchestrator"])
    return paths


def _queue_counts(queue_path: Path) -> dict[str, int]:
    payload = _read_json(queue_path)
    tasks = list(payload.get("tasks", []))
    return {
        "pending": sum(1 for t in tasks if t.get("state") == "pending"),
        "in_progress": sum(1 for t in tasks if t.get("state") == "in_progress"),
        "completed": sum(1 for t in tasks if t.get("state") == "completed"),
        "failed": sum(1 for t in tasks if t.get("state") == "failed"),
    }


def _should_wait_for_validation(queue_counts: dict[str, int]) -> bool:
    """Strict sequential mode: do not generate while validation is outstanding."""
    pending = int(queue_counts.get("pending", 0))
    in_progress = int(queue_counts.get("in_progress", 0))
    return (pending + in_progress) > 0


def _collect_feedback_items(log_path: Path, limit: int = 6000, max_items: int = 24) -> list[dict[str, Any]]:
    rows = _tail_lines(log_path, limit)
    if not rows:
        return []

    items: list[dict[str, Any]] = []
    for raw in reversed(rows):
        try:
            row = json.loads(raw)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        event = str(row.get("event", ""))
        if event == "task_result":
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            items.append(
                {
                    "event": "task_result",
                    "strategy_name": str(row.get("strategy_name", "")),
                    "bar_config": str(row.get("bar_config", "")),
                    "verdict": str(row.get("verdict", "")),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "trade_count": metrics.get("trade_count"),
                },
            )
        elif event == "task_error":
            items.append(
                {
                    "event": "task_error",
                    "strategy_name": str(row.get("strategy_name", "")),
                    "error": str(row.get("error", ""))[:240],
                },
            )

        if len(items) >= max_items:
            break

    return items


def _collect_feedback_items_from_handoffs(
    handoffs_path: Path,
    *,
    max_items: int = 24,
) -> list[dict[str, Any]]:
    if not handoffs_path.exists():
        return []
    try:
        payload = _read_json(handoffs_path)
    except Exception:
        return []

    completed = payload.get("completed", [])
    if not isinstance(completed, list):
        return []

    items: list[dict[str, Any]] = []
    for row in reversed(completed):
        if not isinstance(row, dict):
            continue
        if str(row.get("handoff_type", "")) != "validation_request":
            continue
        result = row.get("result")
        payload_row = row.get("payload")
        if not isinstance(result, dict) or not isinstance(payload_row, dict):
            continue

        items.append(
            {
                "event": "validation_result",
                "strategy_name": str(payload_row.get("strategy_name", "")),
                "hypothesis_id": str(payload_row.get("hypothesis_id", "")),
                "overall_verdict": str(result.get("overall_verdict", "")),
                "task_count": result.get("task_count"),
                "pass_count": result.get("pass_count"),
                "fail_count": result.get("fail_count"),
                "error_count": result.get("error_count"),
                "avg_sharpe_ratio": result.get("avg_sharpe_ratio"),
                "avg_trade_count": result.get("avg_trade_count"),
            },
        )
        if len(items) >= max_items:
            break

    return items


def _collect_orchestrator_feedback_items(
    log_path: Path,
    *,
    limit: int = 4000,
    max_items: int = 16,
) -> list[dict[str, Any]]:
    """Collect generation_rejected and generation_error events from orchestrator log."""
    rows = _tail_lines(log_path, limit)
    if not rows:
        return []

    items: list[dict[str, Any]] = []
    for raw in reversed(rows):
        try:
            row = json.loads(raw)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        event = str(row.get("event", ""))
        if event == "generation_rejected":
            errors = row.get("errors", [])
            if isinstance(errors, list) and errors:
                error_str = str(errors[0])[:300]
            elif isinstance(errors, str) and errors:
                error_str = errors[:300]
            else:
                error_str = ""
            items.append({
                "event": "generation_rejected",
                "strategy_name": str(row.get("strategy_name", "")),
                "hypothesis_id": str(row.get("hypothesis_id", "")),
                "error": error_str,
            })
        elif event == "generation_error":
            items.append({
                "event": "generation_error",
                "error": str(row.get("error", ""))[:240],
                "iteration": row.get("iteration"),
            })
        if len(items) >= max_items:
            break

    return items


def _build_merged_feedback_items(
    *,
    handoffs_path: Path,
    research_log_path: Path,
    orchestrator_log_path: Path,
    max_items: int = 40,
) -> list[dict[str, Any]]:
    """Merge feedback from all three sources. Orch errors first (most actionable)."""
    per_source = max(1, max_items // 3)

    handoff_items = _collect_feedback_items_from_handoffs(
        handoffs_path, max_items=per_source
    )
    research_items = _collect_feedback_items(
        research_log_path, max_items=per_source
    )
    orch_items = _collect_orchestrator_feedback_items(
        orchestrator_log_path, max_items=per_source
    )

    # Interleave: orch errors first (most actionable), then validator results, then handoffs
    merged: list[dict[str, Any]] = []
    merged.extend(orch_items)
    merged.extend(research_items)
    merged.extend(handoff_items)
    return merged[:max_items]


def _as_str_list(value: Any, max_items: int = 8, max_len: int = 220) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in value:
        item = str(raw).strip()
        if not item:
            continue
        out.append(item[:max_len])
        if len(out) >= max_items:
            break
    return out


def _normalize_feedback_digest(payload: dict[str, Any]) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("feedback payload must be a JSON object")

    strengths = _as_str_list(payload.get("strengths"))
    weaknesses = _as_str_list(payload.get("weaknesses"))
    error_patterns = _as_str_list(payload.get("error_patterns"))
    guardrails = _as_str_list(payload.get("guardrails"))
    next_focus = _as_str_list(payload.get("next_focus"))

    if not strengths:
        strengths = ["No reliable strengths identified yet."]
    if not weaknesses:
        weaknesses = ["No consistent weaknesses identified yet."]
    if not guardrails:
        guardrails = ["Enforce strict causality and signal contract."]
    if not next_focus:
        next_focus = ["Generate one robust, testable hypothesis."]

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "error_patterns": error_patterns,
        "guardrails": guardrails,
        "next_focus": next_focus,
    }


def _build_feedback_system_prompt() -> str:
    return (
        "You are a quantitative research feedback analyst for an automated NQ E-mini futures alpha discovery system.\n\n"
        "INSTRUMENT CONTEXT:\n"
        "- NQ E-mini Nasdaq-100 futures, tick size 0.25 pts = $5/tick\n"
        "- Transaction cost: $14.50 per round-trip (commission + 1-tick slippage each side)\n"
        "- To break even at 30 trades: need +$435 gross PnL minimum\n"
        "- Validate split: Sep 2024 – Mar 2025 (7 months; includes US election Nov 2024, Fed rate cuts)\n"
        "- Session filter: ETH (03:00–16:00 ET); signals fire on bar close, entry on next open\n\n"
        "PASS CRITERIA (what success looks like):\n"
        "- Sharpe ratio >= 1.5 on validate split\n"
        "- Trade count >= 30 total across validate split\n"
        "- Gauntlet pass: walk-forward consistency, day-of-week stability, permutation test\n\n"
        "COMMON FAILURE PATTERNS TO WATCH FOR:\n"
        "- Overtrading: >200 trades/month destroys PnL via $14.50 RT costs (e.g. Sharpe -4.41 from 853 trades)\n"
        "- Zero signals: entry filter too strict, no trades generated at all\n"
        "- Causality violations: using future bar data (negative shift index, global aggregates)\n"
        "- dtype errors: signal array returned as int32 instead of int8 (contract failure)\n"
        "- Missing column fallbacks: KeyError on feature columns that may be null/absent\n"
        "- Cross-day contamination: rolling operations that bleed across session boundaries\n\n"
        "Given recent experiment events, extract practical guidance for the next hypothesis iteration.\n"
        "Return ONLY a JSON object with keys:\n"
        "- strengths: list[str] — what is working or showing promise\n"
        "- weaknesses: list[str] — recurring failure modes in recent strategies\n"
        "- error_patterns: list[str] — specific technical errors to avoid (exact error messages if available)\n"
        "- guardrails: list[str] — concrete rules the next strategy MUST follow\n"
        "- next_focus: list[str] — specific hypothesis directions most likely to find edge\n"
        "Keep each item concrete, specific, and actionable. Reference actual strategy names and metrics where available."
    )


def _build_feedback_user_prompt(*, feedback_items: list[dict[str, Any]]) -> str:
    if not feedback_items:
        return (
            "No experiment results yet. This is the first iteration.\n"
            "Return conservative defaults emphasising:\n"
            "- strict causality (no lookahead)\n"
            "- moderate signal frequency (30-150 trades over validate split)\n"
            "- robust column access with null fallbacks\n"
            "- return dtype must be np.int8"
        )
    return (
        "Analyze these recent experiment events and output structured guidance for the next hypothesis.\n"
        "Pay special attention to error_patterns — extract exact error messages from generation_rejected "
        "and task_error events so the coder can avoid repeating the same mistakes.\n\n"
        f"{json.dumps(feedback_items, indent=2, default=str)}"
    )


def _normalize_thinker_brief(
    payload: dict[str, Any],
    *,
    mission_bar_configs: list[str],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("thinker payload must be a JSON object")

    hypothesis_id = _slug(str(payload.get("hypothesis_id", "")).strip())
    if not hypothesis_id:
        hypothesis_id = f"hyp_{int(time.time())}"

    strategy_name_hint = _slug(str(payload.get("strategy_name_hint", "")).strip())
    if not strategy_name_hint:
        strategy_name_hint = f"{hypothesis_id}_signal"

    params_template = payload.get("params_template", {})
    if not isinstance(params_template, dict):
        params_template = {}

    thesis = str(payload.get("thesis", "")).strip()[:800]
    entry_logic = str(payload.get("entry_logic", "")).strip()[:1200]
    exit_logic = str(payload.get("exit_logic", "")).strip()[:800]
    risk_controls = _as_str_list(payload.get("risk_controls"), max_items=8, max_len=220)
    anti_lookahead_checks = _as_str_list(payload.get("anti_lookahead_checks"), max_items=8, max_len=220)
    validation_focus = _as_str_list(payload.get("validation_focus"), max_items=8, max_len=220)

    if not thesis:
        thesis = "Event-driven intraday hypothesis with strict causality discipline."
    if not entry_logic:
        entry_logic = "Generate signals from current and past bars only."
    if not exit_logic:
        exit_logic = "Rely on framework PT/SL/timeout controls."
    if not anti_lookahead_checks:
        anti_lookahead_checks = ["No forward shifts.", "No global future aggregates."]
    if not validation_focus:
        validation_focus = ["Sharpe robustness", "Sufficient trade count", "Gauntlet pass likelihood"]

    raw_cfgs = payload.get("bar_configs", [])
    requested = [str(v).strip() for v in raw_cfgs] if isinstance(raw_cfgs, list) else []
    allowed = {str(v).strip() for v in mission_bar_configs}
    chosen = [cfg for cfg in requested if cfg in allowed]
    if not chosen:
        chosen = [mission_bar_configs[0]]

    return {
        "hypothesis_id": hypothesis_id,
        "strategy_name_hint": strategy_name_hint,
        "bar_configs": chosen,
        "params_template": dict(params_template),
        "thesis": thesis,
        "entry_logic": entry_logic,
        "exit_logic": exit_logic,
        "risk_controls": risk_controls,
        "anti_lookahead_checks": anti_lookahead_checks,
        "validation_focus": validation_focus,
    }


def _normalize_coder_payload(
    payload: dict[str, Any],
    *,
    mission_bar_configs: list[str],
    thinker_brief: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("coder payload must be a JSON object")

    strategy_name = _slug(str(payload.get("strategy_name", payload.get("name", ""))).strip())
    if not strategy_name:
        strategy_name = _slug(str(thinker_brief.get("strategy_name_hint", "")))
    if not strategy_name:
        strategy_name = f"llm_strategy_{int(time.time())}"

    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("payload.params must be an object")
    if not params:
        thinker_params = thinker_brief.get("params_template")
        if isinstance(thinker_params, dict):
            params = dict(thinker_params)

    raw_cfgs = payload.get("bar_configs", payload.get("bar_config", []))
    if isinstance(raw_cfgs, str):
        raw_cfgs = [raw_cfgs]
    requested = [str(v).strip() for v in raw_cfgs] if isinstance(raw_cfgs, list) else []
    allowed = {str(v).strip() for v in mission_bar_configs}
    chosen = [cfg for cfg in requested if cfg in allowed]
    if not chosen:
        thinker_cfgs = thinker_brief.get("bar_configs")
        if isinstance(thinker_cfgs, list):
            chosen = [cfg for cfg in thinker_cfgs if cfg in allowed]
    if not chosen:
        chosen = [mission_bar_configs[0]]

    raw_code = payload.get("code")
    if raw_code is None or not str(raw_code).strip():
        for alt in ("python_code", "module_code", "signal_code", "source", "script"):
            alt_val = payload.get(alt)
            if isinstance(alt_val, str) and alt_val.strip():
                raw_code = alt_val
                break
    if raw_code is None:
        raise ValueError("payload.code is required")
    code = _strip_code_fences(str(raw_code))
    if not code.strip():
        raise ValueError("payload.code is empty")

    return {
        "strategy_name": strategy_name,
        "params": dict(params),
        "bar_configs": chosen,
        "code": code.strip() + "\n",
    }


def _choose_module_path(signals_dir: Path, *, strategy_name: str, module_code: str) -> tuple[Path, bool]:
    base = _slug(strategy_name)
    candidate = signals_dir / f"{base}.py"
    if not candidate.exists():
        return candidate, True
    try:
        existing = candidate.read_text(encoding="utf-8")
    except Exception:
        existing = ""
    if existing == module_code:
        return candidate, False

    suffix = 2
    while True:
        alt = signals_dir / f"{base}_{suffix:02d}.py"
        if not alt.exists():
            return alt, True
        try:
            existing_alt = alt.read_text(encoding="utf-8")
        except Exception:
            existing_alt = ""
        if existing_alt == module_code:
            return alt, False
        suffix += 1


def _existing_signal_names(signals_dir: Path) -> list[str]:
    names: list[str] = []
    for path in sorted(signals_dir.glob("*.py")):
        if path.stem == "__init__" or path.stem.startswith("_"):
            continue
        names.append(path.stem)
    return names


def _load_sample_strategy_df(
    *,
    split: str,
    bar_config: str,
    session_filter: str,
    feature_group: str,
    max_files: int = 20,
    max_rows: int = 1200,
) -> pl.DataFrame:
    try:
        parsed = _parse_bar_config(bar_config)
    except ValueError as exc:
        raise RuntimeError(f"Invalid bar_config '{bar_config}': {exc}") from exc
    files = get_split_files(split)
    if not files:
        raise RuntimeError(f"No files found for split={split}")

    last_non_empty: pl.DataFrame | None = None
    for file_path in files[:max_files]:
        try:
            df = load_cached_matrix(
                file_path,
                bar_size=parsed["bar_size"],
                bar_type=parsed["bar_type"],
                bar_threshold=parsed["bar_threshold"],
                include_bar_columns=True,
                session_filter=session_filter,
            )
        except Exception:
            continue
        if len(df) == 0:
            continue
        df = filter_feature_group(df, feature_group)
        label_cols = [c for c in LABEL_COLUMNS if c in df.columns]
        strategy_df = df.drop(label_cols) if label_cols else df
        if len(strategy_df) == 0:
            continue
        last_non_empty = strategy_df.head(max_rows)
        if len(last_non_empty) >= 64:
            return last_non_empty

    if last_non_empty is not None:
        return last_non_empty
    raise RuntimeError(f"Could not load sample feature frame for bar_config={bar_config}")


FEATURE_COMPUTATION_NOTES = [
    "Use precomputed canonical feature columns whenever available; avoid recomputing them in signal code.",
    "sma_ratio_N = close / SMA(close, N) - 1 (rolling mean, min_samples=N).",
    "ema_ratio_N = close / EWM(close, span=N, adjust=False, min_samples=N) - 1.",
    "RSI uses Wilder smoothing (alpha=1/N) from close-to-close gains/losses.",
    "ATR uses true range with Wilder smoothing; session-aware bars and timestamps are already canonicalized.",
    "VWAP-style cumulative metrics reset at session/day boundaries in canonical builders.",
    "Opening-range fields are session-aware; OR-dependent fields are null before OR is ready.",
    "Orderflow/toxicity/footprint features are bar-causal and derived only from events up to each bar close.",
]


def _build_feature_knowledge(
    *,
    mission_bar_configs: list[str],
    split: str,
    session_filter: str,
    feature_group: str,
    sample_cache: dict[str, pl.DataFrame],
) -> dict[str, Any]:
    columns_by_cfg: dict[str, list[str]] = {}
    errors: dict[str, str] = {}

    for bar_config in mission_bar_configs:
        try:
            if bar_config not in sample_cache:
                sample_cache[bar_config] = _load_sample_strategy_df(
                    split=split,
                    bar_config=bar_config,
                    session_filter=session_filter,
                    feature_group=feature_group,
                )
            cols = sorted(str(c) for c in sample_cache[bar_config].columns)
            columns_by_cfg[bar_config] = cols
        except Exception as exc:
            errors[bar_config] = f"{type(exc).__name__}: {exc}"

    if not columns_by_cfg:
        return {
            "schema_version": "1.0",
            "bar_configs": {},
            "common_columns": [],
            "per_bar_extra_columns": {},
            "computation_notes": list(FEATURE_COMPUTATION_NOTES),
            "errors": errors,
        }

    cfg_names = list(columns_by_cfg.keys())
    common = set(columns_by_cfg[cfg_names[0]])
    for cfg in cfg_names[1:]:
        common &= set(columns_by_cfg[cfg])
    common_columns = sorted(common)

    extras: dict[str, list[str]] = {}
    for cfg, cols in columns_by_cfg.items():
        extras[cfg] = [c for c in cols if c not in common]

    counts = {
        cfg: {
            "total_columns": len(cols),
            "extra_columns": len(extras.get(cfg, [])),
        }
        for cfg, cols in columns_by_cfg.items()
    }

    return {
        "schema_version": "1.0",
        "bar_configs": counts,
        "common_columns": common_columns,
        "per_bar_extra_columns": extras,
        "computation_notes": list(FEATURE_COMPUTATION_NOTES),
        "errors": errors,
    }


def _validate_generated_strategy(
    *,
    strategy_name: str,
    signals_dir: Path,
    params: dict[str, Any],
    bar_configs: list[str],
    split: str,
    session_filter: str,
    feature_group: str,
    sample_cache: dict[str, pl.DataFrame],
) -> list[str]:
    module = load_signal_module(strategy_name, signals_dir=signals_dir)
    strategy_fn = getattr(module, "generate_signal", None)
    if not callable(strategy_fn):
        return [f"{strategy_name}: missing callable generate_signal(df, params)"]

    errors: list[str] = []
    for bar_config in bar_configs:
        if bar_config not in sample_cache:
            sample_cache[bar_config] = _load_sample_strategy_df(
                split=split,
                bar_config=bar_config,
                session_filter=session_filter,
                feature_group=feature_group,
            )

        strategy_df = sample_cache[bar_config]
        if len(strategy_df) == 0:
            errors.append(f"{strategy_name}: empty sample frame for {bar_config}")
            continue

        try:
            causality = check_signal_causality(
                generate_fn=strategy_fn,
                df=strategy_df,
                params=params,
                mode="strict",
            )
        except Exception as exc:
            errors.append(f"{strategy_name}: causality check crashed for {bar_config}: {type(exc).__name__}: {exc}")
            continue
        if causality:
            errors.append(f"{strategy_name}: non-causal for {bar_config}: {causality[0]}")
            continue

        try:
            raw = np.asarray(strategy_fn(strategy_df, params))
        except Exception as exc:
            errors.append(f"{strategy_name}: generate_signal failed for {bar_config}: {type(exc).__name__}: {exc}")
            continue
        signal_errors = _validate_signal_array(raw, len(strategy_df))
        if signal_errors:
            errors.append(f"{strategy_name}: contract failed for {bar_config}: {signal_errors[0]}")

    return errors


def _build_thinker_system_prompt() -> str:
    return (
        "You are a quant researcher designing intraday alpha hypotheses for NQ E-mini Nasdaq-100 futures.\n\n"
        "INSTRUMENT & COST REALITY:\n"
        "- Tick size: 0.25 pts = $5/tick. Total round-trip cost: $14.50 (commission + 1-tick slippage/side).\n"
        "- A strategy trading 100 times costs $1,450 in friction. At 500 trades it needs $7,250 gross to break even.\n"
        "- TARGET: 30–150 total signals across the 7-month validate split (Sep 2024–Mar 2025).\n"
        "- Strategies firing >300 signals almost certainly lose to transaction costs. Design for precision, not frequency.\n\n"
        "VALIDATE SPLIT CONTEXT (Sep 2024 – Mar 2025):\n"
        "- Includes: US presidential election (Nov 2024), multiple Fed rate decisions, Q4 2024 rally, early 2025 volatility.\n"
        "- ETH session: 03:00–16:00 ET. Signals must respect session boundaries.\n\n"
        "PASS CRITERIA:\n"
        "- Sharpe ratio >= 1.5, trade count >= 30, gauntlet pass (walk-forward + permutation + day-of-week tests).\n\n"
        "ANTI-LOOKAHEAD RULES (non-negotiable):\n"
        "- Never reference bar N+1 or later. Only use current bar (index i) and past bars.\n"
        "- shift(1) means shift FORWARD in time (previous bar value) — safe. shift(-1) means FUTURE — forbidden.\n"
        "- Rolling aggregates must use only past bars. No global future statistics.\n\n"
        "DESIGN PRINCIPLES:\n"
        "- Prefer event-driven, sparse signals (a few high-conviction entries per week over noisy daily signals).\n"
        "- Use precomputed canonical features as primary building blocks — do not re-derive what is already there.\n"
        "- Every entry condition should have a clear microstructure rationale (why does this predict price direction?).\n"
        "- Include explicit exit logic: profit target in ticks, stop loss in ticks, max holding bars.\n"
        "- Include cooldown / reentry prevention to avoid chasing the same move multiple times.\n\n"
        "Return ONLY a JSON object with keys:\n"
        "- hypothesis_id: str (short slug, e.g. 'cvd_fade_va_reject_001')\n"
        "- strategy_name_hint: str (Python-safe slug)\n"
        "- thesis: str (1-2 sentences: what market inefficiency does this exploit and why does it exist?)\n"
        "- bar_configs: list[str] (choose from allowed configs; pick those that match signal frequency target)\n"
        "- params_template: object (all numeric thresholds with sensible defaults)\n"
        "- entry_logic: str (precise conditions using named feature columns; specify long/short separately)\n"
        "- exit_logic: str (PT ticks, SL ticks, max_bars; rely on framework controls)\n"
        "- risk_controls: list[str] (e.g. max trades/day, time-of-day filter, spread filter)\n"
        "- anti_lookahead_checks: list[str] (explicit list of checks the coder must verify)\n"
        "- validation_focus: list[str] (what metrics/patterns would confirm this hypothesis has real edge)\n\n"
        "Before producing the final JSON, internally brainstorm at least three distinct hypotheses, "
        "evaluate each against the cost reality and pass criteria, then select the strongest. "
        "Output only the final selected hypothesis JSON."
    )


def _build_thinker_user_prompt(
    *,
    mission: dict[str, Any],
    existing_strategies: list[str],
    feedback_digest: dict[str, list[str]],
    feature_knowledge: dict[str, Any] | None = None,
) -> str:
    bar_configs = mission.get("bar_configs", ["tick_610"])
    current_focus = mission.get("current_focus", [])
    if not isinstance(current_focus, list):
        current_focus = []
    objective = str(mission.get("objective", "Discover robust intraday alpha signals."))
    avoid = ", ".join(existing_strategies[-50:]) if existing_strategies else "(none)"
    focus_blob = "\n".join(f"- {str(x)}" for x in current_focus) if current_focus else "- none provided"
    prompt = (
        f"Mission objective:\n{objective}\n\n"
        f"Allowed bar_configs: {bar_configs}\n"
        f"Preferred session filter: {mission.get('session_filter', 'eth')}\n"
        f"Feature group: {mission.get('feature_group', 'all')}\n"
        f"Current focus:\n{focus_blob}\n\n"
        f"Existing strategy files to avoid duplicating:\n{avoid}\n\n"
        f"Structured feedback digest:\n{json.dumps(feedback_digest, indent=2)}\n\n"
        "Design exactly one hypothesis that is implementable by a separate coding model."
    )
    if feature_knowledge:
        prompt += (
            "\n\nAVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN\n"
            f"{json.dumps(feature_knowledge, indent=2, sort_keys=True, default=str)}\n"
            "AVAILABLE_PRECOMPUTED_FEATURES_JSON_END\n"
            "Use these precomputed feature columns as the primary building blocks."
        )
    return prompt


def _build_coder_system_prompt() -> str:
    return """\
You are a Python quant signal coder for NQ E-mini Nasdaq-100 futures.
You receive a structured hypothesis from a thinker model. Implement ONLY that plan — nothing more.

Return ONLY a JSON object with exactly these keys: strategy_name, bar_configs, params, code.

CRITICAL: dtype=np.int8 IS MANDATORY
generate_signal MUST return np.ndarray with dtype=np.int8 and values in {-1, 0, 1}.
Use: signal = np.where(long_cond, 1, np.where(short_cond, -1, 0)).astype(np.int8)
Do NOT omit .astype(np.int8). np.where returns int64 by default — that WILL fail validation.

WORKING EXAMPLE MODULE (study this structure exactly):

from __future__ import annotations
from typing import Any
import numpy as np
import polars as pl

DEFAULT_PARAMS: dict[str, Any] = {
    "fast_span": 8,
    "slow_span": 21,
    "min_distance": 0.0,
}

def generate_signal(df: pl.DataFrame, params: dict[str, Any]) -> np.ndarray:
    cfg = dict(DEFAULT_PARAMS)
    cfg.update(params or {})

    close = np.asarray(df["close"].to_numpy(), dtype=np.float64)
    fast = _ema(close, int(cfg["fast_span"]))
    slow = _ema(close, int(cfg["slow_span"]))

    dist = fast - slow
    prev = np.roll(dist, 1)
    prev[0] = 0.0

    min_distance = float(cfg["min_distance"])
    cross_up = (prev <= 0.0) & (dist > min_distance)
    cross_down = (prev >= 0.0) & (dist < -min_distance)

    # ALWAYS end with .astype(np.int8)
    signal = np.where(cross_up, 1, np.where(cross_down, -1, 0)).astype(np.int8)
    return signal

STRATEGY_METADATA = {
    "name": "example_ema_turn",
    "version": "1.0",
    "features_required": ["close"],
    "description": "EMA crossover signal.",
}

REQUIREMENTS FOR `code`:
- Complete Python module (no placeholders, no `pass`)
- Allowed imports ONLY: numpy as np, polars as pl, typing (Any, Optional, etc.), __future__
- Defines DEFAULT_PARAMS: dict[str, Any] — all tunable scalars here
- Defines STRATEGY_METADATA: dict with name, version, features_required, description
- Defines generate_signal(df: pl.DataFrame, params: dict[str, Any]) -> np.ndarray
  * Returns array of dtype=np.int8, values strictly in {-1, 0, 1}, length == len(df)
  * No NaN values — fill any intermediate NaN before the final signal array
  * Merge params into defaults at start: cfg = dict(DEFAULT_PARAMS); cfg.update(params or {})
- Deterministic: no random, no time.time(), no uuid
- No I/O: no open(), no os, no sys, no subprocess, no network, no filesystem
- No lookahead: NEVER use shift(-1) or any negative offset (reads the future)
  * Correct: df["col"].shift(1)  -- previous bar value
  * WRONG:   df["col"].shift(-1) -- future bar value (instant disqualification)
- Safe fallbacks for missing columns:
  * if "col_name" not in df.columns: use np.zeros(len(df), dtype=np.float64)
  * All precomputed features may be absent -- never crash on KeyError

COMMON MISTAKES THAT WILL FAIL VALIDATION:
1. Missing .astype(np.int8) -> dtype validation fails (int64 != int8)
2. Negative shift: df["x"].shift(-1) -> causality check fails immediately
3. KeyError on missing column -> crashes at import time
4. NaN in output array -> contract validation fails
5. Wrong length: len(signal) != len(df) -> contract validation fails
6. Importing os, sys, requests, json, random -> forbidden imports check fails
"""


def _build_coder_handoff(
    *,
    thinker_brief: dict[str, Any],
    mission: dict[str, Any],
    thinker_payload_hash: str,
) -> dict[str, Any]:
    if not isinstance(thinker_brief, dict):
        raise ValueError("thinker_brief must be a dict")
    hypothesis_id = str(thinker_brief.get("hypothesis_id", "")).strip()
    strategy_name_hint = str(thinker_brief.get("strategy_name_hint", "")).strip()
    if not hypothesis_id:
        raise ValueError("thinker_brief.hypothesis_id is required")
    if not strategy_name_hint:
        raise ValueError("thinker_brief.strategy_name_hint is required")

    allowed = mission.get("bar_configs", ["tick_610"])
    allowed_cfgs = [str(v).strip() for v in allowed] if isinstance(allowed, list) else ["tick_610"]

    def _clip_text(value: Any, limit: int) -> str:
        raw = str(value or "").strip()
        if len(raw) <= limit:
            return raw
        if limit <= 3:
            return raw[:limit]
        return raw[: limit - 3].rstrip() + "..."

    def _clip_list(value: Any, *, max_items: int, max_len: int) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = _clip_text(item, max_len)
            if not text:
                continue
            out.append(text)
            if len(out) >= max_items:
                break
        return out

    return {
        "handoff_version": "1.0",
        "source_role": "quant_thinker",
        "payload_hash": str(thinker_payload_hash),
        "mission_constraints": {
            "allowed_bar_configs": allowed_cfgs,
            "preferred_session_filter": str(mission.get("session_filter", "eth")),
            "feature_group": str(mission.get("feature_group", "all")),
        },
        "hypothesis": {
            "hypothesis_id": hypothesis_id,
            "strategy_name_hint": strategy_name_hint,
            "bar_configs": list(thinker_brief.get("bar_configs", [])),
            "params_template": dict(thinker_brief.get("params_template", {}))
            if isinstance(thinker_brief.get("params_template"), dict)
            else {},
            "thesis": _clip_text(thinker_brief.get("thesis", ""), 520),
            "entry_logic": _clip_text(thinker_brief.get("entry_logic", ""), 780),
            "exit_logic": _clip_text(thinker_brief.get("exit_logic", ""), 520),
            "risk_controls": _clip_list(thinker_brief.get("risk_controls", []), max_items=6, max_len=180),
            "anti_lookahead_checks": _clip_list(
                thinker_brief.get("anti_lookahead_checks", []),
                max_items=6,
                max_len=180,
            ),
            "validation_focus": _clip_list(thinker_brief.get("validation_focus", []), max_items=6, max_len=180),
        },
    }


def _build_coder_user_prompt(
    *,
    thinker_handoff: dict[str, Any],
    feature_knowledge: dict[str, Any] | None = None,
) -> str:
    prompt = (
        "Implement exactly the handoff below as a signal module.\n"
        "Use only this handoff JSON as source of truth.\n\n"
        "THINKER_HANDOFF_JSON_BEGIN\n"
        f"{json.dumps(thinker_handoff, indent=2, sort_keys=True, default=str)}\n"
        "THINKER_HANDOFF_JSON_END\n"
    )
    if feature_knowledge:
        prompt += (
            "\nAVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN\n"
            f"{json.dumps(feature_knowledge, indent=2, sort_keys=True, default=str)}\n"
            "AVAILABLE_PRECOMPUTED_FEATURES_JSON_END\n"
            "Prefer these precomputed features directly; only compute local fallbacks when a column is missing."
        )
    return prompt


_REPAIR_CODE_MAX_CHARS = 4000


def _build_coder_repair_user_prompt(
    *,
    thinker_handoff: dict[str, Any],
    previous_code: str,
    validation_errors: list[str],
    common_columns: list[str],
) -> str:
    """Build a targeted repair prompt for the coder when generated code fails validation."""
    code_snippet = previous_code
    if len(code_snippet) > _REPAIR_CODE_MAX_CHARS:
        code_snippet = code_snippet[:_REPAIR_CODE_MAX_CHARS] + "\n... [truncated]"

    errors_blob = "\n".join(f"  - {e}" for e in validation_errors)
    cols_hint = ", ".join(common_columns[:40]) if common_columns else "(see feature knowledge)"

    return (
        "Your previously generated code FAILED validation. Fix ONLY the listed errors.\n\n"
        "ORIGINAL_THINKER_HANDOFF_JSON_BEGIN\n"
        f"{json.dumps(thinker_handoff, indent=2, sort_keys=True, default=str)}\n"
        "ORIGINAL_THINKER_HANDOFF_JSON_END\n\n"
        "PREVIOUS_CODE_BEGIN\n"
        f"{code_snippet}\n"
        "PREVIOUS_CODE_END\n\n"
        f"VALIDATION_ERRORS:\n{errors_blob}\n\n"
        f"AVAILABLE_COLUMNS_HINT (common across bar configs): {cols_hint}\n\n"
        "Return corrected JSON with same schema: strategy_name, bar_configs, params, code.\n"
        "Fix ONLY the validation errors above. Do not change the overall strategy logic."
    )


def _task_id(strategy_name: str, bar_config: str, params: dict[str, Any], code_hash: str) -> str:
    digest = _sha256_text(
        f"{strategy_name}|{bar_config}|{json.dumps(params, sort_keys=True, separators=(',', ':'))}|{code_hash}",
    )[:12]
    return f"llm_{_slug(strategy_name)}_{_slug(bar_config)}_{digest}"


def _build_task(
    *,
    strategy_name: str,
    split: str,
    bar_config: str,
    params: dict[str, Any],
    mission: dict[str, Any],
    code_hash: str,
    iteration: int,
    hypothesis_id: str,
) -> dict[str, Any]:
    max_retries = int(mission.get("max_retries", 2))
    timeout_minutes = int(mission.get("task_timeout_minutes", 30))
    heartbeat_seconds = int(mission.get("heartbeat_interval_seconds", 300))
    return {
        "task_id": _task_id(strategy_name, bar_config, params, code_hash),
        "state": "pending",
        "assigned_to": None,
        "strategy_name": strategy_name,
        "split": split,
        "bar_config": bar_config,
        "params": dict(params),
        "retries": 0,
        "max_retries": max_retries,
        "timeout_minutes": timeout_minutes,
        "heartbeat_interval_seconds": heartbeat_seconds,
        "run_gauntlet": bool(mission.get("run_gauntlet", True)),
        "write_candidate": bool(mission.get("write_candidates", True)),
        "source": {
            "agent": "llm_orchestrator",
            "iteration": int(iteration),
            "hypothesis_id": str(hypothesis_id),
            "code_hash": code_hash,
        },
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _maybe_write(path: Path, content: str, *, allow_overwrite: bool = True) -> None:
    if path.exists():
        try:
            current = path.read_text(encoding="utf-8")
        except Exception:
            current = None
        if current == content:
            return
        if not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing module path: {path}")
    _atomic_write_text(path, content)


def _cfg_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _build_llm_client(
    *,
    provider: str,
    model: str,
    agent_cfg: dict[str, Any],
    root: Path,
) -> LLMRawClient:
    raw_provider = str(provider).strip().lower()
    if raw_provider not in {"claude", "claude_cli", "claude_code"}:
        raise ValueError(
            "Unsupported provider. This orchestrator is configured for Claude Code only "
            "(set `provider: claude_cli`)."
        )

    cli_cfg = _cfg_dict(agent_cfg.get("claude_cli"))
    cli_binary = str(
        cli_cfg.get(
            "binary",
            agent_cfg.get("claude_binary", os.getenv("CLAUDE_CODE_BIN", "claude")),
        ),
    ).strip() or "claude"
    timeout_seconds = float(cli_cfg.get("timeout_seconds", agent_cfg.get("timeout_seconds", 180)))
    retries = int(cli_cfg.get("retries", agent_cfg.get("retries", 2)))
    retry_backoff_seconds = float(cli_cfg.get("retry_backoff_seconds", agent_cfg.get("retry_backoff_seconds", 1.5)))
    workdir_raw = str(cli_cfg.get("workdir", "")).strip()
    workdir = root if not workdir_raw else (Path(workdir_raw) if Path(workdir_raw).is_absolute() else (root / workdir_raw))
    extra_args_raw = cli_cfg.get("extra_args", [])
    extra_args = [str(v) for v in extra_args_raw] if isinstance(extra_args_raw, list) else []

    return ClaudeCodeCLIClient(
        model=model,
        cli_binary=cli_binary,
        timeout_seconds=timeout_seconds,
        max_retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        workdir=workdir,
        extra_args=extra_args,
    )


def _resolve_role_cfg(
    *,
    agent_cfg: dict[str, Any],
    role: str,
    default_temperature: float,
    default_max_output_tokens: int,
) -> dict[str, Any]:
    role_cfg = _cfg_dict(agent_cfg.get(role))
    model = str(role_cfg.get("model", agent_cfg.get("model", ""))).strip()
    if not model:
        raise ValueError(f"agent config requires `{role}.model` (or legacy `model`)")

    return {
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous LLM signal generator + queue enqueuer.")
    parser.add_argument("--mission", type=Path, required=True, help="Mission YAML path.")
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path("configs/agents/llm_orchestrator.yaml"),
        help="Agent config YAML path.",
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="Override agent max_iterations.")
    parser.add_argument("--max-runtime-hours", type=float, default=None, help="Optional runtime cap.")
    parser.add_argument("--poll-seconds", type=int, default=None, help="Sleep between successful iterations.")
    parser.add_argument("--resume", action="store_true", help="Resume orchestrator state from research/.state.")
    parser.add_argument("--dry-run", action="store_true", help="Generate and validate only; do not write files/tasks.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    mission_path = args.mission.resolve()
    if not mission_path.exists():
        raise FileNotFoundError(f"Mission file not found: {mission_path}")
    agent_cfg_path = args.agent_config.resolve() if args.agent_config.is_absolute() else (root / args.agent_config).resolve()
    if not agent_cfg_path.exists():
        raise FileNotFoundError(f"Agent config file not found: {agent_cfg_path}")

    mission = _load_yaml(mission_path)
    agent_cfg = _load_yaml(agent_cfg_path)
    mission_name = str(mission.get("mission_name", mission_path.stem))
    bar_configs_raw = mission.get("bar_configs", ["tick_610"])
    mission_bar_configs = [str(v) for v in bar_configs_raw] if isinstance(bar_configs_raw, list) else ["tick_610"]
    if not mission_bar_configs:
        raise ValueError("mission.bar_configs cannot be empty")

    splits_raw = mission.get("splits_allowed", ["validate"])
    split = str((splits_raw[0] if isinstance(splits_raw, list) and splits_raw else "validate")).lower()
    if split == "test":
        split = "validate"
    session_filter = str(mission.get("session_filter", "eth")).lower()
    feature_group = str(mission.get("feature_group", "all")).lower()

    state_paths = _ensure_runtime_state(root, resume=bool(args.resume), mission_name=mission_name)
    orchestrator_state = _read_json(state_paths["orchestrator"])
    iterations_done = int(orchestrator_state.get("iterations_completed", 0))
    total_tasks = int(orchestrator_state.get("total_tasks_enqueued", 0))
    generated_modules = list(orchestrator_state.get("generated_modules", []))

    runtime_cfg = _cfg_dict(agent_cfg.get("runtime"))
    max_iterations = int(
        args.max_iterations
        if args.max_iterations is not None
        else runtime_cfg.get("max_iterations", agent_cfg.get("max_iterations", 20)),
    )
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")

    max_pending_tasks = int(runtime_cfg.get("max_pending_tasks", agent_cfg.get("max_pending_tasks", 25)))
    poll_seconds = int(
        args.poll_seconds
        if args.poll_seconds is not None
        else runtime_cfg.get("poll_seconds", agent_cfg.get("poll_seconds", 10)),
    )
    max_runtime_hours = (
        float(args.max_runtime_hours)
        if args.max_runtime_hours is not None
        else (
            float(runtime_cfg["max_runtime_hours"])
            if runtime_cfg.get("max_runtime_hours") is not None
            else (float(agent_cfg["max_runtime_hours"]) if agent_cfg.get("max_runtime_hours") is not None else None)
        )
    )
    stage_max_attempts = int(runtime_cfg.get("stage_max_attempts", 3))
    json_repair_attempts = int(runtime_cfg.get("json_repair_attempts", 1))
    semantic_retry_attempts = int(runtime_cfg.get("semantic_retry_attempts", 1))
    max_code_repair_attempts = int(runtime_cfg.get("max_code_repair_attempts", 2))
    stage_backoff_seconds = float(runtime_cfg.get("stage_backoff_seconds", 3.0))
    quota_backoff_seconds = float(runtime_cfg.get("quota_backoff_seconds", 20.0))
    max_backoff_seconds = float(runtime_cfg.get("max_backoff_seconds", 90.0))
    if max_pending_tasks != 1:
        print(
            "Sequential mode active: "
            f"runtime.max_pending_tasks={max_pending_tasks} is ignored (effective=1).",
        )

    feedback_role = _resolve_role_cfg(
        agent_cfg=agent_cfg,
        role="feedback_analyst",
        default_temperature=0.1,
        default_max_output_tokens=1400,
    )
    thinker_role = _resolve_role_cfg(
        agent_cfg=agent_cfg,
        role="quant_thinker",
        default_temperature=0.2,
        default_max_output_tokens=2000,
    )
    coder_role = _resolve_role_cfg(
        agent_cfg=agent_cfg,
        role="coder",
        default_temperature=0.2,
        default_max_output_tokens=2600,
    )

    provider = str(agent_cfg.get("provider", "claude_cli")).strip().lower()
    feedback_client = _build_llm_client(
        provider=provider,
        model=feedback_role["model"],
        agent_cfg=agent_cfg,
        root=root,
    )
    thinker_client = _build_llm_client(
        provider=provider,
        model=thinker_role["model"],
        agent_cfg=agent_cfg,
        root=root,
    )
    coder_client = _build_llm_client(
        provider=provider,
        model=coder_role["model"],
        agent_cfg=agent_cfg,
        root=root,
    )

    signals_dir = root / "research" / "signals"
    handoffs_path = state_paths["handoffs"]
    research_log_path = root / "results" / "logs" / "research_experiments.jsonl"
    orchestrator_log_path = root / "results" / "logs" / "llm_orchestrator.jsonl"
    orchestrator_log_lock = root / "results" / "logs" / "llm_orchestrator.lock"
    run_id = f"llm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{_slug(mission_name)}"
    start_monotonic = time.monotonic()

    set_execution_mode(ExecutionMode.RESEARCH)
    sample_cache: dict[str, pl.DataFrame] = {}
    feature_knowledge = _build_feature_knowledge(
        mission_bar_configs=mission_bar_configs,
        split=split,
        session_filter=session_filter,
        feature_group=feature_group,
        sample_cache=sample_cache,
    )
    feature_knowledge_hash = _sha256_text(
        json.dumps(feature_knowledge, sort_keys=True, separators=(",", ":"), default=str),
    )[:16]
    if feature_knowledge.get("errors"):
        print(f"Feature knowledge build warnings: {feature_knowledge['errors']}")
    else:
        common_count = len(feature_knowledge.get("common_columns", []))
        print(f"Feature knowledge ready: common_columns={common_count} bar_configs={mission_bar_configs}")

    print(
        "LLM orchestrator run_id="
        f"{run_id} mission={mission_name} "
        f"provider={provider} "
        f"models=feedback:{feedback_role['model']} thinker:{thinker_role['model']} coder:{coder_role['model']}",
    )
    print(f"split={split} session_filter={session_filter} feature_group={feature_group}")
    print(f"max_iterations={max_iterations} dry_run={bool(args.dry_run)}")

    stop_reason = "max_iterations_reached"
    while iterations_done < max_iterations:
        if max_runtime_hours is not None:
            elapsed = (time.monotonic() - start_monotonic) / 3600.0
            if elapsed >= max_runtime_hours:
                stop_reason = "max_runtime_reached"
                break

        queue_counts = _queue_counts(state_paths["queue"])
        queued = queue_counts["pending"] + queue_counts["in_progress"]
        if _should_wait_for_validation(queue_counts):
            print(
                "Waiting for validator completion before next generation: "
                f"active_tasks={queued}; sleeping {poll_seconds}s",
            )
            time.sleep(max(1, poll_seconds))
            continue

        existing = _existing_signal_names(signals_dir)
        feedback_items = _build_merged_feedback_items(
            handoffs_path=handoffs_path,
            research_log_path=research_log_path,
            orchestrator_log_path=orchestrator_log_path,
        )

        iteration_no = iterations_done + 1
        print(f"[iteration {iteration_no}/{max_iterations}] running feedback->thinker->coder pipeline")

        try:
            feedback_user_prompt = _build_feedback_user_prompt(feedback_items=feedback_items)
            feedback_generation = _call_stage_json(
                stage_name="feedback_analyst",
                schema_hint="keys: strengths, weaknesses, error_patterns, guardrails, next_focus; each list[str]",
                client=feedback_client,
                system_prompt=_build_feedback_system_prompt(),
                user_prompt=feedback_user_prompt,
                temperature=float(feedback_role["temperature"]),
                max_output_tokens=int(feedback_role["max_output_tokens"]),
                max_attempts=stage_max_attempts,
                json_repair_attempts=json_repair_attempts,
                stage_backoff_seconds=stage_backoff_seconds,
                quota_backoff_seconds=quota_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
            feedback_digest, feedback_generation = _normalize_with_semantic_retry(
                stage_name="feedback_analyst",
                stage_result=feedback_generation,
                normalize_fn=_normalize_feedback_digest,
                client=feedback_client,
                system_prompt=_build_feedback_system_prompt(),
                base_user_prompt=feedback_user_prompt,
                temperature=float(feedback_role["temperature"]),
                max_output_tokens=int(feedback_role["max_output_tokens"]),
                max_semantic_retries=semantic_retry_attempts,
                max_attempts=stage_max_attempts,
                json_repair_attempts=json_repair_attempts,
                stage_backoff_seconds=stage_backoff_seconds,
                quota_backoff_seconds=quota_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
                schema_hint="keys: strengths, weaknesses, error_patterns, guardrails, next_focus; each list[str]",
            )
            feedback_digest_hash = _sha256_text(
                json.dumps(feedback_digest, sort_keys=True, separators=(",", ":")),
            )[:16]

            thinker_user_prompt = _build_thinker_user_prompt(
                mission=mission,
                existing_strategies=existing,
                feedback_digest=feedback_digest,
                feature_knowledge=feature_knowledge,
            )
            thinker_generation = _call_stage_json(
                stage_name="quant_thinker",
                schema_hint=(
                    "keys: hypothesis_id, strategy_name_hint, thesis, bar_configs, params_template, "
                    "entry_logic, exit_logic, risk_controls, anti_lookahead_checks, validation_focus"
                ),
                client=thinker_client,
                system_prompt=_build_thinker_system_prompt(),
                user_prompt=thinker_user_prompt,
                temperature=float(thinker_role["temperature"]),
                max_output_tokens=int(thinker_role["max_output_tokens"]),
                max_attempts=stage_max_attempts,
                json_repair_attempts=json_repair_attempts,
                stage_backoff_seconds=stage_backoff_seconds,
                quota_backoff_seconds=quota_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
            thinker_brief, thinker_generation = _normalize_with_semantic_retry(
                stage_name="quant_thinker",
                stage_result=thinker_generation,
                normalize_fn=lambda payload: _normalize_thinker_brief(
                    payload,
                    mission_bar_configs=mission_bar_configs,
                ),
                client=thinker_client,
                system_prompt=_build_thinker_system_prompt(),
                base_user_prompt=thinker_user_prompt,
                temperature=float(thinker_role["temperature"]),
                max_output_tokens=int(thinker_role["max_output_tokens"]),
                max_semantic_retries=semantic_retry_attempts,
                max_attempts=stage_max_attempts,
                json_repair_attempts=json_repair_attempts,
                stage_backoff_seconds=stage_backoff_seconds,
                quota_backoff_seconds=quota_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
                schema_hint=(
                    "keys: hypothesis_id, strategy_name_hint, thesis, bar_configs, params_template, "
                    "entry_logic, exit_logic, risk_controls, anti_lookahead_checks, validation_focus"
                ),
            )
            thinker_hash = _sha256_text(
                json.dumps(thinker_brief, sort_keys=True, separators=(",", ":")),
            )[:16]
            thinker_handoff = _build_coder_handoff(
                thinker_brief=thinker_brief,
                mission=mission,
                thinker_payload_hash=thinker_hash,
            )

            coder_user_prompt = _build_coder_user_prompt(
                thinker_handoff=thinker_handoff,
                feature_knowledge=feature_knowledge,
            )
            coder_generation = _call_stage_json(
                stage_name="coder",
                schema_hint="keys: strategy_name, bar_configs, params, code",
                client=coder_client,
                system_prompt=_build_coder_system_prompt(),
                user_prompt=coder_user_prompt,
                temperature=float(coder_role["temperature"]),
                max_output_tokens=int(coder_role["max_output_tokens"]),
                max_attempts=stage_max_attempts,
                json_repair_attempts=json_repair_attempts,
                stage_backoff_seconds=stage_backoff_seconds,
                quota_backoff_seconds=quota_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
            normalized, coder_generation = _normalize_with_semantic_retry(
                stage_name="coder",
                stage_result=coder_generation,
                normalize_fn=lambda payload: _normalize_coder_payload(
                    payload,
                    mission_bar_configs=mission_bar_configs,
                    thinker_brief=thinker_brief,
                ),
                client=coder_client,
                system_prompt=_build_coder_system_prompt(),
                base_user_prompt=coder_user_prompt,
                temperature=float(coder_role["temperature"]),
                max_output_tokens=int(coder_role["max_output_tokens"]),
                max_semantic_retries=semantic_retry_attempts,
                max_attempts=stage_max_attempts,
                json_repair_attempts=json_repair_attempts,
                stage_backoff_seconds=stage_backoff_seconds,
                quota_backoff_seconds=quota_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
                schema_hint="keys: strategy_name, bar_configs, params, code",
            )

            code = normalized["code"]
            strategy_name = normalized["strategy_name"]
            params = normalized["params"]
            chosen_bars = normalized["bar_configs"]
            code_hash = _sha256_text(code)[:16]
            hypothesis_id = str(thinker_brief.get("hypothesis_id", "hyp_unknown"))

            module_path, is_new_path = _choose_module_path(
                signals_dir,
                strategy_name=strategy_name,
                module_code=code,
            )
            module_name = module_path.stem

            if args.dry_run:
                validation_errors = []
            else:
                for _ in range(8):
                    try:
                        _maybe_write(module_path, code, allow_overwrite=not is_new_path)
                        break
                    except FileExistsError:
                        module_path, is_new_path = _choose_module_path(
                            signals_dir,
                            strategy_name=strategy_name,
                            module_code=code,
                        )
                        module_name = module_path.stem
                else:
                    raise RuntimeError(
                        f"Could not reserve unique module path for strategy {strategy_name}",
                    )
                validation_errors = _validate_generated_strategy(
                    strategy_name=module_name,
                    signals_dir=signals_dir,
                    params=params,
                    bar_configs=chosen_bars,
                    split=split,
                    session_filter=session_filter,
                    feature_group=feature_group,
                    sample_cache=sample_cache,
                )

            # Inline repair loop: retry coder with injected errors
            _last_coder_generation = coder_generation
            if validation_errors and not args.dry_run and max_code_repair_attempts > 0:
                common_cols = list(feature_knowledge.get("common_columns", []))
                for repair_attempt in range(max_code_repair_attempts):
                    print(
                        f"  repair attempt {repair_attempt + 1}/{max_code_repair_attempts} "
                        f"for {module_name}: {validation_errors[0]}"
                    )
                    try:
                        repair_user_prompt = _build_coder_repair_user_prompt(
                            thinker_handoff=thinker_handoff,
                            previous_code=code,
                            validation_errors=validation_errors,
                            common_columns=common_cols,
                        )
                        repair_generation = _call_stage_json(
                            stage_name="coder_repair",
                            schema_hint="keys: strategy_name, bar_configs, params, code",
                            client=coder_client,
                            system_prompt=_build_coder_system_prompt(),
                            user_prompt=repair_user_prompt,
                            temperature=0.1,
                            max_output_tokens=int(coder_role["max_output_tokens"]),
                            max_attempts=2,
                            json_repair_attempts=json_repair_attempts,
                            stage_backoff_seconds=stage_backoff_seconds,
                            quota_backoff_seconds=quota_backoff_seconds,
                            max_backoff_seconds=max_backoff_seconds,
                        )
                        repaired_normalized, repair_generation = _normalize_with_semantic_retry(
                            stage_name="coder_repair",
                            stage_result=repair_generation,
                            normalize_fn=lambda payload: _normalize_coder_payload(
                                payload,
                                mission_bar_configs=mission_bar_configs,
                                thinker_brief=thinker_brief,
                            ),
                            client=coder_client,
                            system_prompt=_build_coder_system_prompt(),
                            base_user_prompt=repair_user_prompt,
                            temperature=0.1,
                            max_output_tokens=int(coder_role["max_output_tokens"]),
                            max_semantic_retries=semantic_retry_attempts,
                            max_attempts=2,
                            json_repair_attempts=json_repair_attempts,
                            stage_backoff_seconds=stage_backoff_seconds,
                            quota_backoff_seconds=quota_backoff_seconds,
                            max_backoff_seconds=max_backoff_seconds,
                            schema_hint="keys: strategy_name, bar_configs, params, code",
                        )
                        _last_coder_generation = repair_generation

                        # Update working variables with repaired output
                        code = repaired_normalized["code"]
                        params = repaired_normalized["params"]
                        chosen_bars = repaired_normalized["bar_configs"]
                        code_hash = _sha256_text(code)[:16]

                        # Overwrite module file with repaired code
                        _atomic_write_text(module_path, code)

                        # Re-validate repaired code
                        validation_errors = _validate_generated_strategy(
                            strategy_name=module_name,
                            signals_dir=signals_dir,
                            params=params,
                            bar_configs=chosen_bars,
                            split=split,
                            session_filter=session_filter,
                            feature_group=feature_group,
                            sample_cache=sample_cache,
                        )
                        if not validation_errors:
                            print(f"  repair succeeded on attempt {repair_attempt + 1}")
                            break
                    except (LLMClientError, ValueError, RuntimeError, OSError) as repair_exc:
                        print(f"  repair call failed: {type(repair_exc).__name__}: {repair_exc}")
                        break

            if validation_errors:
                if not args.dry_run and is_new_path and module_path.exists():
                    module_path.unlink()
                log_experiment(
                    {
                        "run_id": run_id,
                        "agent": "llm_orchestrator",
                        "event": "generation_rejected",
                        "iteration": iteration_no,
                        "strategy_name": module_name,
                        "bar_configs": chosen_bars,
                        "errors": validation_errors,
                        "hypothesis_id": hypothesis_id,
                        "feedback": {
                            "model": feedback_generation.model,
                            "response_id": feedback_generation.response_id,
                            "usage": feedback_generation.usage,
                            "attempts": feedback_generation.attempts,
                            "repaired": bool(feedback_generation.repaired),
                            "digest_hash": feedback_digest_hash,
                            "payload_hash": _sha256_text(feedback_generation.raw_text),
                        },
                        "thinker": {
                            "model": thinker_generation.model,
                            "response_id": thinker_generation.response_id,
                            "usage": thinker_generation.usage,
                            "attempts": thinker_generation.attempts,
                            "repaired": bool(thinker_generation.repaired),
                            "brief_hash": thinker_hash,
                            "payload_hash": _sha256_text(thinker_generation.raw_text),
                        },
                        "coder": {
                            "model": _last_coder_generation.model,
                            "response_id": _last_coder_generation.response_id,
                            "usage": _last_coder_generation.usage,
                            "attempts": _last_coder_generation.attempts,
                            "repaired": bool(_last_coder_generation.repaired),
                            "payload_hash": _sha256_text(_last_coder_generation.raw_text),
                        },
                    },
                    experiments_path=orchestrator_log_path,
                    lock_path=orchestrator_log_lock,
                )
                print(f"rejected {module_name}: {validation_errors[0]}")
            else:
                enqueued_task_ids: list[str] = []
                for bar_config in chosen_bars:
                    task = _build_task(
                        strategy_name=module_name,
                        split=split,
                        bar_config=bar_config,
                        params=params,
                        mission=mission,
                        code_hash=code_hash,
                        iteration=iteration_no,
                        hypothesis_id=hypothesis_id,
                    )
                    inserted = False
                    if not args.dry_run:
                        inserted, _ = enqueue_task(
                            queue_path=state_paths["queue"],
                            lock_path=state_paths["queue_lock"],
                            task=task,
                        )
                    if inserted:
                        enqueued_task_ids.append(task["task_id"])

                if enqueued_task_ids and not args.dry_run:
                    handoff_material = {
                        "run_id": run_id,
                        "strategy_name": module_name,
                        "hypothesis_id": hypothesis_id,
                        "task_ids": sorted(enqueued_task_ids),
                    }
                    handoff_id = _sha256_text(
                        json.dumps(handoff_material, sort_keys=True, separators=(",", ":")),
                    )
                    append_handoff(
                        handoffs_path=state_paths["handoffs"],
                        lock_path=state_paths["handoffs_lock"],
                        handoff={
                            "handoff_id": handoff_id,
                            "handoff_type": "validation_request",
                            "from_agent": "llm_orchestrator",
                            "to_agent": "validator",
                            "run_id": run_id,
                            "state": "pending",
                            "payload": {
                                "strategy_name": module_name,
                                "task_ids": enqueued_task_ids,
                                "bar_configs": chosen_bars,
                                "hypothesis_id": hypothesis_id,
                            },
                        },
                    )

                total_tasks += len(enqueued_task_ids)
                generated_modules.append(module_name)
                log_experiment(
                    {
                        "run_id": run_id,
                        "agent": "llm_orchestrator",
                        "event": "generation_enqueued",
                        "iteration": iteration_no,
                        "strategy_name": module_name,
                        "hypothesis_id": hypothesis_id,
                        "thinker_brief": thinker_brief,
                        "bar_configs": chosen_bars,
                        "params": params,
                        "task_ids": enqueued_task_ids,
                        "module_path": str(module_path),
                        "feedback": {
                            "model": feedback_generation.model,
                            "response_id": feedback_generation.response_id,
                            "usage": feedback_generation.usage,
                            "attempts": feedback_generation.attempts,
                            "repaired": bool(feedback_generation.repaired),
                            "digest_hash": feedback_digest_hash,
                            "payload_hash": _sha256_text(feedback_generation.raw_text),
                        },
                        "thinker": {
                            "model": thinker_generation.model,
                            "response_id": thinker_generation.response_id,
                            "usage": thinker_generation.usage,
                            "attempts": thinker_generation.attempts,
                            "repaired": bool(thinker_generation.repaired),
                            "brief_hash": thinker_hash,
                            "payload_hash": _sha256_text(thinker_generation.raw_text),
                        },
                        "coder": {
                            "model": _last_coder_generation.model,
                            "response_id": _last_coder_generation.response_id,
                            "usage": _last_coder_generation.usage,
                            "attempts": _last_coder_generation.attempts,
                            "repaired": bool(_last_coder_generation.repaired),
                            "payload_hash": _sha256_text(_last_coder_generation.raw_text),
                        },
                        "feature_knowledge_hash": feature_knowledge_hash,
                        "code_hash": code_hash,
                        "dry_run": bool(args.dry_run),
                    },
                    experiments_path=orchestrator_log_path,
                    lock_path=orchestrator_log_lock,
                )
                print(
                    f"accepted {module_name} [{hypothesis_id}]: "
                    f"enqueued={len(enqueued_task_ids)} bars={chosen_bars}",
                )

        except (LLMClientError, ValueError, RuntimeError) as exc:
            trace = traceback.format_exc()
            log_experiment(
                {
                    "run_id": run_id,
                    "agent": "llm_orchestrator",
                    "event": "generation_error",
                    "iteration": iteration_no,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": trace,
                },
                experiments_path=orchestrator_log_path,
                lock_path=orchestrator_log_lock,
            )
            print(f"generation error: {type(exc).__name__}: {exc}")
            print(trace, file=sys.stderr, flush=True)

        iterations_done += 1
        state = {
            "schema_version": "1.0",
            "mission_name": mission_name,
            "run_id": run_id,
            "iterations_completed": iterations_done,
            "total_tasks_enqueued": total_tasks,
            "generated_modules": generated_modules[-200:],
            "last_updated": _utc_now(),
        }
        atomic_json_write(state_paths["orchestrator"], state)
        if iterations_done < max_iterations:
            time.sleep(max(1, poll_seconds))

    final_counts = _queue_counts(state_paths["queue"])
    summary = {
        "run_id": run_id,
        "mission_name": mission_name,
        "iterations_completed": iterations_done,
        "total_tasks_enqueued": total_tasks,
        "models": {
            "feedback_analyst": feedback_role["model"],
            "quant_thinker": thinker_role["model"],
            "coder": coder_role["model"],
        },
        "queue_counts": final_counts,
        "stop_reason": stop_reason,
        "dry_run": bool(args.dry_run),
        "last_updated": _utc_now(),
    }
    summary_path = root / "results" / "runs" / run_id / "llm_orchestrator_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(summary_path, summary)
    print(f"summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
