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
from research.lib.coordination import append_handoff, enqueue_task, read_json_file
from research.lib.experiments import log_experiment
from research.lib.feature_groups import filter_strategy_inputs
from research.lib.feature_surface import (
    build_feature_surface,
    collect_condition_passthrough,
    describe_referenced_columns,
    diagnose_condition_passthrough,
    format_feature_surface_context,
    format_param_feasibility_context,
    format_referenced_surface_warnings,
)
from research.lib.learning_scorecard import (
    format_learning_context,
    normalize_theme_tag,
    read_learning_scorecard,
    resolve_focus_anchors,
)
from research.lib.llm_client import (
    ClaudeCodeCLIClient,
    LLMClientError,
    LLMRawClient,
    extract_json_object,
)
from research.lib.mission_splits import resolve_research_splits
from research.lib.notebook_guidance import (
    normalize_notebook_query_budget,
    notebook_query_budget_context,
    notebook_research_guidance_context,
)
from research.lib.notebook_runtime import ensure_lane_notebook
from research.lib.notebook_audit import notebook_audit_context, summarize_notebook_queries
from research.lib.runtime_state import (
    ensure_orchestrator_state,
    ensure_shared_state,
    read_orchestrator_state,
    reset_orchestrator_state,
    thinker_memory_lock_path,
    thinker_memory_path,
    write_orchestrator_state,
)
from research.lib.thinker_memory import (
    append_thinker_attempt,
    format_thinker_memory_context,
    read_thinker_memory,
)
from research.lib.thinker_feasibility import (
    ThinkerFeasibilityError,
    assess_entry_condition_feasibility,
    format_feasibility_error,
    normalize_entry_conditions,
    repair_thinker_brief_for_feasibility,
)
from research.signals import check_signal_causality, load_signal_module
from src.framework.api import (
    ExecutionMode,
    get_split_files,
    load_cached_matrix,
    set_execution_mode,
)
from src.framework.data.constants import TICK_SIZE


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_MAX_THINKER_FEASIBILITY_REPAIR_ATTEMPTS = 6


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
            previous_payload: dict[str, Any] = current.payload
            if isinstance(exc, ThinkerFeasibilityError) and isinstance(exc.brief, dict) and exc.brief:
                previous_payload = exc.brief
            repair_prompt = (
                f"{base_user_prompt}\n\n"
                f"Validation error: {exc}\n"
                f"Previous JSON:\n{json.dumps(previous_payload, indent=2, default=str)}\n\n"
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


def _diagnose_zero_signal(df: pl.DataFrame, code: str) -> str:
    """Compute empirical firing rates for columns referenced in strategy code."""
    return describe_referenced_columns(df=df, code=code, max_columns=15)


def _state_mode(*, fresh_state: bool) -> str:
    return "fresh" if fresh_state else "resume"


def _queue_counts(queue_path: Path, queue_lock: Path) -> dict[str, int]:
    payload = read_json_file(
        json_path=queue_path,
        lock_path=queue_lock,
        default_payload={"schema_version": "1.0", "tasks": []},
    )
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
            gauntlet = row.get("gauntlet") if isinstance(row.get("gauntlet"), dict) else {}
            advanced = row.get("advanced_validation") if isinstance(row.get("advanced_validation"), dict) else {}
            items.append(
                {
                    "event": "task_result",
                    "strategy_name": str(row.get("strategy_name", "")),
                    "bar_config": str(row.get("bar_config", "")),
                    "verdict": str(row.get("verdict", "")),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "trade_count": metrics.get("trade_count"),
                    "failed_checks": _failed_gauntlet_checks(gauntlet),
                    "dsr": _extract_dsr(advanced),
                    "alpha_decay_verdict": _extract_alpha_decay_verdict(advanced),
                    "factor_verdict": _extract_factor_verdict(advanced),
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
    handoffs_lock: Path,
    *,
    max_items: int = 24,
) -> list[dict[str, Any]]:
    try:
        payload = read_json_file(
            json_path=handoffs_path,
            lock_path=handoffs_lock,
            default_payload={"schema_version": "1.0", "pending": [], "completed": []},
        )
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
                "pass_fraction": result.get("pass_fraction"),
                "avg_sharpe_ratio": result.get("avg_sharpe_ratio"),
                "avg_trade_count": result.get("avg_trade_count"),
                "passing_bar_configs": result.get("passing_bar_configs"),
                "failing_bar_configs": result.get("failing_bar_configs"),
                "best_bar_config": result.get("best_bar_config"),
                "best_sharpe_ratio": result.get("best_sharpe_ratio"),
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
    handoffs_lock_path: Path,
    research_log_path: Path,
    orchestrator_log_path: Path,
    max_items: int = 40,
) -> list[dict[str, Any]]:
    """Merge feedback from all three sources. Orch errors first (most actionable)."""
    per_source = max(1, max_items // 3)

    handoff_items = _collect_feedback_items_from_handoffs(
        handoffs_path,
        handoffs_lock_path,
        max_items=per_source,
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


def _failed_gauntlet_checks(gauntlet: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for name, payload in gauntlet.items():
        if name in {"overall_verdict", "pass_count", "total_tests"}:
            continue
        if isinstance(payload, dict) and str(payload.get("verdict", "")).upper() == "FAIL":
            failed.append(str(name))
    return failed


def _extract_dsr(advanced_validation: dict[str, Any]) -> float | None:
    dsr_payload = advanced_validation.get("deflated_sharpe")
    if not isinstance(dsr_payload, dict):
        return None
    dsr = dsr_payload.get("dsr")
    try:
        return float(dsr) if dsr is not None else None
    except (TypeError, ValueError):
        return None


def _extract_alpha_decay_verdict(advanced_validation: dict[str, Any]) -> str | None:
    payload = advanced_validation.get("alpha_decay")
    if not isinstance(payload, dict):
        return None
    verdict = str(payload.get("verdict", "")).strip()
    return verdict or None


def _extract_factor_verdict(advanced_validation: dict[str, Any]) -> str | None:
    payload = advanced_validation.get("factor_attribution")
    if not isinstance(payload, dict):
        return None
    verdict = str(payload.get("verdict", "")).strip()
    return verdict or None


def _default_notebook_summary(*, configured: bool) -> dict[str, Any]:
    return {
        "configured": bool(configured),
        "used": False,
        "query_count": 0,
        "success_count": 0,
        "error_count": 0,
        "modes_used": [],
        "mode_counts": {"plain": 0, "research": 0, "deep_research": 0},
        "non_fallback_mode_counts": {"plain": 0, "research": 0, "deep_research": 0},
        "fallback_count": 0,
        "discovered_sources": 0,
        "imported_sources": 0,
        "question_previews": [],
    }


def _resolve_notebook_query_budget(mission: dict[str, Any]) -> dict[str, int]:
    return normalize_notebook_query_budget(mission.get("lane_notebook_query_budget"))


def _persist_notebook_progress(
    *,
    notebook_meta: dict[str, Any] | None,
    notebook_summary: dict[str, Any],
    orchestrator_state: dict[str, Any],
    state_paths: dict[str, Path],
) -> None:
    if notebook_meta is None:
        return

    persisted_state = read_orchestrator_state(state_paths["orchestrator"])
    persisted_meta = (
        dict(persisted_state.get("notebooklm"))
        if isinstance(persisted_state.get("notebooklm"), dict)
        else {}
    )
    if (
        persisted_meta
        and str(persisted_meta.get("notebook_id", "")).strip()
        == str(notebook_meta.get("notebook_id", "")).strip()
    ):
        notebook_meta["seeded"] = bool(persisted_meta.get("seeded", notebook_meta.get("seeded", False)))
        notebook_meta["fresh"] = bool(persisted_meta.get("fresh", notebook_meta.get("fresh", True)))
        notebook_meta["imported_sources"] = max(
            int(notebook_meta.get("imported_sources", 0) or 0),
            int(persisted_meta.get("imported_sources", 0) or 0),
        )
        notebook_meta["seed_query_count"] = max(
            int(notebook_meta.get("seed_query_count", 0) or 0),
            int(persisted_meta.get("seed_query_count", 0) or 0),
        )
        persisted_modes = list(persisted_meta.get("seed_modes_used", []) or [])
        if persisted_modes:
            notebook_meta["seed_modes_used"] = list(dict.fromkeys(persisted_modes + list(notebook_meta.get("seed_modes_used", []) or [])))

    changed = False
    imported_sources = max(
        int(notebook_meta.get("imported_sources", 0) or 0),
        int(notebook_summary.get("imported_sources", 0) or 0),
    )
    if imported_sources != int(notebook_meta.get("imported_sources", 0) or 0):
        notebook_meta["imported_sources"] = imported_sources
        changed = True

    seed_query_count = int(notebook_summary.get("query_count", 0) or 0)
    if seed_query_count > int(notebook_meta.get("seed_query_count", 0) or 0):
        notebook_meta["seed_query_count"] = seed_query_count
        changed = True
    seed_modes_used = list(notebook_summary.get("modes_used", []) or [])
    if seed_modes_used:
        merged_modes = list(dict.fromkeys(list(notebook_meta.get("seed_modes_used", []) or []) + seed_modes_used))
        if merged_modes != list(notebook_meta.get("seed_modes_used", []) or []):
            notebook_meta["seed_modes_used"] = merged_modes
            changed = True
    if imported_sources > 0:
        if not bool(notebook_meta.get("seeded", False)):
            notebook_meta["seeded"] = True
            changed = True
        if bool(notebook_meta.get("fresh", True)):
            notebook_meta["fresh"] = False
            changed = True

    if changed:
        orchestrator_state["notebooklm"] = dict(notebook_meta)
        write_orchestrator_state(state_paths["orchestrator"], orchestrator_state)


_BAR_MIN_SL_TICKS = {"tick_610": 20, "volume_2000": 40, "time_1m": 4}
_TARGET_SIGNAL_RATE_LOW_PCT = 0.05
_TARGET_SIGNAL_RATE_HIGH_PCT = 0.3
_TARGET_SIGNAL_RATE_MID_PCT = (_TARGET_SIGNAL_RATE_LOW_PCT + _TARGET_SIGNAL_RATE_HIGH_PCT) / 2.0


def _bar_range_hint(sample_bar_context: dict[str, Any] | None, bar_config: str) -> str:
    context = sample_bar_context if isinstance(sample_bar_context, dict) else {}
    bar_stats = context.get(str(bar_config).strip()) if isinstance(context.get(str(bar_config).strip()), dict) else {}
    range_ticks = bar_stats.get("range_ticks") if isinstance(bar_stats.get("range_ticks"), dict) else {}
    median = range_ticks.get("median")
    if median is None:
        return "unknown median bar range"
    try:
        return f"median bar range ~{int(round(float(median)))}t"
    except Exception:
        return "unknown median bar range"


def _format_bar_risk_floor_context(runtime_context: dict[str, Any] | None) -> str:
    sample_bar_context = (
        dict((runtime_context or {}).get("sample_bar_context", {}))
        if isinstance((runtime_context or {}).get("sample_bar_context", {}), dict)
        else {}
    )
    lines = [
        "BAR_CONFIG_RISK_FLOORS:",
        "If you include a bar config, your params_template must satisfy its minimum stop floor. "
        "If the stop would be too tight, widen it or drop that bar config from the hypothesis.",
    ]
    for bar_config, min_sl in _BAR_MIN_SL_TICKS.items():
        lines.append(f"- {bar_config}: sl_ticks >= {min_sl} ({_bar_range_hint(sample_bar_context, bar_config)})")
    return "\n".join(lines)


def _normalize_requested_bar_configs(
    raw_cfgs: Any,
    *,
    mission_bar_configs: list[str],
) -> list[str]:
    requested = [str(v).strip() for v in raw_cfgs] if isinstance(raw_cfgs, list) else []
    allowed = {str(v).strip() for v in mission_bar_configs}
    chosen: list[str] = []
    seen: set[str] = set()
    for cfg in requested:
        if not cfg or cfg not in allowed or cfg in seen:
            continue
        chosen.append(cfg)
        seen.add(cfg)
    return chosen


def _risk_floor_error_message(
    *,
    bar_config: str,
    sl_ticks: float,
    sample_bar_context: dict[str, Any] | None,
) -> str:
    return (
        f"sl_ticks={int(sl_ticks)} too tight for {bar_config} "
        f"({_bar_range_hint(sample_bar_context, str(bar_config))}) — "
        f"minimum sl_ticks={_BAR_MIN_SL_TICKS.get(str(bar_config).strip(), 0)} required to survive entry bar noise; "
        f"either widen sl_ticks or remove {bar_config} from bar_configs"
    )


def _prune_bar_configs_for_risk_floor(
    bar_configs: list[str],
    *,
    sl_ticks: float,
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for bar_config in bar_configs:
        min_sl = _BAR_MIN_SL_TICKS.get(str(bar_config).strip(), 0)
        if min_sl and float(sl_ticks) < float(min_sl):
            dropped.append(str(bar_config))
        else:
            kept.append(str(bar_config))
    return kept, dropped


def _bar_config_selection_key(
    rows: list[dict[str, Any]],
    *,
    requested_index: int,
) -> tuple[Any, ...]:
    non_empty_rows = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("status", "")) != "empty_sample"
    ]
    if not non_empty_rows:
        return (1, 0, 0, 0, 0, float("inf"), float("inf"), requested_index)

    failing_rows = [
        row
        for row in non_empty_rows
        if str(row.get("status", "")) not in {"ok", "empty_sample"}
    ]
    dead_count = sum(str(row.get("status", "")) == "dead_feature_primary" for row in failing_rows)
    zero_count = sum(str(row.get("status", "")) == "zero_signal" for row in failing_rows)
    over_count = sum(str(row.get("status", "")) == "over_signal" for row in failing_rows)
    other_count = len(failing_rows) - dead_count - zero_count - over_count
    ok_rates = [
        float(row.get("signal_rate_pct", 0.0) or 0.0)
        for row in non_empty_rows
        if str(row.get("status", "")) == "ok"
    ]
    avg_ok_penalty = (
        sum(abs(rate - _TARGET_SIGNAL_RATE_MID_PCT) for rate in ok_rates) / len(ok_rates)
        if ok_rates else float("inf")
    )
    best_ok_penalty = min(
        (abs(rate - _TARGET_SIGNAL_RATE_MID_PCT) for rate in ok_rates),
        default=float("inf"),
    )
    return (
        0,
        dead_count,
        other_count,
        len(failing_rows),
        zero_count,
        over_count,
        avg_ok_penalty,
        best_ok_penalty,
        requested_index,
    )


def _prune_brief_to_single_bar_config(
    brief: dict[str, Any],
    feasibility_report: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    raw_bar_configs = brief.get("bar_configs")
    bar_configs = [str(cfg).strip() for cfg in raw_bar_configs] if isinstance(raw_bar_configs, list) else []
    if len(bar_configs) <= 1:
        return None, None

    bar_results = feasibility_report.get("bar_results") if isinstance(feasibility_report, dict) else None
    grouped: dict[str, list[dict[str, Any]]] = {cfg: [] for cfg in bar_configs}
    if isinstance(bar_results, list):
        for row in bar_results:
            if not isinstance(row, dict):
                continue
            bar_config = str(row.get("bar_config", "")).strip()
            if bar_config in grouped:
                grouped[bar_config].append(row)

    requested_index = {cfg: idx for idx, cfg in enumerate(bar_configs)}
    chosen = min(
        bar_configs,
        key=lambda cfg: _bar_config_selection_key(
            grouped.get(cfg, []),
            requested_index=requested_index.get(cfg, len(bar_configs)),
        ),
    )
    pruned = dict(brief)
    pruned["bar_configs"] = [chosen]
    dropped = [cfg for cfg in bar_configs if cfg != chosen]
    return pruned, f"{bar_configs} -> {[chosen]} (dropped {dropped})"


def _format_results_table(feedback_items: list[dict[str, Any]]) -> str:
    """Format raw feedback items into a compact results table for the thinker."""
    if not feedback_items:
        return "No experiment results yet — this is iteration 1."

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in feedback_items:
        event = item.get("event", "")
        if event in ("task_result", "validation_result"):
            results.append(item)
        elif event in ("task_error", "generation_rejected", "generation_error"):
            errors.append(item)

    def _sharpe(x: dict[str, Any]) -> float:
        s = x.get("sharpe_ratio") or x.get("avg_sharpe_ratio")
        return float(s) if s is not None else -999.0

    results.sort(key=_sharpe, reverse=True)

    lines: list[str] = []
    if results:
        lines.append("EXPERIMENT RESULTS (sorted best-first):")
        for r in results:
            sharpe = r.get("sharpe_ratio") or r.get("avg_sharpe_ratio")
            trades = r.get("trade_count") or r.get("avg_trade_count")
            verdict = r.get("verdict") or r.get("overall_verdict", "?")
            name = r.get("strategy_name", "?")
            bar = r.get("bar_config", "")

            sharpe_str = f"{float(sharpe):.3f}" if sharpe is not None else "N/A"
            trades_str = str(int(float(trades))) if trades is not None else "N/A"
            bar_str = f" [{bar}]" if bar else ""

            diagnostics: list[str] = []
            failed_checks = [str(x) for x in (r.get("failed_checks") or []) if str(x)]
            if failed_checks:
                diagnostics.append("fails=" + ",".join(failed_checks[:3]))
            pass_count = r.get("pass_count")
            task_count = r.get("task_count")
            if isinstance(pass_count, (int, float)) and isinstance(task_count, (int, float)) and float(task_count) > 1:
                diagnostics.append(f"bars={int(pass_count)}/{int(task_count)}")
            best_bar_config = str(r.get("best_bar_config", "")).strip()
            if best_bar_config:
                diagnostics.append(f"best={best_bar_config}")
            dsr = r.get("dsr")
            if dsr is not None:
                diagnostics.append(f"dsr={float(dsr):.2f}")
            alpha_decay = r.get("alpha_decay_verdict")
            if alpha_decay:
                diagnostics.append(f"decay={alpha_decay}")
            factor_verdict = r.get("factor_verdict")
            if factor_verdict:
                diagnostics.append(f"factor={factor_verdict}")
            diag_str = f" [{' | '.join(diagnostics)}]" if diagnostics else ""

            note = ""
            if sharpe is not None and float(sharpe) > 0 and verdict not in ("PASS",):
                note = f"  ← NEAR-MISS: read research/signals/{name}.py"

            lines.append(
                f"  {name}{bar_str}: {verdict} sharpe={sharpe_str} trades={trades_str}{diag_str}{note}",
            )

    if errors:
        lines.append("\nGENERATION/VALIDATION ERRORS:")
        for e in errors[:8]:
            name = str(e.get("strategy_name") or e.get("hypothesis_id", "?"))
            err = str(e.get("error", ""))[:120]
            lines.append(f"  {name}: {err}")

    return "\n".join(lines)


def _build_learning_context(
    *,
    scorecard_path: Path,
    scorecard_lock: Path,
) -> str:
    scorecard = read_learning_scorecard(scorecard_path, scorecard_lock)
    return format_learning_context(scorecard)


def _format_attempt_params(params: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if key in params:
            out[key] = params[key]
    return out


def _sanitize_attempt_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.startswith("_"):
                continue
            out[key_text] = _sanitize_attempt_value(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_sanitize_attempt_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _select_highlighted_conditions(
    *,
    condition_rows: list[dict[str, Any]],
    failure_type: str,
    max_items: int = 2,
) -> tuple[str, list[dict[str, Any]]]:
    rows = [
        sanitized
        for row in condition_rows
        if isinstance(row, dict)
        for sanitized in [_sanitize_attempt_value(row)]
        if isinstance(sanitized, dict)
    ]
    if not rows:
        return "", []

    if failure_type == "over_signal":
        rows.sort(key=lambda row: (-float(row.get("pass_rate_pct", 0.0)), str(row.get("param_key", ""))))
        return "Loosest", rows[:max_items]

    rows.sort(
        key=lambda row: (
            float(row.get("pass_rate_pct", 0.0)),
            0 if str(row.get("severity", "")) == "blocks_all" else 1,
            str(row.get("param_key", "")),
        )
    )
    blocking = [
        row
        for row in rows
        if str(row.get("severity", "")) in {"blocks_all", "restrictive"}
    ]
    chosen = blocking[:max_items] if blocking else rows[:max_items]
    return "Blocking", chosen


def _build_validation_attempt_record(
    *,
    iteration: int,
    hypothesis_id: str,
    theme_tag: str,
    strategy_name: str,
    bar_configs: list[str],
    params: dict[str, Any],
    validation_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(validation_report, dict):
        return None
    bar_results = validation_report.get("bar_results")
    if not isinstance(bar_results, list) or not bar_results:
        return None

    relevant = None
    for status in (
        "dead_feature_primary",
        "zero_signal",
        "over_signal",
        "contract_failed",
        "causality_failed",
        "runtime_error",
    ):
        relevant = next(
            (row for row in bar_results if isinstance(row, dict) and str(row.get("status", "")) == status),
            None,
        )
        if relevant is not None:
            break
    if relevant is None:
        return None

    failure_type = str(relevant.get("status", "runtime_error"))
    sample_label = str(relevant.get("sample_label", "")).strip()
    bar_config = str(relevant.get("bar_config", "")).strip()
    nonzero = int(relevant.get("nonzero", 0) or 0)
    total = int(relevant.get("total", 0) or 0)
    signal_rate_pct = float(relevant.get("signal_rate_pct", 0.0) or 0.0)
    condition_rows = relevant.get("condition_rows") if isinstance(relevant.get("condition_rows"), list) else []
    conditions_label, highlighted_conditions = _select_highlighted_conditions(
        condition_rows=condition_rows,
        failure_type=failure_type,
    )
    offending_params = _format_attempt_params(
        params,
        [str(row.get("param_key", "")).strip() for row in highlighted_conditions if str(row.get("param_key", "")).strip()],
    )

    if failure_type == "dead_feature_primary":
        summary = str(relevant.get("error", "")).strip()[:180] or f"dead primary feature on {bar_config}."
        status_label = "REJECTED (dead_feature_primary)"
    elif failure_type == "zero_signal":
        summary = f"0/{total} bars on {bar_config} ({sample_label})."
        status_label = "REJECTED (zero_signal)"
    elif failure_type == "over_signal":
        summary = (
            f"{nonzero}/{total} bars on {bar_config} ({signal_rate_pct:.2f}% signal rate; target 0.05–0.3%) "
            f"({sample_label})."
        )
        status_label = "REJECTED (over_signal)"
    else:
        summary = str(relevant.get("error", "")).strip()[:180] or f"{failure_type} on {bar_config} ({sample_label})."
        status_label = f"REJECTED ({failure_type})"

    return {
        "iteration": iteration,
        "hypothesis_id": hypothesis_id,
        "theme_tag": theme_tag,
        "strategy_name": strategy_name,
        "bar_configs": list(bar_configs),
        "status": "rejected_pre_enqueue",
        "status_label": status_label,
        "failure_type": failure_type,
        "summary": summary,
        "conditions_label": conditions_label,
        "highlighted_conditions": highlighted_conditions,
        "offending_params": offending_params,
        "signal_rate_pct": signal_rate_pct,
        "sample_label": sample_label,
        "bar_config": bar_config,
    }


def _build_feasibility_attempt_record(
    *,
    iteration: int,
    hypothesis_id: str,
    theme_tag: str,
    bar_configs: list[str],
    params: dict[str, Any],
    feasibility_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return _build_validation_attempt_record(
        iteration=iteration,
        hypothesis_id=hypothesis_id,
        theme_tag=theme_tag,
        strategy_name="thinker_hypothesis",
        bar_configs=bar_configs,
        params=params,
        validation_report=feasibility_report,
    )


def _build_exception_attempt_record(
    *,
    iteration: int,
    hypothesis_id: str,
    theme_tag: str,
    bar_configs: list[str],
    params: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    if isinstance(exc, ThinkerFeasibilityError):
        brief = exc.brief if isinstance(exc.brief, dict) else {}
        if brief:
            hypothesis_id = str(brief.get("hypothesis_id") or hypothesis_id)
            theme_tag = str(brief.get("theme_tag") or theme_tag)
            if not bar_configs:
                bar_configs = [str(v) for v in brief.get("bar_configs", []) if str(v).strip()]
            if not params:
                raw_params = brief.get("params_template")
                if isinstance(raw_params, dict):
                    params = dict(raw_params)
        record = _build_feasibility_attempt_record(
            iteration=iteration,
            hypothesis_id=hypothesis_id,
            theme_tag=theme_tag,
            bar_configs=bar_configs,
            params=params,
            feasibility_report=exc.report,
        )
        if record is not None:
            return record

    message = str(exc).strip()
    risk_match = re.search(
        r"sl_ticks=(?P<actual>\d+)\s+too tight for\s+(?P<bar>\w+).*minimum sl_ticks=(?P<minimum>\d+)",
        message,
    )
    if risk_match:
        actual = int(risk_match.group("actual"))
        minimum = int(risk_match.group("minimum"))
        bar_config = str(risk_match.group("bar"))
        offending_params = {}
        for key, value in params.items():
            key_text = str(key)
            if key_text.startswith("sl_ticks") and str(actual) == str(value):
                offending_params[key_text] = value
        if not offending_params:
            offending_params["sl_ticks"] = actual
        return {
            "iteration": iteration,
            "hypothesis_id": hypothesis_id,
            "theme_tag": theme_tag,
            "bar_configs": list(bar_configs),
            "status": "generation_error",
            "status_label": "ERROR (invalid_risk_floor)",
            "failure_type": "invalid_risk_floor",
            "summary": f"{bar_config} stop too tight: sl_ticks={actual}, minimum={minimum}.",
            "conditions_label": "",
            "highlighted_conditions": [],
            "offending_params": offending_params,
            "bar_config": bar_config,
        }

    return {
        "iteration": iteration,
        "hypothesis_id": hypothesis_id,
        "theme_tag": theme_tag,
        "bar_configs": list(bar_configs),
        "status": "generation_error",
        "status_label": "ERROR",
        "failure_type": "generation_error",
        "summary": message[:180] or type(exc).__name__,
        "conditions_label": "",
        "highlighted_conditions": [],
        "offending_params": {},
    }


def _build_success_attempt_record(
    *,
    iteration: int,
    hypothesis_id: str,
    theme_tag: str,
    strategy_name: str,
    bar_configs: list[str],
    params: dict[str, Any],
) -> dict[str, Any]:
    param_keys = [
        key
        for key in (
            "sl_ticks",
            "pt_ticks",
            "sl_ticks_tick_610",
            "sl_ticks_volume_2000",
            "sl_ticks_time_1m",
            "pt_ticks_tick_610",
            "pt_ticks_volume_2000",
            "pt_ticks_time_1m",
        )
        if key in params
    ]
    return {
        "iteration": iteration,
        "hypothesis_id": hypothesis_id,
        "theme_tag": theme_tag,
        "strategy_name": strategy_name,
        "bar_configs": list(bar_configs),
        "status": "accepted_for_validation",
        "status_label": "ACCEPTED",
        "failure_type": "accepted",
        "summary": f"accepted for validation on {', '.join(bar_configs)}.",
        "conditions_label": "",
        "highlighted_conditions": [],
        "offending_params": _format_attempt_params(params, param_keys),
    }


def _should_attempt_coder_repair(
    *,
    validation_report: dict[str, Any] | None,
) -> bool:
    if not isinstance(validation_report, dict):
        return True
    bar_results = validation_report.get("bar_results")
    if not isinstance(bar_results, list) or not bar_results:
        return True

    neutral_statuses = {"ok", "empty_sample"}
    hypothesis_level_statuses = {"dead_feature_primary", "zero_signal", "over_signal"}
    actionable_coder_statuses = {
        "module_validation_failed",
        "runtime_error",
        "contract_failed",
        "causality_failed",
    }

    statuses = {
        str(row.get("status", "")).strip()
        for row in bar_results
        if isinstance(row, dict)
    }
    statuses -= neutral_statuses
    if not statuses:
        return False
    if statuses <= hypothesis_level_statuses:
        return False
    if statuses & actionable_coder_statuses:
        return True
    return True


def _normalize_and_assess_thinker_brief(
    payload: dict[str, Any],
    *,
    mission_bar_configs: list[str],
    sample_bar_context: dict[str, Any] | None,
    validation_sample_cache: dict[str, list[tuple[str, pl.DataFrame]]],
) -> dict[str, Any]:
    thinker_brief = _normalize_thinker_brief(
        payload,
        mission_bar_configs=mission_bar_configs,
        sample_bar_context=sample_bar_context,
    )
    current_brief = thinker_brief
    if len(list(current_brief.get("bar_configs", []))) > 1:
        initial_report = assess_entry_condition_feasibility(
            entry_conditions=list(current_brief.get("entry_conditions", [])),
            params_template=dict(current_brief.get("params_template", {})),
            selected_bar_configs=list(current_brief.get("bar_configs", [])),
            validation_sample_cache=validation_sample_cache,
        )
        pruned_brief, prune_action = _prune_brief_to_single_bar_config(current_brief, initial_report)
        if pruned_brief is not None:
            print(f"  thinker discovery bar_config prune: {prune_action}")
            current_brief = pruned_brief
    max_repairs = min(
        _MAX_THINKER_FEASIBILITY_REPAIR_ATTEMPTS,
        max(2, len(list(current_brief.get("entry_conditions", []))) * 2),
    )
    feasibility_report: dict[str, Any] = {"bar_results": []}

    for repair_attempt in range(max_repairs + 1):
        feasibility_report = assess_entry_condition_feasibility(
            entry_conditions=list(current_brief.get("entry_conditions", [])),
            params_template=dict(current_brief.get("params_template", {})),
            selected_bar_configs=list(current_brief.get("bar_configs", [])),
            validation_sample_cache=validation_sample_cache,
        )
        failing = [
            row
            for row in feasibility_report.get("bar_results", [])
            if isinstance(row, dict) and str(row.get("status", "")) not in {"ok", "empty_sample"}
        ]
        if not failing:
            return current_brief
        if repair_attempt >= max_repairs:
            break

        repaired_brief, repair_actions = repair_thinker_brief_for_feasibility(
            current_brief,
            feasibility_report,
        )
        if repaired_brief is None or repaired_brief == current_brief:
            break
        if repair_actions:
            print(
                f"  thinker feasibility auto-repair {repair_attempt + 1}/{max_repairs}: "
                f"{'; '.join(repair_actions)}"
            )
        current_brief = repaired_brief

    raise ThinkerFeasibilityError(
        format_feasibility_error(feasibility_report),
        report=feasibility_report,
        brief=current_brief,
    )


def _normalize_thinker_brief(
    payload: dict[str, Any],
    *,
    mission_bar_configs: list[str],
    sample_bar_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("thinker payload must be a JSON object")

    hypothesis_id = _slug(str(payload.get("hypothesis_id", "")).strip())
    if not hypothesis_id:
        hypothesis_id = f"hyp_{int(time.time())}"

    strategy_name_hint = _slug(str(payload.get("strategy_name_hint", "")).strip())
    if not strategy_name_hint:
        strategy_name_hint = f"{hypothesis_id}_signal"

    raw_theme_tag = str(payload.get("theme_tag", "")).strip()
    if not raw_theme_tag:
        raise ValueError("theme_tag is required")
    theme_tag = normalize_theme_tag(raw_theme_tag)

    params_template = payload.get("params_template", {})
    if not isinstance(params_template, dict):
        params_template = {}

    entry_conditions = normalize_entry_conditions(
        payload.get("entry_conditions"),
        params_template=params_template,
    )

    raw_cfgs = payload.get("bar_configs", [])
    chosen = _normalize_requested_bar_configs(
        raw_cfgs,
        mission_bar_configs=mission_bar_configs,
    )
    if not chosen:
        chosen = [mission_bar_configs[0]]

    # Enforce minimum 1.5:1 RR and bar-type-aware minimum SL
    _pt = params_template.get("pt_ticks")
    _sl = params_template.get("sl_ticks")
    if _pt is not None and _sl is not None:
        try:
            pt_val = float(_pt)
            sl_val = float(_sl)
            if sl_val <= 0:
                raise ValueError(f"sl_ticks must be > 0, got {sl_val}")
            if pt_val < 1.5 * sl_val:
                raise ValueError(
                    f"RR violation: pt_ticks={pt_val} < 1.5 × sl_ticks={sl_val} "
                    f"(minimum 1.5:1 required, got {pt_val/sl_val:.2f}:1)"
                )
            chosen, dropped = _prune_bar_configs_for_risk_floor(chosen, sl_ticks=sl_val)
            if dropped and not chosen:
                raise ValueError(
                    _risk_floor_error_message(
                        bar_config=dropped[0],
                        sl_ticks=sl_val,
                        sample_bar_context=sample_bar_context,
                    )
                )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pt_ticks/sl_ticks in params_template: {exc}") from exc

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

    return {
        "hypothesis_id": hypothesis_id,
        "theme_tag": theme_tag,
        "strategy_name_hint": strategy_name_hint,
        "bar_configs": chosen,
        "params_template": dict(params_template),
        "entry_conditions": entry_conditions,
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
    chosen = _normalize_requested_bar_configs(
        raw_cfgs,
        mission_bar_configs=mission_bar_configs,
    )
    thinker_cfgs = _normalize_requested_bar_configs(
        thinker_brief.get("bar_configs", []),
        mission_bar_configs=mission_bar_configs,
    )
    thinker_primary = thinker_cfgs[0] if thinker_cfgs else ""
    if thinker_primary and thinker_primary in chosen:
        chosen = [thinker_primary]
    elif chosen:
        chosen = [chosen[0]]
    elif thinker_primary:
        chosen = [thinker_primary]
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
        strategy_df = filter_strategy_inputs(df, feature_group)
        if len(strategy_df) == 0:
            continue
        last_non_empty = strategy_df.head(max_rows)
        if len(last_non_empty) >= 64:
            return last_non_empty

    if last_non_empty is not None:
        return last_non_empty
    raise RuntimeError(f"Could not load sample feature frame for bar_config={bar_config}")


def _validation_file_indices(total_files: int, *, target_count: int = 3) -> list[int]:
    if total_files <= 0:
        return []
    if total_files <= target_count:
        return list(range(total_files))
    indices = np.linspace(0, total_files - 1, num=target_count, dtype=int).tolist()
    return sorted(set(int(idx) for idx in indices))


def _load_validation_strategy_samples(
    *,
    split: str,
    bar_config: str,
    session_filter: str,
    feature_group: str,
    max_rows: int = 1200,
    target_count: int = 10,
) -> list[tuple[str, pl.DataFrame]]:
    try:
        parsed = _parse_bar_config(bar_config)
    except ValueError as exc:
        raise RuntimeError(f"Invalid bar_config '{bar_config}': {exc}") from exc

    files = get_split_files(split)
    if not files:
        raise RuntimeError(f"No files found for split={split}")

    samples: list[tuple[str, pl.DataFrame]] = []
    seen_paths: set[str] = set()
    candidate_indices = _validation_file_indices(len(files), target_count=target_count)
    candidate_indices.extend(idx for idx in range(len(files)) if idx not in candidate_indices)

    for idx in candidate_indices:
        file_path = files[idx]
        key = str(file_path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
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
        strategy_df = filter_strategy_inputs(df, feature_group)
        if len(strategy_df) == 0:
            continue
        label = Path(file_path).stem
        samples.append((label, strategy_df.head(max_rows)))
        if len(samples) >= target_count:
            break

    if samples:
        return samples
    raise RuntimeError(f"Could not load validation sample frames for bar_config={bar_config}")


FEATURE_COMPUTATION_NOTES = [
    "Use precomputed canonical feature columns whenever available; avoid recomputing them in signal code.",
    "sma_ratio_N = close / SMA(close, N) - 1 (rolling mean, min_samples=N).",
    "ema_ratio_N = close / EWM(close, span=N, adjust=False, min_samples=N) - 1.",
    "RSI uses Wilder smoothing (alpha=1/N) from close-to-close gains/losses.",
    "ATR uses true range with Wilder smoothing; session-aware bars and timestamps are already canonicalized.",
    "VWAP-style cumulative metrics reset at session/day boundaries in canonical builders.",
    "Opening-range fields are session-aware; OR-dependent fields are null before OR is ready.",
    "Orderflow/toxicity/footprint features are bar-causal and derived only from events up to each bar close.",
    "Volume-profile session fields (`poc_price`, `va_high`, `va_low`, `position_in_va`) are expanding-session and causal within the active session.",
    "Previous-session profile levels are separate columns (`prev_day_poc`, `prev_day_vah`, `prev_day_val`, `dist_prev_*`, `prev_day_va_position`).",
    "Profile distance features such as `poc_distance` and `rolling_poc_distance` are ATR-normalized; `*_raw` variants stay in points.",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FEATURE_CATALOG_PATH = _REPO_ROOT / "research" / "feature_catalog.md"
_SKILLS_DIR = _REPO_ROOT / ".claude" / "skills"


def _load_skill(name: str) -> str:
    """Load a skill SKILL.md, stripping the YAML front-matter."""
    path = _SKILLS_DIR / name / "SKILL.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    # Strip YAML front-matter (--- ... ---)
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].lstrip("\n")
    return text.strip()


def _load_feature_catalog() -> list[str]:
    """Load feature catalog lines, stripping comments and blank lines."""
    if not _FEATURE_CATALOG_PATH.exists():
        return []
    lines = []
    for raw in _FEATURE_CATALOG_PATH.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


FEATURE_CATALOG_LINES: list[str] = _load_feature_catalog()


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
        "feature_catalog": FEATURE_CATALOG_LINES,
        "errors": errors,
    }


def _build_feature_surface(
    *,
    mission_bar_configs: list[str],
    split: str,
    session_filter: str,
    feature_group: str,
    validation_sample_cache: dict[str, list[tuple[str, pl.DataFrame]]],
) -> dict[str, Any]:
    samples_by_bar_config: dict[str, list[tuple[str, pl.DataFrame]]] = {}

    for bar_config in mission_bar_configs:
        if bar_config not in validation_sample_cache:
            validation_sample_cache[bar_config] = _load_validation_strategy_samples(
                split=split,
                bar_config=bar_config,
                session_filter=session_filter,
                feature_group=feature_group,
            )
        samples_by_bar_config[bar_config] = validation_sample_cache[bar_config]

    return build_feature_surface(samples_by_bar_config=samples_by_bar_config)


def _format_feature_surface_summary(
    feature_surface: dict[str, Any],
    *,
    selected_bar_configs: list[str],
) -> str:
    by_bar_config = feature_surface.get("by_bar_config")
    if not isinstance(by_bar_config, dict) or not by_bar_config:
        return "Feature surface unavailable."

    parts: list[str] = []
    for bar_config in selected_bar_configs:
        payload = by_bar_config.get(bar_config)
        if not isinstance(payload, dict):
            continue
        dead_count = len(payload.get("dead_features", []) or [])
        sparse_count = len(payload.get("sparse_features", []) or [])
        total_rows = int(payload.get("total_rows", 0) or 0)
        parts.append(
            f"{bar_config}: rows={total_rows} dead={dead_count} sparse={sparse_count}",
        )
    return "; ".join(parts) if parts else "Feature surface unavailable."


def _thinker_handoff_text_fragments(thinker_handoff: dict[str, Any]) -> list[str]:
    hypothesis = thinker_handoff.get("hypothesis")
    if not isinstance(hypothesis, dict):
        return []

    fragments: list[str] = []
    for key in ("thesis", "entry_logic", "exit_logic"):
        value = hypothesis.get(key)
        if isinstance(value, str) and value.strip():
            fragments.append(value)
    for key in ("risk_controls", "anti_lookahead_checks", "validation_focus"):
        values = hypothesis.get(key)
        if isinstance(values, list):
            fragments.extend(str(item) for item in values if str(item).strip())
    entry_conditions = hypothesis.get("entry_conditions")
    if isinstance(entry_conditions, list):
        for row in entry_conditions:
            if not isinstance(row, dict):
                continue
            feature = str(row.get("feature", "")).strip()
            op = str(row.get("op", "")).strip()
            role = str(row.get("role", "")).strip()
            if feature and op:
                fragments.append(f"{feature} {op} ({role})")
    return fragments


def _sample_bar_context(df: pl.DataFrame) -> dict[str, Any]:
    context: dict[str, Any] = {
        "sample_rows": int(len(df)),
        "columns": int(len(df.columns)),
    }
    if not {"high", "low"}.issubset(set(df.columns)) or len(df) == 0:
        return context

    high = np.asarray(df["high"].to_numpy(), dtype=np.float64)
    low = np.asarray(df["low"].to_numpy(), dtype=np.float64)
    range_ticks = (high - low) / float(TICK_SIZE)
    range_ticks = range_ticks[np.isfinite(range_ticks) & (range_ticks >= 0.0)]
    if range_ticks.size == 0:
        return context

    context["range_ticks"] = {
        "median": float(np.percentile(range_ticks, 50)),
        "p25": float(np.percentile(range_ticks, 25)),
        "p75": float(np.percentile(range_ticks, 75)),
        "p90": float(np.percentile(range_ticks, 90)),
    }
    return context


def _build_runtime_context(
    *,
    mission: dict[str, Any],
    mission_bar_configs: list[str],
    search_split: str,
    selection_split: str | None,
    session_filter: str,
    feature_group: str,
    sample_cache: dict[str, pl.DataFrame],
) -> dict[str, Any]:
    bar_context: dict[str, Any] = {}
    for bar_config in mission_bar_configs:
        sample_df = sample_cache.get(bar_config)
        if sample_df is not None:
            bar_context[bar_config] = _sample_bar_context(sample_df)

    return {
        "schema_version": "1.0",
        "objective": str(mission.get("objective", "")),
        "split": search_split,
        "search_split": search_split,
        "selection_split": selection_split,
        "feedback_split": search_split,
        "session_filter": session_filter,
        "feature_group": feature_group,
        "allowed_bar_configs": list(mission_bar_configs),
        "target_sharpe": float(mission.get("target_sharpe", 0.0)),
        "min_trade_count": int(mission.get("min_trade_count", 1)),
        "gauntlet_tests": [
            "shuffle",
            "walk_forward",
            "regime",
            "signal_perturbation",
            "cost_sensitivity",
            "decay",
            "trade_count",
        ],
        "selection_policy": "Selection split is a gate only and is not fed back into thinker learning.",
        "sample_bar_context": bar_context,
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
    validation_sample_cache: dict[str, list[tuple[str, pl.DataFrame]]] | None = None,
    sample_cache: dict[str, pl.DataFrame] | None = None,
    code: str = "",
) -> tuple[list[str], dict[str, Any]]:
    try:
        module = load_signal_module(strategy_name, signals_dir=signals_dir)
    except Exception as exc:
        return [f"{strategy_name}: module validation failed: {type(exc).__name__}: {exc}"], {
            "strategy_name": strategy_name,
            "bar_results": [
                {
                    "status": "module_validation_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ],
        }
    strategy_fn = getattr(module, "generate_signal", None)
    if not callable(strategy_fn):
        return [f"{strategy_name}: missing callable generate_signal(df, params)"], {
            "strategy_name": strategy_name,
            "bar_results": [
                {
                    "status": "module_validation_failed",
                    "error": "missing callable generate_signal(df, params)",
                }
            ],
        }

    if validation_sample_cache is None:
        validation_sample_cache = {}
    if sample_cache:
        for bar_config, df in sample_cache.items():
            validation_sample_cache.setdefault(str(bar_config), [("cached_sample", df)])

    errors: list[str] = []
    report: dict[str, Any] = {"strategy_name": strategy_name, "bar_results": []}
    for bar_config in bar_configs:
        if bar_config not in validation_sample_cache:
            validation_sample_cache[bar_config] = _load_validation_strategy_samples(
                split=split,
                bar_config=bar_config,
                session_filter=session_filter,
                feature_group=feature_group,
            )
        for sample_label, strategy_df in validation_sample_cache[bar_config]:
            if len(strategy_df) == 0:
                errors.append(f"{strategy_name}: empty sample frame for {bar_config} ({sample_label})")
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": sample_label,
                        "status": "empty_sample",
                        "error": "empty sample frame",
                    }
                )
                continue

            try:
                causality = check_signal_causality(
                    generate_fn=strategy_fn,
                    df=strategy_df,
                    params=params,
                    mode="strict",
                )
            except Exception as exc:
                errors.append(
                    f"{strategy_name}: causality check crashed for {bar_config} "
                    f"({sample_label}): {type(exc).__name__}: {exc}",
                )
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": sample_label,
                        "status": "causality_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if causality:
                errors.append(f"{strategy_name}: non-causal for {bar_config} ({sample_label}): {causality[0]}")
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": sample_label,
                        "status": "causality_failed",
                        "error": str(causality[0]),
                    }
                )
                continue

            try:
                raw = np.asarray(strategy_fn(strategy_df, params))
            except Exception as exc:
                errors.append(
                    f"{strategy_name}: generate_signal failed for {bar_config} "
                    f"({sample_label}): {type(exc).__name__}: {exc}",
                )
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": sample_label,
                        "status": "runtime_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            signal_errors = _validate_signal_array(raw, len(strategy_df))
            if signal_errors:
                errors.append(
                    f"{strategy_name}: contract failed for {bar_config} ({sample_label}): {signal_errors[0]}",
                )
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": sample_label,
                        "status": "contract_failed",
                        "error": str(signal_errors[0]),
                    }
                )
                continue

            nonzero = int(np.count_nonzero(raw))
            total = len(raw)
            rate_pct = 100.0 * nonzero / total if total > 0 else 0.0
            condition_rows = collect_condition_passthrough(
                df=strategy_df, code=code, params=params,
            )
            condition_diag = diagnose_condition_passthrough(
                df=strategy_df, code=code, params=params,
            )
            if nonzero == 0:
                diag = _diagnose_zero_signal(strategy_df, code)
                errors.append(
                    f"{strategy_name}: signal_rate=0.0% for {bar_config} ({sample_label}) "
                    f"(0/{total} bars non-zero — target 0.05–0.3%). "
                    f"Column statistics for referenced features:\n{diag}\n"
                    + (f"{condition_diag}\n" if condition_diag else "")
                    + f"ACTION REQUIRED: Relax the condition(s) marked BLOCKS ALL / RESTRICTIVE above "
                    f"to produce at least 1 signal in this {total}-bar sample."
                )
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": sample_label,
                        "status": "zero_signal",
                        "nonzero": nonzero,
                        "total": total,
                        "signal_rate_pct": rate_pct,
                        "condition_rows": condition_rows,
                    }
                )
            elif rate_pct > 2.0:
                errors.append(
                    f"{strategy_name}: signal_rate={rate_pct:.2f}% for {bar_config} ({sample_label}) "
                    f"({nonzero}/{total} bars non-zero — target 0.05–0.3%). "
                    + (f"{condition_diag}\n" if condition_diag else "")
                    + "Tighten the loosest entry filter(s) to avoid cost destruction."
                )
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": sample_label,
                        "status": "over_signal",
                        "nonzero": nonzero,
                        "total": total,
                        "signal_rate_pct": rate_pct,
                        "condition_rows": condition_rows,
                    }
                )
            else:
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": sample_label,
                        "status": "ok",
                        "nonzero": nonzero,
                        "total": total,
                        "signal_rate_pct": rate_pct,
                    }
                )

    return errors, report


def _build_thinker_system_prompt() -> str:
    skill = _load_skill("notebook-alpha-research")
    parts = [
        "You are the quant-thinker agent for the NQ alpha discovery loop.",
        "Treat the runtime mission context in the user prompt as the source of truth for split, session filter,",
        "thresholds, and allowed bar configs. Return only the required JSON object.",
    ]
    if skill:
        parts.append(f"\n## Notebook Alpha Research Skill\n\n{skill}")
    return "\n".join(parts)


def _disable_notebooklm_runtime_mission(mission: dict[str, Any]) -> dict[str, Any]:
    runtime_mission = dict(mission)
    runtime_mission.pop("notebooklm", None)
    runtime_mission["notebooklm_disabled_for_run"] = True
    runtime_mission["notebooklm_notebook_url"] = ""
    runtime_mission["notebook_research_guidance"] = ""
    runtime_mission["lane_notebook_query_budget"] = {
        "max_total_queries": 0,
        "max_research_queries": 0,
        "max_deep_research_queries": 0,
    }
    return runtime_mission


def _build_thinker_user_prompt(
    *,
    mission: dict[str, Any],
    existing_strategies: list[str],
    feedback_items: list[dict[str, Any]],
    learning_context: str = "",
    thinker_memory_context: str = "",
    feature_surface_context: str = "",
    param_feasibility_context: str = "",
    runtime_context: dict[str, Any] | None = None,
    feature_knowledge: dict[str, Any] | None = None,
) -> str:
    bar_configs = mission.get("bar_configs", ["tick_610"])
    current_focus = mission.get("current_focus", [])
    if not isinstance(current_focus, list):
        current_focus = []
    notebooklm_url = str(mission.get("notebooklm_notebook_url", "")).strip()
    notebook_query_budget = _resolve_notebook_query_budget(mission)
    notebook_research_guidance = str(mission.get("notebook_research_guidance", "")).strip()
    objective = str(mission.get("objective", "Discover robust intraday alpha signals."))
    focus_anchors = resolve_focus_anchors(mission)
    avoid = ", ".join(existing_strategies[-50:]) if existing_strategies else "(none)"
    focus_blob = "\n".join(f"- {str(x)}" for x in current_focus) if current_focus else "- none provided"
    results_table = _format_results_table(feedback_items)
    prompt_parts = [
        f"Mission objective:\n{objective}\n\n"
        f"Allowed bar_configs: {bar_configs}\n"
        f"Preferred session filter: {mission.get('session_filter', 'eth')}\n"
        f"Feature group: {mission.get('feature_group', 'all')}\n"
        f"Current focus:\n{focus_blob}\n\n"
        f"Existing strategy files to avoid duplicating:\n{avoid}\n\n"
        f"Recent experiment results:\n{results_table}\n\n"
    ]
    if focus_anchors:
        prompt_parts.append(
            "Current focus anchors (soft guidance, not a fixed taxonomy):\n"
            + "\n".join(f"- {tag}" for tag in focus_anchors)
            + "\n\n",
        )
    prompt_parts.append(
        "Return exactly one concise `theme_tag` in snake_case. Reuse a focus anchor if it fits; "
        "otherwise introduce a new precise tag if the evidence points elsewhere.\n\n",
    )
    prompt_parts.append(
        "Return exactly one selected `bar_config` inside `bar_configs` (list length 1). "
        "Do not spread one hypothesis across multiple bar types in the same iteration.\n\n",
    )
    prompt_parts.append(
        "You must also return `entry_conditions`: a machine-checkable list of the core gates that must be true before a signal can fire.\n"
        "Use 2-6 conditions total. Supported ops: `>`, `>=`, `<`, `<=`, `between`, `bool_true`, `bool_false`.\n"
        "Each condition must use an exact feature name and include `role` = `primary` or `confirmation`.\n"
        "For comparison ops, reference a numeric key from `params_template` via `param_key`.\n"
        "For `between`, use `param_key_low` and `param_key_high`.\n"
        "Primary conditions must be independently plausible on the provided sample stats. Do not set thresholds far outside observed feature bands.\n"
        "Avoid dead features and avoid numeric cutoffs that would pass 0 bars or nearly every bar on the sample.\n"
        "These conditions are checked against live sample data before coding. If they cannot fire, the hypothesis will be auto-repaired or rejected before coder runs.\n\n",
    )
    if learning_context.strip():
        prompt_parts.append(f"{learning_context}\n\n")
    if thinker_memory_context.strip():
        prompt_parts.append(f"{thinker_memory_context}\n\n")
    if feature_surface_context.strip():
        prompt_parts.append(f"{feature_surface_context}\n\n")
    if param_feasibility_context.strip():
        prompt_parts.append(f"{param_feasibility_context}\n\n")
    prompt_parts.append(f"{_format_bar_risk_floor_context(runtime_context)}\n\n")
    prompt_parts.append("Design exactly one hypothesis that is implementable by a separate coding model.")
    prompt = "".join(prompt_parts)
    if runtime_context:
        prompt += (
            "\n\nRUNTIME_MISSION_CONTEXT_JSON_BEGIN\n"
            f"{json.dumps(runtime_context, indent=2, sort_keys=True, default=str)}\n"
            "RUNTIME_MISSION_CONTEXT_JSON_END\n"
            "Treat this JSON as the source of truth for the active split, session filter, thresholds, and sample bar stats."
        )
    if notebooklm_url:
        import os as _os
        cwd = _os.getcwd()
        commands = [
            f'  uv --directory "{cwd}" run python scripts/query_notebook.py --notebook-url "{notebooklm_url}" "question"',
        ]
        if int(notebook_query_budget.get("max_research_queries", 0) or 0) > 0:
            commands.append(
                f'  uv --directory "{cwd}" run python scripts/query_notebook.py --notebook-url "{notebooklm_url}" --research "question"',
            )
        if int(notebook_query_budget.get("max_deep_research_queries", 0) or 0) > 0:
            commands.append(
                f'  uv --directory "{cwd}" run python scripts/query_notebook.py --notebook-url "{notebooklm_url}" --deep-research "question"',
            )
        prompt += (
            "\n\nKNOWLEDGE_BASE_COMMAND (copy these exact commands — do not modify the path):\n"
            + "\n".join(commands)
            + "\n"
            "This notebook is private to your lane and persists across iterations for this lane only. "
            "You must choose the research direction yourself based on the mission, recent results, and what the notebook already contains. "
            f"Hard runtime budget this iteration: at most {int(notebook_query_budget.get('max_research_queries', 0) or 0)} --research query "
            f"and {int(notebook_query_budget.get('max_total_queries', 0) or 0)} total notebook queries. "
            "Use the single --research query only for one pointed source-enrichment question, then use plain queries only if they are necessary to sharpen the hypothesis. "
            "Once the budget is spent, stop querying and finalize the hypothesis."
        )
        if int(notebook_query_budget.get("max_deep_research_queries", 0) or 0) <= 0:
            prompt += "\n--deep-research is disabled for the autonomy thinker loop because it is too slow for iteration-time use."
        if notebook_research_guidance:
            prompt += (
                "\nNotebookLM research guidance:\n"
                f"- {notebook_research_guidance}"
            )
    elif bool(mission.get("notebooklm_disabled_for_run", False)):
        prompt += (
            "\n\nNotebookLM is disabled for this run. Use your built-in knowledge plus the provided runtime, "
            "feature-surface, and feature-catalog context only."
        )
    if feature_knowledge:
        prompt += (
            "\n\nAVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN\n"
            f"{json.dumps(feature_knowledge, indent=2, sort_keys=True, default=str)}\n"
            "AVAILABLE_PRECOMPUTED_FEATURES_JSON_END\n"
            "The JSON above contains: common_columns (all available column names), "
            "feature_catalog (column | formula | interpretation for ~60 key features), "
            "and computation_notes. Study feature_catalog carefully before designing entry conditions — "
            "it tells you exactly what each column measures and its typical range."
        )
    return prompt


def _build_coder_system_prompt() -> str:
    skill = _load_skill("nq-signal-coding-contract")
    parts = [
        "You are the nq-signal-coder agent for the NQ alpha discovery loop.",
        "Follow the user prompt exactly. Return only the required JSON object with keys",
        "`strategy_name`, `bar_configs`, `params`, and `code`.",
        "Treat `hypothesis.entry_conditions` as the required core entry gates. The final code should implement",
        "those conditions directly and use `entry_logic` only to refine direction or execution details.",
        "Preserve the handoff's single selected bar_config; do not broaden the strategy to additional bar types.",
        "Hard runtime reminders: the final signal array must be dtype `np.int8` and should end with",
        "`.astype(np.int8)`, and `shift(-1)` is forbidden.",
    ]
    if skill:
        parts.append(f"\n## Signal Coding Contract Skill\n\n{skill}")
    return "\n".join(parts)


def _build_coder_handoff(
    *,
    thinker_brief: dict[str, Any],
    mission: dict[str, Any],
    thinker_payload_hash: str,
    runtime_context: dict[str, Any] | None = None,
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
            "split": str((runtime_context or {}).get("split", mission.get("split", "validate"))),
            "search_split": str((runtime_context or {}).get("search_split", mission.get("search_split", "validate"))),
            "selection_split": (runtime_context or {}).get("selection_split", mission.get("selection_split")),
            "target_sharpe": float((runtime_context or {}).get("target_sharpe", mission.get("target_sharpe", 0.0))),
            "min_trade_count": int((runtime_context or {}).get("min_trade_count", mission.get("min_trade_count", 1))),
            "sample_bar_context": dict((runtime_context or {}).get("sample_bar_context", {})),
        },
        "hypothesis": {
            "hypothesis_id": hypothesis_id,
            "theme_tag": str(thinker_brief.get("theme_tag", "")),
            "strategy_name_hint": strategy_name_hint,
            "bar_configs": list(thinker_brief.get("bar_configs", [])),
            "params_template": dict(thinker_brief.get("params_template", {}))
            if isinstance(thinker_brief.get("params_template"), dict)
            else {},
            "entry_conditions": list(thinker_brief.get("entry_conditions", []))
            if isinstance(thinker_brief.get("entry_conditions"), list)
            else [],
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
    feature_surface_warning: str = "",
) -> str:
    prompt = (
        "Implement exactly the handoff below as a signal module.\n"
        "Use only this handoff JSON as source of truth.\n\n"
        "Keep the handoff's single selected `bar_config` unchanged.\n\n"
        "THINKER_HANDOFF_JSON_BEGIN\n"
        f"{json.dumps(thinker_handoff, indent=2, sort_keys=True, default=str)}\n"
        "THINKER_HANDOFF_JSON_END\n"
    )
    if feature_surface_warning.strip():
        prompt += f"\n{feature_surface_warning}\n"
    if feature_knowledge:
        prompt += (
            "\nAVAILABLE_PRECOMPUTED_FEATURES_JSON_BEGIN\n"
            f"{json.dumps(feature_knowledge, indent=2, sort_keys=True, default=str)}\n"
            "AVAILABLE_PRECOMPUTED_FEATURES_JSON_END\n"
            "The JSON above contains feature_catalog: column | formula | interpretation for ~60 key features. "
            "Use the exact column names from common_columns. "
            "Prefer precomputed features directly; only compute local fallbacks when a column is absent."
        )
    return prompt


_REPAIR_CODE_MAX_CHARS = 4000


def _build_coder_repair_user_prompt(
    *,
    thinker_handoff: dict[str, Any],
    previous_code: str,
    validation_errors: list[str],
    common_columns: list[str],
    feature_surface_warning: str = "",
) -> str:
    """Build a targeted repair prompt for the coder when generated code fails validation."""
    code_snippet = previous_code
    if len(code_snippet) > _REPAIR_CODE_MAX_CHARS:
        # Keep the tail — generate_signal body (where errors live) appears after imports/DEFAULT_PARAMS
        code_snippet = "[... truncated ...]\n" + code_snippet[-_REPAIR_CODE_MAX_CHARS:]

    errors_blob = "\n".join(f"  - {e}" for e in validation_errors)
    cols_hint = ", ".join(common_columns[:40]) if common_columns else "(see feature knowledge)"

    has_zero_rate = any("signal_rate=0" in e for e in validation_errors)
    helper_contract_error = any(
        "safe_f64_col" in e or "to_numpy()" in e or "copy=False" in e
        for e in validation_errors
    )
    if has_zero_rate:
        fix_instruction = (
            "CRITICAL: Your strategy generated ZERO signals.\n"
            "The column statistics in VALIDATION_ERRORS above show empirical firing rates "
            "for the features your code references.\n"
            "ACTION: RELAX the most restrictive threshold value(s) in DEFAULT_PARAMS so the "
            "strategy fires at least 1 signal in the sample (target rate: 0.05–0.3% of bars).\n"
            "Do NOT add new conditions — only loosen existing threshold values."
        )
    elif helper_contract_error:
        fix_instruction = (
            "Fix the repo helper-contract violation exactly.\n"
            "Use `safe_f64_col(df, \"col\", fill=0.0)` instead of direct dataframe `.to_numpy()` extraction, "
            "and do not use `np.nan_to_num(..., copy=False)`.\n"
            "Keep the strategy logic the same; only rewrite the column extraction / sanitization path so it passes validation."
        )
    else:
        fix_instruction = (
            "Fix ONLY the validation errors above. Do not change the overall strategy logic."
        )

    return (
        "Your previously generated code FAILED validation.\n\n"
        "ORIGINAL_THINKER_HANDOFF_JSON_BEGIN\n"
        f"{json.dumps(thinker_handoff, indent=2, sort_keys=True, default=str)}\n"
        "ORIGINAL_THINKER_HANDOFF_JSON_END\n\n"
        + (f"{feature_surface_warning}\n\n" if feature_surface_warning.strip() else "")
        + (
        "PREVIOUS_CODE_BEGIN\n"
        f"{code_snippet}\n"
        "PREVIOUS_CODE_END\n\n"
        f"VALIDATION_ERRORS:\n{errors_blob}\n\n"
        f"AVAILABLE_COLUMNS_HINT (common across bar configs): {cols_hint}\n\n"
        f"{fix_instruction}\n\n"
        "Return corrected JSON with same schema: strategy_name, bar_configs, params, code."
        )
    )


def _task_id(strategy_name: str, bar_config: str, params: dict[str, Any], code_hash: str) -> str:
    digest = _sha256_text(
        f"{strategy_name}|{bar_config}|{json.dumps(params, sort_keys=True, separators=(',', ':'))}|{code_hash}",
    )[:12]
    return f"llm_{_slug(strategy_name)}_{_slug(bar_config)}_{digest}"


def _build_task(
    *,
    strategy_name: str,
    search_split: str,
    selection_split: str | None,
    bar_config: str,
    params: dict[str, Any],
    mission: dict[str, Any],
    code_hash: str,
    iteration: int,
    hypothesis_id: str,
    theme_tag: str,
) -> dict[str, Any]:
    max_retries = int(mission.get("max_retries", 2))
    timeout_minutes = int(mission.get("task_timeout_minutes", 30))
    heartbeat_seconds = int(mission.get("heartbeat_interval_seconds", 300))

    # Promote engine exit params from params dict to task top-level so research.py
    # can pass them directly to run_backtest(). Canonical names: pt_ticks, sl_ticks,
    # max_bars. Convert ticks → points for profit_target/stop_loss.
    _pt_ticks = params.get("pt_ticks")
    _sl_ticks = params.get("sl_ticks")
    _exit_bars = params.get("max_bars") or params.get("exit_bars")
    profit_target = float(_pt_ticks) * TICK_SIZE if _pt_ticks is not None else None
    stop_loss = float(_sl_ticks) * TICK_SIZE if _sl_ticks is not None else None
    exit_bars = int(_exit_bars) if _exit_bars is not None else None

    return {
        "task_id": _task_id(strategy_name, bar_config, params, code_hash),
        "state": "pending",
        "assigned_to": None,
        "strategy_name": strategy_name,
        "split": search_split,
        "search_split": search_split,
        "selection_split": selection_split,
        "bar_config": bar_config,
        "theme_tag": str(theme_tag),
        "params": dict(params),
        "exit_bars": exit_bars,
        "profit_target": profit_target,
        "stop_loss": stop_loss,
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
            "theme_tag": str(theme_tag),
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

    cli_cfg = _cfg_dict(agent_cfg.get("claude_cli"))
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
    retry_backoff_seconds = float(cli_cfg.get("retry_backoff_seconds", agent_cfg.get("retry_backoff_seconds", 1.5)))
    disable_slash_commands_raw = cli_cfg.get("disable_slash_commands", agent_cfg.get("disable_slash_commands", True))
    disable_slash_commands = bool(disable_slash_commands_raw)
    if role_disable_slash_commands is not None:
        disable_slash_commands = bool(role_disable_slash_commands)
    workdir_raw = str(cli_cfg.get("workdir", "")).strip()
    workdir = root if not workdir_raw else (Path(workdir_raw) if Path(workdir_raw).is_absolute() else (root / workdir_raw))
    global_extra_args_raw = cli_cfg.get("extra_args", [])
    global_extra_args = [str(v) for v in global_extra_args_raw] if isinstance(global_extra_args_raw, list) else []
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

    role_cli_cfg = _cfg_dict(role_cfg.get("claude_cli"))
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
    parser.add_argument("--resume", action="store_true", help="Resume orchestrator state from research/.state (default behavior).")
    parser.add_argument(
        "--fresh-state",
        action="store_true",
        help="Reset only this orchestrator lane's state file before starting.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate and validate only; do not write files/tasks.")
    parser.add_argument("--lane", type=str, default=None, help="Lane ID (e.g. A, B, C) for parallel execution. Namespaces state and log files.")
    parser.add_argument(
        "--disable-notebooklm",
        action="store_true",
        help="Disable NotebookLM for this orchestrator run and rely only on built-in model knowledge.",
    )
    args = parser.parse_args()
    if args.resume and args.fresh_state:
        parser.error("Use at most one of --resume or --fresh-state.")

    lane_id: str | None = None
    if args.lane is not None:
        lane_id = re.sub(r"[^a-zA-Z0-9]", "", str(args.lane)).upper() or None
        if not lane_id:
            raise ValueError("--lane must contain at least one alphanumeric character")

    root = Path(__file__).resolve().parent.parent
    mission_path = args.mission.resolve()
    if not mission_path.exists():
        raise FileNotFoundError(f"Mission file not found: {mission_path}")
    agent_cfg_path = args.agent_config.resolve() if args.agent_config.is_absolute() else (root / args.agent_config).resolve()
    if not agent_cfg_path.exists():
        raise FileNotFoundError(f"Agent config file not found: {agent_cfg_path}")

    mission = _load_yaml(mission_path)
    runtime_mission = (
        _disable_notebooklm_runtime_mission(mission)
        if bool(args.disable_notebooklm)
        else dict(mission)
    )
    agent_cfg = _load_yaml(agent_cfg_path)
    mission_name = str(runtime_mission.get("mission_name", mission_path.stem))
    bar_configs_raw = runtime_mission.get("bar_configs", ["tick_610"])
    mission_bar_configs = [str(v) for v in bar_configs_raw] if isinstance(bar_configs_raw, list) else ["tick_610"]
    if not mission_bar_configs:
        raise ValueError("mission.bar_configs cannot be empty")

    split_plan = resolve_research_splits(runtime_mission)
    search_split = str(split_plan["search_split"])
    selection_split = (
        str(split_plan["selection_split"])
        if split_plan["selection_split"] is not None
        else None
    )
    session_filter = str(runtime_mission.get("session_filter", "eth")).lower()
    feature_group = str(runtime_mission.get("feature_group", "all")).lower()

    state_mode = _state_mode(fresh_state=bool(args.fresh_state))
    shared_paths = ensure_shared_state(root, mission_name=mission_name)
    orchestrator_state_file = (
        reset_orchestrator_state(root, mission_name=mission_name, lane_id=lane_id)
        if state_mode == "fresh"
        else ensure_orchestrator_state(root, mission_name=mission_name, lane_id=lane_id)
    )
    state_paths = {
        "queue": shared_paths["queue"],
        "queue_lock": shared_paths["queue_lock"],
        "handoffs": shared_paths["handoffs"],
        "handoffs_lock": shared_paths["handoffs_lock"],
        "scorecard": shared_paths["scorecard"],
        "scorecard_lock": shared_paths["scorecard_lock"],
        "orchestrator": orchestrator_state_file,
        "thinker_memory": thinker_memory_path(root, lane_id=lane_id),
        "thinker_memory_lock": thinker_memory_lock_path(root, lane_id=lane_id),
    }
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
    thinker_client = _build_llm_client(
        provider=provider,
        model=thinker_role["model"],
        agent_cfg=agent_cfg,
        root=root,
        role_agent_name=thinker_role.get("agent_name"),
        role_extra_args=thinker_role.get("extra_args"),
        role_disable_slash_commands=thinker_role.get("disable_slash_commands"),
        role_timeout_seconds=thinker_role.get("timeout_seconds"),
    )
    coder_client = _build_llm_client(
        provider=provider,
        model=coder_role["model"],
        agent_cfg=agent_cfg,
        root=root,
        role_agent_name=coder_role.get("agent_name"),
        role_extra_args=coder_role.get("extra_args"),
        role_disable_slash_commands=coder_role.get("disable_slash_commands"),
        role_timeout_seconds=coder_role.get("timeout_seconds"),
    )

    signals_dir = root / "research" / "signals"
    handoffs_path = state_paths["handoffs"]
    research_log_path = root / "results" / "logs" / "research_experiments.jsonl"
    log_suffix = f"_{lane_id}" if lane_id else ""
    orchestrator_log_path = root / "results" / "logs" / f"llm_orchestrator{log_suffix}.jsonl"
    orchestrator_log_lock = root / "results" / "logs" / f"llm_orchestrator{log_suffix}.lock"
    run_id = f"llm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{_slug(mission_name)}{log_suffix}"
    start_monotonic = time.monotonic()

    notebook_bootstrap = (
        ensure_lane_notebook(
            mission=runtime_mission,
            lane_id=lane_id,
            state_payload=orchestrator_state,
            run_id=run_id,
        )
        if not args.disable_notebooklm
        else {
            "configured": False,
            "mode": "disabled",
            "notebook": None,
            "mission_overrides": {},
        }
    )
    active_mission = dict(runtime_mission)
    active_mission.update(dict(notebook_bootstrap.get("mission_overrides", {})))
    notebook_research_guidance = str(active_mission.get("notebook_research_guidance", "")).strip()
    notebook_query_budget = _resolve_notebook_query_budget(active_mission)
    notebook_meta = (
        dict(notebook_bootstrap["notebook"])
        if isinstance(notebook_bootstrap.get("notebook"), dict)
        else None
    )
    orchestrator_state["run_id"] = run_id
    if notebook_meta is not None:
        orchestrator_state["notebooklm"] = notebook_meta
    elif "notebooklm" in orchestrator_state:
        orchestrator_state.pop("notebooklm", None)
    write_orchestrator_state(state_paths["orchestrator"], orchestrator_state)
    if notebook_meta is not None:
        log_experiment(
            {
                "run_id": run_id,
                "agent": "llm_orchestrator",
                "event": "notebooklm_setup",
                "lane_id": lane_id,
                "notebooklm": notebook_meta,
            },
            experiments_path=orchestrator_log_path,
            lock_path=orchestrator_log_lock,
        )

    set_execution_mode(ExecutionMode.RESEARCH)
    sample_cache: dict[str, pl.DataFrame] = {}
    validation_sample_cache: dict[str, list[tuple[str, pl.DataFrame]]] = {}
    feature_knowledge = _build_feature_knowledge(
        mission_bar_configs=mission_bar_configs,
        split=search_split,
        session_filter=session_filter,
        feature_group=feature_group,
        sample_cache=sample_cache,
    )
    feature_surface = _build_feature_surface(
        mission_bar_configs=mission_bar_configs,
        split=search_split,
        session_filter=session_filter,
        feature_group=feature_group,
        validation_sample_cache=validation_sample_cache,
    )
    feature_surface_context = format_feature_surface_context(
        feature_surface,
        selected_bar_configs=mission_bar_configs,
    )
    param_feasibility_context = format_param_feasibility_context(
        feature_surface,
        selected_bar_configs=mission_bar_configs,
    )
    runtime_context = _build_runtime_context(
        mission=active_mission,
        mission_bar_configs=mission_bar_configs,
        search_split=search_split,
        selection_split=selection_split,
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
        "Feature surface ready: "
        + _format_feature_surface_summary(
            feature_surface,
            selected_bar_configs=mission_bar_configs,
        ),
    )

    print(
        "LLM orchestrator run_id="
        f"{run_id} mission={mission_name} "
        f"provider={provider} "
        f"models=thinker:{thinker_role['model']} coder:{coder_role['model']}",
    )
    print(
        f"search_split={search_split} selection_split={selection_split or 'none'} "
        f"session_filter={session_filter} feature_group={feature_group}",
    )
    if notebook_meta is not None:
        print(
            "notebooklm="
            f"{notebook_meta.get('mode')} id={notebook_meta.get('notebook_id')} "
            f"seeded={bool(notebook_meta.get('seeded', False))} "
            f"imported_sources={notebook_meta.get('imported_sources', 0)}",
        )
    elif bool(args.disable_notebooklm):
        print("notebooklm=disabled")
    print(f"state_mode={state_mode}")
    print(f"max_iterations={max_iterations} dry_run={bool(args.dry_run)}")

    stop_reason = "max_iterations_reached"
    notebooklm_configured = bool(str(active_mission.get("notebooklm_notebook_url", "")).strip())
    while iterations_done < max_iterations:
        if max_runtime_hours is not None:
            elapsed = (time.monotonic() - start_monotonic) / 3600.0
            if elapsed >= max_runtime_hours:
                stop_reason = "max_runtime_reached"
                break

        queue_counts = _queue_counts(state_paths["queue"], state_paths["queue_lock"])
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
            handoffs_lock_path=state_paths["handoffs_lock"],
            research_log_path=research_log_path,
            orchestrator_log_path=orchestrator_log_path,
        )
        learning_context = _build_learning_context(
            scorecard_path=state_paths["scorecard"],
            scorecard_lock=state_paths["scorecard_lock"],
        )
        thinker_memory_context = format_thinker_memory_context(
            read_thinker_memory(
                path=state_paths["thinker_memory"],
                lock_path=state_paths["thinker_memory_lock"],
                lane_id=lane_id,
                window_size=3,
            )
        )

        iteration_no = iterations_done + 1
        print(f"[iteration {iteration_no}/{max_iterations}] running thinker->coder pipeline")
        notebook_summary = _default_notebook_summary(configured=notebooklm_configured)
        thinker_brief: dict[str, Any] = {}
        hypothesis_id = f"iter_{iteration_no}"
        theme_tag = "unknown"
        chosen_bars: list[str] = []
        params: dict[str, Any] = {}
        strategy_name = ""

        try:
            thinker_user_prompt = _build_thinker_user_prompt(
                mission=active_mission,
                existing_strategies=existing,
                feedback_items=feedback_items,
                learning_context=learning_context,
                thinker_memory_context=thinker_memory_context,
                feature_surface_context=feature_surface_context,
                param_feasibility_context=param_feasibility_context,
                runtime_context=runtime_context,
                feature_knowledge=feature_knowledge,
            )
            with notebook_audit_context(
                run_id=run_id,
                iteration=iteration_no,
                stage="quant_thinker",
                lane_id=lane_id,
                orchestrator_state_path=str(state_paths["orchestrator"]),
            ), notebook_research_guidance_context(notebook_research_guidance), notebook_query_budget_context(
                notebook_query_budget,
            ):
                thinker_generation = _call_stage_json(
                    stage_name="quant_thinker",
                    schema_hint=(
                        "keys: hypothesis_id, theme_tag, strategy_name_hint, thesis, bar_configs, params_template, "
                        "entry_conditions, entry_logic, exit_logic, risk_controls, anti_lookahead_checks, validation_focus"
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
                    normalize_fn=lambda payload: _normalize_and_assess_thinker_brief(
                        payload,
                        mission_bar_configs=mission_bar_configs,
                        sample_bar_context=runtime_context.get("sample_bar_context"),
                        validation_sample_cache=validation_sample_cache,
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
                "keys: hypothesis_id, theme_tag, strategy_name_hint, thesis, bar_configs, params_template, "
                        "entry_conditions, entry_logic, exit_logic, risk_controls, anti_lookahead_checks, validation_focus"
                    ),
                )
            if notebooklm_configured:
                notebook_summary = _default_notebook_summary(configured=True)
                notebook_summary.update(
                    summarize_notebook_queries(
                        run_id=run_id,
                        iteration=iteration_no,
                        stage="quant_thinker",
                        lane_id=lane_id,
                    ),
                )
                _persist_notebook_progress(
                    notebook_meta=notebook_meta,
                    notebook_summary=notebook_summary,
                    orchestrator_state=orchestrator_state,
                    state_paths=state_paths,
                )
            thinker_hash = _sha256_text(
                json.dumps(thinker_brief, sort_keys=True, separators=(",", ":")),
            )[:16]
            thinker_handoff = _build_coder_handoff(
                thinker_brief=thinker_brief,
                mission=active_mission,
                thinker_payload_hash=thinker_hash,
                runtime_context=runtime_context,
            )
            feature_surface_warning = format_referenced_surface_warnings(
                surface=feature_surface,
                selected_bar_configs=list(thinker_handoff["hypothesis"].get("bar_configs", [])),
                text_fragments=_thinker_handoff_text_fragments(thinker_handoff),
            )

            coder_user_prompt = _build_coder_user_prompt(
                thinker_handoff=thinker_handoff,
                feature_knowledge=feature_knowledge,
                feature_surface_warning=feature_surface_warning,
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
            theme_tag = str(thinker_brief.get("theme_tag", "other"))

            module_path, is_new_path = _choose_module_path(
                signals_dir,
                strategy_name=strategy_name,
                module_code=code,
            )
            module_name = module_path.stem

            validation_report: dict[str, Any] | None = None
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
                validation_errors, validation_report = _validate_generated_strategy(
                    strategy_name=module_name,
                    signals_dir=signals_dir,
                    params=params,
                    bar_configs=chosen_bars,
                    split=search_split,
                    session_filter=session_filter,
                    feature_group=feature_group,
                    validation_sample_cache=validation_sample_cache,
                    code=code,
                )

            # Inline repair loop: retry coder with injected errors
            _last_coder_generation = coder_generation
            should_repair = _should_attempt_coder_repair(validation_report=validation_report)
            if (
                validation_errors
                and not args.dry_run
                and max_code_repair_attempts > 0
                and should_repair
            ):
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
                            feature_surface_warning=format_referenced_surface_warnings(
                                surface=feature_surface,
                                selected_bar_configs=chosen_bars,
                                text_fragments=[code],
                            ),
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
                        validation_errors, validation_report = _validate_generated_strategy(
                            strategy_name=module_name,
                            signals_dir=signals_dir,
                            params=params,
                            bar_configs=chosen_bars,
                            split=search_split,
                            session_filter=session_filter,
                            feature_group=feature_group,
                            validation_sample_cache=validation_sample_cache,
                            code=code,
                        )
                        if not validation_errors:
                            print(f"  repair succeeded on attempt {repair_attempt + 1}")
                            break
                    except (LLMClientError, ValueError, RuntimeError, OSError) as repair_exc:
                        print(f"  repair call failed: {type(repair_exc).__name__}: {repair_exc}")
                        break
            elif validation_errors and not args.dry_run and not should_repair:
                print(
                    "  skipping coder repair: validation failure is hypothesis-level "
                    "(use thinker memory to adjust next iteration)"
                )

            if validation_errors:
                if not args.dry_run and is_new_path and module_path.exists():
                    module_path.unlink()
                attempt_record = _build_validation_attempt_record(
                    iteration=iteration_no,
                    hypothesis_id=hypothesis_id,
                    theme_tag=theme_tag,
                    strategy_name=module_name,
                    bar_configs=chosen_bars,
                    params=params,
                    validation_report=validation_report,
                )
                if attempt_record is not None:
                    append_thinker_attempt(
                        path=state_paths["thinker_memory"],
                        lock_path=state_paths["thinker_memory_lock"],
                        lane_id=lane_id,
                        attempt=attempt_record,
                        window_size=3,
                    )
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
                        "theme_tag": str(thinker_brief.get("theme_tag", "other")),
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
                        "notebooklm": notebook_summary,
                        "validation_report": validation_report,
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
                        search_split=search_split,
                        selection_split=selection_split,
                        bar_config=bar_config,
                        params=params,
                        mission=active_mission,
                        code_hash=code_hash,
                        iteration=iteration_no,
                        hypothesis_id=hypothesis_id,
                        theme_tag=theme_tag,
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
                                "theme_tag": theme_tag,
                            },
                        },
                    )

                total_tasks += len(enqueued_task_ids)
                generated_modules.append(module_name)
                append_thinker_attempt(
                    path=state_paths["thinker_memory"],
                    lock_path=state_paths["thinker_memory_lock"],
                    lane_id=lane_id,
                    attempt=_build_success_attempt_record(
                        iteration=iteration_no,
                        hypothesis_id=hypothesis_id,
                        theme_tag=theme_tag,
                        strategy_name=module_name,
                        bar_configs=chosen_bars,
                        params=params,
                    ),
                    window_size=3,
                )
                log_experiment(
                    {
                        "run_id": run_id,
                        "agent": "llm_orchestrator",
                        "event": "generation_enqueued",
                        "iteration": iteration_no,
                        "strategy_name": module_name,
                        "hypothesis_id": hypothesis_id,
                        "theme_tag": theme_tag,
                        "thinker_brief": thinker_brief,
                        "bar_configs": chosen_bars,
                        "params": params,
                        "task_ids": enqueued_task_ids,
                        "module_path": str(module_path),
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
                        "notebooklm": notebook_summary,
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
            if notebooklm_configured:
                notebook_summary = _default_notebook_summary(configured=True)
                notebook_summary.update(
                    summarize_notebook_queries(
                        run_id=run_id,
                        iteration=iteration_no,
                        stage="quant_thinker",
                        lane_id=lane_id,
                    ),
                )
                _persist_notebook_progress(
                    notebook_meta=notebook_meta,
                    notebook_summary=notebook_summary,
                    orchestrator_state=orchestrator_state,
                    state_paths=state_paths,
                )
            append_thinker_attempt(
                path=state_paths["thinker_memory"],
                lock_path=state_paths["thinker_memory_lock"],
                lane_id=lane_id,
                attempt=_build_exception_attempt_record(
                    iteration=iteration_no,
                    hypothesis_id=hypothesis_id,
                    theme_tag=theme_tag,
                    bar_configs=chosen_bars,
                    params=params,
                    exc=exc,
                ),
                window_size=3,
            )
            log_experiment(
                {
                    "run_id": run_id,
                    "agent": "llm_orchestrator",
                    "event": "generation_error",
                    "iteration": iteration_no,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": trace,
                    "notebooklm": notebook_summary,
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
            "notebooklm": notebook_meta,
            "last_updated": _utc_now(),
        }
        persisted_state = read_orchestrator_state(state_paths["orchestrator"])
        persisted_meta = (
            dict(persisted_state.get("notebooklm"))
            if isinstance(persisted_state.get("notebooklm"), dict)
            else {}
        )
        if (
            notebook_meta is not None
            and persisted_meta
            and str(persisted_meta.get("notebook_id", "")).strip()
            == str(notebook_meta.get("notebook_id", "")).strip()
        ):
            state["notebooklm"] = persisted_meta
        write_orchestrator_state(state_paths["orchestrator"], state)
        if iterations_done < max_iterations:
            time.sleep(max(1, poll_seconds))

    final_counts = _queue_counts(state_paths["queue"], state_paths["queue_lock"])
    summary = {
        "run_id": run_id,
        "mission_name": mission_name,
        "iterations_completed": iterations_done,
        "total_tasks_enqueued": total_tasks,
        "models": {
            "quant_thinker": thinker_role["model"],
            "coder": coder_role["model"],
        },
        "queue_counts": final_counts,
        "stop_reason": stop_reason,
        "notebooklm": notebook_meta,
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
