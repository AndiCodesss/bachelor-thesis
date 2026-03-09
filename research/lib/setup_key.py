"""Stable setup-key helpers for concrete entry-setup tracking."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _condition_signature(row: dict[str, Any]) -> dict[str, str]:
    feature = str(row.get("feature") or row.get("column") or "").strip()
    operator = str(row.get("op") or row.get("operator") or "").strip()
    role = str(row.get("role") or "").strip().lower() or "confirmation"
    kind = "compare"
    if operator == "between":
        kind = "between"
    elif operator in {"bool_true", "bool_false"}:
        kind = "boolean"
    return {
        "feature": feature,
        "operator": operator,
        "role": role,
        "kind": kind,
    }


def _exit_profile(params: dict[str, Any]) -> list[str]:
    keys = [
        "sl_ticks",
        "pt_ticks",
        "max_bars",
        "exit_bars",
        "profit_target",
        "stop_loss",
        "profit_target_return",
        "stop_loss_return",
        "min_bars_between",
    ]
    return [key for key in keys if key in params]


def _direction_logic_hash(entry_logic: Any) -> str:
    text = str(entry_logic or "").strip().lower()
    if not text:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def build_setup_signature(
    *,
    bar_config: str,
    theme_tag: str,
    entry_conditions: list[dict[str, Any]] | None,
    params: dict[str, Any] | None,
    entry_logic: Any = "",
) -> dict[str, Any]:
    conditions: list[dict[str, str]] = []
    if isinstance(entry_conditions, list):
        for raw in entry_conditions:
            if not isinstance(raw, dict):
                continue
            signature = _condition_signature(raw)
            if not signature["feature"] or not signature["operator"]:
                continue
            conditions.append(signature)
    conditions.sort(key=lambda item: (item["feature"], item["operator"], item["role"], item["kind"]))

    params_obj = dict(params) if isinstance(params, dict) else {}
    return {
        "bar_config": str(bar_config).strip(),
        "theme_tag": _slug(theme_tag),
        "conditions": conditions,
        "condition_count": len(conditions),
        "exit_profile": _exit_profile(params_obj),
        "direction_logic_hash": _direction_logic_hash(entry_logic),
    }


def describe_setup_signature(signature: dict[str, Any]) -> str:
    if not isinstance(signature, dict):
        return "unknown setup"
    bar_config = str(signature.get("bar_config", "")).strip() or "unknown_bar"
    conditions = signature.get("conditions") if isinstance(signature.get("conditions"), list) else []
    parts: list[str] = []
    for raw in conditions[:3]:
        if not isinstance(raw, dict):
            continue
        feature = str(raw.get("feature", "")).strip()
        operator = str(raw.get("operator", "")).strip()
        role = str(raw.get("role", "")).strip()
        if not feature or not operator:
            continue
        role_suffix = f" ({role})" if role else ""
        parts.append(f"{feature} {operator}{role_suffix}")
    if len(conditions) > 3:
        parts.append(f"+{len(conditions) - 3} more")
    condition_blob = ", ".join(parts) if parts else "no_conditions"
    exit_profile = signature.get("exit_profile") if isinstance(signature.get("exit_profile"), list) else []
    exit_blob = ",".join(str(v) for v in exit_profile[:3]) if exit_profile else "basic_exit"
    return f"{bar_config} | {condition_blob} | exit={exit_blob}"


def build_setup_key(
    *,
    bar_config: str,
    theme_tag: str,
    entry_conditions: list[dict[str, Any]] | None,
    params: dict[str, Any] | None,
    entry_logic: Any = "",
) -> tuple[str, str]:
    signature = build_setup_signature(
        bar_config=bar_config,
        theme_tag=theme_tag,
        entry_conditions=entry_conditions,
        params=params,
        entry_logic=entry_logic,
    )
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()[:16]
    return f"setup_{digest}", describe_setup_signature(signature)


def task_setup_identity(task: dict[str, Any]) -> tuple[str, str]:
    source = task.get("source") if isinstance(task.get("source"), dict) else {}
    setup_key = str(task.get("setup_key") or source.get("setup_key") or "").strip()
    setup_label = str(task.get("setup_label") or source.get("setup_label") or "").strip()
    if setup_key:
        return setup_key, (setup_label or setup_key)

    params = task.get("params") if isinstance(task.get("params"), dict) else {}
    bar_config = str(task.get("bar_config", "")).strip()
    strategy_name = str(task.get("strategy_name", "")).strip()
    digest = hashlib.sha256(
        json.dumps(
            {
                "bar_config": bar_config,
                "strategy_name": strategy_name,
                "param_keys": sorted(str(key) for key in params.keys()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    ).hexdigest()[:16]
    fallback_key = f"setup_{digest}"
    fallback_label = f"{bar_config or 'unknown_bar'} | {strategy_name or 'unknown_strategy'}"
    return fallback_key, fallback_label


__all__ = [
    "build_setup_key",
    "build_setup_signature",
    "describe_setup_signature",
    "task_setup_identity",
]
