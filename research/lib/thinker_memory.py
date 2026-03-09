from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import portalocker

from research.lib.atomic_io import atomic_json_write


def empty_thinker_memory(*, lane_id: str | None, window_size: int = 3) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "lane_id": lane_id,
        "window_size": max(1, int(window_size)),
        "recent_attempts": [],
    }


def read_thinker_memory(
    *,
    path: Path,
    lock_path: Path,
    lane_id: str | None,
    window_size: int = 3,
) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return empty_thinker_memory(lane_id=lane_id, window_size=window_size)
    with portalocker.Lock(lock_path, mode="a", timeout=5):
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return empty_thinker_memory(lane_id=lane_id, window_size=window_size)
    if not isinstance(payload, dict):
        return empty_thinker_memory(lane_id=lane_id, window_size=window_size)
    attempts = payload.get("recent_attempts")
    if not isinstance(attempts, list):
        payload["recent_attempts"] = []
    payload["lane_id"] = payload.get("lane_id", lane_id)
    payload["window_size"] = max(1, int(payload.get("window_size", window_size)))
    return payload


def append_thinker_attempt(
    *,
    path: Path,
    lock_path: Path,
    lane_id: str | None,
    attempt: dict[str, Any],
    window_size: int = 3,
) -> dict[str, Any]:
    payload = read_thinker_memory(
        path=path,
        lock_path=lock_path,
        lane_id=lane_id,
        window_size=window_size,
    )
    attempts = list(payload.get("recent_attempts", []))
    attempts.append(_sanitize_for_storage(dict(attempt)))
    payload["recent_attempts"] = attempts[-max(1, int(payload.get("window_size", window_size))):]
    with portalocker.Lock(lock_path, mode="a", timeout=5):
        atomic_json_write(path, payload)
    return payload


def _sanitize_for_storage(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.startswith("_"):
                continue
            out[key_text] = _sanitize_for_storage(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_storage(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_thinker_memory_context(
    payload: dict[str, Any],
    *,
    max_attempts: int = 3,
) -> str:
    attempts = payload.get("recent_attempts")
    if not isinstance(attempts, list) or not attempts:
        return ""

    lines = ["RECENT_ATTEMPT_MEMORY (last 3 iterations, this lane):"]
    for row in reversed(attempts[-max_attempts:]):
        if not isinstance(row, dict):
            continue
        iteration = int(row.get("iteration", 0) or 0)
        theme_tag = str(row.get("theme_tag", "unknown")).strip() or "unknown"
        status_label = str(row.get("status_label", "UNKNOWN")).strip() or "UNKNOWN"
        summary = str(row.get("summary", "")).strip() or "no summary"
        lines.append(f"[iter {iteration}] {theme_tag} -> {status_label}: {summary}")

        conditions = row.get("highlighted_conditions")
        if isinstance(conditions, list) and conditions:
            label = str(row.get("conditions_label", "Blocking")).strip() or "Blocking"
            formatted = []
            for condition in conditions[:2]:
                if not isinstance(condition, dict):
                    continue
                formatted.append(
                    f"{condition.get('column')} {condition.get('operator')} {condition.get('threshold')} "
                    f"({float(condition.get('pass_rate_pct', 0.0)):.1f}% pass)"
                )
            if formatted:
                lines.append(f"  {label}: " + ", ".join(formatted))

        params = row.get("offending_params")
        if isinstance(params, dict) and params:
            blob = ", ".join(f"{key}={value}" for key, value in list(params.items())[:3])
            lines.append(f"  Params: {blob}")

    return "\n".join(lines)


__all__ = [
    "append_thinker_attempt",
    "empty_thinker_memory",
    "format_thinker_memory_context",
    "read_thinker_memory",
]
