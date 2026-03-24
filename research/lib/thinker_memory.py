from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import portalocker

from research.lib.atomic_io import atomic_json_write


def empty_thinker_memory(*, lane_id: str | None, window_size: int = 3) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
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
        mechanism = str(row.get("mechanism", "")).strip()
        status_label = str(row.get("status_label", "UNKNOWN")).strip() or "UNKNOWN"
        summary = str(row.get("summary", "")).strip() or "no summary"
        if len(mechanism) > 96:
            mechanism = mechanism[:93].rstrip() + "..."
        header = f"[iter {iteration}] theme={theme_tag}"
        if mechanism:
            header += f" | mechanism={mechanism}"
        lines.append(f"{header} -> {status_label}: {summary}")

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

        conflicts = row.get("cross_sample_conflicts")
        if isinstance(conflicts, list):
            for conflict in conflicts[:2]:
                text = str(conflict).strip()
                if text:
                    lines.append(f"  Cross-sample: {text}")

    cooldowns = derive_primary_feature_cooldowns(payload)
    if cooldowns:
        lines.append("RECENT_PRIMARY_FEATURE_COOLDOWN:")
        for item in cooldowns[:3]:
            feature = str(item.get("feature", "")).strip() or str(item.get("family_key", "")).strip()
            reason = str(item.get("reason", "")).strip() or "recent repeated blocker"
            lines.append(
                f"  Avoid reusing `{feature}` as a primary gate next iteration unless the setup is materially different "
                f"({reason})."
            )

    return "\n".join(lines)


def primary_feature_cooldown_key(feature: str) -> str:
    name = str(feature or "").strip().lower()
    if not name:
        return ""
    if name in {"prev_day_va_position", "position_in_va", "rolling_va_position"} or "va_position" in name:
        return "value_area_position"
    if name.startswith("dist_prev_") or name in {"prev_day_poc", "prev_day_vah", "prev_day_val"}:
        return "previous_session_value_reference"
    if name in {"value_acceptance_rate_8", "va_rejection_score", "price_discovery_score", "failed_auction_score"}:
        return "auction_state_anchor"
    return name


def derive_primary_feature_cooldowns(
    payload: dict[str, Any],
    *,
    min_score: int = 2,
    max_items: int = 4,
) -> list[dict[str, Any]]:
    attempts = payload.get("recent_attempts")
    if not isinstance(attempts, list) or not attempts:
        return []

    scores: dict[str, int] = defaultdict(int)
    exemplars: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for row in attempts:
        if not isinstance(row, dict):
            continue
        highlighted = row.get("highlighted_conditions")
        if not isinstance(highlighted, list) or not highlighted:
            continue
        conflicts = [str(item).strip() for item in (row.get("cross_sample_conflicts") or []) if str(item).strip()]
        failure_type = str(row.get("failure_type", "")).strip().lower()
        for condition in highlighted:
            if not isinstance(condition, dict):
                continue
            role = str(condition.get("role", "primary")).strip().lower() or "primary"
            if role != "primary":
                continue
            feature = str(condition.get("column") or condition.get("feature") or "").strip()
            if not feature:
                continue
            family_key = primary_feature_cooldown_key(feature)
            if not family_key:
                continue
            score = 0
            if conflicts:
                score = 2
                reasons.setdefault(family_key, "cross-sample feasibility conflict")
            elif failure_type in {"zero_signal", "dead_feature_primary"}:
                severity = str(condition.get("severity", "")).strip().lower()
                pass_rate = float(condition.get("pass_rate_pct", 0.0) or 0.0)
                if severity == "blocks_all" or pass_rate <= 0.0:
                    score = 1
                    reasons.setdefault(family_key, "recent zero-signal blocker")
            if score <= 0:
                continue
            scores[family_key] += score
            exemplars.setdefault(family_key, feature)

    ranked: list[dict[str, Any]] = []
    for family_key, score in scores.items():
        if score < int(min_score):
            continue
        ranked.append(
            {
                "family_key": family_key,
                "feature": exemplars.get(family_key, family_key),
                "score": int(score),
                "reason": reasons.get(family_key, "recent repeated blocker"),
            }
        )
    ranked.sort(key=lambda row: (-int(row.get("score", 0)), str(row.get("feature", ""))))
    return ranked[:max(1, int(max_items))]


__all__ = [
    "append_thinker_attempt",
    "derive_primary_feature_cooldowns",
    "empty_thinker_memory",
    "format_thinker_memory_context",
    "primary_feature_cooldown_key",
    "read_thinker_memory",
]
