from __future__ import annotations

import re
from typing import Any

_SUPPORTED_EXPECTED_SIDES = {"long", "short", "both"}
_MIN_TEXT_LENGTHS = {
    "event": 24,
    "mechanism": 20,
    "expected_regime": 20,
    "macro_location": 20,
    "micro_trigger": 20,
    "post_cost_rationale": 28,
    "falsification": 48,
    "novelty_vs_recent_failures": 28,
}
_MAX_TEXT_LENGTHS = {
    "event": 220,
    "mechanism": 180,
    "expected_regime": 220,
    "macro_location": 220,
    "micro_trigger": 220,
    "post_cost_rationale": 320,
    "falsification": 420,
    "novelty_vs_recent_failures": 320,
}
_TAUTOLOGICAL_FALSIFICATION_PATTERNS = (
    "avg_trade_pnl",
    "average trade pnl",
    "pnl is negative",
    "pnl turns negative",
    "not profitable",
    "does not work",
    "if it loses money",
    "if the strategy loses",
    "if sharpe",
    "if there is no edge",
    "if the edge is not positive",
    "if performance is poor",
)


class ThinkerResearchContractError(ValueError):
    def __init__(self, message: str, *, brief: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.brief = dict(brief or {})


def _slug(value: str, *, max_len: int = 72) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    if not slug:
        return "unknown_mechanism"
    return slug[:max_len].rstrip("_") or "unknown_mechanism"


def _clip_text(value: Any, *, max_len: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip()


def _feature_names(entry_conditions: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in entry_conditions:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature", "")).strip()
        if feature:
            names.append(feature)
    return names


def _error(
    message: str,
    *,
    raw_brief: dict[str, Any] | None,
) -> ThinkerResearchContractError:
    return ThinkerResearchContractError(message, brief=_sanitize_raw_brief(raw_brief))


def _sanitize_raw_brief(raw_brief: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw_brief, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "event",
        "mechanism",
        "expected_side",
        "expected_horizon_bars",
        "expected_regime",
        "macro_location",
        "micro_trigger",
        "post_cost_rationale",
        "falsification",
        "novelty_vs_recent_failures",
    ):
        value = raw_brief.get(key)
        if value is None:
            continue
        out[key] = value
    return out


def _normalized_text_field(
    raw_brief: dict[str, Any],
    *,
    field: str,
) -> str:
    text = _clip_text(raw_brief.get(field, ""), max_len=_MAX_TEXT_LENGTHS[field])
    min_len = _MIN_TEXT_LENGTHS[field]
    if len(text) < min_len:
        raise _error(
            f"research_brief.{field} must be specific and at least {min_len} characters",
            raw_brief=raw_brief,
        )
    return text


def _normalize_expected_side(raw_brief: dict[str, Any]) -> str:
    value = str(raw_brief.get("expected_side", "")).strip().lower()
    if value not in _SUPPORTED_EXPECTED_SIDES:
        allowed = ", ".join(sorted(_SUPPORTED_EXPECTED_SIDES))
        raise _error(
            f"research_brief.expected_side must be one of {allowed}",
            raw_brief=raw_brief,
        )
    return value


def _normalize_expected_horizon(
    raw_brief: dict[str, Any],
    *,
    allowed_horizons: list[int],
) -> int:
    try:
        horizon = int(raw_brief.get("expected_horizon_bars"))
    except (TypeError, ValueError) as exc:
        raise _error(
            "research_brief.expected_horizon_bars must be an integer from the mission horizon set",
            raw_brief=raw_brief,
        ) from exc

    if horizon <= 0:
        raise _error(
            "research_brief.expected_horizon_bars must be > 0",
            raw_brief=raw_brief,
        )
    if allowed_horizons and horizon not in allowed_horizons:
        raise _error(
            "research_brief.expected_horizon_bars must be one of "
            + ", ".join(str(value) for value in allowed_horizons),
            raw_brief=raw_brief,
        )
    return horizon


def _falsification_mentions_feature(
    falsification: str,
    *,
    entry_conditions: list[dict[str, Any]],
) -> bool:
    lower_text = falsification.lower()
    for feature in _feature_names(entry_conditions):
        if re.search(rf"\b{re.escape(feature.lower())}\b", lower_text):
            return True
    return False


def _is_tautological_falsification(text: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in _TAUTOLOGICAL_FALSIFICATION_PATTERNS)


def normalize_research_brief(
    raw_brief: Any,
    *,
    entry_conditions: list[dict[str, Any]],
    allowed_horizons: list[int],
) -> dict[str, Any]:
    if not isinstance(raw_brief, dict):
        raise ThinkerResearchContractError("research_brief must be an object", brief={})

    event = _normalized_text_field(raw_brief, field="event")
    mechanism = _normalized_text_field(raw_brief, field="mechanism")
    expected_regime = _normalized_text_field(raw_brief, field="expected_regime")
    macro_location = _normalized_text_field(raw_brief, field="macro_location")
    micro_trigger = _normalized_text_field(raw_brief, field="micro_trigger")
    post_cost_rationale = _normalized_text_field(raw_brief, field="post_cost_rationale")
    falsification = _normalized_text_field(raw_brief, field="falsification")
    novelty = _normalized_text_field(raw_brief, field="novelty_vs_recent_failures")
    expected_side = _normalize_expected_side(raw_brief)
    expected_horizon_bars = _normalize_expected_horizon(
        raw_brief,
        allowed_horizons=sorted({int(value) for value in allowed_horizons if int(value) > 0}),
    )

    if not _falsification_mentions_feature(falsification, entry_conditions=entry_conditions):
        feature_blob = ", ".join(_feature_names(entry_conditions)[:4]) or "an entry condition feature"
        raise _error(
            "research_brief.falsification must reference at least one entry condition feature by name "
            f"(for example: {feature_blob})",
            raw_brief=raw_brief,
        )
    if _is_tautological_falsification(falsification):
        raise _error(
            "research_brief.falsification cannot be tautological or phrased only as poor backtest performance",
            raw_brief=raw_brief,
        )

    return {
        "event": event,
        "mechanism": mechanism,
        "mechanism_key": _slug(mechanism),
        "expected_side": expected_side,
        "expected_horizon_bars": expected_horizon_bars,
        "expected_regime": expected_regime,
        "macro_location": macro_location,
        "micro_trigger": micro_trigger,
        "post_cost_rationale": post_cost_rationale,
        "falsification": falsification,
        "novelty_vs_recent_failures": novelty,
    }


__all__ = [
    "ThinkerResearchContractError",
    "normalize_research_brief",
]
