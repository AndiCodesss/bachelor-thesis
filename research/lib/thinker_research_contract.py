from __future__ import annotations

import re
from typing import Any

_SUPPORTED_EXPECTED_SIDES = {"long", "short", "both"}
_MIN_TEXT_LENGTHS = {
    "event": 24,
    "mechanism": 20,
    "market_regime": 20,
    "structural_location": 20,
    "micro_trigger": 20,
    "post_cost_rationale": 28,
    "falsification": 48,
    "novelty_vs_recent_failures": 28,
}
_MAX_TEXT_LENGTHS = {
    "event": 220,
    "mechanism": 180,
    "market_regime": 220,
    "structural_location": 220,
    "micro_trigger": 220,
    "post_cost_rationale": 320,
    "falsification": 420,
    "novelty_vs_recent_failures": 320,
}
_LEGACY_FIELD_ALIASES = {
    "market_regime": ("expected_regime",),
    "structural_location": ("macro_location",),
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
        "market_regime",
        "structural_location",
        "micro_trigger",
        "post_cost_rationale",
        "falsification",
        "novelty_vs_recent_failures",
    ):
        value = _raw_field_value(raw_brief, field=key)
        if value is None:
            continue
        out[key] = value
    return out


def _raw_field_value(raw_brief: dict[str, Any], *, field: str) -> Any:
    if field in raw_brief:
        direct = raw_brief.get(field)
        if direct is not None and str(direct).strip():
            return direct
    for alias in _LEGACY_FIELD_ALIASES.get(field, ()):
        if alias in raw_brief:
            alias_value = raw_brief.get(alias)
            if alias_value is not None and str(alias_value).strip():
                return alias_value
    return None


def _normalized_text_field(
    raw_brief: dict[str, Any],
    *,
    field: str,
) -> str:
    text = _clip_text(_raw_field_value(raw_brief, field=field), max_len=_MAX_TEXT_LENGTHS[field])
    min_len = _MIN_TEXT_LENGTHS[field]
    if len(text) < min_len:
        raise _error(
            f"research_brief.{field} must be specific and at least {min_len} characters",
            raw_brief=raw_brief,
        )
    return text


def _normalized_text_field_with_fallback(
    raw_brief: dict[str, Any],
    *,
    field: str,
    fallback_text: str,
) -> str:
    text = _clip_text(_raw_field_value(raw_brief, field=field), max_len=_MAX_TEXT_LENGTHS[field])
    min_len = _MIN_TEXT_LENGTHS[field]
    if len(text) >= min_len:
        return text
    repaired = _clip_text(fallback_text, max_len=_MAX_TEXT_LENGTHS[field])
    if len(repaired) >= min_len:
        return repaired
    raise _error(
        f"research_brief.{field} must be specific and at least {min_len} characters",
        raw_brief=raw_brief,
    )


def _format_threshold(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6g}"


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


def _condition_failure_clause(
    condition: dict[str, Any],
    *,
    params_template: dict[str, Any],
) -> str:
    feature = str(condition.get("feature", "")).strip() or "entry_feature"
    op = str(condition.get("op", "")).strip()
    if op in {">", ">=", "<", "<="}:
        param_key = str(condition.get("param_key", "")).strip()
        threshold = _format_threshold(params_template.get(param_key))
        direction = {
            ">": "stay above",
            ">=": "stay at or above",
            "<": "stay below",
            "<=": "stay at or below",
        }.get(op, "satisfy")
        if threshold:
            return f"{feature} does not {direction} {threshold}"
        return f"{feature} no longer satisfies its required {op} gate"
    if op == "between":
        low = _format_threshold(params_template.get(str(condition.get("param_key_low", "")).strip()))
        high = _format_threshold(params_template.get(str(condition.get("param_key_high", "")).strip()))
        if low and high:
            return f"{feature} leaves the {low} to {high} band"
        return f"{feature} leaves its required band"
    if op == "bool_true":
        return f"{feature} is not true on the trigger bar"
    if op == "bool_false":
        return f"{feature} turns true on the trigger bar"
    return f"{feature} no longer supports the setup"


def _join_clauses(clauses: list[str]) -> str:
    items = [str(item).strip() for item in clauses if str(item).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return f"{'; '.join(items[:-1])}; or {items[-1]}"


def _synthesized_falsification(
    raw_brief: dict[str, Any],
    *,
    entry_conditions: list[dict[str, Any]],
    params_template: dict[str, Any],
) -> str:
    clauses = [
        _condition_failure_clause(condition, params_template=params_template)
        for condition in entry_conditions[:3]
        if isinstance(condition, dict)
    ]
    clause_blob = _join_clauses(clauses)
    mechanism = _clip_text(_raw_field_value(raw_brief, field="mechanism"), max_len=96)
    thesis_label = f"{mechanism} thesis" if mechanism else "setup thesis"
    if clause_blob:
        return (
            f"If {clause_blob} on the entry bars, the {thesis_label} is wrong "
            "and the trade should be skipped."
        )
    return "If the required entry features do not confirm on the entry bars, the setup thesis is wrong."


def _synthesized_post_cost_rationale(raw_brief: dict[str, Any]) -> str:
    mechanism = _clip_text(_raw_field_value(raw_brief, field="mechanism"), max_len=96)
    micro_trigger = _clip_text(_raw_field_value(raw_brief, field="micro_trigger"), max_len=120)
    try:
        horizon = int(raw_brief.get("expected_horizon_bars"))
    except (TypeError, ValueError):
        horizon = 0
    horizon_blob = f"{horizon}-bar" if horizon > 0 else "short-horizon"
    if mechanism and micro_trigger:
        return (
            f"The setup is sparse and event-driven: once {micro_trigger} confirms the {mechanism} thesis, "
            f"the expected {horizon_blob} move can still outrun one-turn costs."
        )
    if mechanism:
        return (
            f"The setup is selective enough that the {mechanism} thesis should only trigger when the move "
            f"has enough urgency to clear one-turn costs over a {horizon_blob} holding window."
        )
    return "The setup is selective and event-driven enough that a short burst can still clear one-turn costs."


def _synthesized_novelty(raw_brief: dict[str, Any], *, entry_conditions: list[dict[str, Any]]) -> str:
    mechanism = _clip_text(_raw_field_value(raw_brief, field="mechanism"), max_len=96)
    structural_location = _clip_text(_raw_field_value(raw_brief, field="structural_location"), max_len=120)
    micro_trigger = _clip_text(_raw_field_value(raw_brief, field="micro_trigger"), max_len=120)
    feature_blob = ", ".join(_feature_names(entry_conditions)[:3])
    if mechanism and structural_location and feature_blob:
        return (
            "Unlike recent broad threshold ideas, this setup is specific to "
            f"{structural_location} and the {mechanism} mechanism, with explicit confirmation from {feature_blob}."
        )
    if mechanism and micro_trigger:
        return (
            "This is narrower than a generic threshold stack because it requires "
            f"{micro_trigger} to express the {mechanism} mechanism."
        )
    return "This is narrower than recent broad threshold ideas because it requires a specific location plus trigger."


def _normalize_falsification_field(
    raw_brief: dict[str, Any],
    *,
    entry_conditions: list[dict[str, Any]],
    params_template: dict[str, Any],
) -> str:
    falsification = _clip_text(_raw_field_value(raw_brief, field="falsification"), max_len=_MAX_TEXT_LENGTHS["falsification"])
    min_len = _MIN_TEXT_LENGTHS["falsification"]
    if (
        len(falsification) >= min_len
        and _falsification_mentions_feature(falsification, entry_conditions=entry_conditions)
        and not _is_tautological_falsification(falsification)
    ):
        return falsification

    repaired = _clip_text(
        _synthesized_falsification(
            raw_brief,
            entry_conditions=entry_conditions,
            params_template=params_template,
        ),
        max_len=_MAX_TEXT_LENGTHS["falsification"],
    )
    if (
        len(repaired) >= min_len
        and _falsification_mentions_feature(repaired, entry_conditions=entry_conditions)
        and not _is_tautological_falsification(repaired)
    ):
        return repaired

    if len(falsification) < min_len:
        raise _error(
            f"research_brief.falsification must be specific and at least {min_len} characters",
            raw_brief=raw_brief,
        )
    if not _falsification_mentions_feature(falsification, entry_conditions=entry_conditions):
        feature_blob = ", ".join(_feature_names(entry_conditions)[:4]) or "an entry condition feature"
        raise _error(
            "research_brief.falsification must reference at least one entry condition feature by name "
            f"(for example: {feature_blob})",
            raw_brief=raw_brief,
        )
    raise _error(
        "research_brief.falsification cannot be tautological or phrased only as poor backtest performance",
        raw_brief=raw_brief,
    )


def normalize_research_brief(
    raw_brief: Any,
    *,
    entry_conditions: list[dict[str, Any]],
    allowed_horizons: list[int],
    params_template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_brief, dict):
        raise ThinkerResearchContractError("research_brief must be an object", brief={})

    event = _normalized_text_field(raw_brief, field="event")
    mechanism = _normalized_text_field(raw_brief, field="mechanism")
    market_regime = _normalized_text_field(raw_brief, field="market_regime")
    structural_location = _normalized_text_field(raw_brief, field="structural_location")
    micro_trigger = _normalized_text_field(raw_brief, field="micro_trigger")
    post_cost_rationale = _normalized_text_field_with_fallback(
        raw_brief,
        field="post_cost_rationale",
        fallback_text=_synthesized_post_cost_rationale(raw_brief),
    )
    falsification = _normalize_falsification_field(
        raw_brief,
        entry_conditions=entry_conditions,
        params_template=dict(params_template or {}),
    )
    novelty = _normalized_text_field_with_fallback(
        raw_brief,
        field="novelty_vs_recent_failures",
        fallback_text=_synthesized_novelty(raw_brief, entry_conditions=entry_conditions),
    )
    expected_side = _normalize_expected_side(raw_brief)
    expected_horizon_bars = _normalize_expected_horizon(
        raw_brief,
        allowed_horizons=sorted({int(value) for value in allowed_horizons if int(value) > 0}),
    )

    return {
        "event": event,
        "mechanism": mechanism,
        "mechanism_key": _slug(mechanism),
        "expected_side": expected_side,
        "expected_horizon_bars": expected_horizon_bars,
        "market_regime": market_regime,
        "structural_location": structural_location,
        "micro_trigger": micro_trigger,
        "post_cost_rationale": post_cost_rationale,
        "falsification": falsification,
        "novelty_vs_recent_failures": novelty,
    }


__all__ = [
    "ThinkerResearchContractError",
    "normalize_research_brief",
]
