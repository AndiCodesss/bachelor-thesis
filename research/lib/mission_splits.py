"""Mission split resolution for research discovery workflows."""

from __future__ import annotations

from typing import Any


ALLOWED_RESEARCH_SPLITS = frozenset({"train", "validate"})
ALLOWED_PROMOTION_SPLITS = frozenset({"test"})


def _normalize_split(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    return raw or None


def resolve_research_splits(
    mission: dict[str, Any],
    *,
    default_search_split: str = "validate",
) -> dict[str, str | None]:
    """Resolve search/selection split semantics for research mode.

    Legacy missions may still provide ``splits_allowed``. In that case the first
    entry is treated as the search split and no separate selection split is used.
    """
    if not isinstance(mission, dict):
        raise TypeError("mission must be a dict")

    raw_search = mission.get("search_split")
    if raw_search is None:
        legacy = mission.get("splits_allowed", [default_search_split])
        if isinstance(legacy, list):
            legacy_norm = {_normalize_split(v) for v in legacy if _normalize_split(v)}
            if "test" in legacy_norm:
                raise ValueError(
                    "splits_allowed may not include 'test' in research mode; "
                    "use promotion_split: test instead",
                )
        if isinstance(legacy, list) and legacy:
            raw_search = legacy[0]
        else:
            raw_search = default_search_split

    search_split = _normalize_split(raw_search)
    if search_split not in ALLOWED_RESEARCH_SPLITS:
        allowed = ", ".join(sorted(ALLOWED_RESEARCH_SPLITS))
        raise ValueError(
            f"research search_split must be one of [{allowed}], got {search_split!r}",
        )

    selection_split = _normalize_split(mission.get("selection_split"))
    if selection_split is not None:
        if selection_split not in ALLOWED_RESEARCH_SPLITS:
            allowed = ", ".join(sorted(ALLOWED_RESEARCH_SPLITS))
            raise ValueError(
                f"research selection_split must be one of [{allowed}], got {selection_split!r}",
            )
        if selection_split == search_split:
            raise ValueError("selection_split must differ from search_split")

    promotion_split = _normalize_split(mission.get("promotion_split"))
    if promotion_split is not None and promotion_split not in ALLOWED_PROMOTION_SPLITS:
        allowed = ", ".join(sorted(ALLOWED_PROMOTION_SPLITS))
        raise ValueError(
            f"promotion_split must be one of [{allowed}], got {promotion_split!r}",
        )

    return {
        "search_split": search_split,
        "selection_split": selection_split,
        # Discovery feedback must stay on the search split to avoid selection leakage.
        "feedback_split": search_split,
        "promotion_split": promotion_split,
    }
