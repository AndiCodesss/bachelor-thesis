from __future__ import annotations

from research.lib.mission_splits import resolve_research_splits


def test_resolve_research_splits_defaults_search_to_train() -> None:
    out = resolve_research_splits({})
    assert out["search_split"] == "train"
    assert out["selection_split"] is None
    assert out["feedback_split"] == "train"
