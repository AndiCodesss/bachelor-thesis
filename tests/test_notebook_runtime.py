from __future__ import annotations

from research.lib.notebook_runtime import (
    ensure_lane_notebook,
    notebook_id_from_url,
    notebook_url_from_id,
    resolve_notebooklm_config,
)


def _mission() -> dict:
    return {
        "mission_name": "alpha-discovery",
        "objective": "Search train split for robust, event-driven NQ scalp candidates",
        "bar_configs": ["tick_610", "time_1m"],
        "session_filter": "rth",
        "notebooklm": {
            "mode": "lane_fresh",
            "required": True,
            "title_prefix": "NQ Alpha Discovery",
            "require_research_on_fresh": True,
            "fresh_query_mode": "research",
            "min_fresh_imports": 1,
            "iteration_query_budget": {
                "max_total_queries": 3,
                "max_research_queries": 1,
                "max_deep_research_queries": 0,
            },
            "research_guidance": "Use high-quality trusted sources.",
        },
    }


def test_notebook_url_roundtrip():
    notebook_id = "abc123"
    url = notebook_url_from_id(notebook_id)
    assert url.endswith("/abc123")
    assert notebook_id_from_url(url) == notebook_id


def test_resolve_notebooklm_config_supports_lane_fresh_mode():
    cfg = resolve_notebooklm_config(_mission())
    assert cfg["enabled"] is True
    assert cfg["mode"] == "lane_fresh"
    assert cfg["required"] is True
    assert cfg["require_research_on_fresh"] is True
    assert cfg["bootstrap_queries"] == []
    assert cfg["fresh_query_mode"] == "research"
    assert cfg["min_fresh_imports"] == 1
    assert cfg["iteration_query_budget"] == {
        "max_total_queries": 3,
        "max_research_queries": 1,
        "max_deep_research_queries": 0,
    }
    assert cfg["research_guidance"] == "Use high-quality trusted sources."


def test_ensure_lane_notebook_reuses_existing_lane_notebook_without_network_calls():
    mission = _mission()
    state = {
        "notebooklm": {
            "mode": "lane_fresh",
            "notebook_id": "nb_lane_a",
            "notebook_url": "https://notebooklm.google.com/notebook/nb_lane_a",
            "seeded": False,
        },
    }
    out = ensure_lane_notebook(
        mission=mission,
        lane_id="A",
        state_payload=state,
        run_id="run_1",
    )
    assert out["notebook"]["notebook_id"] == "nb_lane_a"
    assert out["notebook"]["fresh"] is False
    assert out["mission_overrides"]["notebooklm_notebook_url"].endswith("nb_lane_a")
    assert out["mission_overrides"]["lane_notebook_requires_research"] is True
    assert out["mission_overrides"]["lane_notebook_seed_requirements"]["preferred_mode"] == "research"
    assert out["mission_overrides"]["lane_notebook_seed_requirements"]["accepted_modes"] == ["research"]
    assert out["mission_overrides"]["lane_notebook_seed_requirements"]["min_imported_sources"] == 1
    assert out["mission_overrides"]["lane_notebook_query_budget"] == {
        "max_total_queries": 3,
        "max_research_queries": 1,
        "max_deep_research_queries": 0,
    }
    assert out["mission_overrides"]["notebook_research_guidance"] == "Use high-quality trusted sources."
