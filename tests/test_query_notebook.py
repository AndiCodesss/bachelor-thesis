from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "query_notebook.py"
    spec = importlib.util.spec_from_file_location("query_notebook_module", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_query_budget_error_blocks_second_research_query(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "load_notebook_query_budget",
        lambda: {
            "max_total_queries": 3,
            "max_research_queries": 1,
            "max_deep_research_queries": 0,
        },
    )
    monkeypatch.setattr(
        mod,
        "audit_context_from_env",
        lambda: {
            "run_id": "run_1",
            "iteration": 2,
            "stage": "quant_thinker",
            "lane_id": "A",
        },
    )
    monkeypatch.setattr(
        mod,
        "summarize_notebook_queries",
        lambda **_: {
            "query_count": 1,
            "mode_counts": {"plain": 0, "research": 1, "deep_research": 0},
        },
    )

    error = mod._query_budget_error("research")
    assert error is not None
    assert "research budget exhausted" in error


def test_query_budget_error_blocks_when_total_budget_spent(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "load_notebook_query_budget",
        lambda: {
            "max_total_queries": 3,
            "max_research_queries": 1,
            "max_deep_research_queries": 0,
        },
    )
    monkeypatch.setattr(
        mod,
        "audit_context_from_env",
        lambda: {
            "run_id": "run_1",
            "iteration": 2,
            "stage": "quant_thinker",
            "lane_id": "A",
        },
    )
    monkeypatch.setattr(
        mod,
        "summarize_notebook_queries",
        lambda **_: {
            "query_count": 3,
            "mode_counts": {"plain": 2, "research": 1, "deep_research": 0},
        },
    )

    error = mod._query_budget_error("plain")
    assert error is not None
    assert "budget exhausted" in error


def test_query_budget_error_disables_deep_research(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "load_notebook_query_budget",
        lambda: {
            "max_total_queries": 3,
            "max_research_queries": 1,
            "max_deep_research_queries": 0,
        },
    )
    monkeypatch.setattr(
        mod,
        "audit_context_from_env",
        lambda: {
            "run_id": "run_1",
            "iteration": 2,
            "stage": "quant_thinker",
            "lane_id": "A",
        },
    )
    monkeypatch.setattr(
        mod,
        "summarize_notebook_queries",
        lambda **_: {
            "query_count": 0,
            "mode_counts": {"plain": 0, "research": 0, "deep_research": 0},
        },
    )

    error = mod._query_budget_error("deep_research")
    assert error == "deep research is disabled for this autonomy iteration"
