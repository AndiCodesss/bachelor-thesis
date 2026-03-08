"""NotebookLM query guidance helpers."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from typing import Any, Iterator


ENV_NOTEBOOK_RESEARCH_GUIDANCE = "NOTEBOOK_RESEARCH_GUIDANCE"
ENV_NOTEBOOK_QUERY_BUDGET = "NOTEBOOK_QUERY_BUDGET"


@contextmanager
def notebook_research_guidance_context(guidance: str | None) -> Iterator[None]:
    previous = os.environ.get(ENV_NOTEBOOK_RESEARCH_GUIDANCE)
    try:
        value = str(guidance or "").strip()
        if value:
            os.environ[ENV_NOTEBOOK_RESEARCH_GUIDANCE] = value
        else:
            os.environ.pop(ENV_NOTEBOOK_RESEARCH_GUIDANCE, None)
        yield
    finally:
        if previous is None:
            os.environ.pop(ENV_NOTEBOOK_RESEARCH_GUIDANCE, None)
        else:
            os.environ[ENV_NOTEBOOK_RESEARCH_GUIDANCE] = previous


def load_notebook_research_guidance() -> str:
    return str(os.getenv(ENV_NOTEBOOK_RESEARCH_GUIDANCE, "")).strip()


def normalize_notebook_query_budget(value: Any) -> dict[str, int]:
    payload = dict(value) if isinstance(value, dict) else {}
    return {
        "max_total_queries": max(0, int(payload.get("max_total_queries", 3) or 0)),
        "max_research_queries": max(0, int(payload.get("max_research_queries", 1) or 0)),
        "max_deep_research_queries": max(0, int(payload.get("max_deep_research_queries", 0) or 0)),
    }


@contextmanager
def notebook_query_budget_context(budget: dict[str, Any] | None) -> Iterator[None]:
    previous = os.environ.get(ENV_NOTEBOOK_QUERY_BUDGET)
    try:
        if budget:
            os.environ[ENV_NOTEBOOK_QUERY_BUDGET] = json.dumps(
                normalize_notebook_query_budget(budget),
                sort_keys=True,
            )
        else:
            os.environ.pop(ENV_NOTEBOOK_QUERY_BUDGET, None)
        yield
    finally:
        if previous is None:
            os.environ.pop(ENV_NOTEBOOK_QUERY_BUDGET, None)
        else:
            os.environ[ENV_NOTEBOOK_QUERY_BUDGET] = previous


def load_notebook_query_budget() -> dict[str, int]:
    raw = str(os.getenv(ENV_NOTEBOOK_QUERY_BUDGET, "")).strip()
    if not raw:
        return normalize_notebook_query_budget({})
    try:
        payload = json.loads(raw)
    except Exception:
        return normalize_notebook_query_budget({})
    return normalize_notebook_query_budget(payload)


__all__ = [
    "ENV_NOTEBOOK_RESEARCH_GUIDANCE",
    "ENV_NOTEBOOK_QUERY_BUDGET",
    "load_notebook_research_guidance",
    "load_notebook_query_budget",
    "normalize_notebook_query_budget",
    "notebook_query_budget_context",
    "notebook_research_guidance_context",
]
