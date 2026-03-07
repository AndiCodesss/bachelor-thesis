"""NotebookLM query guidance helpers."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Iterator


ENV_NOTEBOOK_RESEARCH_GUIDANCE = "NOTEBOOK_RESEARCH_GUIDANCE"


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


__all__ = [
    "ENV_NOTEBOOK_RESEARCH_GUIDANCE",
    "load_notebook_research_guidance",
    "notebook_research_guidance_context",
]
