from __future__ import annotations

from research.lib.notebook_guidance import (
    ENV_NOTEBOOK_RESEARCH_GUIDANCE,
    load_notebook_research_guidance,
    notebook_research_guidance_context,
)


def test_notebook_research_guidance_context_restores_environment(monkeypatch):
    monkeypatch.delenv(ENV_NOTEBOOK_RESEARCH_GUIDANCE, raising=False)
    with notebook_research_guidance_context("Use high-quality trusted sources."):
        assert load_notebook_research_guidance() == "Use high-quality trusted sources."
    assert load_notebook_research_guidance() == ""
