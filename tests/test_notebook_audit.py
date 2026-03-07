from __future__ import annotations

import os
from pathlib import Path

from research.lib.notebook_audit import (
    audit_context_from_env,
    log_notebook_query,
    notebook_audit_context,
    summarize_notebook_queries,
)


def test_notebook_audit_context_restores_environment():
    original_run = os.environ.get("NOTEBOOK_AUDIT_RUN_ID")
    os.environ["NOTEBOOK_AUDIT_RUN_ID"] = "outer"
    try:
        with notebook_audit_context(
            run_id="run_123",
            iteration=7,
            stage="quant_thinker",
            lane_id="B",
        ):
            ctx = audit_context_from_env()
            assert ctx["run_id"] == "run_123"
            assert ctx["iteration"] == 7
            assert ctx["stage"] == "quant_thinker"
            assert ctx["lane_id"] == "B"
            assert ctx["agent"] == "llm_orchestrator"
        assert os.environ.get("NOTEBOOK_AUDIT_RUN_ID") == "outer"
    finally:
        if original_run is None:
            os.environ.pop("NOTEBOOK_AUDIT_RUN_ID", None)
        else:
            os.environ["NOTEBOOK_AUDIT_RUN_ID"] = original_run


def test_summarize_notebook_queries_filters_current_iteration(tmp_path: Path):
    audit_path = tmp_path / "notebook_queries.jsonl"
    lock_path = tmp_path / "notebook_queries.lock"

    with notebook_audit_context(run_id="run_a", iteration=3, stage="quant_thinker", lane_id="A"):
        log_notebook_query(
            notebook_id="nb1",
            mode="plain",
            question="What orderflow conditions matter?",
            status="success",
            duration_seconds=0.4,
            answer_chars=120,
            audit_path=audit_path,
            lock_path=lock_path,
        )
        log_notebook_query(
            notebook_id="nb1",
            mode="research",
            question="Find documented NQ scalp thresholds",
            status="success",
            duration_seconds=5.0,
            answer_chars=240,
            discovered_sources=5,
            approved_sources=3,
            imported_sources=3,
            rejected_sources=2,
            approved_domains=["cmegroup.com", "ssrn.com"],
            audit_path=audit_path,
            lock_path=lock_path,
        )

    with notebook_audit_context(run_id="run_a", iteration=4, stage="quant_thinker", lane_id="A"):
        log_notebook_query(
            notebook_id="nb1",
            mode="plain",
            question="This should not be counted",
            status="error",
            duration_seconds=0.2,
            error="timeout",
            audit_path=audit_path,
            lock_path=lock_path,
        )

    summary = summarize_notebook_queries(
        run_id="run_a",
        iteration=3,
        stage="quant_thinker",
        lane_id="A",
        audit_path=audit_path,
    )
    assert summary["used"] is True
    assert summary["query_count"] == 2
    assert summary["success_count"] == 2
    assert summary["error_count"] == 0
    assert summary["mode_counts"]["plain"] == 1
    assert summary["mode_counts"]["research"] == 1
    assert summary["non_fallback_mode_counts"]["research"] == 1
    assert summary["fallback_count"] == 0
    assert summary["discovered_sources"] == 5
    assert summary["approved_sources"] == 3
    assert summary["imported_sources"] == 3
    assert summary["rejected_sources"] == 2
    assert summary["approved_domains"] == ["cmegroup.com", "ssrn.com"]
    assert len(summary["question_previews"]) == 2
