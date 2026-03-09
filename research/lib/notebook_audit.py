"""NotebookLM query audit helpers for local research runs."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import portalocker

from research.lib.experiments import log_experiment


DEFAULT_AUDIT_PATH = Path("results/logs/notebook_queries.jsonl")
DEFAULT_AUDIT_LOCK_PATH = Path("results/logs/notebook_queries.lock")

ENV_RUN_ID = "NOTEBOOK_AUDIT_RUN_ID"
ENV_ITERATION = "NOTEBOOK_AUDIT_ITERATION"
ENV_STAGE = "NOTEBOOK_AUDIT_STAGE"
ENV_LANE_ID = "NOTEBOOK_AUDIT_LANE_ID"
ENV_AGENT = "NOTEBOOK_AUDIT_AGENT"
ENV_ORCHESTRATOR_STATE_PATH = "NOTEBOOK_AUDIT_ORCHESTRATOR_STATE_PATH"


def _question_hash(question: str) -> str:
    return hashlib.sha256(str(question).encode("utf-8")).hexdigest()


def audit_context_from_env() -> dict[str, Any]:
    """Read the current NotebookLM audit context from environment variables."""
    iteration_raw = os.getenv(ENV_ITERATION)
    iteration: int | None
    try:
        iteration = int(iteration_raw) if iteration_raw is not None else None
    except (TypeError, ValueError):
        iteration = None
    return {
        "run_id": os.getenv(ENV_RUN_ID) or None,
        "iteration": iteration,
        "stage": os.getenv(ENV_STAGE) or None,
        "lane_id": os.getenv(ENV_LANE_ID) or None,
        "agent": os.getenv(ENV_AGENT) or None,
        "orchestrator_state_path": os.getenv(ENV_ORCHESTRATOR_STATE_PATH) or None,
    }


@contextmanager
def notebook_audit_context(
    *,
    run_id: str,
    iteration: int,
    stage: str,
    lane_id: str | None = None,
    agent: str = "llm_orchestrator",
    orchestrator_state_path: str | None = None,
) -> Iterator[None]:
    """Temporarily expose audit metadata to child NotebookLM query processes."""
    updates = {
        ENV_RUN_ID: str(run_id),
        ENV_ITERATION: str(int(iteration)),
        ENV_STAGE: str(stage),
        ENV_LANE_ID: str(lane_id) if lane_id is not None else None,
        ENV_AGENT: str(agent),
        ENV_ORCHESTRATOR_STATE_PATH: str(orchestrator_state_path) if orchestrator_state_path else None,
    }
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def log_notebook_query(
    *,
    notebook_id: str,
    mode: str,
    question: str,
    status: str,
    duration_seconds: float,
    answer_chars: int = 0,
    discovered_sources: int = 0,
    imported_sources: int = 0,
    fallback_to_plain: bool = False,
    error: str | None = None,
    audit_path: Path | str = DEFAULT_AUDIT_PATH,
    lock_path: Path | str = DEFAULT_AUDIT_LOCK_PATH,
) -> bool:
    """Append one NotebookLM audit row."""
    row = {
        "event": "notebook_query",
        "notebook_id": str(notebook_id).strip(),
        "mode": str(mode).strip() or "plain",
        "status": str(status).strip() or "unknown",
        "duration_seconds": float(max(0.0, duration_seconds)),
        "answer_chars": int(max(0, answer_chars)),
        "discovered_sources": int(max(0, discovered_sources)),
        "imported_sources": int(max(0, imported_sources)),
        "fallback_to_plain": bool(fallback_to_plain),
        "question_hash": _question_hash(question),
        "question_preview": str(question).strip()[:180],
    }
    row.update(audit_context_from_env())
    if error:
        row["error"] = str(error)[:400]
    return log_experiment(
        row,
        experiments_path=audit_path,
        lock_path=lock_path,
        dedupe_window=1,
    )


def _read_recent_rows(audit_path: Path, lock_path: Path, limit: int) -> list[dict[str, Any]]:
    if not audit_path.exists():
        return []
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(lock_path, mode="a", timeout=5):
        with open(audit_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    rows: list[dict[str, Any]] = []
    for raw in lines[-max(1, int(limit)) :]:
        try:
            row = json.loads(raw)
        except Exception:
            continue
        if isinstance(row, dict) and str(row.get("event", "")) == "notebook_query":
            rows.append(row)
    return rows


def summarize_notebook_queries(
    *,
    run_id: str,
    iteration: int,
    stage: str,
    lane_id: str | None = None,
    audit_path: Path | str = DEFAULT_AUDIT_PATH,
    lock_path: Path | str = DEFAULT_AUDIT_LOCK_PATH,
    limit: int = 500,
) -> dict[str, Any]:
    """Summarize NotebookLM queries for one orchestrator iteration/stage."""
    path = Path(audit_path)
    lock = Path(lock_path)
    rows = _read_recent_rows(path, lock, limit=max(1, int(limit)))
    matched: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("run_id", "")) != str(run_id):
            continue
        try:
            row_iteration = int(row.get("iteration"))
        except (TypeError, ValueError):
            continue
        if row_iteration != int(iteration):
            continue
        if str(row.get("stage", "")) != str(stage):
            continue
        if lane_id is not None and str(row.get("lane_id", "")) != str(lane_id):
            continue
        matched.append(row)

    mode_counts = {"plain": 0, "research": 0, "deep_research": 0}
    non_fallback_mode_counts = {"plain": 0, "research": 0, "deep_research": 0}
    question_previews: list[str] = []
    discovered_sources = 0
    imported_sources = 0
    error_count = 0
    fallback_count = 0
    for row in matched:
        mode = str(row.get("mode", "")).strip().lower()
        if mode in mode_counts:
            mode_counts[mode] += 1
            if (
                str(row.get("status", "")).strip().lower() == "success"
                and not bool(row.get("fallback_to_plain", False))
            ):
                non_fallback_mode_counts[mode] += 1
        discovered_sources += int(row.get("discovered_sources", 0) or 0)
        imported_sources += int(row.get("imported_sources", 0) or 0)
        if str(row.get("status", "")).strip().lower() != "success":
            error_count += 1
        if bool(row.get("fallback_to_plain", False)):
            fallback_count += 1
        preview = str(row.get("question_preview", "")).strip()
        if preview and preview not in question_previews:
            question_previews.append(preview)

    modes_used = [mode for mode, count in mode_counts.items() if count > 0]
    return {
        "used": bool(matched),
        "query_count": len(matched),
        "success_count": len(matched) - error_count,
        "error_count": error_count,
        "modes_used": modes_used,
        "mode_counts": mode_counts,
        "non_fallback_mode_counts": non_fallback_mode_counts,
        "fallback_count": fallback_count,
        "discovered_sources": discovered_sources,
        "imported_sources": imported_sources,
        "question_previews": question_previews[:3],
    }


__all__ = [
    "DEFAULT_AUDIT_LOCK_PATH",
    "DEFAULT_AUDIT_PATH",
    "audit_context_from_env",
    "log_notebook_query",
    "notebook_audit_context",
    "summarize_notebook_queries",
]
