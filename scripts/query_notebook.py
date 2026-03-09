#!/usr/bin/env python3
"""Query a NotebookLM notebook and print the answer to stdout.

Modes:
    Default  — query existing sources (fast)
    --research — web search → import top sources → ask (self-enriching)
    --deep-research — thorough web search → import top sources → ask

Usage:
    uv run python scripts/query_notebook.py --notebook-url URL "question"
    uv run python scripts/query_notebook.py --notebook-url URL --research "question"
    uv run python scripts/query_notebook.py --notebook-url URL --deep-research "question"

Authentication:
    Run 'uv run notebooklm login' once to store credentials in
    ~/.notebooklm/storage_state.json. All subsequent queries use those cookies.

Note on --research / --deep-research:
    URL-backed discovered sources are permanently imported into the notebook,
    enriching it for all future queries. Research runs can be guided with an
    environment-provided source-quality instruction.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.lib.notebook_audit import audit_context_from_env, summarize_notebook_queries
from research.lib.notebook_audit import log_notebook_query
from research.lib.notebook_guidance import (
    load_notebook_query_budget,
    load_notebook_research_guidance,
)
from research.lib.runtime_state import read_orchestrator_state, write_orchestrator_state

_RESEARCH_POLL_INTERVAL = 5    # seconds between status polls
_FAST_RESEARCH_TIMEOUT = 300   # seconds max for fast mode (5 min)
_DEEP_RESEARCH_TIMEOUT = 900   # seconds max for deep mode (15 min)


def _query_budget_error(mode: str) -> str | None:
    budget = load_notebook_query_budget()
    context = audit_context_from_env()
    run_id = context.get("run_id")
    iteration = context.get("iteration")
    stage = context.get("stage")
    if not run_id or iteration is None or not stage:
        return None

    summary = summarize_notebook_queries(
        run_id=str(run_id),
        iteration=int(iteration),
        stage=str(stage),
        lane_id=context.get("lane_id"),
    )
    max_total = int(budget.get("max_total_queries", 0) or 0)
    if max_total >= 0 and int(summary.get("query_count", 0) or 0) >= max_total:
        return (
            "iteration notebook query budget exhausted "
            f"({summary.get('query_count', 0)}/{max_total} total queries already used)"
        )

    mode_counts = summary.get("mode_counts") if isinstance(summary.get("mode_counts"), dict) else {}
    if mode == "research":
        max_research = int(budget.get("max_research_queries", 0) or 0)
        if int(mode_counts.get("research", 0) or 0) >= max_research:
            return (
                "iteration notebook research budget exhausted "
                f"({mode_counts.get('research', 0)}/{max_research} --research queries already used)"
            )
    if mode == "deep_research":
        max_deep = int(budget.get("max_deep_research_queries", 0) or 0)
        if int(mode_counts.get("deep_research", 0) or 0) >= max_deep:
            return (
                "deep research is disabled for this autonomy iteration"
                if max_deep == 0
                else (
                    "iteration notebook deep-research budget exhausted "
                    f"({mode_counts.get('deep_research', 0)}/{max_deep} --deep-research queries already used)"
                )
            )
    return None


def _persist_import_progress_if_applicable(
    *,
    notebook_id: str,
    mode: str,
    imported_sources: int,
    fallback_to_plain: bool,
) -> None:
    if fallback_to_plain:
        return
    if imported_sources <= 0:
        return

    context = audit_context_from_env()
    state_path = str(context.get("orchestrator_state_path") or "").strip()
    if not state_path:
        return

    payload = read_orchestrator_state(Path(state_path))
    notebook_meta = payload.get("notebooklm")
    if not isinstance(notebook_meta, dict):
        return
    if str(notebook_meta.get("notebook_id", "")).strip() != str(notebook_id).strip():
        return

    notebook_meta["imported_sources"] = int(notebook_meta.get("imported_sources", 0) or 0) + int(imported_sources)
    seed_modes_used = list(notebook_meta.get("seed_modes_used", []) or [])
    notebook_meta["seed_query_count"] = int(notebook_meta.get("seed_query_count", 0) or 0) + 1
    if mode not in seed_modes_used:
        seed_modes_used.append(mode)
    notebook_meta["seed_modes_used"] = seed_modes_used
    notebook_meta["seeded"] = True
    notebook_meta["fresh"] = False

    payload["notebooklm"] = notebook_meta
    write_orchestrator_state(Path(state_path), payload)


def _notebook_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


async def _ask(notebook_id: str, question: str) -> tuple[str, dict[str, object]]:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        result = await client.chat.ask(notebook_id, question)
        return result.answer, {
            "discovered_sources": 0,
            "imported_sources": 0,
            "fallback_to_plain": False,
        }


async def _research_and_ask(
    notebook_id: str,
    question: str,
    *,
    mode: str,
) -> tuple[str, dict[str, object]]:
    """Search the web for new sources, import them, then ask the question."""
    from notebooklm import NotebookLMClient

    timeout = _FAST_RESEARCH_TIMEOUT if mode == "fast" else _DEEP_RESEARCH_TIMEOUT

    async with await NotebookLMClient.from_storage() as client:
        # 1. Start research
        print(f"[NotebookLM: starting {mode} web research...]", file=sys.stderr)
        research_guidance = load_notebook_research_guidance()
        research_question = question
        if research_guidance:
            research_question = (
                f"{research_guidance}\n\n"
                f"Research question: {question}"
            )
        task = await client.research.start(notebook_id, research_question, source="web", mode=mode)
        if not task:
            print("[NotebookLM: research failed to start, falling back to plain query]",
                  file=sys.stderr)
            result = await client.chat.ask(notebook_id, question)
            return result.answer, {
                "discovered_sources": 0,
                "imported_sources": 0,
                "fallback_to_plain": True,
            }

        # 2. Poll until completed
        deadline = time.monotonic() + timeout
        status = {}
        while time.monotonic() < deadline:
            await asyncio.sleep(_RESEARCH_POLL_INTERVAL)
            status = await client.research.poll(notebook_id)
            if status.get("status") == "completed":
                break
            print("[NotebookLM: research in progress...]", file=sys.stderr)
        else:
            print(f"[NotebookLM: research timed out after {timeout}s, querying existing sources]",
                  file=sys.stderr)
            result = await client.chat.ask(notebook_id, question)
            return result.answer, {
                "discovered_sources": 0,
                "imported_sources": 0,
                "fallback_to_plain": True,
            }

        # 3. Import all discovered URL-backed sources
        sources = [s for s in status.get("sources", []) if s.get("url")]
        if sources:
            task_id = status.get("task_id") or task.get("task_id")
            imported = await client.research.import_sources(notebook_id, task_id, sources)
            print(f"[NotebookLM: imported {len(imported)} new source(s) into notebook]",
                  file=sys.stderr)
            imported_count = len(imported)
        else:
            print("[NotebookLM: no new sources found to import]", file=sys.stderr)
            imported_count = 0

        # 4. Ask now that sources are enriched
        result = await client.chat.ask(notebook_id, question)
        return result.answer, {
            "discovered_sources": len(sources),
            "imported_sources": int(imported_count),
            "fallback_to_plain": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a NotebookLM notebook.")
    parser.add_argument("question", help="Question to ask the notebook")
    parser.add_argument("--notebook-url", required=True, help="NotebookLM notebook URL")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--research",
        action="store_true",
        help="Run fast web research to find and import new sources before answering",
    )
    mode_group.add_argument(
        "--deep-research",
        action="store_true",
        help="Run thorough deep web research before answering (slower, up to ~15 min)",
    )
    args = parser.parse_args()

    notebook_id = _notebook_id_from_url(args.notebook_url)
    mode = "plain"
    if args.research:
        mode = "research"
    elif args.deep_research:
        mode = "deep_research"
    budget_error = _query_budget_error(mode)
    if budget_error:
        print(
            f"[NotebookLM query blocked: {budget_error}. Use existing notebook context and finalize the hypothesis.]",
            file=sys.stderr,
        )
        sys.exit(2)
    start = time.monotonic()
    try:
        if args.research:
            answer, meta = asyncio.run(
                _research_and_ask(
                    notebook_id,
                    args.question,
                    mode="fast",
                ),
            )
        elif args.deep_research:
            answer, meta = asyncio.run(
                _research_and_ask(
                    notebook_id,
                    args.question,
                    mode="deep",
                ),
            )
        else:
            answer, meta = asyncio.run(_ask(notebook_id, args.question))
        try:
            log_notebook_query(
                notebook_id=notebook_id,
                mode=mode,
                question=args.question,
                status="success",
                duration_seconds=time.monotonic() - start,
                answer_chars=len(answer),
                discovered_sources=int(meta.get("discovered_sources", 0)),
                imported_sources=int(meta.get("imported_sources", 0)),
                fallback_to_plain=bool(meta.get("fallback_to_plain", False)),
            )
        except Exception:
            pass
        try:
            _persist_import_progress_if_applicable(
                notebook_id=notebook_id,
                mode=mode,
                imported_sources=int(meta.get("imported_sources", 0) or 0),
                fallback_to_plain=bool(meta.get("fallback_to_plain", False)),
            )
        except Exception:
            pass
        print(answer)
    except FileNotFoundError:
        try:
            log_notebook_query(
                notebook_id=notebook_id,
                mode=mode,
                question=args.question,
                status="error",
                duration_seconds=time.monotonic() - start,
                error="auth_not_found",
            )
        except Exception:
            pass
        print(
            "[NotebookLM: auth not found. Run 'uv run notebooklm login' to authenticate.]",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        try:
            log_notebook_query(
                notebook_id=notebook_id,
                mode=mode,
                question=args.question,
                status="error",
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )
        except Exception:
            pass
        print(f"[NotebookLM query failed: {e}]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
