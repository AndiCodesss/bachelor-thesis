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
    Approved discovered sources are permanently imported into the notebook,
    enriching it for all future queries. Low-signal domains and URLs are
    filtered through the active NotebookLM source policy before import.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.lib.notebook_audit import log_notebook_query
from research.lib.notebook_policy import filter_research_sources, load_source_policy_from_env

_RESEARCH_POLL_INTERVAL = 5    # seconds between status polls
_FAST_RESEARCH_TIMEOUT = 300   # seconds max for fast mode (5 min)
_DEEP_RESEARCH_TIMEOUT = 900   # seconds max for deep mode (15 min)


def _notebook_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


async def _ask(notebook_id: str, question: str) -> tuple[str, dict[str, object]]:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        result = await client.chat.ask(notebook_id, question)
        return result.answer, {
            "discovered_sources": 0,
            "approved_sources": 0,
            "imported_sources": 0,
            "approved_domains": [],
            "fallback_to_plain": False,
        }


async def _research_and_ask(
    notebook_id: str,
    question: str,
    *,
    mode: str,
    source_policy: dict[str, object],
) -> tuple[str, dict[str, object]]:
    """Search the web for new sources, import them, then ask the question."""
    from notebooklm import NotebookLMClient

    timeout = _FAST_RESEARCH_TIMEOUT if mode == "fast" else _DEEP_RESEARCH_TIMEOUT

    async with await NotebookLMClient.from_storage() as client:
        # 1. Start research
        print(f"[NotebookLM: starting {mode} web research...]", file=sys.stderr)
        task = await client.research.start(notebook_id, question, source="web", mode=mode)
        if not task:
            print("[NotebookLM: research failed to start, falling back to plain query]",
                  file=sys.stderr)
            result = await client.chat.ask(notebook_id, question)
            return result.answer, {
                "discovered_sources": 0,
                "approved_sources": 0,
                "imported_sources": 0,
                "approved_domains": [],
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
                "approved_sources": 0,
                "imported_sources": 0,
                "approved_domains": [],
                "fallback_to_plain": True,
            }

        # 3. Import only approved sources
        approved_sources, filter_stats = filter_research_sources(status.get("sources", []), source_policy)
        if approved_sources:
            task_id = status.get("task_id") or task.get("task_id")
            imported = await client.research.import_sources(notebook_id, task_id, approved_sources)
            print(f"[NotebookLM: imported {len(imported)} new source(s) into notebook]",
                  file=sys.stderr)
            imported_count = len(imported)
        else:
            print(
                "[NotebookLM: no approved sources found to import "
                f"(rejected={int(filter_stats.get('rejected_sources', 0))})]",
                file=sys.stderr,
            )
            imported_count = 0

        # 4. Ask now that sources are enriched
        result = await client.chat.ask(notebook_id, question)
        return result.answer, {
            "discovered_sources": int(filter_stats.get("discovered_sources", 0)),
            "approved_sources": int(filter_stats.get("approved_sources", 0)),
            "imported_sources": int(imported_count),
            "approved_domains": list(filter_stats.get("approved_domains", [])),
            "rejected_sources": int(filter_stats.get("rejected_sources", 0)),
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
    start = time.monotonic()
    source_policy = load_source_policy_from_env()

    try:
        if args.research:
            answer, meta = asyncio.run(
                _research_and_ask(
                    notebook_id,
                    args.question,
                    mode="fast",
                    source_policy=source_policy,
                ),
            )
        elif args.deep_research:
            answer, meta = asyncio.run(
                _research_and_ask(
                    notebook_id,
                    args.question,
                    mode="deep",
                    source_policy=source_policy,
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
                approved_sources=int(meta.get("approved_sources", 0)),
                imported_sources=int(meta.get("imported_sources", 0)),
                rejected_sources=int(meta.get("rejected_sources", 0)),
                approved_domains=list(meta.get("approved_domains", [])),
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
