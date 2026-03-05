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
    Discovered sources are permanently imported into the notebook, enriching it
    for all future queries. Use when the plain query returns a shallow answer.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

_RESEARCH_POLL_INTERVAL = 5    # seconds between status polls
_FAST_RESEARCH_TIMEOUT = 300   # seconds max for fast mode (5 min)
_DEEP_RESEARCH_TIMEOUT = 900   # seconds max for deep mode (15 min)


def _notebook_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


async def _ask(notebook_id: str, question: str) -> str:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        result = await client.chat.ask(notebook_id, question)
        return result.answer


async def _research_and_ask(notebook_id: str, question: str, *, mode: str) -> str:
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
            return result.answer

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
            return result.answer

        # 3. Import all found sources (filter URL-less entries; deep research may produce some)
        sources = [s for s in status.get("sources", []) if s.get("url")]
        if sources:
            task_id = status.get("task_id") or task.get("task_id")
            imported = await client.research.import_sources(notebook_id, task_id, sources)
            print(f"[NotebookLM: imported {len(imported)} new source(s) into notebook]",
                  file=sys.stderr)
        else:
            print("[NotebookLM: no new sources found to import]", file=sys.stderr)

        # 4. Ask now that sources are enriched
        result = await client.chat.ask(notebook_id, question)
        return result.answer


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
        help="Run thorough deep web research before answering (slower, ~5 min)",
    )
    args = parser.parse_args()

    notebook_id = _notebook_id_from_url(args.notebook_url)

    try:
        if args.research:
            answer = asyncio.run(_research_and_ask(notebook_id, args.question, mode="fast"))
        elif args.deep_research:
            answer = asyncio.run(_research_and_ask(notebook_id, args.question, mode="deep"))
        else:
            answer = asyncio.run(_ask(notebook_id, args.question))
        print(answer)
    except FileNotFoundError:
        print(
            "[NotebookLM: auth not found. Run 'uv run notebooklm login' to authenticate.]",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"[NotebookLM query failed: {e}]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
