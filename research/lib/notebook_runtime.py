"""NotebookLM runtime helpers for lane-local autonomy notebooks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any


_NOTEBOOK_URL_PREFIX = "https://notebooklm.google.com/notebook/"
_RESEARCH_POLL_INTERVAL_SECONDS = 5
_FAST_RESEARCH_TIMEOUT_SECONDS = 300
_DEEP_RESEARCH_TIMEOUT_SECONDS = 900


def notebook_url_from_id(notebook_id: str) -> str:
    notebook_id = str(notebook_id).strip()
    if not notebook_id:
        raise ValueError("notebook_id is required")
    return f"{_NOTEBOOK_URL_PREFIX}{notebook_id}"


def notebook_id_from_url(url: str) -> str:
    value = str(url or "").strip().rstrip("/")
    if not value:
        raise ValueError("notebook URL is required")
    notebook_id = value.rsplit("/", 1)[-1]
    if not notebook_id:
        raise ValueError(f"Invalid NotebookLM URL: {url}")
    return notebook_id


def resolve_notebooklm_config(mission: dict[str, Any]) -> dict[str, Any]:
    raw = mission.get("notebooklm")
    if isinstance(raw, dict):
        cfg = dict(raw)
        mode = str(cfg.get("mode", "shared")).strip().lower() or "shared"
        if mode not in {"shared", "lane_fresh"}:
            raise ValueError(f"Unsupported notebooklm.mode '{mode}'")
        notebook_url = str(
            cfg.get("notebook_url") or mission.get("notebooklm_notebook_url", ""),
        ).strip()
        bootstrap_queries = _clean_text_list(cfg.get("bootstrap_queries") or [])
        return {
            "enabled": bool(notebook_url) or mode == "lane_fresh",
            "mode": mode,
            "required": bool(cfg.get("required", mode == "lane_fresh")),
            "notebook_url": notebook_url,
            "title_prefix": str(cfg.get("title_prefix", mission.get("mission_name", "NotebookLM"))).strip()
            or str(mission.get("mission_name", "NotebookLM")).strip()
            or "NotebookLM",
            "bootstrap_queries": bootstrap_queries,
            "bootstrap_mode": _normalize_seed_mode(cfg.get("bootstrap_mode", "deep")),
            "require_research_on_fresh": bool(cfg.get("require_research_on_fresh", mode == "lane_fresh")),
            "min_bootstrap_successes": max(0, int(cfg.get("min_bootstrap_successes", 0))),
        }

    notebook_url = str(mission.get("notebooklm_notebook_url", "")).strip()
    return {
        "enabled": bool(notebook_url),
        "mode": "shared",
        "required": False,
        "notebook_url": notebook_url,
        "title_prefix": str(mission.get("mission_name", "NotebookLM")).strip() or "NotebookLM",
        "bootstrap_queries": [],
        "bootstrap_mode": "deep",
        "require_research_on_fresh": False,
        "min_bootstrap_successes": 0,
    }


def ensure_lane_notebook(
    *,
    mission: dict[str, Any],
    lane_id: str | None,
    state_payload: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    cfg = resolve_notebooklm_config(mission)
    result = {
        "configured": bool(cfg.get("enabled", False)),
        "mode": str(cfg.get("mode", "shared")),
        "notebook": None,
        "mission_overrides": {},
    }
    if not cfg.get("enabled", False):
        return result

    if cfg["mode"] == "shared":
        notebook_url = str(cfg.get("notebook_url", "")).strip()
        if not notebook_url:
            if cfg.get("required", False):
                raise RuntimeError("NotebookLM is required but no shared notebook URL is configured")
            return result
        notebook_meta = {
            "configured": True,
            "mode": "shared",
            "notebook_id": notebook_id_from_url(notebook_url),
            "notebook_url": notebook_url,
            "lane_id": lane_id,
            "fresh": False,
            "seeded": True,
        }
        result["notebook"] = notebook_meta
        result["mission_overrides"] = {
            "notebooklm_notebook_url": notebook_url,
            "lane_notebook_requires_research": False,
        }
        return result

    existing = state_payload.get("notebooklm") if isinstance(state_payload.get("notebooklm"), dict) else None
    if existing:
        existing_id = str(existing.get("notebook_id", "")).strip()
        existing_url = str(existing.get("notebook_url", "")).strip()
        if existing_id and existing_url and str(existing.get("mode", "")) == "lane_fresh":
            notebook_meta = dict(existing)
            notebook_meta["fresh"] = False
            result["notebook"] = notebook_meta
            result["mission_overrides"] = {
                "notebooklm_notebook_url": existing_url,
                "lane_notebook_requires_research": bool(
                    cfg.get("require_research_on_fresh", True) and not bool(existing.get("seeded", False)),
                ),
            }
            return result

    notebook_meta = _create_and_optionally_seed_lane_notebook(
        title=_build_lane_notebook_title(str(cfg.get("title_prefix", "NotebookLM")), lane_id=lane_id),
        lane_id=lane_id,
        run_id=run_id,
        bootstrap_queries=list(cfg.get("bootstrap_queries", [])),
        bootstrap_mode=str(cfg.get("bootstrap_mode", "deep")),
        min_bootstrap_successes=int(cfg.get("min_bootstrap_successes", 0)),
        required=bool(cfg.get("required", False)),
    )
    result["notebook"] = notebook_meta
    result["mission_overrides"] = {
        "notebooklm_notebook_url": str(notebook_meta["notebook_url"]),
        "lane_notebook_requires_research": bool(cfg.get("require_research_on_fresh", True) and not notebook_meta.get("seeded", False)),
    }
    return result


def _build_lane_notebook_title(title_prefix: str, *, lane_id: str | None) -> str:
    prefix = str(title_prefix or "NotebookLM").strip() or "NotebookLM"
    lane_suffix = f" lane {lane_id}" if lane_id else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d %H%M%S UTC")
    return f"{prefix}{lane_suffix} {stamp}".strip()


def _create_and_optionally_seed_lane_notebook(
    *,
    title: str,
    lane_id: str | None,
    run_id: str,
    bootstrap_queries: list[str],
    bootstrap_mode: str,
    min_bootstrap_successes: int,
    required: bool,
) -> dict[str, Any]:
    return asyncio.run(
        _create_and_optionally_seed_lane_notebook_async(
            title=title,
            lane_id=lane_id,
            run_id=run_id,
            bootstrap_queries=bootstrap_queries,
            bootstrap_mode=bootstrap_mode,
            min_bootstrap_successes=min_bootstrap_successes,
            required=required,
        ),
    )


async def _create_and_optionally_seed_lane_notebook_async(
    *,
    title: str,
    lane_id: str | None,
    run_id: str,
    bootstrap_queries: list[str],
    bootstrap_mode: str,
    min_bootstrap_successes: int,
    required: bool,
) -> dict[str, Any]:
    from notebooklm import NotebookLMClient

    async with await NotebookLMClient.from_storage() as client:
        notebook = await client.notebooks.create(title)
        notebook_id = str(notebook.id)
        notebook_url = notebook_url_from_id(notebook_id)
        bootstrap_results: list[dict[str, Any]] = []
        bootstrap_successes = 0

        for query in bootstrap_queries:
            row = {
                "query": str(query),
                "mode": bootstrap_mode,
                "status": "pending",
                "imported_sources": 0,
            }
            try:
                imported = await _seed_research_query(
                    client=client,
                    notebook_id=notebook_id,
                    question=str(query),
                    mode=bootstrap_mode,
                )
                row["status"] = "success"
                row["imported_sources"] = int(imported)
                bootstrap_successes += 1
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            bootstrap_results.append(row)

    if required and bootstrap_successes < max(0, int(min_bootstrap_successes)):
        raise RuntimeError(
            "NotebookLM bootstrap did not complete enough successful bootstrap queries "
            f"({bootstrap_successes} < {max(0, int(min_bootstrap_successes))})",
        )

    imported_sources = sum(int(row.get("imported_sources", 0) or 0) for row in bootstrap_results)
    return {
        "configured": True,
        "mode": "lane_fresh",
        "run_id": str(run_id),
        "lane_id": lane_id,
        "title": title,
        "notebook_id": notebook_id,
        "notebook_url": notebook_url,
        "fresh": True,
        "seeded": bootstrap_successes > 0,
        "bootstrap_mode": bootstrap_mode,
        "bootstrap_queries": list(bootstrap_queries),
        "bootstrap_results": bootstrap_results,
        "bootstrap_successes": bootstrap_successes,
        "imported_sources": imported_sources,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


async def _seed_research_query(
    *,
    client: Any,
    notebook_id: str,
    question: str,
    mode: str,
) -> int:
    mode = _normalize_seed_mode(mode)
    timeout = (
        _FAST_RESEARCH_TIMEOUT_SECONDS
        if mode == "fast"
        else _DEEP_RESEARCH_TIMEOUT_SECONDS
    )
    task = await client.research.start(notebook_id, question, source="web", mode=mode)
    if not task:
        raise RuntimeError("research failed to start")

    deadline = asyncio.get_running_loop().time() + timeout
    status: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(_RESEARCH_POLL_INTERVAL_SECONDS)
        status = await client.research.poll(notebook_id)
        state = str(status.get("status", "")).strip().lower()
        if state == "completed":
            break
        if state == "no_research":
            raise RuntimeError("research failed to initialize")
    else:
        raise TimeoutError(f"research timed out after {timeout} seconds")

    sources = [row for row in status.get("sources", []) if row.get("url")]
    if not sources:
        return 0
    task_id = status.get("task_id") or task.get("task_id")
    if not task_id:
        raise RuntimeError("research completed without task_id")
    imported = await client.research.import_sources(notebook_id, task_id, sources)
    return len(imported)


def _normalize_seed_mode(value: Any) -> str:
    mode = str(value or "deep").strip().lower()
    if mode not in {"fast", "deep"}:
        raise ValueError(f"Unsupported NotebookLM bootstrap_mode '{value}'")
    return mode


def _clean_text_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            out.append(text)
    return out


__all__ = [
    "ensure_lane_notebook",
    "notebook_id_from_url",
    "notebook_url_from_id",
    "resolve_notebooklm_config",
]
