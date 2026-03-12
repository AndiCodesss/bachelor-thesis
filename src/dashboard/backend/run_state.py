from __future__ import annotations

from typing import Any

MAX_RUN_LOG_LINES = 4000

RunRecord = dict[str, Any]


def new_run_record(*, run_type: str, cmd: str) -> RunRecord:
    return {
        "logs": [],
        "status": "running",
        "cmd": cmd,
        "type": run_type,
    }


def append_run_log(run: RunRecord, line: str) -> None:
    logs = run.setdefault("logs", [])
    logs.append(line)
    overflow = len(logs) - MAX_RUN_LOG_LINES
    if overflow > 0:
        del logs[:overflow]


def set_run_status(run: RunRecord, status: str) -> None:
    run["status"] = status


def summarize_run(run_id: str, run: RunRecord) -> dict[str, Any]:
    return {
        "id": run_id,
        "type": run["type"],
        "status": run["status"],
        "cmd": run["cmd"],
        "log_count": len(run.get("logs", [])),
    }


__all__ = [
    "MAX_RUN_LOG_LINES",
    "RunRecord",
    "append_run_log",
    "new_run_record",
    "set_run_status",
    "summarize_run",
]
