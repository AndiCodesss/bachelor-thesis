from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from research.lib.script_support import tail_lines


def recent_json_events(
    path: Path,
    event_names: set[str],
    *,
    limit: int,
    scan_lines: int = 3000,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in reversed(tail_lines(path, scan_lines)):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict) and str(row.get("event", "")) in event_names:
            out.append(row)
            if len(out) >= limit:
                break
    return list(reversed(out))


def optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return None


def optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def recent_task_snapshot(row: dict[str, Any]) -> dict[str, object]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    search_result = row.get("search_result") if isinstance(row.get("search_result"), dict) else {}
    edge_surface = row.get("edge_surface") if isinstance(row.get("edge_surface"), dict) else {}
    if not edge_surface and isinstance(search_result.get("edge_surface"), dict):
        edge_surface = search_result.get("edge_surface")
    global_probe = (
        edge_surface.get("global_probe")
        if isinstance(edge_surface.get("global_probe"), dict)
        else {}
    )

    return {
        "strategy": str(row.get("strategy_name", "")),
        "bar": str(row.get("bar_config", "")),
        "timestamp": str(row.get("timestamp", "")),
        "verdict": str(row.get("verdict", "UNKNOWN")),
        "failure_code": str(row.get("failure_code", "")),
        "signal_count": optional_int(search_result.get("signal_count")),
        "edge_events": optional_int(global_probe.get("base_event_count")) if global_probe else None,
        "edge_status": str(edge_surface.get("status", "")) if edge_surface else "",
        "best_horizon_bars": optional_int(global_probe.get("best_horizon_bars")) if global_probe else None,
        "best_avg_trade_pnl": optional_float(global_probe.get("best_avg_trade_pnl")) if global_probe else None,
        "backtest_trades": optional_int(metrics.get("trade_count")),
        "net_pnl": optional_float(metrics.get("net_pnl")),
        "sharpe": optional_float(metrics.get("sharpe_ratio")),
    }


def summarize_tool_input(tool_name: str, input_data: dict[str, Any]) -> str:
    if tool_name == "Grep":
        parts: list[str] = []
        if "pattern" in input_data:
            parts.append(f"pattern={str(input_data['pattern'])[:60]}")
        if "path" in input_data:
            parts.append(f"path={Path(str(input_data['path'])).name}")
        return " ".join(parts) or str(input_data)[:100]
    if tool_name in ("Read", "Write", "Edit"):
        path = input_data.get("file_path", "")
        return Path(str(path)).name if path else str(input_data)[:100]
    if tool_name == "Bash":
        return str(input_data.get("command", ""))[:100]
    if tool_name == "Glob":
        return str(input_data.get("pattern", ""))[:100]
    return str(input_data)[:100]


def parse_thinker_events(jsonl_path: Path, *, max_events: int = 40) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    pending_tools: dict[str, str] = {}

    lines = tail_lines(jsonl_path, max(200, max_events * 5))
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        msg_type = obj.get("type")
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        if msg_type == "assistant":
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        events.append({"type": "text", "content": text[:800]})
                elif item_type == "tool_use":
                    tool_name = str(item.get("name", ""))
                    tool_id = str(item.get("id", ""))
                    raw_input = item.get("input", {})
                    summary = summarize_tool_input(
                        tool_name,
                        raw_input if isinstance(raw_input, dict) else {},
                    )
                    if tool_id:
                        pending_tools[tool_id] = tool_name
                    events.append({"type": "tool_call", "tool": tool_name, "summary": summary})
        elif msg_type == "user":
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "tool_result":
                    continue
                tool_id = str(item.get("tool_use_id", ""))
                tool_name = pending_tools.get(tool_id, "")
                result_content = item.get("content", "")
                if isinstance(result_content, list):
                    text_parts = [
                        chunk.get("text", "")
                        for chunk in result_content
                        if isinstance(chunk, dict) and chunk.get("type") == "text"
                    ]
                    result_content = " ".join(text_parts)
                summary = str(result_content)[:200].replace("\n", " ").strip()
                events.append({"type": "tool_result", "tool": tool_name, "summary": summary})

    return events[-max_events:]


def get_sessions_dir(project_root: Path) -> Path:
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return projects_dir / "unknown"
    candidates = [d for d in projects_dir.iterdir() if d.is_dir() and project_root.name in d.name]
    if candidates:
        return max(candidates, key=lambda d: len(d.name))
    slug = str(project_root).replace("/", "-").replace(" ", "-").lstrip("-")
    return projects_dir / slug


def find_thinker_session_file(project_root: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None

    thinker_pids: list[str] = []
    for line in result.stdout.splitlines():
        if "claude -p " in line:
            parts = line.split()
            if parts:
                thinker_pids.append(parts[1])

    sessions_dir = get_sessions_dir(project_root)
    for pid in thinker_pids:
        try:
            fd_dir = Path(f"/proc/{pid}/fd")
            for fd in fd_dir.iterdir():
                try:
                    target = fd.resolve()
                except (PermissionError, OSError):
                    continue
                if target.suffix == ".jsonl" and str(sessions_dir) in str(target) and target.exists():
                    return target
        except (PermissionError, OSError):
            continue
    return None


def find_fallback_session_file(project_root: Path) -> Path | None:
    sessions_dir = get_sessions_dir(project_root)
    if not sessions_dir.exists():
        return None
    files = list(sessions_dir.glob("*.jsonl"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def session_file_last_updated(session_file: Path) -> str | None:
    try:
        mtime = session_file.stat().st_mtime
    except Exception:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


__all__ = [
    "find_fallback_session_file",
    "find_thinker_session_file",
    "optional_float",
    "optional_int",
    "parse_thinker_events",
    "recent_json_events",
    "recent_task_snapshot",
    "session_file_last_updated",
    "summarize_tool_input",
]
