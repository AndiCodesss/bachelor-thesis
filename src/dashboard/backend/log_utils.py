from __future__ import annotations

import json
import math
from pathlib import Path
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
        result = float(value)
        return result if math.isfinite(result) else None
    try:
        result = float(str(value))
    except Exception:
        return None
    return result if math.isfinite(result) else None


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

__all__ = [
    "optional_float",
    "optional_int",
    "recent_json_events",
    "recent_task_snapshot",
]
