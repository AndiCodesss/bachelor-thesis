from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from research.lib.coordination import read_json_file_if_exists
from research.lib.script_support import load_yaml_dict

from src.dashboard.backend.log_utils import recent_json_events, recent_task_snapshot


def load_mission_name(mission_ref: str, *, project_root: Path) -> str:
    mission_path = Path(mission_ref)
    if not mission_path.is_absolute():
        mission_path = project_root / mission_path
    mission_path = mission_path.resolve()
    if not mission_path.exists():
        return mission_path.stem
    try:
        payload = load_yaml_dict(mission_path)
    except Exception:
        return mission_path.stem
    return str(payload.get("mission_name", mission_path.stem))


def read_runtime_json(path: Path, default_payload: dict[str, Any]) -> dict[str, Any]:
    return read_json_file_if_exists(
        json_path=path,
        lock_path=path.with_suffix(".lock"),
        default_payload=default_payload,
    )


def collect_autonomy_status(*, project_root: Path) -> dict[str, Any]:
    state_dir = project_root / "research" / ".state"
    logs_dir = project_root / "results" / "logs"
    queue_path = state_dir / "experiment_queue.json"
    budget_path = state_dir / "mission_budget.json"
    exp_log = logs_dir / "research_experiments.jsonl"

    metrics: dict[str, Any] = {
        "queue": {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0},
        "budget": {"experiments_run": 0, "max_experiments": "n/a", "failures": {}},
        "financial": {
            "tested": 0,
            "avg_net_pnl": 0.0,
            "avg_sharpe": 0.0,
            "pass_rate_pct": 0.0,
            "best": None,
            "worst": None,
        },
        "active_hypotheses": [],
        "recent_results": [],
    }

    try:
        queue = read_runtime_json(queue_path, {"schema_version": "1.0", "tasks": []})
        tasks = queue.get("tasks", [])
        for task in tasks:
            state = task.get("state")
            if state in metrics["queue"]:
                metrics["queue"][state] += 1

        hypothesis_counts: dict[str, int] = {}
        for task in tasks:
            if task.get("state") not in {"pending", "in_progress"}:
                continue
            source = task.get("source")
            hypothesis_id = ""
            if isinstance(source, dict):
                hypothesis_id = str(source.get("hypothesis_id", "")).strip()
            if hypothesis_id:
                hypothesis_counts[hypothesis_id] = hypothesis_counts.get(hypothesis_id, 0) + 1

        metrics["active_hypotheses"] = [
            {"id": hypothesis_id, "tasks": count}
            for hypothesis_id, count in sorted(hypothesis_counts.items(), key=lambda item: -item[1])[:4]
        ]
    except Exception:
        pass

    try:
        budget = read_runtime_json(
            budget_path,
            {"experiments_run": 0, "max_experiments": "n/a", "failures_by_type": {}},
        )
        metrics["budget"]["experiments_run"] = budget.get("experiments_run", 0)
        metrics["budget"]["max_experiments"] = budget.get("max_experiments", "n/a")
        metrics["budget"]["failures"] = budget.get("failures_by_type", {})
    except Exception:
        pass

    try:
        recent_tasks = recent_json_events(exp_log, {"task_result"}, limit=10, scan_lines=5000)
        financial_points: list[dict[str, Any]] = []
        pass_count = 0
        recent_results: list[dict[str, object]] = []

        for row in recent_tasks:
            metrics_row = row.get("metrics")
            recent_results.append(recent_task_snapshot(row))
            if isinstance(metrics_row, dict):
                net_pnl = metrics_row.get("net_pnl")
                sharpe_ratio = metrics_row.get("sharpe_ratio")
                if isinstance(net_pnl, (int, float)) and isinstance(sharpe_ratio, (int, float)):
                    financial_points.append(
                        {
                            "strategy": str(row.get("strategy_name", "")),
                            "bar": str(row.get("bar_config", "")),
                            "net_pnl": float(net_pnl),
                            "sharpe": float(sharpe_ratio),
                            "trades": metrics_row.get("trade_count"),
                        }
                    )
            if str(row.get("verdict", "")) == "PASS":
                pass_count += 1

        if financial_points:
            metrics["financial"] = {
                "tested": len(financial_points),
                "avg_net_pnl": mean(point["net_pnl"] for point in financial_points),
                "avg_sharpe": mean(point["sharpe"] for point in financial_points),
                "pass_rate_pct": (100.0 * pass_count / len(recent_tasks)) if recent_tasks else 0.0,
                "best": max(financial_points, key=lambda point: point["net_pnl"]),
                "worst": min(financial_points, key=lambda point: point["net_pnl"]),
            }
        metrics["recent_results"] = list(reversed(recent_results[-5:]))
    except Exception:
        pass

    return metrics


def list_signals(*, project_root: Path) -> list[dict[str, str]]:
    exp_log = project_root / "results" / "logs" / "research_experiments.jsonl"
    signals: list[dict[str, str]] = []
    seen: set[str] = set()

    try:
        recent_tasks = recent_json_events(exp_log, {"task_result"}, limit=100, scan_lines=15000)
        for row in recent_tasks:
            strategy = str(row.get("strategy_name", ""))
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            signals.append(
                {
                    "strategy": strategy,
                    "verdict": str(row.get("verdict", "UNKNOWN")),
                    "timestamp": str(row.get("timestamp", "")),
                    "bar_config": str(row.get("bar_config", "")),
                }
            )
    except Exception:
        pass

    return sorted(signals, key=lambda row: row["timestamp"], reverse=True)


def get_signal_details(*, project_root: Path, strategy_name: str) -> dict[str, Any]:
    exp_log = project_root / "results" / "logs" / "research_experiments.jsonl"
    signal_path = project_root / "research" / "signals" / f"{strategy_name}.py"

    experiment_data: dict[str, Any] | None = None
    try:
        recent_tasks = recent_json_events(exp_log, {"task_result"}, limit=100, scan_lines=15000)
        for row in recent_tasks:
            if str(row.get("strategy_name", "")) == strategy_name:
                experiment_data = row
                break
    except Exception:
        experiment_data = None

    code = ""
    try:
        if signal_path.exists():
            code = signal_path.read_text(encoding="utf-8")
        else:
            code = f"# Source code for {strategy_name}.py not found on disk."
    except Exception as exc:
        code = f"# Error reading file: {exc}"

    if not experiment_data:
        experiment_data = {
            "strategy_name": strategy_name,
            "error": "No matching experiment metrics found in logs.",
        }

    return {
        "strategy": strategy_name,
        "code": code,
        "metrics": experiment_data.get("metrics", {}),
        "gauntlet": experiment_data.get("gauntlet", {}),
        "verdict": experiment_data.get("verdict", "UNKNOWN"),
        "timestamp": experiment_data.get("timestamp", ""),
    }


__all__ = [
    "collect_autonomy_status",
    "get_signal_details",
    "list_signals",
    "load_mission_name",
    "read_runtime_json",
]
