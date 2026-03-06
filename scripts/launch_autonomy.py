#!/usr/bin/env python3
"""Launch validator + LLM orchestrator in tmux and show live run status."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from statistics import mean
from typing import Any, NamedTuple

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.lib.runtime_state import clear_orchestrator_state, reset_shared_state


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


class OrchestratorWorker(NamedTuple):
    lane_id: str | None
    session_name: str
    out_path: Path
    log_path: Path


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        values[key] = value
    return values


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _run_quiet(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _tmux_has_session(name: str) -> bool:
    proc = _run_quiet(["tmux", "has-session", "-t", name])
    return proc.returncode == 0


def _tmux_kill_session(name: str) -> None:
    if _tmux_has_session(name):
        subprocess.run(["tmux", "kill-session", "-t", name], check=False)


def _tmux_new_session(name: str, shell_command: str) -> None:
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", name, f"bash -lc {shlex.quote(shell_command)}"],
        check=True,
    )


def _tmux_interrupt_session(name: str) -> None:
    """Send Ctrl+C to the first pane of a session."""
    if not _tmux_has_session(name):
        return
    subprocess.run(["tmux", "send-keys", "-t", name, "C-c"], check=False)


def _shutdown_sessions_gracefully(
    sessions: list[str],
    *,
    wait_seconds: float = 3.0,
    poll_seconds: float = 0.2,
) -> list[str]:
    """Attempt graceful session shutdown; force kill leftovers."""
    targets = [s for s in sessions if _tmux_has_session(s)]
    if not targets:
        return []

    for session in targets:
        _tmux_interrupt_session(session)

    deadline = time.monotonic() + max(0.5, float(wait_seconds))
    try:
        while time.monotonic() < deadline:
            remaining = [s for s in targets if _tmux_has_session(s)]
            if not remaining:
                return []
            time.sleep(max(0.1, float(poll_seconds)))
    except KeyboardInterrupt:
        # If another Ctrl+C arrives during shutdown wait, continue with force-kill path.
        pass

    remaining = [s for s in targets if _tmux_has_session(s)]
    for session in remaining:
        _tmux_kill_session(session)
    return remaining


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_yaml(path: Path, default: Any) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _tail_lines(path: Path, max_lines: int = 300) -> list[str]:
    if not path.exists():
        return []
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return data[-max_lines:] if max_lines > 0 else data


def _last_json_event(path: Path, event_names: set[str] | None = None) -> dict[str, Any] | None:
    for raw in reversed(_tail_lines(path, max_lines=1200)):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        if event_names and str(row.get("event", "")) not in event_names:
            continue
        return row
    return None


def _recent_json_events(path: Path, *, event_names: set[str], limit: int, scan_lines: int = 3000) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in reversed(_tail_lines(path, max_lines=scan_lines)):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        if str(row.get("event", "")) not in event_names:
            continue
        out.append(row)
        if len(out) >= limit:
            break
    return list(reversed(out))


def _last_json_event_across(paths: list[Path], event_names: set[str] | None = None) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    latest_ts = ""
    for path in paths:
        row = _last_json_event(path, event_names)
        if not isinstance(row, dict):
            continue
        ts = str(row.get("timestamp", ""))
        if latest is None or ts >= latest_ts:
            latest = row
            latest_ts = ts
    return latest


def _recent_json_events_across(
    paths: list[Path],
    *,
    event_names: set[str],
    limit: int,
    scan_lines: int = 3000,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in paths:
        merged.extend(_recent_json_events(path, event_names=event_names, limit=limit, scan_lines=scan_lines))
    merged.sort(key=lambda row: str(row.get("timestamp", "")), reverse=True)
    return list(reversed(merged[:limit]))


def _last_nonempty_line(path: Path) -> str:
    for raw in reversed(_tail_lines(path, max_lines=200)):
        if raw.strip():
            return raw.strip()
    return "(no output yet)"


def _status_tag(up: bool) -> str:
    return f"{GREEN}UP{RESET}" if up else f"{RED}DOWN{RESET}"


def _lane_ids(lane_count: int) -> list[str | None]:
    count = max(1, min(10, int(lane_count)))
    if count == 1:
        return [None]
    return [chr(ord("A") + i) for i in range(count)]


def _orchestrator_workers(*, session_prefix: str, logs_dir: Path, lane_count: int) -> list[OrchestratorWorker]:
    workers: list[OrchestratorWorker] = []
    for lane_id in _lane_ids(lane_count):
        suffix = f"_{lane_id}" if lane_id else ""
        session_name = f"{session_prefix}_orchestrator{suffix.lower()}"
        workers.append(
            OrchestratorWorker(
                lane_id=lane_id,
                session_name=session_name,
                out_path=logs_dir / f"llm_orchestrator{suffix}.out",
                log_path=logs_dir / f"llm_orchestrator{suffix}.jsonl",
            )
        )
    return workers


def _fmt_num(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "n/a"


def _kill_criteria_hits(budget: dict[str, Any]) -> list[str]:
    kill = budget.get("kill_criteria")
    fails = budget.get("failures_by_type")
    if not isinstance(kill, dict) or not isinstance(fails, dict):
        return []
    hits: list[str] = []
    for verdict, threshold in kill.items():
        try:
            k = int(threshold)
            v = int(fails.get(verdict, 0))
        except Exception:
            continue
        if v >= k:
            hits.append(f"{verdict}:{v}/{k}")
    return hits


def _extract_recent_task_results(path: Path, limit: int = 12) -> list[dict[str, Any]]:
    return _recent_json_events(path, event_names={"task_result"}, limit=limit, scan_lines=5000)


def _financial_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    points: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        net_pnl = metrics.get("net_pnl")
        sharpe = metrics.get("sharpe_ratio")
        if not isinstance(net_pnl, (int, float)) or not isinstance(sharpe, (int, float)):
            continue
        points.append(
            {
                "strategy": str(row.get("strategy_name", "")),
                "bar": str(row.get("bar_config", "")),
                "net_pnl": float(net_pnl),
                "sharpe": float(sharpe),
                "trades": metrics.get("trade_count"),
                "timestamp": str(row.get("timestamp", "")),
                "verdict": str(row.get("verdict", "")),
            }
        )
    if not points:
        return {"count": 0}
    best = max(points, key=lambda x: x["net_pnl"])
    worst = min(points, key=lambda x: x["net_pnl"])
    pass_count = sum(1 for row in rows if str(row.get("verdict", "")) == "PASS")
    return {
        "count": len(points),
        "unique_strategies": len({f"{p['strategy']}|{p['bar']}" for p in points}),
        "avg_net_pnl": mean(p["net_pnl"] for p in points),
        "avg_sharpe": mean(p["sharpe"] for p in points),
        "pass_rate_pct": (100.0 * pass_count / len(rows)) if rows else 0.0,
        "best": best,
        "worst": worst,
    }


def _gauntlet_snapshot(last_task_result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(last_task_result, dict):
        return {"available": False}
    gauntlet = last_task_result.get("gauntlet")
    if not isinstance(gauntlet, dict):
        return {"available": False}
    failing: list[str] = []
    passing: list[str] = []
    for key, value in gauntlet.items():
        if key in {"overall_verdict", "pass_count", "total_tests"}:
            continue
        if isinstance(value, dict):
            verdict = str(value.get("verdict", ""))
            if verdict == "FAIL":
                failing.append(key)
            elif verdict == "PASS":
                passing.append(key)
    return {
        "available": True,
        "overall_verdict": str(gauntlet.get("overall_verdict", "n/a")),
        "pass_count": gauntlet.get("pass_count", "n/a"),
        "total_tests": gauntlet.get("total_tests", "n/a"),
        "failing": failing,
        "passing": passing,
    }


def _render_dashboard(
    *,
    root: Path,
    orchestrator_workers: list[OrchestratorWorker],
    validator_session: str,
    mission: Path,
    agent_config: Path,
    mission_max_experiments: int | None,
    validator_enabled: bool,
    orchestrator_enabled: bool,
) -> None:
    state_dir = root / "research" / ".state"
    logs_dir = root / "results" / "logs"
    queue_path = state_dir / "experiment_queue.json"
    budget_path = state_dir / "mission_budget.json"
    exp_log = logs_dir / "research_experiments.jsonl"
    worker_out = logs_dir / "research_worker.out"
    orchestrator_log_paths = [worker.log_path for worker in orchestrator_workers]

    queue = _read_json(queue_path, {"tasks": []})
    tasks = queue.get("tasks", []) if isinstance(queue, dict) else []
    pending = sum(1 for t in tasks if isinstance(t, dict) and t.get("state") == "pending")
    in_progress = sum(1 for t in tasks if isinstance(t, dict) and t.get("state") == "in_progress")
    completed = sum(1 for t in tasks if isinstance(t, dict) and t.get("state") == "completed")
    failed = sum(1 for t in tasks if isinstance(t, dict) and t.get("state") == "failed")

    budget = _read_json(
        budget_path,
        {
            "experiments_run": 0,
            "max_experiments": None,
            "failures_by_type": {},
            "kill_criteria": {},
        },
    )
    experiments_run = budget.get("experiments_run", 0)
    max_experiments = (
        int(mission_max_experiments)
        if isinstance(mission_max_experiments, int)
        else budget.get("max_experiments", "n/a")
    )
    failures = budget.get("failures_by_type", {})
    if not isinstance(failures, dict):
        failures = {}

    last_task = _last_json_event(exp_log, {"task_result", "task_error"})
    last_run_end = _last_json_event(exp_log, {"run_end"})
    last_llm = _last_json_event_across(
        orchestrator_log_paths,
        {"generation_enqueued", "generation_rejected", "generation_error"},
    )
    last_task_result = _last_json_event(exp_log, {"task_result"})
    recent_task_results = _extract_recent_task_results(exp_log, limit=10)
    fin = _financial_snapshot(recent_task_results)
    gshot = _gauntlet_snapshot(last_task_result)
    recent_hypotheses = _recent_json_events_across(
        orchestrator_log_paths,
        event_names={"generation_enqueued", "generation_rejected", "generation_error"},
        limit=5,
        scan_lines=4000,
    )
    live_hypothesis_counts: dict[str, int] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if str(task.get("state", "")) not in {"pending", "in_progress"}:
            continue
        source = task.get("source")
        if not isinstance(source, dict):
            continue
        hypothesis_id = str(source.get("hypothesis_id", "")).strip()
        if hypothesis_id:
            live_hypothesis_counts[hypothesis_id] = live_hypothesis_counts.get(hypothesis_id, 0) + 1

    orchestrator_statuses = {
        worker.session_name: _tmux_has_session(worker.session_name)
        for worker in orchestrator_workers
    }
    val_up = _tmux_has_session(validator_session) if validator_enabled else False

    print("\033[2J\033[H", end="")
    print(f"{BOLD}{CYAN}Autonomy Launcher Dashboard{RESET}")
    print(f"{DIM}{_now_utc()}{RESET}")
    print("")
    print(f"{BOLD}Mission:{RESET} {mission}")
    print(f"{BOLD}Agent Config:{RESET} {agent_config}")
    print("")
    print(f"{BOLD}Sessions{RESET}")
    if orchestrator_enabled:
        for worker in orchestrator_workers:
            lane_label = worker.lane_id or "default"
            print(
                f"  LLM orchestrator [{worker.session_name}] lane={lane_label}: "
                f"{_status_tag(orchestrator_statuses.get(worker.session_name, False))}"
            )
    if validator_enabled:
        print(f"  Validator      [{validator_session}]: {_status_tag(val_up)}")
    print("")
    print(f"{BOLD}Queue{RESET}")
    print(f"  pending={pending}  in_progress={in_progress}  completed={completed}  failed={failed}")
    if live_hypothesis_counts:
        top_h = sorted(live_hypothesis_counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
        print("  active_hypotheses: " + ", ".join(f"{hid} ({n} tasks)" for hid, n in top_h))
    print("")
    print(f"{BOLD}Budget{RESET}")
    fail_blob = ", ".join(f"{k}:{v}" for k, v in sorted(failures.items())) or "(none)"
    print(f"  experiments_run={experiments_run} / max={max_experiments}")
    print(f"  failures={fail_blob}")
    hits = _kill_criteria_hits(budget if isinstance(budget, dict) else {})
    if hits:
        print(f"  {YELLOW}kill criteria reached: {', '.join(hits)}{RESET}")
    print("")
    print(f"{BOLD}Last Validator Event{RESET}")
    if isinstance(last_task, dict):
        event = str(last_task.get("event", ""))
        strategy = str(last_task.get("strategy_name", ""))
        bar = str(last_task.get("bar_config", ""))
        verdict = str(last_task.get("verdict", ""))
        ts = str(last_task.get("timestamp", ""))
        if event == "task_result":
            metrics = last_task.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            print(
                "  "
                f"{ts}  {event}  {strategy} [{bar}]  verdict={verdict}  "
                f"net_pnl={_fmt_num(metrics.get('net_pnl'))}  "
                f"sharpe={_fmt_num(metrics.get('sharpe_ratio'))}  "
                f"trades={metrics.get('trade_count', 'n/a')}"
            )
        else:
            err = str(last_task.get("error", ""))[:180]
            print(f"  {ts}  {event}  {strategy}  verdict={verdict}  error={err}")
    else:
        print("  (no task events yet)")
    print("")
    print(f"{BOLD}Last LLM Event{RESET}")
    if isinstance(last_llm, dict):
        event = str(last_llm.get("event", ""))
        strategy = str(last_llm.get("strategy_name", ""))
        iteration = last_llm.get("iteration", "n/a")
        ts = str(last_llm.get("timestamp", ""))
        if event == "generation_enqueued":
            task_ids = last_llm.get("task_ids")
            n_tasks = len(task_ids) if isinstance(task_ids, list) else "n/a"
            print(f"  {ts}  {event}  {strategy}  iteration={iteration}  enqueued_tasks={n_tasks}")
        elif event == "generation_rejected":
            errs = last_llm.get("errors")
            msg = str(errs[0]) if isinstance(errs, list) and errs else ""
            print(f"  {ts}  {event}  {strategy}  iteration={iteration}  reason={msg[:180]}")
        else:
            err = str(last_llm.get("error", ""))
            print(f"  {ts}  {event}  iteration={iteration}  error={err[:180]}")
    else:
        print("  (no llm events yet)")
    print("")
    print(f"{BOLD}Hypothesis Updates{RESET}")
    if recent_hypotheses:
        for row in reversed(recent_hypotheses):
            event = str(row.get("event", ""))
            ts = str(row.get("timestamp", ""))
            iteration = row.get("iteration", "n/a")
            hypothesis_id = str(row.get("hypothesis_id", "")).strip()
            strategy = str(row.get("strategy_name", "")).strip()
            if event == "generation_enqueued":
                task_ids = row.get("task_ids")
                n_tasks = len(task_ids) if isinstance(task_ids, list) else "n/a"
                print(
                    f"  {ts}  iter={iteration}  {event}  "
                    f"hyp={hypothesis_id or 'n/a'}  strategy={strategy or 'n/a'}  tasks={n_tasks}"
                )
            elif event == "generation_rejected":
                errs = row.get("errors")
                reason = str(errs[0]) if isinstance(errs, list) and errs else ""
                print(
                    f"  {ts}  iter={iteration}  {event}  "
                    f"hyp={hypothesis_id or 'n/a'}  strategy={strategy or 'n/a'}  reason={reason[:120]}"
                )
            else:
                err = str(row.get("error", ""))
                print(f"  {ts}  iter={iteration}  {event}  error={err[:120]}")
    else:
        print("  (no recent hypothesis updates)")
    print("")
    print(f"{BOLD}Gauntlet Snapshot (latest task_result){RESET}")
    if gshot.get("available"):
        print(
            "  "
            f"overall={gshot['overall_verdict']}  "
            f"pass_count={gshot['pass_count']}/{gshot['total_tests']}"
        )
        failing = gshot.get("failing", [])
        passing = gshot.get("passing", [])
        if failing:
            print("  failing_tests: " + ", ".join(failing))
        if passing:
            print("  passing_tests: " + ", ".join(passing))
    else:
        print("  (no gauntlet data yet)")
    print("")
    print(f"{BOLD}Financial Snapshot (last {len(recent_task_results)} task results){RESET}")
    if fin.get("count", 0):
        print(
            "  "
            f"avg_net_pnl={_fmt_num(fin['avg_net_pnl'])}  "
            f"avg_sharpe={_fmt_num(fin['avg_sharpe'])}  "
            f"pass_rate={_fmt_num(fin['pass_rate_pct'])}%  "
            f"tested={fin['count']}  unique_strats={fin['unique_strategies']}"
        )
        best = fin["best"]
        worst = fin["worst"]
        print(
            "  "
            f"best_net={_fmt_num(best['net_pnl'])} ({best['strategy']} [{best['bar']}], "
            f"sharpe={_fmt_num(best['sharpe'])}, trades={best['trades']})"
        )
        print(
            "  "
            f"worst_net={_fmt_num(worst['net_pnl'])} ({worst['strategy']} [{worst['bar']}], "
            f"sharpe={_fmt_num(worst['sharpe'])}, trades={worst['trades']})"
        )
    else:
        print("  (no financial metrics yet)")
    print("")
    print(f"{BOLD}Recent Output{RESET}")
    if orchestrator_enabled:
        for worker in orchestrator_workers:
            label = f"llm_orchestrator_{worker.lane_id}.out" if worker.lane_id else "llm_orchestrator.out"
            print(f"  {label}: {_last_nonempty_line(worker.out_path)}")
    if validator_enabled:
        print(f"  research_worker.out:  {_last_nonempty_line(worker_out)}")
    if isinstance(last_run_end, dict):
        print("")
        print(f"{BOLD}Last Run End{RESET}")
        print(
            "  "
            f"{last_run_end.get('timestamp', '')}  "
            f"stop_reason={last_run_end.get('stop_reason', '')}  "
            f"run_id={last_run_end.get('run_id', '')}"
        )
    print("")
    attach_cmds: list[str] = []
    stop_cmds: list[str] = []
    if orchestrator_enabled:
        for worker in orchestrator_workers:
            attach_cmds.append(f"tmux attach -t {worker.session_name}")
            stop_cmds.append(f"tmux kill-session -t {worker.session_name}")
    if validator_enabled:
        attach_cmds.append(f"tmux attach -t {validator_session}")
        stop_cmds.append(f"tmux kill-session -t {validator_session}")
    if attach_cmds:
        print(f"{DIM}Attach: {' | '.join(attach_cmds)}{RESET}")
    if stop_cmds:
        print(f"{DIM}Stop:   {'; '.join(stop_cmds)}{RESET}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch and monitor autonomous research workers.")
    parser.add_argument(
        "--mission",
        type=Path,
        default=Path("configs/missions/alpha-discovery.yaml"),
        help="Mission YAML path.",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path("configs/agents/llm_orchestrator.yaml"),
        help="LLM orchestrator config YAML path.",
    )
    parser.add_argument("--session-prefix", default="alpha", help="tmux session prefix.")
    parser.add_argument("--poll-seconds", type=int, default=3, help="Dashboard refresh interval seconds.")
    parser.add_argument("--no-monitor", action="store_true", help="Launch and exit without dashboard loop.")
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Do not kill existing tmux sessions with the same names before launch.",
    )
    parser.add_argument(
        "--fresh-state",
        action="store_true",
        help="Reset runtime state before launching workers.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="On Ctrl+C, stop monitoring but leave tmux workers running.",
    )
    parser.add_argument(
        "--allow-bootstrap",
        action="store_true",
        help="Allow validator to auto-bootstrap tasks from research/signals when queue is empty.",
    )
    parser.add_argument("--validator-only", action="store_true", help="Launch only validator worker.")
    parser.add_argument("--orchestrator-only", action="store_true", help="Launch only LLM orchestrator worker.")
    parser.add_argument("--lane-count", type=int, default=1, help="Number of orchestrator lanes to launch (1-10).")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.validator_only and args.orchestrator_only:
        parser.error("Use at most one of --validator-only or --orchestrator-only.")
    launch_validator = not args.orchestrator_only
    launch_orchestrator = not args.validator_only

    root = Path(__file__).resolve().parent.parent
    mission = args.mission if args.mission.is_absolute() else (root / args.mission)
    agent_cfg = args.agent_config if args.agent_config.is_absolute() else (root / args.agent_config)
    mission = mission.resolve()
    agent_cfg = agent_cfg.resolve()

    if not mission.exists():
        raise FileNotFoundError(f"Mission file not found: {mission}")
    if launch_orchestrator and not agent_cfg.exists():
        raise FileNotFoundError(f"Agent config file not found: {agent_cfg}")

    required_cmds = ["tmux", "uv", "bash"]
    if launch_orchestrator:
        required_cmds.append("claude")
    for cmd in required_cmds:
        if not _command_exists(cmd):
            raise RuntimeError(f"Required command not found on PATH: {cmd}")

    env_file = root / ".env"
    if not env_file.exists():
        raise FileNotFoundError(f"Missing .env file at {env_file}. Copy .env.example first.")
    env_values = _parse_env_file(env_file)
    nq_path = str(os.getenv("NQ_DATA_PATH", env_values.get("NQ_DATA_PATH", ""))).strip()
    if not nq_path:
        raise RuntimeError("NQ_DATA_PATH is empty. Set it in .env before launch.")
    if not Path(nq_path).exists():
        raise RuntimeError(f"NQ_DATA_PATH does not exist: {nq_path}")

    mission_payload = _read_yaml(mission, default={})
    mission_max_experiments: int | None = None
    mission_name = mission.stem
    if isinstance(mission_payload, dict):
        mission_name = str(mission_payload.get("mission_name", mission.stem))
        raw = mission_payload.get("max_experiments")
        try:
            mission_max_experiments = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            mission_max_experiments = None

    logs_dir = root / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    fresh_state = bool(args.fresh_state or args.no_resume)
    if fresh_state and args.no_restart:
        parser.error("--fresh-state cannot be combined with --no-restart.")
    if fresh_state and not launch_validator:
        parser.error("--fresh-state requires launching the validator or running scripts/run_cleanup.py --force.")

    lane_count = max(1, min(10, int(args.lane_count)))
    orchestrator_workers = (
        _orchestrator_workers(
            session_prefix=args.session_prefix,
            logs_dir=logs_dir,
            lane_count=lane_count,
        )
        if launch_orchestrator
        else []
    )
    val_session = f"{args.session_prefix}_validator"

    if not args.no_restart:
        for worker in orchestrator_workers:
            _tmux_kill_session(worker.session_name)
        if launch_validator:
            _tmux_kill_session(val_session)

    if fresh_state:
        reset_shared_state(root, mission_name=mission_name)
        clear_orchestrator_state(root)

    state_flag = " --resume"
    bootstrap_flag = "" if args.allow_bootstrap else " --no-bootstrap"
    mission_q = shlex.quote(str(mission))
    agent_q = shlex.quote(str(agent_cfg))
    root_q = shlex.quote(str(root))

    if launch_validator and not _tmux_has_session(val_session):
        validator_cmd = (
            f"cd {root_q}; "
            "set -a; source .env; set +a; "
            f"uv run python -u scripts/research.py --mission {mission_q} --auto-mode "
            f"--worker-agent validator{state_flag}{bootstrap_flag} 2>&1 | tee -a results/logs/research_worker.out"
        )
        _tmux_new_session(val_session, validator_cmd)

    for worker in orchestrator_workers:
        if _tmux_has_session(worker.session_name):
            continue
        lane_flag = f" --lane {worker.lane_id}" if worker.lane_id else ""
        orchestrator_cmd = (
            f"cd {root_q}; "
            "set -a; source .env; set +a; "
            f"uv run python -u scripts/llm_orchestrator.py --mission {mission_q} "
            f"--agent-config {agent_q}{state_flag}{lane_flag} 2>&1 | tee -a {shlex.quote(str(worker.out_path))}"
        )
        _tmux_new_session(worker.session_name, orchestrator_cmd)

    print(f"{GREEN}Launched workers successfully.{RESET}")
    if launch_orchestrator:
        for worker in orchestrator_workers:
            lane_label = worker.lane_id or "default"
            print(f"  orchestrator session ({lane_label}): {worker.session_name}")
    if launch_validator:
        print(f"  validator session:    {val_session}")
    print("")

    if args.no_monitor:
        print("Monitoring disabled (--no-monitor).")
        return 0

    down_cycles = 0
    poll_seconds = max(1, int(args.poll_seconds))
    try:
        while True:
            _render_dashboard(
                root=root,
                orchestrator_workers=orchestrator_workers,
                validator_session=val_session,
                mission=mission,
                agent_config=agent_cfg,
                mission_max_experiments=mission_max_experiments,
                validator_enabled=launch_validator,
                orchestrator_enabled=launch_orchestrator,
            )
            orch_up = (
                any(_tmux_has_session(worker.session_name) for worker in orchestrator_workers)
                if launch_orchestrator
                else False
            )
            val_up = _tmux_has_session(val_session) if launch_validator else False
            all_down = (not launch_orchestrator or not orch_up) and (not launch_validator or not val_up)
            if all_down:
                down_cycles += 1
                if down_cycles >= 2:
                    print("")
                    print(f"{YELLOW}Both requested sessions are down. Exiting monitor.{RESET}")
                    return 0
            else:
                down_cycles = 0
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("")
        if args.keep_running:
            print("Monitor stopped (Ctrl+C). Workers keep running in tmux.")
            return 0

        sessions_to_stop: list[str] = []
        if launch_orchestrator:
            sessions_to_stop.extend(worker.session_name for worker in orchestrator_workers)
        if launch_validator:
            sessions_to_stop.append(val_session)

        print("Monitor stopped (Ctrl+C). Shutting down workers gracefully...")
        forced = _shutdown_sessions_gracefully(sessions_to_stop)
        if forced:
            print(f"{YELLOW}Forced session shutdown: {', '.join(forced)}{RESET}")
        else:
            print(f"{GREEN}Workers stopped cleanly.{RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
