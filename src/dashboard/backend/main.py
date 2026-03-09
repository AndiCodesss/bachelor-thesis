from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import uuid
import os
import signal
import sys
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from research.lib.coordination import read_json_file_if_exists
from src.framework.data.bar_configs import BAR_CONFIGS, bar_config_label

project_root = Path(__file__).resolve().parent.parent.parent.parent

AVAILABLE_BARS = [bar_config_label(cfg) for cfg in BAR_CONFIGS]
LOCAL_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

app = FastAPI(title="NQ-Alpha Central Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LOCAL_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class CacheRunRequest(BaseModel):
    splits: list[str]
    session_filter: str
    exec_mode: str
    bar_filter: str | None = None
    clean: bool = False

class AutonomyRunRequest(BaseModel):
    mission: str = "configs/missions/alpha-discovery.yaml"
    agent_config: str = "configs/agents/llm_orchestrator.yaml"
    no_resume: bool = False
    allow_bootstrap: bool = False
    use_notebooklm: bool = True
    validator_only: bool = False
    orchestrator_only: bool = False
    lane_count: int = Field(default=2, ge=1, le=10)

class CleanupRunRequest(BaseModel):
    force: bool = False

runs = {}


def _load_mission_name(mission_ref: str) -> str:
    import yaml

    mission_path = Path(mission_ref)
    if not mission_path.is_absolute():
        mission_path = project_root / mission_path
    mission_path = mission_path.resolve()
    if not mission_path.exists():
        return mission_path.stem
    try:
        payload = yaml.safe_load(mission_path.read_text(encoding="utf-8"))
    except Exception:
        return mission_path.stem
    if isinstance(payload, dict):
        return str(payload.get("mission_name", mission_path.stem))
    return mission_path.stem


def _reset_runtime_state(mission_name: str) -> None:
    from research.lib.runtime_state import clear_orchestrator_state, reset_shared_state

    reset_shared_state(project_root, mission_name=mission_name)
    clear_orchestrator_state(project_root)


def _read_runtime_json(path: Path, default_payload: dict) -> dict:
    return read_json_file_if_exists(
        json_path=path,
        lock_path=path.with_suffix(".lock"),
        default_payload=default_payload,
    )

# --- Cache Config ---
@app.get("/api/config/cache")
def get_cache_config():
    return {
        "splits": ["train", "validate", "test", "all"],
        "session_filters": ["eth", "rth", "both"],
        "exec_modes": ["auto", "research", "promotion"],
        "bar_filters": AVAILABLE_BARS,
    }

# --- Autonomy Status Parsers ---
def _tail_lines(path: Path, max_lines: int = 300) -> list[str]:
    if not path.exists():
        return []
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return data[-max_lines:] if max_lines > 0 else data

def _recent_json_events(path: Path, event_names: set[str], limit: int, scan_lines: int = 3000):
    out = []
    for raw in reversed(_tail_lines(path, max_lines=scan_lines)):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict) and str(row.get("event", "")) in event_names:
                out.append(row)
                if len(out) >= limit:
                    break
        except Exception:
            continue
    return list(reversed(out))


def _terminate_process(proc) -> None:
    pid = getattr(proc, "pid", None)
    if pid is None:
        return
    if getattr(proc, "returncode", None) is not None:
        return

    if os.name != "nt":
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(3)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        except Exception:
            pass

    try:
        proc.terminate()
    except Exception:
        pass
    time.sleep(3)
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def _process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _terminate_process_group(procs) -> None:
    """Send SIGTERM to all process groups, wait once, then SIGKILL all."""
    pgids = []
    for proc in procs:
        pid = getattr(proc, "pid", None)
        if pid is None or getattr(proc, "returncode", None) is not None:
            continue
        try:
            pgids.append(os.getpgid(pid))
        except OSError:
            continue

    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass

    time.sleep(3)

    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass


def _subprocess_exec_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}

# --- Thinker Session Helpers ---

def _get_sessions_dir() -> Path:
    """Find the Claude sessions directory for this project."""
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return projects_dir / "unknown"
    # Claude slugifies the project path by replacing / and special chars with -
    # Find the directory that contains "bachelor" (our project name)
    candidates = [d for d in projects_dir.iterdir() if d.is_dir() and "bachelor" in d.name]
    if candidates:
        # Pick the one with the longest name match (most specific)
        return max(candidates, key=lambda d: len(d.name))
    # Fallback: compute slug by replacing / with - and spaces with -
    slug = str(project_root).replace("/", "-").replace(" ", "-").lstrip("-")
    return projects_dir / slug

def _summarize_tool_input(tool_name: str, input_data: dict) -> str:
    """Compact human-readable summary of a tool call's input."""
    if tool_name == "Grep":
        parts = []
        if "pattern" in input_data:
            parts.append(f"pattern={str(input_data['pattern'])[:60]}")
        if "path" in input_data:
            parts.append(f"path={Path(str(input_data['path'])).name}")
        return " ".join(parts) or str(input_data)[:100]
    elif tool_name in ("Read", "Write", "Edit"):
        path = input_data.get("file_path", "")
        return Path(str(path)).name if path else str(input_data)[:100]
    elif tool_name == "Bash":
        return str(input_data.get("command", ""))[:100]
    elif tool_name == "Glob":
        return str(input_data.get("pattern", ""))[:100]
    else:
        return str(input_data)[:100]


def _parse_thinker_events(jsonl_path: Path, max_events: int = 40) -> list[dict]:
    """Parse a thinker JSONL session file into structured events."""
    events: list[dict] = []
    pending_tools: dict[str, str] = {}  # tool_use id -> tool name

    lines = _tail_lines(jsonl_path, max_lines=max(200, max_events * 5))
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
                    inp = item.get("input", {})
                    if not isinstance(inp, dict):
                        inp = {}
                    summary = _summarize_tool_input(tool_name, inp)
                    if tool_id:
                        pending_tools[tool_id] = tool_name
                    events.append({"type": "tool_call", "tool": tool_name, "summary": summary})

        elif msg_type == "user":
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "tool_result":
                    tool_id = str(item.get("tool_use_id", ""))
                    tool_name = pending_tools.get(tool_id, "")
                    result_content = item.get("content", "")
                    if isinstance(result_content, list):
                        text_parts = [
                            c.get("text", "") for c in result_content
                            if isinstance(c, dict) and c.get("type") == "text"
                        ]
                        result_content = " ".join(text_parts)
                    summary = str(result_content)[:200].replace("\n", " ").strip()
                    events.append({"type": "tool_result", "tool": tool_name, "summary": summary})

    return events[-max_events:]


def _find_thinker_session_file() -> Optional[Path]:
    """Find JSONL open by an active 'claude -p' (thinker) process."""
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        thinker_pids = []
        for line in result.stdout.splitlines():
            if "claude -p " in line:
                parts = line.split()
                if parts:
                    thinker_pids.append(parts[1])

        for pid in thinker_pids:
            try:
                fd_dir = Path(f"/proc/{pid}/fd")
                for fd in fd_dir.iterdir():
                    try:
                        target = fd.resolve()
                        if (target.suffix == ".jsonl"
                                and str(_get_sessions_dir()) in str(target)
                                and target.exists()):
                            return target
                    except (PermissionError, OSError):
                        continue
            except (PermissionError, OSError):
                continue
    except Exception:
        pass
    return None


def _find_fallback_session_file() -> Optional[Path]:
    """Most recently modified JSONL in sessions directory."""
    sessions_dir = _get_sessions_dir()
    if not sessions_dir.exists():
        return None
    files = list(sessions_dir.glob("*.jsonl"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


@app.get("/api/autonomy/thinker")
def get_thinker_activity():
    session_file = _find_thinker_session_file()
    is_active = session_file is not None

    if not is_active:
        session_file = _find_fallback_session_file()
        if not session_file:
            return {"events": [], "session_id": None, "is_active": False, "last_updated": None}

    events = _parse_thinker_events(session_file)

    last_updated = None
    try:
        mtime = session_file.stat().st_mtime
        last_updated = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        pass

    return {
        "events": events,
        "session_id": session_file.stem,
        "is_active": is_active,
        "last_updated": last_updated,
    }


@app.get("/api/autonomy/status")
def get_autonomy_status():
    state_dir = project_root / "research" / ".state"
    logs_dir = project_root / "results" / "logs"
    queue_path = state_dir / "experiment_queue.json"
    budget_path = state_dir / "mission_budget.json"
    exp_log = logs_dir / "research_experiments.jsonl"

    # Defaults
    metrics = {
        "queue": {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0},
        "budget": {"experiments_run": 0, "max_experiments": "n/a", "failures": {}},
        "financial": {
            "tested": 0, "avg_net_pnl": 0.0, "avg_sharpe": 0.0, "pass_rate_pct": 0.0,
            "best": None, "worst": None
        },
        "active_hypotheses": []
    }

    # Queue
    try:
        queue = _read_runtime_json(queue_path, {"schema_version": "1.0", "tasks": []})
        tasks = queue.get("tasks", [])
        for t in tasks:
            state = t.get("state")
            if state in metrics["queue"]:
                metrics["queue"][state] += 1

        hyp_counts = {}
        for t in tasks:
            if t.get("state") in ["pending", "in_progress"]:
                hid = str(t.get("source", {}).get("hypothesis_id", "")).strip()
                if hid:
                    hyp_counts[hid] = hyp_counts.get(hid, 0) + 1
        metrics["active_hypotheses"] = [
            {"id": k, "tasks": v}
            for k, v in sorted(hyp_counts.items(), key=lambda x: -x[1])[:4]
        ]
    except Exception:
        pass

    # Budget
    try:
        b = _read_runtime_json(budget_path, {"experiments_run": 0, "max_experiments": "n/a", "failures_by_type": {}})
        metrics["budget"]["experiments_run"] = b.get("experiments_run", 0)
        metrics["budget"]["max_experiments"] = b.get("max_experiments", "n/a")
        metrics["budget"]["failures"] = b.get("failures_by_type", {})
    except Exception:
        pass

    # Financial Snapshot
    try:
        recent_tasks = _recent_json_events(exp_log, {"task_result"}, limit=10, scan_lines=5000)
        points = []
        pass_count = 0
        for row in recent_tasks:
            m = row.get("metrics")
            if not isinstance(m, dict):
                continue
            npnl, sharpe = m.get("net_pnl"), m.get("sharpe_ratio")
            if isinstance(npnl, (int, float)) and isinstance(sharpe, (int, float)):
                points.append({
                    "strategy": str(row.get("strategy_name", "")),
                    "bar": str(row.get("bar_config", "")),
                    "net_pnl": float(npnl),
                    "sharpe": float(sharpe),
                    "trades": m.get("trade_count")
                })
            if str(row.get("verdict", "")) == "PASS":
                pass_count += 1

        if points:
            metrics["financial"] = {
                "tested": len(points),
                "avg_net_pnl": mean(p["net_pnl"] for p in points),
                "avg_sharpe": mean(p["sharpe"] for p in points),
                "pass_rate_pct": (100.0 * pass_count / len(recent_tasks)) if recent_tasks else 0.0,
                "best": max(points, key=lambda x: x["net_pnl"]),
                "worst": min(points, key=lambda x: x["net_pnl"])
            }
    except Exception:
        pass

    return metrics

# --- Signals Explorer ---
@app.get("/api/signals")
def list_signals():
    logs_dir = project_root / "results" / "logs"
    exp_log = logs_dir / "research_experiments.jsonl"

    signals = []
    seen = set()
    try:
        # We want to scan back a good chunk to get a historical list
        recent_tasks = _recent_json_events(exp_log, {"task_result"}, limit=100, scan_lines=15000)
        for row in recent_tasks:
            strat = str(row.get("strategy_name", ""))
            if not strat or strat in seen:
                continue

            seen.add(strat)
            signals.append({
                "strategy": strat,
                "verdict": row.get("verdict", "UNKNOWN"),
                "timestamp": row.get("timestamp", ""),
                "bar_config": row.get("bar_config", "")
            })
    except Exception:
        pass

    return sorted(signals, key=lambda x: x["timestamp"], reverse=True)

@app.get("/api/signals/{strategy_name}")
def get_signal_details(strategy_name: str):
    logs_dir = project_root / "results" / "logs"
    exp_log = logs_dir / "research_experiments.jsonl"
    signals_dir = project_root / "research" / "signals"

    # 1. Fetch exact JSON event payload
    experiment_data = None
    try:
        # scan for the specific strategy
        recent_tasks = _recent_json_events(exp_log, {"task_result"}, limit=100, scan_lines=15000)
        for row in recent_tasks:
            if str(row.get("strategy_name", "")) == strategy_name:
                experiment_data = row
                break
    except Exception:
        pass

    # 2. Fetch the actual .py code
    code = ""
    try:
        target_py = signals_dir / f"{strategy_name}.py"
        if target_py.exists():
            code = target_py.read_text(encoding="utf-8")
        else:
            code = f"# Source code for {strategy_name}.py not found on disk."
    except Exception as e:
        code = f"# Error reading file: {str(e)}"

    if not experiment_data:
        # Provide fallback if they clicked a generated signal file without an experiment log
        experiment_data = {"strategy_name": strategy_name, "error": "No matching experiment metrics found in logs."}

    return {
        "strategy": strategy_name,
        "code": code,
        "metrics": experiment_data.get("metrics", {}),
        "gauntlet": experiment_data.get("gauntlet", {}),
        "verdict": experiment_data.get("verdict", "UNKNOWN"),
        "timestamp": experiment_data.get("timestamp", "")
    }

# --- Shared Background Execution ---
async def execute_in_background(run_id: str, cmd: list[str]):
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, # Merge stderr into stdout
            env=os.environ.copy(),
            **_subprocess_exec_kwargs(),
        )
        runs[run_id]["process"] = process

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode('utf-8', errors='replace')
            runs[run_id]["logs"].append(text)

        await process.wait()
        runs[run_id]["status"] = "completed" if process.returncode == 0 else "failed"
    except Exception as e:
        runs[run_id]["logs"].append(f"\nError reading logs: {str(e)}")
        runs[run_id]["status"] = "failed"

async def direct_run_autonomy(run_id: str, commands: list[dict]):
    active_procs = []

    runs[run_id]["process"] = None
    runs[run_id]["processes"] = []

    try:
        for cmd_info in commands:
            cmd = cmd_info["cmd"]
            name = cmd_info["name"]
            runs[run_id]["logs"].append(f"--- Starting {name} ---\n")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=os.environ.copy(),
                **_subprocess_exec_kwargs(),
            )
            active_procs.append((name, proc))
            runs[run_id]["processes"].append(proc)

        async def read_stream(name, stream):
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode('utf-8', errors='replace')
                runs[run_id]["logs"].append(f"[{name}] {text}")

        readers = [asyncio.create_task(read_stream(name, p.stdout)) for name, p in active_procs]

        await asyncio.gather(*readers)

        failures = []
        for name, p in active_procs:
            return_code = await p.wait()
            if return_code != 0:
                failures.append((name, return_code))

        if failures:
            runs[run_id]["status"] = "failed"
            details = ", ".join(f"{name} exited {code}" for name, code in failures)
            runs[run_id]["logs"].append(f"\n--- Autonomy processes failed: {details} ---\n")
        else:
            runs[run_id]["status"] = "completed"
            runs[run_id]["logs"].append("\n--- Autonomy processes completed ---\n")

    except Exception as e:
        runs[run_id]["logs"].append(f"\nError running autonomy: {str(e)}\n")
        runs[run_id]["status"] = "failed"
    finally:
        for p in runs[run_id].get("processes", []):
            if p.returncode is None:
                _terminate_process(p)

# --- Endpoints ---

@app.post("/api/run/cache")
async def start_cache_run(req: CacheRunRequest):
    run_id = f"cache-{uuid.uuid4().hex[:8]}"
    cmd = [sys.executable, "scripts/cache_runner.py"]
    if req.splits:
        cmd.extend(["--split", ",".join(req.splits)])
    cmd.extend(["--session-filter", req.session_filter])
    cmd.extend(["--execution-mode", req.exec_mode])
    if req.bar_filter:
        cmd.extend(["--bar-filter", req.bar_filter])
    if req.clean:
        cmd.append("--clean")

    runs[run_id] = {"logs": [], "status": "running", "cmd": " ".join(cmd), "type": "cache"}
    asyncio.create_task(execute_in_background(run_id, cmd))
    return {"run_id": run_id, "status": "started", "cmd": " ".join(cmd)}

@app.post("/api/run/autonomy")
async def start_autonomy_run(req: AutonomyRunRequest):
    run_id = f"autonomy-{uuid.uuid4().hex[:8]}"

    commands = _build_autonomy_commands(req)
    fresh_state = bool(req.no_resume)

    if req.validator_only and req.orchestrator_only:
        raise HTTPException(status_code=400, detail="Cannot request validator-only and orchestrator-only at the same time.")
    if fresh_state and req.orchestrator_only:
        raise HTTPException(status_code=400, detail="Fresh state requires launching the validator or running cleanup first.")

    if fresh_state:
        mission_name = _load_mission_name(req.mission)
        _reset_runtime_state(mission_name)

    if not commands:
        raise HTTPException(status_code=400, detail="No autonomy workers selected.")

    runs[run_id] = {"logs": [], "status": "running", "cmd": f"Directly running {len(commands)} background processes", "type": "autonomy"}
    asyncio.create_task(direct_run_autonomy(run_id, commands))
    return {"run_id": run_id, "status": "started", "cmd": runs[run_id]["cmd"]}


def _build_autonomy_commands(req: AutonomyRunRequest) -> list[dict[str, list[str] | str]]:
    commands: list[dict[str, list[str] | str]] = []
    state_flag = ["--resume"]

    if not req.orchestrator_only:
        val_cmd = [
            sys.executable,
            "-u",
            "scripts/research.py",
            "--mission",
            req.mission,
            "--auto-mode",
            "--worker-agent",
            "validator",
            *state_flag,
        ]
        if not req.allow_bootstrap:
            val_cmd.append("--no-bootstrap")
        commands.append({"name": "Validator", "cmd": val_cmd})

    if not req.validator_only:
        lane_count = max(1, min(10, req.lane_count))
        for i in range(lane_count):
            lane_letter = chr(ord("A") + i)
            orch_cmd = [
                sys.executable,
                "-u",
                "scripts/llm_orchestrator.py",
                "--mission",
                req.mission,
                "--agent-config",
                req.agent_config,
                "--lane",
                lane_letter,
                *state_flag,
            ]
            if not req.use_notebooklm:
                orch_cmd.append("--disable-notebooklm")
            commands.append({"name": f"Orchestrator-{lane_letter}", "cmd": orch_cmd})

    return commands

@app.post("/api/run/cleanup")
async def start_cleanup_run(req: CleanupRunRequest):
    run_id = f"cleanup-{uuid.uuid4().hex[:8]}"
    cmd = [sys.executable, "scripts/run_cleanup.py"]
    if req.force:
        cmd.append("--force")

    runs[run_id] = {"logs": [], "status": "running", "cmd": " ".join(cmd), "type": "cleanup"}
    asyncio.create_task(execute_in_background(run_id, cmd))
    return {"run_id": run_id, "status": "started", "cmd": " ".join(cmd)}


@app.get("/api/runs")
def list_runs():
    return [
        {
            "id": r_id,
            "type": r_info["type"],
            "status": r_info["status"],
            "cmd": r_info["cmd"],
            "log_count": len(r_info["logs"])
        }
        for r_id, r_info in runs.items()
    ]

@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in runs:
        return {"error": "Run not found"}
    r_info = runs[run_id]
    return {
        "id": run_id,
        "type": r_info["type"],
        "status": r_info["status"],
        "cmd": r_info["cmd"],
        "log_count": len(r_info["logs"])
    }

@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    if run_id not in runs:
        return {"error": "Run not found"}

    r_info = runs[run_id]
    if r_info["status"] != "running":
        return {"error": "Run is not active"}

    process = r_info.get("process")
    if process:
        try:
            _terminate_process(process)
            r_info["status"] = "failed"
            r_info["logs"].append("\n--- Process forcefully terminated by user ---")

            return {"status": "stopped"}
        except Exception as e:
            return {"error": str(e)}

    if r_info.get("type") == "autonomy" and "processes" in r_info:
        _terminate_process_group(r_info["processes"])
        r_info["status"] = "failed"
        r_info["logs"].append("\n--- Processes forcefully terminated by user ---")
        return {"status": "stopped"}

    return {"error": "Process handle missing"}

@app.websocket("/ws/runs/{run_id}/logs")
async def websocket_logs(websocket: WebSocket, run_id: str):
    await websocket.accept()
    if run_id not in runs:
        await websocket.close(code=1008, reason="Run not found")
        return

    run_info = runs[run_id]
    sent_lines = 0

    try:
        while True:
            # Send all new lines
            current_logs = run_info["logs"]
            while sent_lines < len(current_logs):
                await websocket.send_text(current_logs[sent_lines])
                sent_lines += 1

            if run_info["status"] != "running" and sent_lines == len(run_info["logs"]):
                await websocket.send_text(f"\n--- Process exited with status: {run_info['status']} ---\n")
                break

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\nWebSocket error: {str(e)}\n")
            await websocket.close(code=1011)
        except Exception:
            pass
