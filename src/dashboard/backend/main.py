from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import uuid
import os
import sys
import json
from pathlib import Path
from statistics import mean

# Fix path to import from src
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.framework.features_canonical.builder import BAR_CONFIGS
    def get_bar_label(cfg):
        if cfg["bar_type"] == "time":
            return cfg["bar_size"]
        if cfg["bar_type"] == "volume":
            return f"vol_{cfg['bar_threshold']}"
        return f"{cfg['bar_type']}_{cfg['bar_threshold']}"
    AVAILABLE_BARS = [get_bar_label(cfg) for cfg in BAR_CONFIGS]
except ImportError:
    AVAILABLE_BARS = []

app = FastAPI(title="NQ-Alpha Central Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    no_restart: bool = False
    no_resume: bool = False
    allow_bootstrap: bool = False
    validator_only: bool = False
    orchestrator_only: bool = False

class CleanupRunRequest(BaseModel):
    force: bool = False

runs = {}

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
    if not path.exists(): return []
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return data[-max_lines:] if max_lines > 0 else data

def _recent_json_events(path: Path, event_names: set[str], limit: int, scan_lines: int = 3000):
    out = []
    for raw in reversed(_tail_lines(path, max_lines=scan_lines)):
        line = raw.strip()
        if not line: continue
        try:
            row = json.loads(line)
            if isinstance(row, dict) and str(row.get("event", "")) in event_names:
                out.append(row)
                if len(out) >= limit: break
        except Exception: continue
    return list(reversed(out))

# --- Thinker Session Helpers ---

SESSIONS_DIR = Path.home() / ".claude" / "projects" / "-mnt-c-Users-Andreas-Oberd-rfer-Downloads-bachelor"


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

    lines = _tail_lines(jsonl_path, max_lines=200)
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


@app.get("/api/autonomy/status")
def get_autonomy_status():
    state_dir = project_root / "research" / ".state"
    logs_dir = project_root / "results" / "logs"
    queue_path = state_dir / "experiment_queue.json"
    budget_path = state_dir / "mission_budget.json"
    exp_log = logs_dir / "research_experiments.jsonl"
    llm_log = logs_dir / "llm_orchestrator.jsonl"
    
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
        if queue_path.exists():
            queue = json.loads(queue_path.read_text())
            tasks = queue.get("tasks", [])
            for t in tasks:
                state = t.get("state")
                if state in metrics["queue"]: metrics["queue"][state] += 1
            
            # Active hypotheses
            hyp_counts = {}
            for t in tasks:
                if t.get("state") in ["pending", "in_progress"]:
                    hid = str(t.get("source", {}).get("hypothesis_id", "")).strip()
                    if hid: hyp_counts[hid] = hyp_counts.get(hid, 0) + 1
            metrics["active_hypotheses"] = [{"id": k, "tasks": v} for k, v in sorted(hyp_counts.items(), key=lambda x: -x[1])[:4]]
    except Exception: pass

    # Budget
    try:
        if budget_path.exists():
            b = json.loads(budget_path.read_text())
            metrics["budget"]["experiments_run"] = b.get("experiments_run", 0)
            metrics["budget"]["max_experiments"] = b.get("max_experiments", "n/a")
            metrics["budget"]["failures"] = b.get("failures_by_type", {})
    except Exception: pass

    # Financial Snapshot
    try:
        recent_tasks = _recent_json_events(exp_log, {"task_result"}, limit=10, scan_lines=5000)
        points = []
        pass_count = 0
        for row in recent_tasks:
            m = row.get("metrics")
            if not isinstance(m, dict): continue
            npnl, sharpe = m.get("net_pnl"), m.get("sharpe_ratio")
            if isinstance(npnl, (int, float)) and isinstance(sharpe, (int, float)):
                points.append({
                    "strategy": str(row.get("strategy_name", "")),
                    "bar": str(row.get("bar_config", "")),
                    "net_pnl": float(npnl),
                    "sharpe": float(sharpe),
                    "trades": m.get("trade_count")
                })
            if str(row.get("verdict", "")) == "PASS": pass_count += 1
        
        if points:
            metrics["financial"] = {
                "tested": len(points),
                "avg_net_pnl": mean(p["net_pnl"] for p in points),
                "avg_sharpe": mean(p["sharpe"] for p in points),
                "pass_rate_pct": (100.0 * pass_count / len(recent_tasks)) if recent_tasks else 0.0,
                "best": max(points, key=lambda x: x["net_pnl"]),
                "worst": min(points, key=lambda x: x["net_pnl"])
            }
    except Exception: pass

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
            if not strat or strat in seen: continue
            
            seen.add(strat)
            signals.append({
                "strategy": strat,
                "verdict": row.get("verdict", "UNKNOWN"),
                "timestamp": row.get("timestamp", ""),
                "bar_config": row.get("bar_config", "")
            })
    except Exception: pass
    
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
    except Exception: pass
    
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
            env=os.environ.copy()
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
    import subprocess
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
                env=os.environ.copy()
            )
            active_procs.append((name, proc))
            runs[run_id]["processes"].append(proc)
            
        async def read_stream(name, stream):
            while True:
                line = await stream.readline()
                if not line: break
                text = line.decode('utf-8', errors='replace')
                runs[run_id]["logs"].append(f"[{name}] {text}")

        readers = [asyncio.create_task(read_stream(name, p.stdout)) for name, p in active_procs]
        
        await asyncio.gather(*readers)
        
        for name, p in active_procs:
            await p.wait()
            
        runs[run_id]["status"] = "completed"
        runs[run_id]["logs"].append("\n--- Autonomy processes completed ---\n")
        
    except Exception as e:
        runs[run_id]["logs"].append(f"\nError running autonomy: {str(e)}\n")
        runs[run_id]["status"] = "failed"
    finally:
        for p in runs[run_id].get("processes", []):
            try:
                p.terminate()
            except:
                pass

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
    
    commands = []
    resume_flag = [] if req.no_resume else ["--resume"]
    
    if not req.orchestrator_only:
        val_cmd = [sys.executable, "-u", "scripts/research.py", "--mission", req.mission, "--auto-mode", "--worker-agent", "validator"] + resume_flag
        if not req.allow_bootstrap:
            val_cmd.append("--no-bootstrap")
        commands.append({"name": "Validator", "cmd": val_cmd})
        
    if not req.validator_only:
        orch_cmd = [sys.executable, "-u", "scripts/llm_orchestrator.py", "--mission", req.mission, "--agent-config", req.agent_config] + resume_flag
        commands.append({"name": "Orchestrator", "cmd": orch_cmd})

    runs[run_id] = {"logs": [], "status": "running", "cmd": f"Directly running {len(commands)} background processes", "type": "autonomy"}
    asyncio.create_task(direct_run_autonomy(run_id, commands))
    return {"run_id": run_id, "status": "started", "cmd": runs[run_id]["cmd"]}

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
            process.terminate()
            r_info["status"] = "failed"
            r_info["logs"].append("\n--- Process forcefully terminated by user ---")
            
            return {"status": "stopped"}
        except Exception as e:
            return {"error": str(e)}
            
    if r_info.get("type") == "autonomy" and "processes" in r_info:
        for p in r_info["processes"]:
            try:
                p.terminate()
            except:
                pass
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
        except:
            pass
