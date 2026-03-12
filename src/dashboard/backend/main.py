from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.dashboard.backend.run_state import (
    RunRecord,
    append_run_log,
    new_run_record,
    set_run_status,
    summarize_run,
)
from src.dashboard.backend.services import (
    collect_autonomy_status,
    get_signal_details as get_signal_details_payload,
    list_signals as list_signal_rows,
)
from src.framework.data.bar_configs import BAR_CONFIGS, bar_config_label

project_root = Path(__file__).resolve().parent.parent.parent.parent

AVAILABLE_BARS = [bar_config_label(cfg) for cfg in BAR_CONFIGS]
LOCAL_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

app = FastAPI(title="NQ-Alpha Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LOCAL_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


runs: dict[str, RunRecord] = {}


def _reset_runtime_state(mission_ref: str) -> None:
    from research.lib.runtime_state import clear_orchestrator_state, reset_shared_state
    from research.lib.script_support import load_yaml_dict, mission_state_fingerprint

    mission_path = Path(mission_ref)
    if not mission_path.is_absolute():
        mission_path = project_root / mission_path
    mission_path = mission_path.resolve()
    mission_name = mission_path.stem
    mission_fingerprint: str | None = None
    if mission_path.exists():
        try:
            payload = load_yaml_dict(mission_path)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            mission_name = str(payload.get("mission_name", mission_path.stem))
            mission_fingerprint = mission_state_fingerprint(payload)
    reset_shared_state(
        project_root,
        mission_name=mission_name,
        mission_fingerprint=mission_fingerprint,
    )
    clear_orchestrator_state(project_root)


def _terminate_process(proc) -> None:
    pid = getattr(proc, "pid", None)
    if pid is None or getattr(proc, "returncode", None) is not None:
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
    pgids: list[int] = []
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
        except (ProcessLookupError, OSError):
            pass


def _subprocess_exec_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _lane_letters(lane_count: int) -> list[str]:
    count = max(1, min(10, int(lane_count)))
    return [chr(ord("A") + idx) for idx in range(count)]


def _build_autonomy_commands(req: AutonomyRunRequest) -> list[dict[str, list[str] | str]]:
    commands: list[dict[str, list[str] | str]] = []
    state_flag = [] if req.no_resume else ["--resume"]

    if not req.orchestrator_only:
        validator_cmd = [
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
            validator_cmd.append("--no-bootstrap")
        commands.append({"name": "Validator", "cmd": validator_cmd})

    if not req.validator_only:
        for lane_letter in _lane_letters(req.lane_count):
            orchestrator_cmd = [
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
                orchestrator_cmd.append("--disable-notebooklm")
            commands.append({"name": f"Orchestrator-{lane_letter}", "cmd": orchestrator_cmd})

    return commands


@app.get("/api/config/cache")
def get_cache_config():
    return {
        "splits": ["train", "validate", "test", "all"],
        "session_filters": ["eth", "rth", "both"],
        "exec_modes": ["auto", "research", "promotion"],
        "bar_filters": AVAILABLE_BARS,
    }


@app.get("/api/autonomy/status")
def get_autonomy_status():
    return collect_autonomy_status(project_root=project_root)


@app.get("/api/signals")
def list_signals():
    return list_signal_rows(project_root=project_root)


@app.get("/api/signals/{strategy_name}")
def get_signal_details(strategy_name: str):
    return get_signal_details_payload(project_root=project_root, strategy_name=strategy_name)


async def execute_in_background(run_id: str, cmd: list[str]):
    run = runs[run_id]
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=os.environ.copy(),
            **_subprocess_exec_kwargs(),
        )
        run["process"] = process

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            append_run_log(run, line.decode("utf-8", errors="replace"))

        await process.wait()
        set_run_status(run, "completed" if process.returncode == 0 else "failed")
    except Exception as exc:
        append_run_log(run, f"\nError reading logs: {exc}")
        set_run_status(run, "failed")


async def direct_run_autonomy(run_id: str, commands: list[dict[str, list[str] | str]]):
    run = runs[run_id]
    active_procs: list[tuple[str, asyncio.subprocess.Process]] = []
    run["process"] = None
    run["processes"] = []

    try:
        for command_spec in commands:
            name = str(command_spec["name"])
            cmd = list(command_spec["cmd"])
            append_run_log(run, f"--- Starting {name} ---\n")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=os.environ.copy(),
                **_subprocess_exec_kwargs(),
            )
            active_procs.append((name, proc))
            run["processes"].append(proc)

        async def _read_stream(name: str, stream) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                append_run_log(run, f"[{name}] {line.decode('utf-8', errors='replace')}")

        readers = [asyncio.create_task(_read_stream(name, proc.stdout)) for name, proc in active_procs]
        await asyncio.gather(*readers)

        failures: list[tuple[str, int]] = []
        for name, proc in active_procs:
            return_code = await proc.wait()
            if return_code != 0:
                failures.append((name, return_code))

        if failures:
            details = ", ".join(f"{name} exited {code}" for name, code in failures)
            append_run_log(run, f"\n--- Autonomy processes failed: {details} ---\n")
            set_run_status(run, "failed")
        else:
            append_run_log(run, "\n--- Autonomy processes completed ---\n")
            set_run_status(run, "completed")
    except Exception as exc:
        append_run_log(run, f"\nError running autonomy: {exc}\n")
        set_run_status(run, "failed")
    finally:
        for proc in run.get("processes", []):
            if proc.returncode is None:
                _terminate_process(proc)


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

    runs[run_id] = new_run_record(run_type="cache", cmd=" ".join(cmd))
    asyncio.create_task(execute_in_background(run_id, cmd))
    return {"run_id": run_id, "status": "started", "cmd": " ".join(cmd)}


@app.post("/api/run/autonomy")
async def start_autonomy_run(req: AutonomyRunRequest):
    run_id = f"autonomy-{uuid.uuid4().hex[:8]}"
    commands = _build_autonomy_commands(req)
    fresh_state = bool(req.no_resume)

    if req.validator_only and req.orchestrator_only:
        raise HTTPException(
            status_code=400,
            detail="Cannot request validator-only and orchestrator-only at the same time.",
        )
    if fresh_state and req.orchestrator_only:
        raise HTTPException(
            status_code=400,
            detail="Fresh state requires launching the validator or running cleanup first.",
        )
    if not commands:
        raise HTTPException(status_code=400, detail="No autonomy workers selected.")

    if fresh_state:
        _reset_runtime_state(req.mission)

    runs[run_id] = new_run_record(
        run_type="autonomy",
        cmd=f"Directly running {len(commands)} background processes",
    )
    asyncio.create_task(direct_run_autonomy(run_id, commands))
    return {"run_id": run_id, "status": "started", "cmd": runs[run_id]["cmd"]}


@app.post("/api/run/cleanup")
async def start_cleanup_run(req: CleanupRunRequest):
    run_id = f"cleanup-{uuid.uuid4().hex[:8]}"
    cmd = [sys.executable, "scripts/run_cleanup.py"]
    if req.force:
        cmd.append("--force")

    runs[run_id] = new_run_record(run_type="cleanup", cmd=" ".join(cmd))
    asyncio.create_task(execute_in_background(run_id, cmd))
    return {"run_id": run_id, "status": "started", "cmd": " ".join(cmd)}


@app.get("/api/runs")
def list_runs():
    return [summarize_run(run_id, run) for run_id, run in runs.items()]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = runs.get(run_id)
    if run is None:
        return {"error": "Run not found"}
    return summarize_run(run_id, run)


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    run = runs.get(run_id)
    if run is None:
        return {"error": "Run not found"}
    if run["status"] != "running":
        return {"error": "Run is not active"}

    process = run.get("process")
    if process:
        try:
            _terminate_process(process)
        except Exception as exc:
            return {"error": str(exc)}
        set_run_status(run, "failed")
        append_run_log(run, "\n--- Process forcefully terminated by user ---")
        return {"status": "stopped"}

    if run.get("type") == "autonomy" and "processes" in run:
        _terminate_process_group(run["processes"])
        set_run_status(run, "failed")
        append_run_log(run, "\n--- Processes forcefully terminated by user ---")
        return {"status": "stopped"}

    return {"error": "Process handle missing"}


@app.websocket("/ws/runs/{run_id}/logs")
async def websocket_logs(websocket: WebSocket, run_id: str):
    await websocket.accept()
    run = runs.get(run_id)
    if run is None:
        await websocket.close(code=1008, reason="Run not found")
        return

    sent_lines = 0
    try:
        while True:
            current_logs = run["logs"]
            while sent_lines < len(current_logs):
                await websocket.send_text(current_logs[sent_lines])
                sent_lines += 1

            if run["status"] != "running" and sent_lines == len(run["logs"]):
                await websocket.send_text(f"\n--- Process exited with status: {run['status']} ---\n")
                break

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(f"\nWebSocket error: {exc}\n")
            await websocket.close(code=1011)
        except Exception:
            pass
