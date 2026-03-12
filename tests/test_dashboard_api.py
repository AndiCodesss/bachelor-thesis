from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.dashboard.backend.main as dashboard_main


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dashboard_main, "project_root", tmp_path)
    dashboard_main.runs.clear()
    return TestClient(dashboard_main.app)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cache_config_and_localhost_cors(client: TestClient):
    resp = client.get("/api/config/cache")
    assert resp.status_code == 200
    assert resp.json()["bar_filters"] == ["tick_610", "vol_2000", "1m"]

    preflight = client.options(
        "/api/config/cache",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"

    blocked = client.options(
        "/api/config/cache",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert blocked.status_code == 400
    assert "access-control-allow-origin" not in blocked.headers


def test_autonomy_status_reads_runtime_state_and_budget(client: TestClient):
    state_dir = dashboard_main.project_root / "research" / ".state"
    logs_dir = dashboard_main.project_root / "results" / "logs"

    _write_json(
        state_dir / "experiment_queue.json",
        {
            "schema_version": "1.0",
            "tasks": [
                {"task_id": "t1", "state": "pending", "source": {"hypothesis_id": "hyp-alpha"}},
                {"task_id": "t2", "state": "in_progress", "source": {"hypothesis_id": "hyp-alpha"}},
                {"task_id": "t3", "state": "in_progress", "source": {"hypothesis_id": "hyp-beta"}},
                {"task_id": "t4", "state": "completed", "source": {"hypothesis_id": "hyp-gamma"}},
                {"task_id": "t5", "state": "failed", "source": {"hypothesis_id": "hyp-delta"}},
            ],
        },
    )
    _write_json(
        state_dir / "mission_budget.json",
        {
            "schema_version": "1.0",
            "experiments_run": 17,
            "max_experiments": 50,
            "failures_by_type": {"FAIL": 3, "ERROR": 1},
            "kill_criteria": {"FAIL": 10, "ERROR": 5},
        },
    )
    _write_jsonl(
        logs_dir / "research_experiments.jsonl",
        [
            {
                "event": "task_result",
                "timestamp": "2026-03-06T10:15:00+00:00",
                "strategy_name": "alpha_retest",
                "bar_config": "tick_610",
                "verdict": "PASS",
                "search_result": {
                    "signal_count": 18,
                    "edge_surface": {
                        "status": "global_edge",
                        "global_probe": {
                            "base_event_count": 18,
                            "best_horizon_bars": 5,
                            "best_avg_trade_pnl": 32.5,
                        },
                    },
                },
                "metrics": {"net_pnl": 1200.0, "sharpe_ratio": 1.8, "trade_count": 14},
            },
            {
                "event": "task_result",
                "timestamp": "2026-03-06T10:16:00+00:00",
                "strategy_name": "failed_breakout",
                "bar_config": "1m",
                "verdict": "FAIL",
                "failure_code": "no_edge",
                "search_result": {
                    "signal_count": 142,
                    "edge_surface": {
                        "status": "no_edge",
                        "global_probe": {
                            "base_event_count": 140,
                            "best_horizon_bars": 1,
                            "best_avg_trade_pnl": -8.25,
                        },
                    },
                },
                "metrics": {"net_pnl": -300.0, "sharpe_ratio": -0.4, "trade_count": 0},
            },
        ],
    )

    resp = client.get("/api/autonomy/status")
    assert resp.status_code == 200
    data = resp.json()

    assert data["queue"] == {"pending": 1, "in_progress": 2, "completed": 1, "failed": 1}
    assert data["budget"] == {
        "experiments_run": 17,
        "max_experiments": 50,
        "failures": {"FAIL": 3, "ERROR": 1},
    }
    assert data["active_hypotheses"][0] == {"id": "hyp-alpha", "tasks": 2}
    assert data["financial"]["tested"] == 2
    assert data["financial"]["avg_net_pnl"] == 450.0
    assert data["financial"]["avg_sharpe"] == 0.7
    assert data["financial"]["pass_rate_pct"] == 50.0
    assert data["financial"]["best"]["strategy"] == "alpha_retest"
    assert data["financial"]["worst"]["strategy"] == "failed_breakout"
    assert data["recent_results"][0]["strategy"] == "failed_breakout"
    assert data["recent_results"][0]["signal_count"] == 142
    assert data["recent_results"][0]["edge_events"] == 140
    assert data["recent_results"][0]["edge_status"] == "no_edge"
    assert data["recent_results"][0]["backtest_trades"] == 0
    assert data["recent_results"][1]["strategy"] == "alpha_retest"
    assert data["recent_results"][1]["best_horizon_bars"] == 5


def test_dashboard_endpoints_sanitize_nan_metrics(client: TestClient):
    logs_dir = dashboard_main.project_root / "results" / "logs"

    _write_jsonl(
        logs_dir / "research_experiments.jsonl",
        [
            {
                "event": "task_result",
                "timestamp": "2026-03-06T10:15:00+00:00",
                "strategy_name": "nan_case",
                "bar_config": "1m",
                "verdict": "PASS",
                "metrics": {"net_pnl": 100.0, "sharpe_ratio": float("nan"), "trade_count": 1},
            },
        ],
    )

    status_resp = client.get("/api/autonomy/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["financial"]["tested"] == 0
    assert status_data["recent_results"][0]["strategy"] == "nan_case"
    assert status_data["recent_results"][0]["sharpe"] is None

    detail_resp = client.get("/api/signals/nan_case")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["metrics"]["sharpe_ratio"] is None
    assert detail_data["metrics"]["net_pnl"] == 100.0


def test_signals_endpoints_return_recent_results_and_source_code(client: TestClient):
    logs_dir = dashboard_main.project_root / "results" / "logs"
    signals_dir = dashboard_main.project_root / "research" / "signals"

    _write_jsonl(
        logs_dir / "research_experiments.jsonl",
        [
            {
                "event": "task_result",
                "timestamp": "2026-03-06T10:15:00+00:00",
                "strategy_name": "mean_rev_probe",
                "bar_config": "tick_610",
                "verdict": "PASS",
                "metrics": {"net_pnl": 900.0, "sharpe_ratio": 1.2},
                "gauntlet": {"verdict": "PASS"},
            },
            {
                "event": "task_result",
                "timestamp": "2026-03-06T10:14:00+00:00",
                "strategy_name": "mean_rev_probe",
                "bar_config": "1m",
                "verdict": "FAIL",
                "metrics": {"net_pnl": -10.0, "sharpe_ratio": -0.1},
                "gauntlet": {"verdict": "FAIL"},
            },
            {
                "event": "task_result",
                "timestamp": "2026-03-06T10:16:00+00:00",
                "strategy_name": "open_drive_hold",
                "bar_config": "vol_2000",
                "verdict": "NEEDS_WORK",
                "metrics": {"net_pnl": 250.0, "sharpe_ratio": 0.6},
                "gauntlet": {"verdict": "FAIL"},
            },
        ],
    )
    signals_dir.mkdir(parents=True, exist_ok=True)
    (signals_dir / "mean_rev_probe.py").write_text(
        "def generate_signals(df):\n    return df\n",
        encoding="utf-8",
    )

    listing = client.get("/api/signals")
    assert listing.status_code == 200
    rows = listing.json()
    assert [row["strategy"] for row in rows] == ["open_drive_hold", "mean_rev_probe"]
    assert rows[1]["verdict"] == "PASS"
    assert rows[1]["bar_config"] == "tick_610"

    detail = client.get("/api/signals/mean_rev_probe")
    assert detail.status_code == 200
    data = detail.json()
    assert data["strategy"] == "mean_rev_probe"
    assert data["verdict"] == "PASS"
    assert data["metrics"]["net_pnl"] == 900.0
    assert "def generate_signals" in data["code"]


def test_autonomy_run_rejects_invalid_worker_selection(client: TestClient):
    resp = client.post(
        "/api/run/autonomy",
        json={"validator_only": True, "orchestrator_only": True},
    )
    assert resp.status_code == 400
    assert "validator-only" in resp.json()["detail"]


def test_build_autonomy_commands_disables_notebooklm_only_for_orchestrators():
    req = dashboard_main.AutonomyRunRequest(
        mission="configs/missions/alpha-discovery.yaml",
        agent_config="configs/agents/llm_orchestrator.yaml",
        use_notebooklm=False,
        lane_count=2,
    )

    commands = dashboard_main._build_autonomy_commands(req)

    assert commands[0]["name"] == "Validator"
    assert "--disable-notebooklm" not in commands[0]["cmd"]
    assert commands[1]["name"] == "Orchestrator-A"
    assert commands[2]["name"] == "Orchestrator-B"
    assert commands[1]["cmd"][-1] == "--disable-notebooklm"
    assert commands[2]["cmd"][-1] == "--disable-notebooklm"


def test_build_autonomy_commands_omits_resume_for_fresh_state():
    req = dashboard_main.AutonomyRunRequest(no_resume=True, lane_count=1)

    commands = dashboard_main._build_autonomy_commands(req)

    assert "--resume" not in commands[0]["cmd"]
    assert "--resume" not in commands[1]["cmd"]


def test_stop_run_uses_tree_termination_for_single_process(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    stopped: list[object] = []

    def fake_terminate(proc) -> None:
        stopped.append(proc)

    process = object()
    dashboard_main.runs["run-1"] = {
        "type": "cache",
        "status": "running",
        "cmd": ["uv", "run", "python"],
        "logs": [],
        "process": process,
    }
    monkeypatch.setattr(dashboard_main, "_terminate_process", fake_terminate)

    resp = client.post("/api/runs/run-1/stop")

    assert resp.status_code == 200
    assert resp.json() == {"status": "stopped"}
    assert stopped == [process]
    assert dashboard_main.runs["run-1"]["status"] == "failed"
    assert "forcefully terminated by user" in "".join(dashboard_main.runs["run-1"]["logs"])


def test_execute_in_background_launches_in_own_process_group(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    class _FakeStdout:
        async def readline(self):
            return b""

    class _FakeProcess:
        def __init__(self):
            self.stdout = _FakeStdout()
            self.returncode = 0

        async def wait(self):
            return 0

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(dashboard_main.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    dashboard_main.runs.clear()
    dashboard_main.runs["run-2"] = {
        "type": "cache",
        "status": "running",
        "cmd": ["uv", "run", "python"],
        "logs": [],
    }

    asyncio.run(dashboard_main.execute_in_background("run-2", ["uv", "run", "python"]))

    assert captured["cmd"] == ("uv", "run", "python")
    kwargs = captured["kwargs"]
    for key, value in dashboard_main._subprocess_exec_kwargs().items():
        assert kwargs[key] == value


def test_terminate_process_group_uses_ctrl_break_on_windows(monkeypatch: pytest.MonkeyPatch):
    class _FakeProc:
        def __init__(self):
            self.pid = 1234
            self.returncode = None
            self.sent: list[int] = []
            self.terminated = False
            self.killed = False

        def send_signal(self, sig: int) -> None:
            self.sent.append(sig)
            self.returncode = 0

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    proc = _FakeProc()
    monkeypatch.setattr(dashboard_main.os, "name", "nt", raising=False)
    monkeypatch.setattr(dashboard_main.signal, "CTRL_BREAK_EVENT", 777, raising=False)
    monkeypatch.setattr(dashboard_main.time, "sleep", lambda _seconds: None)

    dashboard_main._terminate_process_group([proc])

    assert proc.sent == [777]
    assert proc.terminated is False
    assert proc.killed is False
