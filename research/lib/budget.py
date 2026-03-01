"""Persistent mission budget control for autonomous research runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.lib.atomic_io import atomic_json_write


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MissionBudget:
    max_experiments: int
    kill_criteria: dict[str, int]
    state_file: Path
    mission_name: str | None = None
    reset_on_mission_change: bool = True

    def __post_init__(self) -> None:
        self.state_file = Path(self.state_file)
        self.started_at = _utc_now()
        if self.state_file.exists():
            import json

            with open(self.state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            saved_mission = payload.get("mission_name")
            if (
                self.reset_on_mission_change
                and self.mission_name
                and saved_mission
                and str(saved_mission) != str(self.mission_name)
            ):
                self.experiments_run = 0
                self.failures_by_type = {}
                self.started_at = _utc_now()
                self._save()
            else:
                self.experiments_run = int(payload.get("experiments_run", 0))
                self.failures_by_type = {
                    str(k): int(v) for k, v in dict(payload.get("failures_by_type", {})).items()
                }
                self.started_at = str(payload.get("started_at") or self.started_at)
        else:
            self.experiments_run = 0
            self.failures_by_type: dict[str, int] = {}
            self._save()

    def _save(self) -> None:
        payload = {
            "schema_version": "1.0",
            "mission_name": self.mission_name,
            "experiments_run": int(self.experiments_run),
            "failures_by_type": dict(self.failures_by_type),
            "started_at": self.started_at,
            "last_updated": _utc_now(),
        }
        atomic_json_write(self.state_file, payload)

    def check_budget(self) -> tuple[bool, str]:
        if self.experiments_run >= int(self.max_experiments):
            return False, f"max_experiments_reached:{self.max_experiments}"
        for verdict, limit in self.kill_criteria.items():
            if int(self.failures_by_type.get(verdict, 0)) >= int(limit):
                return False, f"kill_criteria_reached:{verdict}:{limit}"
        return True, ""

    def record_experiment(self, verdict: str) -> None:
        self.experiments_run += 1
        if verdict in {"FAIL", "ERROR", "ABANDON"}:
            self.failures_by_type[verdict] = self.failures_by_type.get(verdict, 0) + 1
        self._save()

    def snapshot(self) -> dict[str, Any]:
        return {
            "mission_name": self.mission_name,
            "experiments_run": int(self.experiments_run),
            "max_experiments": int(self.max_experiments),
            "failures_by_type": dict(self.failures_by_type),
            "kill_criteria": dict(self.kill_criteria),
            "started_at": self.started_at,
        }
