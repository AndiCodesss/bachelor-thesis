from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.lib.candidates import write_candidate


def test_only_validator_can_write_candidate(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        write_candidate(
            agent_name="creative-researcher",
            candidate_data={"strategy_id": "s1"},
            candidate_dir=tmp_path,
        )


def test_candidate_write_once_policy(tmp_path: Path) -> None:
    out = write_candidate(
        agent_name="validator",
        candidate_data={"strategy_id": "alpha_1", "validation_metrics": {"sharpe": 1.2}},
        candidate_dir=tmp_path,
    )
    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["strategy_id"] == "alpha_1"

    with pytest.raises(FileExistsError):
        write_candidate(
            agent_name="validator",
            candidate_data={"strategy_id": "alpha_1"},
            candidate_dir=tmp_path,
        )

