from __future__ import annotations

from pathlib import Path

import pytest

from research.lib.promotion import compute_file_hash, verify_candidate_artifacts


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_verify_candidate_artifacts_passes(tmp_path: Path) -> None:
    root = tmp_path
    signal = root / "research/signals/s1.py"
    lock = root / "configs/framework_lock.json"
    _write(signal, "def generate_signal(df, params):\n    return [0]\n")
    _write(lock, '{"schema_version": 1}')

    candidate = {
        "strategy_id": "s1",
        "artifacts": {
            "signal_file": "research/signals/s1.py",
            "signal_file_hash": compute_file_hash(signal),
        },
        "provenance": {
            "framework_lock_hash": compute_file_hash(lock),
            "git_commit": "abc123",
        },
    }

    out = verify_candidate_artifacts(
        candidate=candidate,
        project_root=root,
        framework_manifest_path=lock,
        current_git_commit="abc123",
    )
    assert out["ok"] is True
    assert out["warnings"] == []


def test_verify_candidate_artifacts_rejects_signal_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path
    signal = root / "research/signals/s1.py"
    lock = root / "configs/framework_lock.json"
    _write(signal, "def generate_signal(df, params):\n    return [0]\n")
    _write(lock, '{"schema_version": 1}')

    candidate = {
        "strategy_id": "s1",
        "artifacts": {
            "signal_file": "research/signals/s1.py",
            "signal_file_hash": "0" * 64,
        },
        "provenance": {
            "framework_lock_hash": compute_file_hash(lock),
        },
    }

    with pytest.raises(ValueError, match="Signal hash mismatch"):
        verify_candidate_artifacts(
            candidate=candidate,
            project_root=root,
            framework_manifest_path=lock,
            current_git_commit="abc123",
        )

