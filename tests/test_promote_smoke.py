from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def test_promote_entrypoint_verified_only_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    signal = root / "research/signals/example_ema_turn.py"
    lock = root / "configs/framework_lock.json"
    candidate = tmp_path / "candidate.json"
    out = tmp_path / "promotion_report.json"

    payload = {
        "strategy_id": "example_ema_turn_test",
        "artifacts": {
            "signal_file": str(signal),
            "signal_file_hash": _sha(signal),
        },
        "provenance": {
            "framework_lock_hash": _sha(lock),
        },
    }
    with open(candidate, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/promote.py",
            "--candidate",
            str(candidate),
            "--framework-lock-manifest",
            str(lock),
            "--framework-lock-mode",
            "warn",
            "--verify-only",
            "--out",
            str(out),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["status"] == "VERIFIED_ONLY"
    assert report["artifact_verification"]["ok"] is True
