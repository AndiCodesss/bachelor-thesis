"""Candidate persistence helpers with validator-only write policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research.lib.atomic_io import atomic_json_write


def compute_file_hash(path: Path | str) -> str:
    """Return full SHA-256 hash for a file."""
    file_path = Path(path)
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_candidate(
    *,
    agent_name: str,
    candidate_data: dict[str, Any],
    candidate_dir: Path | str = Path("research/candidates"),
) -> Path:
    """Write immutable candidate JSON (validator only; write-once)."""
    if agent_name != "validator":
        raise PermissionError("Only validator can write candidates")
    strategy_id = str(candidate_data.get("strategy_id", "")).strip()
    if not strategy_id:
        raise ValueError("candidate_data.strategy_id is required")

    out_dir = Path(candidate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{strategy_id}.json"
    if out.exists():
        raise FileExistsError(
            f"Candidate already exists: {out}. "
            "Create a new strategy_id for a new version.",
        )

    payload = dict(candidate_data)
    payload.setdefault("schema_version", "1.0")
    atomic_json_write(out, payload)

    # Best effort read-only hardening for immutable candidate artifacts.
    try:
        out.chmod(0o444)
    except OSError:
        pass

    return out


def load_candidate(path: Path | str) -> dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Candidate file must contain a JSON object")
    return payload

