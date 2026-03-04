"""Promotion-time verification helpers for candidate artifacts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def compute_file_hash(path: Path | str) -> str:
    """Compute full SHA-256 hash for file bytes."""
    file_path = Path(path)
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_hash(raw: str) -> str:
    val = str(raw).strip().lower()
    if val.startswith("sha256:"):
        return val.split(":", 1)[1]
    return val


def _resolve_path(raw: str, project_root: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (project_root / p)


def get_git_commit(project_root: Path, *, timeout_seconds: float = 10.0) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out reading git commit after {float(timeout_seconds):.1f}s",
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError("Unable to read git commit via `git rev-parse HEAD`")
    return proc.stdout.strip()


def verify_candidate_artifacts(
    *,
    candidate: dict[str, Any],
    project_root: Path,
    framework_manifest_path: Path,
    current_git_commit: str | None = None,
) -> dict[str, Any]:
    """Verify candidate artifact hashes + framework lock provenance."""
    root = Path(project_root).resolve()
    manifest = Path(framework_manifest_path).resolve()

    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Candidate missing artifacts object")
    provenance = candidate.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Candidate missing provenance object")

    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    signal_file = artifacts.get("signal_file")
    signal_hash = artifacts.get("signal_file_hash")
    if not signal_file or not signal_hash:
        raise ValueError("Candidate artifacts must contain signal_file and signal_file_hash")
    signal_path = _resolve_path(str(signal_file), root)
    if not signal_path.exists():
        raise FileNotFoundError(f"Signal file not found: {signal_path}")

    actual_signal_hash = compute_file_hash(signal_path)
    expected_signal_hash = _normalize_hash(str(signal_hash))
    if actual_signal_hash != expected_signal_hash:
        raise ValueError(
            f"Signal hash mismatch for {signal_path}: expected {expected_signal_hash}, got {actual_signal_hash}",
        )
    checks.append({"check": "signal_file_hash", "ok": True, "path": str(signal_path)})

    # Verify all optional artifact hash pairs: <artifact_key> + <artifact_key>_hash.
    for key, value in artifacts.items():
        if not key.endswith("_hash"):
            continue
        artifact_key = key[: -len("_hash")]
        if artifact_key == "signal_file":
            continue
        artifact_path_raw = artifacts.get(artifact_key)
        if not artifact_path_raw:
            continue
        artifact_path = _resolve_path(str(artifact_path_raw), root)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
        actual = compute_file_hash(artifact_path)
        expected = _normalize_hash(str(value))
        if actual != expected:
            raise ValueError(
                f"Artifact hash mismatch for {artifact_path}: expected {expected}, got {actual}",
            )
        checks.append({"check": key, "ok": True, "path": str(artifact_path)})

    expected_lock_hash = provenance.get("framework_lock_hash")
    if not expected_lock_hash:
        raise ValueError("Candidate provenance missing framework_lock_hash")
    actual_lock_hash = compute_file_hash(manifest)
    if actual_lock_hash != _normalize_hash(str(expected_lock_hash)):
        raise ValueError(
            "Framework lock hash mismatch: "
            f"candidate={expected_lock_hash}, current={actual_lock_hash}",
        )
    checks.append({"check": "framework_lock_hash", "ok": True, "path": str(manifest)})

    expected_commit = str(provenance.get("git_commit", "")).strip()
    if expected_commit:
        observed_commit = current_git_commit or get_git_commit(root)
        if observed_commit != expected_commit:
            warnings.append(
                "Git commit differs from candidate provenance: "
                f"candidate={expected_commit}, current={observed_commit}",
            )
        checks.append({"check": "git_commit", "ok": observed_commit == expected_commit})

    return {
        "ok": True,
        "checks": checks,
        "warnings": warnings,
    }
