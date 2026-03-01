"""Tests for framework hash-lock manifest helpers."""

from __future__ import annotations

from pathlib import Path

from src.framework.security.framework_lock import build_manifest, verify_manifest, save_manifest


def test_framework_lock_verify_passes_and_detects_drift(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    file_a = root / "a.txt"
    file_b = root / "b.txt"
    file_a.write_text("alpha\n", encoding="utf-8")
    file_b.write_text("beta\n", encoding="utf-8")

    manifest = build_manifest(project_root=root, rel_paths=["a.txt", "b.txt"])
    manifest_path = root / "lock.json"
    save_manifest(manifest, manifest_path)

    ok = verify_manifest(manifest_path=manifest_path, project_root=root)
    assert ok["ok"] is True
    assert ok["verified_file_count"] == 2
    assert ok["missing_files"] == []
    assert ok["modified_files"] == []

    file_b.write_text("beta changed\n", encoding="utf-8")
    drift = verify_manifest(manifest_path=manifest_path, project_root=root)
    assert drift["ok"] is False
    assert len(drift["modified_files"]) == 1
    assert drift["modified_files"][0]["path"] == "b.txt"


def test_framework_lock_verify_detects_missing_file(tmp_path: Path):
    root = tmp_path / "proj2"
    root.mkdir(parents=True, exist_ok=True)
    file_a = root / "a.txt"
    file_a.write_text("alpha\n", encoding="utf-8")

    manifest = build_manifest(project_root=root, rel_paths=["a.txt"])
    manifest_path = root / "lock.json"
    save_manifest(manifest, manifest_path)

    file_a.unlink()
    out = verify_manifest(manifest_path=manifest_path, project_root=root)
    assert out["ok"] is False
    assert out["missing_files"] == ["a.txt"]

