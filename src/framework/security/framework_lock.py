"""Framework integrity lock via deterministic file-hash manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CORE_FILES: list[str] = [
    "src/framework/api.py",
    "src/framework/backtest/costs.py",
    "src/framework/backtest/engine.py",
    "src/framework/backtest/metrics.py",
    "src/framework/backtest/validators.py",
    "src/framework/data/bars.py",
    "src/framework/data/constants.py",
    "src/framework/data/loader.py",
    "src/framework/data/splits.py",
    "src/framework/features_canonical/aggressor.py",
    "src/framework/features_canonical/book.py",
    "src/framework/features_canonical/builder.py",
    "src/framework/features_canonical/footprint.py",
    "src/framework/features_canonical/labels.py",
    "src/framework/features_canonical/microstructure.py",
    "src/framework/features_canonical/microstructure_v2.py",
    "src/framework/features_canonical/momentum.py",
    "src/framework/features_canonical/ohlcv_indicators.py",
    "src/framework/features_canonical/opening_range.py",
    "src/framework/features_canonical/orderflow.py",
    "src/framework/features_canonical/pipeline.py",
    "src/framework/features_canonical/statistical.py",
    "src/framework/features_canonical/toxicity.py",
    "src/framework/features_canonical/volume_profile.py",
    "src/framework/security/framework_lock.py",
    "src/framework/validation/alpha_decay.py",
    "src/framework/validation/factor_attribution.py",
    "src/framework/validation/robustness.py",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_rel(path: str | Path) -> str:
    return str(Path(path).as_posix()).lstrip("./")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest(*, project_root: Path, rel_paths: list[str]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    seen: set[str] = set()
    files: list[dict[str, Any]] = []
    for raw in rel_paths:
        rel = _normalize_rel(raw)
        if rel in seen:
            continue
        seen.add(rel)
        abs_path = (root / rel).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Cannot lock missing file: {abs_path}")
        if not abs_path.is_file():
            raise ValueError(f"Cannot lock non-file path: {abs_path}")
        files.append(
            {
                "path": rel,
                "sha256": _sha256_file(abs_path),
                "size_bytes": int(abs_path.stat().st_size),
            }
        )

    files.sort(key=lambda x: str(x["path"]))
    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "project_root": str(root),
        "file_count": int(len(files)),
        "files": files,
    }


def save_manifest(manifest: dict[str, Any], out_path: Path) -> None:
    target = Path(out_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid manifest payload (not object): {path}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Invalid manifest payload (missing files[]): {path}")
    return payload


def verify_manifest(*, manifest_path: Path, project_root: Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    root = Path(project_root).resolve()
    payload = load_manifest(path)
    files = payload["files"]

    missing: list[str] = []
    modified: list[dict[str, str]] = []
    ok_count = 0

    for row in files:
        rel = _normalize_rel(row.get("path", ""))
        expected = str(row.get("sha256", ""))
        if not rel or not expected:
            modified.append(
                {
                    "path": rel or "<missing-path>",
                    "expected_sha256": expected or "<missing-sha256>",
                    "actual_sha256": "<invalid-manifest-row>",
                }
            )
            continue
        abs_path = (root / rel).resolve()
        if not abs_path.exists():
            missing.append(rel)
            continue
        actual = _sha256_file(abs_path)
        if actual != expected:
            modified.append(
                {
                    "path": rel,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                }
            )
            continue
        ok_count += 1

    ok = (len(missing) == 0) and (len(modified) == 0)
    return {
        "ok": bool(ok),
        "manifest_path": str(path),
        "project_root": str(root),
        "manifest_file_count": int(len(files)),
        "verified_file_count": int(ok_count),
        "missing_files": missing,
        "modified_files": modified,
    }
