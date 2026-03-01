#!/usr/bin/env python3
"""Verify framework integrity manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.framework.security.framework_lock import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify framework lock manifest.")
    parser.add_argument(
        "--manifest",
        default="configs/framework_lock.json",
        help="Manifest path (project-relative or absolute).",
    )
    parser.add_argument(
        "--mode",
        choices=["warn", "error"],
        default="error",
        help="error: non-zero exit on mismatch; warn: print warnings only.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (project_root / manifest_path).resolve()

    result = verify_manifest(manifest_path=manifest_path, project_root=project_root)
    status = "PASS" if result["ok"] else "FAIL"
    print(
        f"Framework lock: {status} "
        f"(verified={result['verified_file_count']}/{result['manifest_file_count']})",
    )
    if result["missing_files"]:
        print("Missing files:")
        for rel in result["missing_files"]:
            print(f"  - {rel}")
    if result["modified_files"]:
        print("Modified files:")
        for row in result["modified_files"]:
            print(f"  - {row['path']}")
            print(f"    expected={row['expected_sha256']}")
            print(f"    actual={row['actual_sha256']}")

    if (not result["ok"]) and args.mode == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
