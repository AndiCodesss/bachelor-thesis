#!/usr/bin/env python3
"""Build framework integrity manifest (sha256 hash lock file)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.framework.security.framework_lock import DEFAULT_CORE_FILES, build_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build framework lock manifest.")
    parser.add_argument(
        "--out",
        default="configs/framework_lock.json",
        help="Output manifest path (project-relative or absolute).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit file list (project-relative). Defaults to hardened core set.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (project_root / out_path).resolve()

    rel_files = list(args.files) if args.files else list(DEFAULT_CORE_FILES)
    manifest = build_manifest(project_root=project_root, rel_paths=rel_files)
    save_manifest(manifest, out_path=out_path)

    print(f"Manifest written: {out_path}")
    print(f"Locked files: {manifest['file_count']}")
    for row in manifest["files"]:
        print(f"  - {row['path']}")


if __name__ == "__main__":
    main()
