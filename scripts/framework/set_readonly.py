#!/usr/bin/env python3
"""Toggle framework files writable/read-only using manifest paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.framework.security.framework_lock import load_manifest


def _is_writable(path: Path) -> bool:
    mode = path.stat().st_mode
    return bool(mode & stat.S_IWUSR)


def _set_writable(path: Path, writable: bool) -> None:
    mode = path.stat().st_mode
    if writable:
        mode |= stat.S_IWUSR
    else:
        mode &= ~stat.S_IWUSR
        mode &= ~stat.S_IWGRP
        mode &= ~stat.S_IWOTH
    os.chmod(path, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set framework files read-only/writable.")
    parser.add_argument(
        "--manifest",
        default="configs/framework_lock.json",
        help="Manifest path (project-relative or absolute).",
    )
    parser.add_argument("--mode", choices=["lock", "unlock", "status"], default="status")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (project_root / manifest_path).resolve()
    payload = load_manifest(manifest_path)

    files = payload["files"]
    changed = 0
    for row in files:
        rel = str(row["path"])
        path = (project_root / rel).resolve()
        if not path.exists():
            print(f"MISSING {rel}")
            continue
        before = _is_writable(path)
        if args.mode == "lock":
            _set_writable(path, writable=False)
        elif args.mode == "unlock":
            _set_writable(path, writable=True)
        after = _is_writable(path)
        if before != after:
            changed += 1
        print(f"{'WRITE' if after else 'READONLY'} {rel}")

    print(f"Mode={args.mode} files={len(files)} changed={changed}")


if __name__ == "__main__":
    main()
