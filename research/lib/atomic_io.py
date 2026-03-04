"""Crash-safe atomic JSON file operations for research coordination state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def atomic_json_write(file_path: Path | str, data: dict[str, Any]) -> None:
    """Atomically write JSON with durability guarantees.

    Sequence:
    1. write to temp file in target directory
    2. fsync temp file
    3. os.replace(temp, target)
    4. fsync parent directory (best effort)
    """
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )

    try:
        try:
            f = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise

        with f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, target)

        # Best-effort metadata sync for full crash durability.
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Some filesystems/platforms do not support directory fsync.
            pass
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
