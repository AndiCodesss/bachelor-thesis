from __future__ import annotations

import json
from pathlib import Path

from research.lib.atomic_io import atomic_json_write


def test_atomic_json_write_creates_file(tmp_path: Path):
    out = tmp_path / "state.json"
    payload = {"a": 1, "b": "x"}
    atomic_json_write(out, payload)
    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        assert json.load(f) == payload


def test_atomic_json_write_overwrites_with_valid_json(tmp_path: Path):
    out = tmp_path / "state.json"
    atomic_json_write(out, {"version": 1})
    atomic_json_write(out, {"version": 2, "ok": True})
    with open(out, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["version"] == 2
    assert payload["ok"] is True
