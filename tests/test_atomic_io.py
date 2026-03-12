from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import research.lib.atomic_io as atomic_io_mod
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


def test_atomic_json_write_sanitizes_non_finite_numbers(tmp_path: Path):
    out = tmp_path / "state.json"
    atomic_json_write(
        out,
        {
            "nan_value": float("nan"),
            "inf_value": float("inf"),
            "nested": [1.0, float("-inf")],
        },
    )
    with open(out, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload == {
        "nan_value": None,
        "inf_value": None,
        "nested": [1.0, None],
    }


def test_atomic_json_write_closes_fd_if_fdopen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    out = tmp_path / "state.json"
    real_close = os.close
    closed_fds: list[int] = []

    def _fake_fdopen(_fd, *_args, **_kwargs):
        raise OSError("fdopen boom")

    def _tracking_close(fd: int) -> None:
        closed_fds.append(int(fd))
        real_close(fd)

    monkeypatch.setattr(atomic_io_mod.os, "fdopen", _fake_fdopen)
    monkeypatch.setattr(atomic_io_mod.os, "close", _tracking_close)

    with pytest.raises(OSError, match="fdopen boom"):
        atomic_io_mod.atomic_json_write(out, {"a": 1})

    assert len(closed_fds) == 1
