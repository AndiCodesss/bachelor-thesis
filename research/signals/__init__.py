"""Signal discovery and deterministic strategy identity helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import polars as pl

SignalFn = Callable[[pl.DataFrame, dict[str, Any]], np.ndarray]


def _signals_dir(signals_dir: Path | None = None) -> Path:
    return Path(signals_dir) if signals_dir is not None else Path(__file__).resolve().parent


def load_signal_module(strategy_name: str, signals_dir: Path | None = None) -> ModuleType:
    """Dynamically load one strategy module from research/signals."""
    directory = _signals_dir(signals_dir)
    path = directory / f"{strategy_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Strategy module not found: {path}")
    spec = importlib.util.spec_from_file_location(f"research.signals.{strategy_name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_signals(signals_dir: Path | None = None) -> dict[str, SignalFn]:
    """Discover all strategy files exposing `generate_signal(df, params)`."""
    directory = _signals_dir(signals_dir)
    found: dict[str, SignalFn] = {}
    for path in sorted(directory.glob("*.py")):
        if path.stem in {"__init__"} or path.stem.startswith("_"):
            continue
        module = load_signal_module(path.stem, directory)
        fn = getattr(module, "generate_signal", None)
        if callable(fn):
            found[path.stem] = fn
    return found


def get_strategy_metadata(strategy_name: str, signals_dir: Path | None = None) -> dict[str, Any]:
    """Return strategy metadata if present, else empty dict."""
    module = load_signal_module(strategy_name, signals_dir)
    raw = getattr(module, "STRATEGY_METADATA", {})
    return raw if isinstance(raw, dict) else {}


def compute_strategy_id(
    strategy_name: str,
    params: dict[str, Any],
    strategy_function: SignalFn,
    bar_config: str = "",
    session_filter: str = "",
) -> str:
    """Stable strategy id from name + params + source + bar config + session."""
    params_blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    params_hash = hashlib.sha256(params_blob.encode("utf-8")).hexdigest()[:8]
    source = inspect.getsource(strategy_function)
    code_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]
    env_key = f"{bar_config}|{session_filter}".strip("|")
    if env_key:
        env_hash = hashlib.sha256(env_key.encode("utf-8")).hexdigest()[:6]
        return f"{strategy_name}_{params_hash}_{code_hash}_{env_hash}"
    return f"{strategy_name}_{params_hash}_{code_hash}"


__all__ = [
    "SignalFn",
    "discover_signals",
    "load_signal_module",
    "get_strategy_metadata",
    "compute_strategy_id",
]
