"""Signal discovery, strategy identity, and signal contract helpers."""

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
_CAUSALITY_MIN_PREFIX_BARS = 32


def _invoke_generate_signal(
    *,
    generate_fn: Callable[..., Any],
    df: pl.DataFrame,
    params: dict[str, Any],
    accepts_state: bool,
    model_state: Any | None,
) -> np.ndarray:
    if accepts_state:
        raw = generate_fn(df, params, model_state)
    else:
        raw = generate_fn(df, params)
    return np.asarray(raw, dtype=np.float64).reshape(-1)


def _normalize_signal_for_causality(arr: np.ndarray, mode: str) -> tuple[np.ndarray | None, str | None]:
    if mode == "strict":
        if np.isnan(arr).any():
            return None, "signal contains NaN"
        unique_vals = set(np.unique(arr).tolist())
        if not unique_vals.issubset({-1.0, 0.0, 1.0}):
            return None, f"signal contains invalid values: {sorted(unique_vals)}"
        return arr.astype(np.int8, copy=False), None

    if mode == "sign":
        normalized = np.sign(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.int8)
        return normalized, None

    raise ValueError(f"Unknown causality mode '{mode}'. Use 'strict' or 'sign'.")


def check_signal_causality(
    *,
    generate_fn: Callable[..., Any],
    df: pl.DataFrame,
    params: dict[str, Any],
    accepts_state: bool = False,
    model_state: Any | None = None,
    mode: str = "strict",
    min_prefix_bars: int = _CAUSALITY_MIN_PREFIX_BARS,
    full_signal: np.ndarray | None = None,
) -> list[str]:
    """Prefix-invariance causality check for signal functions.

    A causal signal must produce identical outputs for the first K bars whether
    computed on the first K bars only, or on the full frame and then sliced.
    """
    n_rows = len(df)
    if n_rows < max(int(min_prefix_bars) + 1, 2):
        return []

    if full_signal is None:
        full_raw = _invoke_generate_signal(
            generate_fn=generate_fn,
            df=df,
            params=params,
            accepts_state=accepts_state,
            model_state=model_state,
        )
    else:
        full_raw = np.asarray(full_signal, dtype=np.float64).reshape(-1)

    if len(full_raw) != n_rows:
        return [f"signal length {len(full_raw)} != expected {n_rows}"]

    full_norm, err = _normalize_signal_for_causality(full_raw, mode)
    if err is not None:
        return [err]
    assert full_norm is not None

    candidate_cuts = {
        int(round(n_rows * 0.50)),
        int(round(n_rows * 0.75)),
        int(round(n_rows * 0.90)),
        n_rows - 1,
    }
    cuts = sorted(
        cut for cut in candidate_cuts
        if int(min_prefix_bars) <= cut < n_rows
    )
    if not cuts:
        return []

    for cut in cuts:
        prefix_df = df[:cut]
        prefix_raw = _invoke_generate_signal(
            generate_fn=generate_fn,
            df=prefix_df,
            params=params,
            accepts_state=accepts_state,
            model_state=model_state,
        )
        if len(prefix_raw) != cut:
            return [f"prefix signal length {len(prefix_raw)} != expected {cut}"]

        prefix_norm, err = _normalize_signal_for_causality(prefix_raw, mode)
        if err is not None:
            return [err]
        assert prefix_norm is not None

        mismatch_idx = np.flatnonzero(full_norm[:cut] != prefix_norm)
        if mismatch_idx.size > 0:
            i = int(mismatch_idx[0])
            return [
                "non-causal signal: prefix mismatch at "
                f"bar={i}, cut={cut}, full={int(full_norm[i])}, prefix={int(prefix_norm[i])}",
            ]

    return []


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
    "check_signal_causality",
]
