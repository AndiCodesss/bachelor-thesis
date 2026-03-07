"""Signal discovery, strategy identity, and signal contract helpers."""

from __future__ import annotations

import ast
import builtins as py_builtins
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType
from types import MappingProxyType
from typing import Any, Callable
from collections.abc import Mapping

import numpy as np
import polars as pl

SignalFn = Callable[..., np.ndarray]
CAUSALITY_MIN_PREFIX_BARS = 32
_SAFE_IMPORT_ROOTS = {"__future__", "math", "numpy", "polars", "typing"}
_SAFE_EXACT_IMPORTS = {"research.signals"}
_BANNED_CALL_ROOTS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_BANNED_ATTR_CALLS = {
    "dump",
    "dumps",
    "load",
    "read_csv",
    "read_database",
    "read_excel",
    "read_ipc",
    "read_json",
    "read_ndjson",
    "read_parquet",
    "save",
    "savez",
    "savez_compressed",
    "scan_csv",
    "scan_ipc",
    "scan_ndjson",
    "scan_parquet",
}
_SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "Exception",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "ValueError",
    "zip",
)


class ModelStateCloneError(TypeError):
    """Raised when model_state cannot be isolated for causality checks."""


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _is_dunder_name(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _is_mutable_default(node: ast.AST | None) -> bool:
    return isinstance(node, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp))


def _freeze_constant(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze_constant(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_constant(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_constant(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_constant(v) for v in value)
    return value


def safe_f64_col(df: pl.DataFrame, name: str, fill: float = 0.0) -> np.ndarray:
    """Return a writable float64 array for one strategy column."""
    default = float(fill)
    if str(name) not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    arr = np.array(df[str(name)].to_numpy(), dtype=np.float64, copy=True)
    return np.nan_to_num(arr, nan=default, posinf=default, neginf=default)


def session_start_mask(df: pl.DataFrame, *, ts_col: str = "ts_event") -> np.ndarray:
    """Mark the first bar of each session using US/Eastern session dates."""
    out = np.zeros(len(df), dtype=bool)
    if len(out) == 0:
        return out
    out[0] = True
    if ts_col not in df.columns or len(out) == 1:
        return out
    ts = df[ts_col].cast(pl.Datetime("us", "UTC"))
    dates = np.asarray(ts.dt.convert_time_zone("US/Eastern").dt.date().to_numpy())
    out[1:] = dates[1:] != dates[:-1]
    return out


def signal_from_conditions(long_cond: Any, short_cond: Any) -> np.ndarray:
    """Build the final strategy signal array in the framework contract format."""
    long_mask = np.asarray(long_cond, dtype=bool).reshape(-1)
    short_mask = np.asarray(short_cond, dtype=bool).reshape(-1)
    if long_mask.shape != short_mask.shape:
        raise ValueError("long_cond and short_cond must have matching shapes")
    return np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(np.int8)


def _restricted_import(
    name: str,
    globals_dict: dict[str, Any] | None = None,
    locals_dict: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
    if level != 0:
        raise ImportError("relative imports are not allowed in strategy modules")
    if name in _SAFE_EXACT_IMPORTS:
        return py_builtins.__import__(name, globals_dict, locals_dict, fromlist, level)
    root = name.split(".", 1)[0]
    if root not in _SAFE_IMPORT_ROOTS:
        raise ImportError(f"disallowed import root '{root}' in strategy module")
    return py_builtins.__import__(name, globals_dict, locals_dict, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    safe = {name: getattr(py_builtins, name) for name in _SAFE_BUILTIN_NAMES}
    safe["__import__"] = _restricted_import
    return safe


def _validate_signal_source(source: str, path: Path) -> None:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"Invalid strategy syntax in {path}: {exc.msg}") from exc

    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _SAFE_EXACT_IMPORTS:
                    continue
                root = alias.name.split(".", 1)[0]
                if root not in _SAFE_IMPORT_ROOTS:
                    raise ValueError(f"Strategy module {path} uses disallowed import '{alias.name}'")
            continue
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if node.level == 0 and module_name in _SAFE_EXACT_IMPORTS:
                continue
            root = module_name.split(".", 1)[0]
            if node.level != 0 or root not in _SAFE_IMPORT_ROOTS:
                raise ValueError(f"Strategy module {path} uses disallowed import from '{node.module}'")
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(node, ast.AsyncFunctionDef):
                raise ValueError(f"Strategy module {path} may not define async functions")
            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
            if any(_is_mutable_default(default) for default in defaults):
                raise ValueError(f"Strategy module {path} uses mutable default arguments")
            continue
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            raise ValueError(f"Strategy module {path} uses unsupported top-level statement '{type(node).__name__}'")

        for target in targets:
            if not isinstance(target, ast.Name) or not target.id.isupper():
                raise ValueError(
                    f"Strategy module {path} may only assign uppercase constants at module scope",
                )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError(f"Strategy module {path} may not mutate module/global scope")
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
            raise ValueError(f"Strategy module {path} may only import at module scope")
        if isinstance(node, ast.Attribute) and _is_dunder_name(node.attr):
            raise ValueError(
                f"Strategy module {path} uses disallowed dunder attribute '{node.attr}'",
            )
        if isinstance(node, ast.Call):
            call_name = _dotted_name(node.func)
            if isinstance(node.func, ast.Attribute):
                leaf = node.func.attr
                if _is_dunder_name(leaf) or leaf in _BANNED_ATTR_CALLS:
                    raise ValueError(f"Strategy module {path} uses disallowed call '{leaf}'")
                if leaf == "to_numpy":
                    raise ValueError(
                        f"Strategy module {path} may not call to_numpy(); use safe_f64_col(...) instead",
                    )
            if call_name == "np.nan_to_num":
                for kw in node.keywords:
                    if kw.arg == "copy" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                        raise ValueError(
                            f"Strategy module {path} may not use np.nan_to_num(..., copy=False)",
                        )
            if call_name is None:
                continue
            root = call_name.split(".", 1)[0]
            leaf = call_name.rsplit(".", 1)[-1]
            if root in _BANNED_CALL_ROOTS or call_name in _BANNED_CALL_ROOTS or leaf in _BANNED_ATTR_CALLS:
                raise ValueError(f"Strategy module {path} uses disallowed call '{call_name}'")


def _clone_model_state(model_state: Any | None) -> Any | None:
    if model_state is None:
        return None
    try:
        return copy.deepcopy(model_state)
    except Exception as exc:
        raise ModelStateCloneError(
            "model_state must be deepcopyable for causality checks",
        ) from exc


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
    min_prefix_bars: int = CAUSALITY_MIN_PREFIX_BARS,
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
        try:
            full_raw = _invoke_generate_signal(
                generate_fn=generate_fn,
                df=df,
                params=params,
                accepts_state=accepts_state,
                model_state=_clone_model_state(model_state),
            )
        except ModelStateCloneError as exc:
            return [str(exc)]
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
        try:
            prefix_raw = _invoke_generate_signal(
                generate_fn=generate_fn,
                df=prefix_df,
                params=params,
                accepts_state=accepts_state,
                model_state=_clone_model_state(model_state),
            )
        except ModelStateCloneError as exc:
            return [str(exc)]
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
    source = path.read_text(encoding="utf-8")
    _validate_signal_source(source, path)
    module = ModuleType(f"research.signals.{strategy_name}")
    module.__file__ = str(path)
    module.__package__ = "research.signals"
    module.__dict__["__builtins__"] = _safe_builtins()
    exec(compile(source, str(path), "exec"), module.__dict__)
    for name, value in list(module.__dict__.items()):
        if name.startswith("__") or callable(value) or isinstance(value, ModuleType):
            continue
        if name.isupper():
            module.__dict__[name] = _freeze_constant(value)
    return module


def discover_signals(signals_dir: Path | None = None) -> dict[str, SignalFn]:
    """Discover all strategy files exposing `generate_signal` or `signal`."""
    directory = _signals_dir(signals_dir)
    found: dict[str, SignalFn] = {}
    for path in sorted(directory.glob("*.py")):
        if path.stem in {"__init__"} or path.stem.startswith("_"):
            continue
        module = load_signal_module(path.stem, directory)
        fn = getattr(module, "generate_signal", None)
        if not callable(fn):
            fn = getattr(module, "signal", None)
        if callable(fn):
            found[path.stem] = fn
    return found


def get_strategy_metadata(strategy_name: str, signals_dir: Path | None = None) -> dict[str, Any]:
    """Return strategy metadata if present, else empty dict."""
    module = load_signal_module(strategy_name, signals_dir)
    raw = getattr(module, "STRATEGY_METADATA", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def compute_strategy_id(
    strategy_name: str,
    params: dict[str, Any],
    strategy_function: SignalFn,
    bar_config: str = "",
    session_filter: str = "",
) -> str:
    """Stable strategy id from name + params + source + bar config + session."""
    params_blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    params_hash = hashlib.sha256(params_blob.encode("utf-8")).hexdigest()[:16]
    source = inspect.getsource(strategy_function)
    code_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    if bar_config or session_filter:
        env_blob = json.dumps(
            {"bar_config": str(bar_config), "session_filter": str(session_filter)},
            sort_keys=True,
            separators=(",", ":"),
        )
        env_hash = hashlib.sha256(env_blob.encode("utf-8")).hexdigest()[:12]
        return f"{strategy_name}_{params_hash}_{code_hash}_{env_hash}"
    return f"{strategy_name}_{params_hash}_{code_hash}"


__all__ = [
    "CAUSALITY_MIN_PREFIX_BARS",
    "SignalFn",
    "discover_signals",
    "load_signal_module",
    "get_strategy_metadata",
    "compute_strategy_id",
    "check_signal_causality",
    "safe_f64_col",
    "session_start_mask",
    "signal_from_conditions",
]
