"""Shared helpers for CLI scripts and local runtime utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml

_VALID_SESSION_FILTERS = frozenset({"eth", "rth"})
_VALID_FEATURE_GROUPS = frozenset({"all", "ohlcv"})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml_dict(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML object expected: {path}")
    return payload


def load_json_dict(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        rows = f.readlines()
    return rows[-limit:] if limit > 0 else rows


def state_mode(*, fresh_state: bool) -> str:
    return "fresh" if fresh_state else "resume"


def normalize_session_filter(value: Any, *, default: str = "eth") -> str:
    raw = str(value).strip().lower()
    fallback = str(default).strip().lower() or "eth"
    if raw in _VALID_SESSION_FILTERS:
        return raw
    if fallback in _VALID_SESSION_FILTERS:
        return fallback
    return "eth"


def normalize_feature_group(value: Any, *, default: str = "all") -> str:
    raw = str(value).strip().lower()
    fallback = str(default).strip().lower() or "all"
    if raw in _VALID_FEATURE_GROUPS:
        return raw
    if fallback in _VALID_FEATURE_GROUPS:
        return fallback
    return "all"


def parse_bar_config(bar_config: str) -> dict[str, Any]:
    raw = str(bar_config).strip().lower()
    match = re.fullmatch(r"(tick|volume|vol|time)_(.+)", raw)
    if not match:
        raise ValueError(f"Unsupported bar_config '{bar_config}'")
    kind, suffix = match.groups()

    if kind == "time":
        if not re.fullmatch(r"[1-9][0-9]*[mh]", suffix):
            raise ValueError(f"Invalid time bar size in '{bar_config}'")
        return {"bar_type": "time", "bar_size": suffix, "bar_threshold": None}

    if not suffix.isdigit() or int(suffix) <= 0:
        raise ValueError(f"Invalid bar threshold in '{bar_config}'")

    bar_type = "volume" if kind in {"volume", "vol"} else "tick"
    return {"bar_type": bar_type, "bar_size": "5m", "bar_threshold": int(suffix)}


def validate_signal_array(signal: np.ndarray, expected_len: int) -> list[str]:
    errors: list[str] = []
    if signal.ndim != 1:
        return [f"signal must be 1D, got ndim={signal.ndim}"]
    if len(signal) != expected_len:
        return [f"signal length {len(signal)} != expected {expected_len}"]
    if np.isnan(signal).any():
        errors.append("signal contains NaN")
    uniq = set(np.unique(signal).tolist())
    if not uniq.issubset({-1, 0, 1}):
        errors.append(f"signal contains invalid values: {sorted(uniq)}")
    return errors


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, set):
        normalized = [_normalize_json_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return value


def mission_state_payload(mission: dict[str, Any]) -> dict[str, Any]:
    from research.lib.mission_splits import resolve_research_splits

    split_plan = resolve_research_splits(mission)
    raw_bar_configs = mission.get("bar_configs", ["volume_2000"])
    bar_configs = [str(v) for v in raw_bar_configs] if isinstance(raw_bar_configs, list) else ["volume_2000"]
    selection_gate = mission.get("selection_gate")
    advanced_validation = mission.get("advanced_validation")
    edge_surface = mission.get("edge_surface")
    backtest = mission.get("backtest")
    max_files = mission.get("max_files_per_task")
    return _normalize_json_value(
        {
            "mission_name": str(mission.get("mission_name", "")).strip(),
            "search_split": split_plan["search_split"],
            "selection_split": split_plan["selection_split"],
            "promotion_split": split_plan["promotion_split"],
            "session_filter": normalize_session_filter(mission.get("session_filter", "eth")),
            "feature_group": normalize_feature_group(mission.get("feature_group", "all")),
            "bar_configs": bar_configs,
            "target_sharpe": float(mission.get("target_sharpe", 0.0)),
            "min_trade_count": int(mission.get("min_trade_count", 1)),
            "run_gauntlet": bool(mission.get("run_gauntlet", True)),
            "write_candidates": bool(mission.get("write_candidates", True)),
            "max_files_per_task": int(max_files) if max_files is not None else None,
            "backtest": dict(backtest) if isinstance(backtest, dict) else {},
            "selection_gate": dict(selection_gate) if isinstance(selection_gate, dict) else {},
            "advanced_validation": (
                dict(advanced_validation) if isinstance(advanced_validation, dict) else {}
            ),
            "edge_surface": dict(edge_surface) if isinstance(edge_surface, dict) else {},
        }
    )


def mission_state_fingerprint(mission: dict[str, Any]) -> str:
    payload = mission_state_payload(mission)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = [
    "load_json_dict",
    "load_yaml_dict",
    "mission_state_fingerprint",
    "mission_state_payload",
    "normalize_feature_group",
    "normalize_session_filter",
    "parse_bar_config",
    "state_mode",
    "tail_lines",
    "utc_now_iso",
    "validate_signal_array",
]
