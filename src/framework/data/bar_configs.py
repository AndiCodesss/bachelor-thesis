"""Lightweight shared bar-configuration metadata."""

from __future__ import annotations

from typing import Any


# Default bar configurations for NQ E-mini futures.
BAR_CONFIGS: list[dict[str, Any]] = [
    {"bar_type": "tick", "bar_size": "5m", "bar_threshold": 610},
    {"bar_type": "volume", "bar_size": "5m", "bar_threshold": 2000},
    {"bar_type": "time", "bar_size": "1m", "bar_threshold": None},
]


def bar_config_label(config: dict[str, Any]) -> str:
    """Human-readable bar config label: tick_610, vol_2000, 1m."""
    bar_type = str(config.get("bar_type", "")).strip().lower()
    if bar_type == "time":
        return str(config.get("bar_size", "")).strip()
    if bar_type == "volume":
        return f"vol_{config.get('bar_threshold')}"
    return f"{bar_type}_{config.get('bar_threshold')}"
