"""Example event-driven signal for framework contract testing.

Computes fast and slow EMAs from the close price and generates
crossover signals: long when fast crosses above slow, short on
the inverse cross.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

DEFAULT_PARAMS: dict[str, Any] = {
    "fast_span": 8,
    "slow_span": 21,
    "min_distance": 0.0,
}


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average computed in-place from close prices."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def generate_signal(df: pl.DataFrame, params: dict[str, Any]) -> np.ndarray:
    """Long on fast EMA crossing above slow EMA, short on inverse cross."""
    cfg = dict(DEFAULT_PARAMS)
    cfg.update(params or {})

    close = np.asarray(df["close"].to_numpy(), dtype=np.float64)
    fast = _ema(close, int(cfg["fast_span"]))
    slow = _ema(close, int(cfg["slow_span"]))

    dist = fast - slow
    prev = np.roll(dist, 1)
    prev[0] = 0.0

    min_distance = float(cfg["min_distance"])
    cross_up = (prev <= 0.0) & (dist > min_distance)
    cross_down = (prev >= 0.0) & (dist < -min_distance)

    signal = np.where(cross_up, 1, np.where(cross_down, -1, 0)).astype(np.int8)
    return signal


STRATEGY_METADATA = {
    "name": "example_ema_turn",
    "version": "1.0",
    "features_required": ["close"],
    "description": "EMA crossover signal for contract and pipeline smoke testing.",
}

