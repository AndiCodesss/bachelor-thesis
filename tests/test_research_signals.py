from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from research.signals import compute_strategy_id, discover_signals


def _df(n: int = 32) -> pl.DataFrame:
    start = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=i) for i in range(n)]
    c = np.linspace(21000.0, 21010.0, n)
    return pl.DataFrame(
        {
            "ts_event": ts,
            "open": c,
            "high": c + 1.0,
            "low": c - 1.0,
            "close": c,
            "volume": np.full(n, 1000),
            "ema_8": c + 0.05,
            "ema_21": c - 0.05,
        },
    )


def test_discover_signals_and_contract() -> None:
    signals = discover_signals()
    assert "example_ema_turn" in signals
    out = signals["example_ema_turn"](_df(), {})
    assert isinstance(out, np.ndarray)
    assert len(out) == 32
    assert set(np.unique(out).tolist()).issubset({-1, 0, 1})


def test_strategy_id_is_deterministic() -> None:
    signals = discover_signals()
    fn = signals["example_ema_turn"]
    p1 = {"x": 1, "y": 2}
    p2 = {"y": 2, "x": 1}
    a = compute_strategy_id("example_ema_turn", p1, fn)
    b = compute_strategy_id("example_ema_turn", p2, fn)
    c = compute_strategy_id("example_ema_turn", {"x": 2, "y": 2}, fn)
    assert a == b
    assert a != c

