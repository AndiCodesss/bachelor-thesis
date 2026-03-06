from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from research.signals import (
    check_signal_causality,
    compute_strategy_id,
    discover_signals,
    load_signal_module,
)


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


def test_strategy_id_differs_by_bar_config() -> None:
    """Same strategy on different bar configs must produce different IDs."""
    signals = discover_signals()
    fn = signals["example_ema_turn"]
    params = {"x": 1}
    id_tick = compute_strategy_id("example_ema_turn", params, fn, bar_config="tick_610")
    id_vol = compute_strategy_id("example_ema_turn", params, fn, bar_config="volume_2000")
    id_time = compute_strategy_id("example_ema_turn", params, fn, bar_config="time_1m")
    assert id_tick != id_vol
    assert id_tick != id_time
    assert id_vol != id_time


def test_strategy_id_differs_by_session_filter() -> None:
    """Same strategy + bar config on different sessions must produce different IDs."""
    signals = discover_signals()
    fn = signals["example_ema_turn"]
    params = {"x": 1}
    id_eth = compute_strategy_id("example_ema_turn", params, fn, bar_config="tick_610", session_filter="eth")
    id_rth = compute_strategy_id("example_ema_turn", params, fn, bar_config="tick_610", session_filter="rth")
    assert id_eth != id_rth


def test_strategy_id_stable_with_no_env() -> None:
    """Backwards compatibility: no bar_config/session still produces a valid ID."""
    signals = discover_signals()
    fn = signals["example_ema_turn"]
    params = {"x": 1}
    id_bare = compute_strategy_id("example_ema_turn", params, fn)
    assert "_" in id_bare
    # example_ema_turn_paramsHash_codeHash = 5 segments
    assert len(id_bare.split("_")) == 5


def test_strategy_id_distinguishes_bar_config_from_session_filter() -> None:
    """bar_config/session_filter placement must not collide for asymmetric inputs."""
    signals = discover_signals()
    fn = signals["example_ema_turn"]
    params = {"x": 1}
    id_cfg = compute_strategy_id("example_ema_turn", params, fn, bar_config="eth", session_filter="")
    id_session = compute_strategy_id("example_ema_turn", params, fn, bar_config="", session_filter="eth")
    assert id_cfg != id_session


def _causal_signal(df: pl.DataFrame, _params: dict) -> np.ndarray:
    close = np.asarray(df["close"].to_numpy(), dtype=np.float64)
    prev = np.roll(close, 1)
    prev[0] = close[0]
    return np.where(close > prev, 1, np.where(close < prev, -1, 0)).astype(np.int8)


def _future_peek_signal(df: pl.DataFrame, _params: dict) -> np.ndarray:
    close = np.asarray(df["close"].to_numpy(), dtype=np.float64)
    nxt = np.roll(close, -1)
    nxt[-1] = close[-1]
    return np.where(nxt > close, 1, np.where(nxt < close, -1, 0)).astype(np.int8)


def _future_peek_continuous(df: pl.DataFrame, _params: dict) -> np.ndarray:
    close = np.asarray(df["close"].to_numpy(), dtype=np.float64)
    nxt = np.roll(close, -1)
    nxt[-1] = close[-1]
    return nxt - close


def test_signal_causality_check_passes_causal_signal() -> None:
    errors = check_signal_causality(
        generate_fn=_causal_signal,
        df=_df(96),
        params={},
        mode="strict",
        min_prefix_bars=8,
    )
    assert errors == []


def test_signal_causality_check_rejects_future_peek_signal() -> None:
    errors = check_signal_causality(
        generate_fn=_future_peek_signal,
        df=_df(96),
        params={},
        mode="strict",
        min_prefix_bars=8,
    )
    assert errors
    assert "non-causal signal" in errors[0]


def test_signal_causality_check_sign_mode_rejects_future_peek_continuous() -> None:
    errors = check_signal_causality(
        generate_fn=_future_peek_continuous,
        df=_df(96),
        params={},
        mode="sign",
        min_prefix_bars=8,
    )
    assert errors
    assert "non-causal signal" in errors[0]


def test_signal_causality_stateful_strategies_receive_fresh_state_per_invocation() -> None:
    """Causality check should not reuse mutable state across full/prefix calls."""

    def _stateful_signal(df: pl.DataFrame, _params: dict, model_state: dict) -> np.ndarray:
        model_state["calls"] = int(model_state.get("calls", 0)) + 1
        if model_state["calls"] != 1:
            raise RuntimeError("state reused across causality invocations")
        return np.zeros(len(df), dtype=np.int8)

    errors = check_signal_causality(
        generate_fn=_stateful_signal,
        df=_df(96),
        params={},
        accepts_state=True,
        model_state={},
        mode="strict",
        min_prefix_bars=8,
    )
    assert errors == []


def test_signal_causality_requires_deepcopyable_state() -> None:
    class _NonDeepcopyable(dict):
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot clone")

    def _stateful_signal(df: pl.DataFrame, _params: dict, model_state: dict) -> np.ndarray:
        model_state["seen"] = len(df)
        return np.zeros(len(df), dtype=np.int8)

    errors = check_signal_causality(
        generate_fn=_stateful_signal,
        df=_df(96),
        params={},
        accepts_state=True,
        model_state=_NonDeepcopyable(),
        mode="strict",
        min_prefix_bars=8,
    )
    assert errors == ["model_state must be deepcopyable for causality checks"]


def test_load_signal_module_rejects_disallowed_import(tmp_path: Path) -> None:
    (tmp_path / "bad_signal.py").write_text(
        "import os\n"
        "import numpy as np\n"
        "def generate_signal(df, params):\n"
        "    return np.zeros(len(df), dtype=np.int8)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disallowed import"):
        load_signal_module("bad_signal", signals_dir=tmp_path)


def test_load_signal_module_rejects_disallowed_file_io_call(tmp_path: Path) -> None:
    (tmp_path / "bad_io_signal.py").write_text(
        "import numpy as np\n"
        "import polars as pl\n"
        "def generate_signal(df, params):\n"
        "    pl.read_parquet('secret.parquet')\n"
        "    return np.zeros(len(df), dtype=np.int8)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disallowed call"):
        load_signal_module("bad_io_signal", signals_dir=tmp_path)


def test_load_signal_module_rejects_mutable_module_scope_state(tmp_path: Path) -> None:
    (tmp_path / "bad_state_signal.py").write_text(
        "import numpy as np\n"
        "CACHE = []\n"
        "def generate_signal(df, params):\n"
        "    CACHE.append(len(df))\n"
        "    return np.zeros(len(df), dtype=np.int8)\n",
        encoding="utf-8",
    )

    with pytest.raises(AttributeError):
        load_signal_module("bad_state_signal", signals_dir=tmp_path).generate_signal(_df(), {})
