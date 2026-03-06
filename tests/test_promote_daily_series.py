from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest


def _load_promote_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "promote.py"
    spec = importlib.util.spec_from_file_location("promote_module", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_promote_daily_series_zero_fills_weekdays():
    mod = _load_promote_module()
    trades = pl.DataFrame(
        {
            "entry_time": [
                datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc),
            ],
            "exit_time": [
                datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),  # Monday
                datetime(2024, 1, 3, 11, 0, tzinfo=timezone.utc),  # Wednesday
            ],
            "entry_price": [100.0, 100.0],
            "exit_price": [101.0, 99.0],
            "direction": [1, 1],
            "size": [1, 1],
        }
    )

    daily_returns = mod._daily_returns_from_trades(trades)
    daily_counts = mod._daily_trade_counts_from_trades(trades)

    # Monday, Tuesday (no trades), Wednesday.
    assert len(daily_returns) == 3
    assert len(daily_counts) == 3
    assert abs(daily_returns[0] - 5.5) < 1e-9
    assert abs(daily_returns[1] - 0.0) < 1e-9
    assert abs(daily_returns[2] + 34.5) < 1e-9
    assert daily_counts == [1, 0, 1]


def test_promote_rejects_non_causal_signal(tmp_path: Path):
    mod = _load_promote_module()

    n = 96
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=i) for i in range(n)]
    close = np.linspace(100.0, 130.0, n)
    bars = pl.DataFrame(
        {
            "ts_event": ts,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000, dtype=np.uint32),
        }
    )

    def _future_peek(df: pl.DataFrame, _params: dict) -> np.ndarray:
        c = np.asarray(df["close"].to_numpy(), dtype=np.float64)
        nxt = np.roll(c, -1)
        nxt[-1] = c[-1]
        return nxt - c

    runtime = mod.SignalRuntime(
        generate_fn=_future_peek,
        generate_accepts_state=False,
        fit_fn=None,
        fit_on_files_fn=None,
    )
    fake_file = tmp_path / "nq_2024-01-02.parquet"
    fake_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="signal causality failed"):
        mod._run_signal_on_files(
            files=[fake_file],
            runtime=runtime,
            signal_params={},
            model_state=None,
            load_bars=lambda _p: bars,
            backtest_kwargs={"entry_on_next_open": False},
        )


def test_promote_requires_min_bars_for_causality(tmp_path: Path):
    mod = _load_promote_module()

    n = 20
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=i) for i in range(n)]
    close = np.linspace(100.0, 102.0, n)
    bars = pl.DataFrame(
        {
            "ts_event": ts,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000, dtype=np.uint32),
        }
    )

    runtime = mod.SignalRuntime(
        generate_fn=lambda df, _params: np.zeros(len(df), dtype=np.int8),
        generate_accepts_state=False,
        fit_fn=None,
        fit_on_files_fn=None,
    )
    fake_file = tmp_path / "nq_2024-01-02.parquet"
    fake_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="requires at least 33 bars"):
        mod._run_signal_on_files(
            files=[fake_file],
            runtime=runtime,
            signal_params={},
            model_state=None,
            load_bars=lambda _p: bars,
            backtest_kwargs={"entry_on_next_open": False},
        )


def test_promote_ohlcv_feature_group_hides_microstructure_columns(tmp_path: Path):
    mod = _load_promote_module()

    n = 40
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=i) for i in range(n)]
    close = np.linspace(100.0, 102.0, n)
    bars = pl.DataFrame(
        {
            "ts_event": ts,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000, dtype=np.uint32),
            "bid_price": close - 0.25,
            "ask_price": close + 0.25,
            "order_flow_imbalance": np.linspace(-1.0, 1.0, n),
            "sma_ratio_8": np.zeros(n, dtype=np.float64),
        }
    )

    seen_columns: list[str] = []

    def _signal(df: pl.DataFrame, _params: dict) -> np.ndarray:
        seen_columns[:] = list(df.columns)
        return np.zeros(len(df), dtype=np.int8)

    runtime = mod.SignalRuntime(
        generate_fn=_signal,
        generate_accepts_state=False,
        fit_fn=None,
        fit_on_files_fn=None,
    )
    fake_file = tmp_path / "nq_2024-01-02.parquet"
    fake_file.write_text("", encoding="utf-8")

    out = mod._run_signal_on_files(
        files=[fake_file],
        runtime=runtime,
        signal_params={},
        model_state=None,
        load_bars=lambda _p: bars,
        backtest_kwargs={"entry_on_next_open": False},
        feature_group="ohlcv",
    )

    assert out["trade_count"] == 0

    assert "bid_price" not in seen_columns
    assert "ask_price" not in seen_columns
    assert "order_flow_imbalance" not in seen_columns
    assert "open" in seen_columns
    assert "high" in seen_columns
    assert "low" in seen_columns
    assert "close" in seen_columns
    assert "volume" in seen_columns
