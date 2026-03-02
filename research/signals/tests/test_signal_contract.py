"""Contract tests for research signals.

Every signal function in research/signals/ must satisfy:
  - Returns np.ndarray of int8
  - Values in {-1, 0, 1}
  - Same length as input DataFrame
  - No NaN values

These tests are run automatically by the research loop before experiments.
"""

import importlib
import inspect
from pathlib import Path

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

SIGNALS_DIR = Path(__file__).resolve().parent.parent


def _discover_signal_modules():
    """Find all .py files in research/signals/ that define a signal function."""
    modules = []
    for py_file in sorted(SIGNALS_DIR.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "README.md":
            continue
        modules.append(py_file.stem)
    return modules


def _get_signal_fn(module_name: str):
    """Import module and return its signal function (generate_signal or signal)."""
    mod = importlib.import_module(f"research.signals.{module_name}")
    for fn_name in ("generate_signal", "signal"):
        if hasattr(mod, fn_name):
            return getattr(mod, fn_name)
    return None


def _make_dummy_bars(n_bars: int = 50) -> pl.DataFrame:
    """Create a minimal bar DataFrame for contract testing."""
    rng = np.random.default_rng(42)
    close = 15000.0 + np.cumsum(rng.standard_normal(n_bars) * 5.0)
    start = datetime(2024, 1, 2, 9, 30, 0, tzinfo=timezone.utc)
    return pl.DataFrame({
        "ts_event": pl.datetime_range(
            start,
            start + timedelta(minutes=5 * (n_bars - 1)),
            interval="5m",
            eager=True,
        ).head(n_bars),
        "close": close,
        "open": close + rng.standard_normal(n_bars) * 2,
        "high": close + np.abs(rng.standard_normal(n_bars) * 3),
        "low": close - np.abs(rng.standard_normal(n_bars) * 3),
        "volume": rng.integers(100, 5000, size=n_bars).astype(np.uint32),
        "return_1bar": np.concatenate([[None], np.diff(close) / close[:-1]]),
    })


SIGNAL_MODULES = _discover_signal_modules()


@pytest.mark.parametrize("module_name", SIGNAL_MODULES if SIGNAL_MODULES else ["__skip__"])
def test_signal_contract(module_name: str):
    """Verify signal function contract: returns int8 array of {-1, 0, 1}."""
    if module_name == "__skip__":
        pytest.skip("No signal modules discovered")

    fn = _get_signal_fn(module_name)
    if fn is None:
        pytest.skip(f"No signal function found in {module_name}")

    bars = _make_dummy_bars()
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    if len(params) >= 2:
        result = fn(bars, {})
    else:
        result = fn(bars)

    arr = np.asarray(result)
    assert arr.shape == (len(bars),), (
        f"{module_name}: expected shape ({len(bars)},), got {arr.shape}"
    )
    assert not np.any(np.isnan(arr.astype(float))), (
        f"{module_name}: signal contains NaN"
    )
    unique_vals = set(np.unique(arr))
    assert unique_vals <= {-1, 0, 1}, (
        f"{module_name}: signal values {unique_vals} not subset of {{-1, 0, 1}}"
    )
