from __future__ import annotations
from typing import Any
import numpy as np
import polars as pl

DEFAULT_PARAMS: dict[str, Any] = {
    "va_probe_below": -0.1,
    "va_probe_above": 1.1,
    "va_reenter_below": 0.02,
    "va_reenter_above": 0.98,
    "min_va_width_pts": 8.0,
    "max_va_width_pts": 80.0,
    "max_arm_bars": 8,
    "min_close_position_long": 0.5,
    "max_close_position_short": 0.5,
    "min_volume_ratio": 1.2,
    "min_session_progress": 0.05,
    "max_session_progress": 0.85,
    "min_spread_filter": 1.0,
    "cooldown_bars": 20,
    "max_trades_per_day": 2,
    "pt_ticks": 16,
    "sl_ticks": 8,
    "max_hold_bars": 60,
}

STRATEGY_METADATA = {
    "name": "va_rejection_state_machine",
    "version": "1.0",
    "features_required": [
        "position_in_va",
        "va_width",
        "close_position",
        "volume_ratio",
        "session_progress",
        "spread",
        "ts_event",
    ],
    "description": (
        "Value Area rejection mean-reversion: fires when price probes outside "
        "prior-session VA boundaries (below VAL or above VAH) then snaps back "
        "inside, targeting rotation toward POC. Dual independent 3-state machines "
        "for long and short with cooldown and daily trade limits."
    ),
}


def generate_signal(df: pl.DataFrame, params: dict[str, Any]) -> np.ndarray:
    cfg = dict(DEFAULT_PARAMS)
    cfg.update(params or {})

    n = len(df)
    signal = np.zeros(n, dtype=np.int8)

    def _col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in df.columns:
            return np.full(n, default, dtype=np.float64)
        arr = df[name].fill_null(default).to_numpy().astype(np.float64)
        np.nan_to_num(arr, nan=default, copy=False)
        return arr

    position_in_va = _col("position_in_va", 0.5)
    va_width = _col("va_width", 0.0)
    close_position = _col("close_position", 0.5)
    volume_ratio = _col("volume_ratio", 1.0)
    session_progress = _col("session_progress", 0.5)
    spread = _col("spread", 0.0)

    # Session boundary detection
    if "ts_event" not in df.columns:
        new_session = np.zeros(n, dtype=bool)
        new_session[0] = True
    else:
        ts = df["ts_event"].cast(pl.Datetime("us", "UTC"))
        dates = ts.dt.convert_time_zone("US/Eastern").dt.date().to_numpy()
        new_session = np.concatenate([[True], dates[1:] != dates[:-1]])

    # Unpack parameters
    va_probe_below = float(cfg["va_probe_below"])
    va_probe_above = float(cfg["va_probe_above"])
    va_reenter_below = float(cfg["va_reenter_below"])
    va_reenter_above = float(cfg["va_reenter_above"])
    min_va_width = float(cfg["min_va_width_pts"])
    max_va_width = float(cfg["max_va_width_pts"])
    max_arm_bars = int(cfg["max_arm_bars"])
    min_close_pos_long = float(cfg["min_close_position_long"])
    max_close_pos_short = float(cfg["max_close_position_short"])
    min_vol_ratio = float(cfg["min_volume_ratio"])
    min_sess_prog = float(cfg["min_session_progress"])
    max_sess_prog = float(cfg["max_session_progress"])
    max_spread = float(cfg["min_spread_filter"])
    cooldown = int(cfg["cooldown_bars"])
    max_daily = int(cfg["max_trades_per_day"])

    # State variables
    long_state = 0   # 0=neutral, 1=armed
    short_state = 0  # 0=neutral, 1=armed
    long_probe_bar = 0
    short_probe_bar = 0
    last_signal_bar = -(cooldown + 1)
    daily_count = 0

    for i in range(1, n):
        # Reset state at session boundaries
        if new_session[i]:
            long_state = 0
            short_state = 0
            long_probe_bar = 0
            short_probe_bar = 0
            daily_count = 0

        # Shared filter conditions (computed once per bar)
        va_ok = min_va_width <= va_width[i] <= max_va_width
        sess_ok = min_sess_prog <= session_progress[i] <= max_sess_prog
        spread_ok = spread[i] <= max_spread
        no_cooldown = (i - last_signal_bar) >= cooldown
        under_daily = daily_count < max_daily

        # --- LONG state machine ---
        if long_state == 0:
            # Arm when price probes below VAL
            if position_in_va[i] < va_probe_below and va_ok and sess_ok:
                long_state = 1
                long_probe_bar = i
        else:  # long_state == 1 (armed)
            if (i - long_probe_bar) > max_arm_bars:
                # Setup expired
                long_state = 0
            elif (
                position_in_va[i] >= va_reenter_below
                and close_position[i] > min_close_pos_long
                and volume_ratio[i] > min_vol_ratio
                and sess_ok
                and spread_ok
                and no_cooldown
                and under_daily
            ):
                # Price snapped back inside VA with bullish conviction bar
                signal[i] = 1
                long_state = 0
                last_signal_bar = i
                daily_count += 1

        # --- SHORT state machine ---
        if short_state == 0:
            # Arm when price probes above VAH
            if position_in_va[i] > va_probe_above and va_ok and sess_ok:
                short_state = 1
                short_probe_bar = i
        else:  # short_state == 1 (armed)
            if (i - short_probe_bar) > max_arm_bars:
                # Setup expired
                short_state = 0
            elif (
                position_in_va[i] <= va_reenter_above
                and close_position[i] < max_close_pos_short
                and volume_ratio[i] > min_vol_ratio
                and sess_ok
                and spread_ok
                and no_cooldown
                and under_daily
            ):
                # Price snapped back inside VA with bearish conviction bar
                signal[i] = -1
                short_state = 0
                last_signal_bar = i
                daily_count += 1

    return signal
