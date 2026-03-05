import numpy as np
import polars as pl

DEFAULT_PARAMS = {
    "vwap_z_entry_long": -2.0,
    "vwap_z_entry_short": 2.0,
    "vwap_z_exit_long": -0.5,
    "vwap_z_exit_short": 0.5,
    "vpin_toxicity_max": 0,
    "atr_norm_14_low": 0.003,
    "atr_norm_14_high": 0.012,
    "session_progress_min": 0.15,
    "session_progress_max": 0.85,
    "delta_intensity_z_cap": 1.5,
    "max_trades_per_day": 8,
    "cooldown_bars": 5,
    "stop_atr_mult": 1.5,
    "max_hold_bars": 40,
}

STRATEGY_METADATA = {
    "strategy_name": "vwap_toxicity_gated_reversion",
    "hypothesis_id": "h007_vwap_revert_toxicity_gate",
    "bar_configs": ["volume_2000"],
    "features_used": [
        "vwap_dev_zscore", "vpin_toxicity", "atr_norm_14",
        "session_progress", "delta_intensity_z", "close",
    ],
}


def _safe_col(df: pl.DataFrame, name: str, default: float) -> np.ndarray:
    if name in df.columns:
        return df[name].fill_null(default).to_numpy().astype(np.float64)
    return np.full(len(df), default, dtype=np.float64)


def _get_dates(df: pl.DataFrame) -> np.ndarray:
    for col in ("ts_event", "ts_close"):
        if col in df.columns:
            try:
                dates = df[col].cast(pl.Date).to_numpy()
                return dates
            except Exception:
                pass
    return np.zeros(len(df), dtype=np.int64)


def generate_signal(df: pl.DataFrame, params: dict) -> np.ndarray:
    p = {**DEFAULT_PARAMS, **params}
    n = len(df)
    signal = np.zeros(n, dtype=np.int32)

    vwap_z = _safe_col(df, "vwap_dev_zscore", 0.0)
    vpin_tox = _safe_col(df, "vpin_toxicity", 1.0)  # default toxic = block
    atr_n = _safe_col(df, "atr_norm_14", 0.0)
    sess_prog = _safe_col(df, "session_progress", 0.5)
    delta_iz = _safe_col(df, "delta_intensity_z", 999.0)
    close = _safe_col(df, "close", 0.0)

    # date tracking for daily trade cap
    dates = _get_dates(df)
    use_dates = not np.all(dates == 0)

    vz_entry_l = p["vwap_z_entry_long"]
    vz_entry_s = p["vwap_z_entry_short"]
    vz_exit_l = p["vwap_z_exit_long"]
    vz_exit_s = p["vwap_z_exit_short"]
    tox_max = p["vpin_toxicity_max"]
    atr_lo = p["atr_norm_14_low"]
    atr_hi = p["atr_norm_14_high"]
    sp_min = p["session_progress_min"]
    sp_max = p["session_progress_max"]
    diz_cap = p["delta_intensity_z_cap"]
    max_td = p["max_trades_per_day"]
    cd_bars = p["cooldown_bars"]
    stop_mult = p["stop_atr_mult"]
    max_hold = p["max_hold_bars"]

    pos = 0  # current position: -1, 0, 1
    entry_price = 0.0
    entry_atr = 0.0
    bars_held = 0
    bars_since_entry = cd_bars + 1  # allow first entry
    daily_trades = 0
    current_date = None

    for i in range(n):
        # reset daily trade count on new date
        if use_dates:
            d = dates[i]
            if d != current_date:
                current_date = d
                daily_trades = 0
        else:
            if i > 0 and sess_prog[i] < sess_prog[i - 1] - 0.3:
                daily_trades = 0

        bars_since_entry += 1

        # --- EXIT LOGIC (evaluate before entry) ---
        if pos != 0:
            bars_held += 1
            should_exit = False

            # session end exit
            if sess_prog[i] > 0.98:
                should_exit = True

            # time stop
            if bars_held >= max_hold:
                should_exit = True

            # stop loss: ATR-adaptive
            if entry_atr > 0.0 and close[i] > 0.0:
                stop_dist = stop_mult * entry_atr * entry_price
                if pos == 1 and close[i] <= entry_price - stop_dist:
                    should_exit = True
                elif pos == -1 and close[i] >= entry_price + stop_dist:
                    should_exit = True

            # profit target: z-score reversion
            if pos == 1 and vwap_z[i] >= vz_exit_l:
                should_exit = True
            elif pos == -1 and vwap_z[i] <= vz_exit_s:
                should_exit = True

            if should_exit:
                pos = 0
                entry_price = 0.0
                entry_atr = 0.0
                bars_held = 0

        # --- ENTRY LOGIC ---
        if pos == 0:
            can_enter = (
                vpin_tox[i] <= tox_max
                and atr_lo <= atr_n[i] <= atr_hi
                and sp_min <= sess_prog[i] <= sp_max
                and abs(delta_iz[i]) < diz_cap
                and daily_trades < max_td
                and bars_since_entry >= cd_bars
            )
            if can_enter:
                if vwap_z[i] < vz_entry_l:
                    pos = 1
                    entry_price = close[i]
                    entry_atr = atr_n[i]
                    bars_held = 0
                    bars_since_entry = 0
                    daily_trades += 1
                elif vwap_z[i] > vz_entry_s:
                    pos = -1
                    entry_price = close[i]
                    entry_atr = atr_n[i]
                    bars_held = 0
                    bars_since_entry = 0
                    daily_trades += 1

        signal[i] = pos

    return signal.astype(np.int8)
