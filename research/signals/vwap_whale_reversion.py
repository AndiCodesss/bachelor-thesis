import numpy as np
import polars as pl

DEFAULT_PARAMS = {
    "vwap_z_entry": 2.0,
    "vwap_z_exit": 0.3,
    "whale_ofi_min": 0.15,
    "whale_participation_min": 0.05,
    "spread_bps_max": 3.0,
    "session_progress_min": 0.08,
    "vol_zscore_max": 2.5,
    "hold_bars": 10,
    "pt_ticks": 12,
    "sl_ticks": 10,
}

STRATEGY_METADATA = {
    "hypothesis_id": "h_vwap_whale_revert_002",
    "strategy_name": "vwap_whale_reversion",
    "bar_configs": ["tick_610", "volume_2000"],
    "description": (
        "Mean-reversion strategy: enters when price deviates >2 sigma from "
        "session VWAP while whale order-flow alignment opposes the deviation, "
        "indicating institutional mean-reversion pressure."
    ),
    "version": "1.0.0",
}


def _safe_col(df: pl.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    """Extract column as float64 numpy array with safe fallback."""
    if name in df.columns:
        return df[name].cast(pl.Float64).fill_null(default).fill_nan(default).to_numpy()
    return np.full(len(df), default, dtype=np.float64)


def generate_signal(df: pl.DataFrame, params: dict) -> np.ndarray:
    """Generate -1/0/+1 signal array based on VWAP deviation + whale flow reversion."""
    p = {**DEFAULT_PARAMS, **params}
    n = len(df)
    signal = np.zeros(n, dtype=np.int32)

    # --- Extract precomputed features with safe fallbacks ---
    vwap_z = _safe_col(df, "vwap_dev_zscore", 0.0)
    whale_ofi = _safe_col(df, "whale_ofi_alignment", 0.0)
    whale_part = _safe_col(df, "whale_participation_rate_30", 0.0)
    spread = _safe_col(df, "spread_bps", 999.0)
    sess_prog = _safe_col(df, "session_progress", 0.0)
    vol_z = _safe_col(df, "vol_zscore", 0.0)
    mins_open = _safe_col(df, "minutes_since_open", 0.0)
    vpin_z = _safe_col(df, "vpin_zscore", 0.0)

    # --- Thresholds ---
    vwap_z_entry = float(p["vwap_z_entry"])
    whale_ofi_min = float(p["whale_ofi_min"])
    whale_part_min = float(p["whale_participation_min"])
    spread_max = float(p["spread_bps_max"])
    sess_min = float(p["session_progress_min"])
    vol_z_max = float(p["vol_zscore_max"])

    # --- Quality gates (shared by long and short) ---
    gate_whale_part = whale_part > whale_part_min
    gate_spread = spread < spread_max
    gate_session = sess_prog > sess_min
    gate_vol = vol_z < vol_z_max
    gate_no_late = sess_prog <= 0.95
    gate_maturity = mins_open > 35.0
    gate_no_toxic = vpin_z <= 2.0

    common_gate = (
        gate_whale_part
        & gate_spread
        & gate_session
        & gate_vol
        & gate_no_late
        & gate_maturity
        & gate_no_toxic
    )

    # --- Long signal: price below VWAP + whale buying ---
    long_cond = (
        (vwap_z < -vwap_z_entry)
        & (whale_ofi > whale_ofi_min)
        & common_gate
    )

    # --- Short signal: price above VWAP + whale selling ---
    short_cond = (
        (vwap_z > vwap_z_entry)
        & (whale_ofi < -whale_ofi_min)
        & common_gate
    )

    signal[long_cond] = 1
    signal[short_cond] = -1

    return signal.astype(np.int8)
