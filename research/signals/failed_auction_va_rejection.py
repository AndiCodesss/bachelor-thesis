import numpy as np
import polars as pl

DEFAULT_PARAMS = {
    "va_long_threshold": 0.2,
    "va_short_threshold": 0.8,
    "absorption_min": 1.0,
    "vpin_max": 0.55,
    "spread_bps_max": 3.0,
    "tape_speed_z_min": 0.3,
    "minutes_since_open_min": 15,
    "adx_max": 30,
    "failed_auction_score_min": 1.0,
    "cooldown_bars": 15,
    "max_trades_per_day": 4,
    "pt_ticks": 6,
    "sl_ticks": 14,
    "time_stop_bars": 40,
    "session_filter": "eth",
}

STRATEGY_METADATA = {
    "strategy_name": "failed_auction_va_rejection",
    "hypothesis_id": "h009_failed_auction_va_rejection",
    "bar_configs": ["tick_610", "volume_2000"],
    "description": "Fades failed auctions beyond value area boundaries when absorption and low VPIN confirm noise-driven probe rejection.",
}


def _safe_col(df: pl.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy().astype(np.float64)
    return np.full(len(df), default, dtype=np.float64)


def _safe_bool_col(df: pl.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        arr = df[name].to_numpy()
        if arr.dtype == bool:
            return arr
        return np.where(np.isnan(arr.astype(np.float64)), False, arr.astype(np.float64) > 0.5)
    return np.zeros(len(df), dtype=bool)


def generate_signal(df: pl.DataFrame, params: dict) -> np.ndarray:
    p = {**DEFAULT_PARAMS, **params}
    n = len(df)
    signal = np.zeros(n, dtype=np.int32)

    # --- load features with safe fallbacks ---
    failed_bear = _safe_bool_col(df, "failed_auction_bear")
    failed_bull = _safe_bool_col(df, "failed_auction_bull")
    position_in_va = _safe_col(df, "position_in_va", default=0.5)
    absorption = _safe_col(df, "absorption_signal", default=0.0)
    vpin = _safe_col(df, "vpin", default=1.0)
    spread_bps = _safe_col(df, "spread_bps", default=999.0)
    tape_speed_z = _safe_col(df, "tape_speed_z", default=0.0)
    minutes_since_open = _safe_col(df, "minutes_since_open", default=0.0)
    adx_14 = _safe_col(df, "adx_14", default=100.0)
    fa_score = _safe_col(df, "failed_auction_score", default=0.0)

    # session date for daily trade counter reset
    if "ts_event" in df.columns:
        ts_event = df["ts_event"]
        if ts_event.dtype == pl.Datetime or str(ts_event.dtype).startswith("Datetime"):
            dates = ts_event.cast(pl.Date).to_numpy()
        elif ts_event.dtype in (pl.Int64, pl.UInt64):
            dates = ts_event.cast(pl.Datetime("ns")).cast(pl.Date).to_numpy()
        else:
            dates = np.zeros(n, dtype="datetime64[D]")
    else:
        dates = np.zeros(n, dtype="datetime64[D]")

    # --- replace NaN with conservative defaults ---
    position_in_va = np.where(np.isfinite(position_in_va), position_in_va, 0.5)
    absorption = np.where(np.isfinite(absorption), absorption, 0.0)
    vpin = np.where(np.isfinite(vpin), vpin, 1.0)
    spread_bps = np.where(np.isfinite(spread_bps), spread_bps, 999.0)
    tape_speed_z = np.where(np.isfinite(tape_speed_z), tape_speed_z, 0.0)
    minutes_since_open = np.where(np.isfinite(minutes_since_open), minutes_since_open, 0.0)
    adx_14 = np.where(np.isfinite(adx_14), adx_14, 100.0)
    fa_score = np.where(np.isfinite(fa_score), fa_score, 0.0)

    # --- common quality gates (vectorised) ---
    quality = (
        (absorption >= p["absorption_min"])
        & (vpin < p["vpin_max"])
        & (spread_bps < p["spread_bps_max"])
        & (tape_speed_z > p["tape_speed_z_min"])
        & (minutes_since_open > p["minutes_since_open_min"])
        & (adx_14 < p["adx_max"])
        & (fa_score >= p["failed_auction_score_min"])
    )

    # --- directional conditions ---
    long_cond = failed_bear & (position_in_va < p["va_long_threshold"]) & quality
    short_cond = failed_bull & (position_in_va > p["va_short_threshold"]) & quality

    # --- apply risk controls: cooldown + daily trade cap (sequential) ---
    cooldown_bars = int(p["cooldown_bars"])
    max_trades = int(p["max_trades_per_day"])

    bars_since_last_signal = cooldown_bars + 1  # start eligible
    current_date = None
    trades_today = 0

    for i in range(n):
        # reset daily counter on new date
        bar_date = dates[i]
        if bar_date != current_date:
            current_date = bar_date
            trades_today = 0

        bars_since_last_signal += 1

        if bars_since_last_signal <= cooldown_bars:
            continue
        if trades_today >= max_trades:
            continue

        if long_cond[i]:
            signal[i] = 1
            bars_since_last_signal = 0
            trades_today += 1
        elif short_cond[i]:
            signal[i] = -1
            bars_since_last_signal = 0
            trades_today += 1

    return signal.astype(np.int8)
