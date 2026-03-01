"""Volume profile features: POC, value area, and profile shape statistics."""
import numpy as np
import polars as pl
from src.framework.data.constants import TICK_SIZE, VP_PROXIMITY_ATR_FRACTION, SWING_VP_LOOKBACK


def _compute_profile(prices: np.ndarray, sizes: np.ndarray):
    """Build volume-at-price histogram, extract POC, value area, and shape stats.

    Returns dict with: poc, va_high, va_low, skew, kurtosis, hvn_count,
    lvn_count, hvn_prices, lvn_prices.
    """
    if len(prices) == 0:
        return {
            "poc": np.nan, "va_high": np.nan, "va_low": np.nan,
            "skew": np.nan, "kurtosis": np.nan, "hvn_count": 0,
            "lvn_count": 0, "hvn_prices": np.array([]), "lvn_prices": np.array([]),
        }

    # Discretize prices to tick grid
    tick_prices = np.round(prices / TICK_SIZE) * TICK_SIZE

    # Build volume-at-price histogram using unique sorted levels
    unique_levels = np.unique(tick_prices)
    vol_at_price = np.zeros(len(unique_levels))
    for i, level in enumerate(unique_levels):
        vol_at_price[i] = sizes[tick_prices == level].sum()

    total_vol = vol_at_price.sum()
    if total_vol == 0:
        return {
            "poc": unique_levels[0], "va_high": unique_levels[0],
            "va_low": unique_levels[0], "skew": 0.0, "kurtosis": 0.0,
            "hvn_count": 0, "lvn_count": 0,
            "hvn_prices": np.array([]), "lvn_prices": np.array([]),
        }

    # POC: price level with max volume
    poc_idx = np.argmax(vol_at_price)
    poc = unique_levels[poc_idx]

    # Value area: expand from POC until 70% of total volume captured
    va_target = total_vol * 0.70
    va_vol = vol_at_price[poc_idx]
    lo_idx = poc_idx
    hi_idx = poc_idx

    while va_vol < va_target and (lo_idx > 0 or hi_idx < len(unique_levels) - 1):
        lo_vol = vol_at_price[lo_idx - 1] if lo_idx > 0 else -1.0
        hi_vol = vol_at_price[hi_idx + 1] if hi_idx < len(unique_levels) - 1 else -1.0
        if hi_vol >= lo_vol:
            hi_idx += 1
            va_vol += hi_vol
        else:
            lo_idx -= 1
            va_vol += lo_vol

    va_low = unique_levels[lo_idx]
    va_high = unique_levels[hi_idx]

    # Profile shape: skewness and kurtosis of volume distribution
    weights = vol_at_price / total_vol
    mean_p = np.sum(unique_levels * weights)
    var_p = np.sum(weights * (unique_levels - mean_p) ** 2)
    std_p = np.sqrt(var_p) if var_p > 0 else 1e-9

    skew = float(np.sum(weights * ((unique_levels - mean_p) / std_p) ** 3)) if var_p > 0 else 0.0
    kurtosis = float(np.sum(weights * ((unique_levels - mean_p) / std_p) ** 4)) if var_p > 0 else 0.0

    # HVN: local peaks (volume > both neighbors)
    # LVN: local minima (volume < both neighbors) — thin zones where price slips through
    hvn_prices_list = []
    lvn_prices_list = []
    for i in range(1, len(vol_at_price) - 1):
        if vol_at_price[i] > vol_at_price[i - 1] and vol_at_price[i] > vol_at_price[i + 1]:
            hvn_prices_list.append(unique_levels[i])
        if vol_at_price[i] < vol_at_price[i - 1] and vol_at_price[i] < vol_at_price[i + 1]:
            lvn_prices_list.append(unique_levels[i])

    return {
        "poc": poc, "va_high": va_high, "va_low": va_low,
        "skew": skew, "kurtosis": kurtosis,
        "hvn_count": len(hvn_prices_list), "lvn_count": len(lvn_prices_list),
        "hvn_prices": np.array(hvn_prices_list), "lvn_prices": np.array(lvn_prices_list),
    }


def _sanitize_bar_vap(
    prices_list: list | None,
    sizes_list: list | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one bar's VAP lists into clean numpy arrays."""
    if prices_list is None or sizes_list is None:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    n = min(len(prices_list), len(sizes_list))
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    prices = np.asarray(prices_list[:n], dtype=np.float64)
    sizes = np.asarray(sizes_list[:n], dtype=np.float64)
    valid = np.isfinite(prices) & np.isfinite(sizes) & (sizes > 0)
    if not np.any(valid):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    return prices[valid], sizes[valid]


def _build_bar_vap_inputs(bars: pl.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build per-bar histogram inputs for profile construction.

    Prefers true traded VAP lists from bars.py footprint pass:
    - vap_prices: list of price levels traded in bar
    - vap_volumes: list of executed volume at each level

    Falls back to bar close + volume (legacy approximation) if VAP lists
    are not available.
    """
    has_vap = "vap_prices" in bars.columns and "vap_volumes" in bars.columns

    if has_vap:
        raw_prices = bars["vap_prices"].to_list()
        raw_sizes = bars["vap_volumes"].to_list()
        prices_per_bar: list[np.ndarray] = []
        sizes_per_bar: list[np.ndarray] = []
        for p_list, s_list in zip(raw_prices, raw_sizes):
            p_arr, s_arr = _sanitize_bar_vap(p_list, s_list)
            prices_per_bar.append(p_arr)
            sizes_per_bar.append(s_arr)
        return prices_per_bar, sizes_per_bar

    closes = bars["close"].to_numpy().astype(np.float64)
    volumes = bars["volume"].to_numpy().astype(np.float64)
    prices_per_bar = [np.array([closes[i]], dtype=np.float64) for i in range(len(bars))]
    sizes_per_bar = [np.array([max(volumes[i], 0.0)], dtype=np.float64) for i in range(len(bars))]
    return prices_per_bar, sizes_per_bar


def _concat_bar_vap_window(
    prices_per_bar: list[np.ndarray],
    sizes_per_bar: list[np.ndarray],
    start: int,
    end_exclusive: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-bar VAP arrays across a bar window."""
    price_chunks = []
    size_chunks = []
    for i in range(start, end_exclusive):
        if len(prices_per_bar[i]) == 0:
            continue
        price_chunks.append(prices_per_bar[i])
        size_chunks.append(sizes_per_bar[i])

    if not price_chunks:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    return np.concatenate(price_chunks), np.concatenate(size_chunks)


def _compute_swing_vp_features(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    safe_atr: np.ndarray,
    prices_per_bar: list[np.ndarray],
    sizes_per_bar: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Swing-anchored volume profile features based on channel breakouts."""
    n = len(close)
    lookback = SWING_VP_LOOKBACK
    max_swing_bars = 30

    swing_poc_dist = np.full(n, np.nan)
    swing_lvn_dist = np.full(n, np.nan)
    swing_hvn_dist = np.full(n, np.nan)
    swing_va_position = np.full(n, np.nan)
    breakout_dir = np.full(n, 0.0)
    bars_since = np.full(n, np.nan)

    cur_dir = 0
    breakout_start = -1
    bars_count = 0

    for i in range(n):
        if i >= lookback:
            ch_high = high[i - lookback:i].max()
            ch_low = low[i - lookback:i].min()

            new_dir = 0
            if close[i] > ch_high:
                new_dir = 1
            elif close[i] < ch_low:
                new_dir = -1

            if new_dir != 0 and new_dir != cur_dir:
                cur_dir = new_dir
                breakout_start = i
                bars_count = 0

        if cur_dir != 0 and breakout_start >= 0:
            bars_count += 1

        breakout_dir[i] = float(cur_dir)

        if cur_dir == 0 or breakout_start < 0:
            continue

        bars_since[i] = float(bars_count)

        if bars_count > max_swing_bars:
            continue

        prices, sizes = _concat_bar_vap_window(
            prices_per_bar, sizes_per_bar, breakout_start, i + 1,
        )
        if len(prices) == 0:
            continue

        prof = _compute_profile(prices, sizes)
        cur_close = close[i]
        cur_atr = safe_atr[i]

        if not np.isnan(prof["poc"]):
            swing_poc_dist[i] = (cur_close - prof["poc"]) / cur_atr

        lvn_prices = prof["lvn_prices"]
        if len(lvn_prices) > 0:
            swing_lvn_dist[i] = np.abs(cur_close - lvn_prices).min() / cur_atr

        hvn_prices = prof["hvn_prices"]
        if len(hvn_prices) > 0:
            swing_hvn_dist[i] = np.abs(cur_close - hvn_prices).min() / cur_atr

        va_range = prof["va_high"] - prof["va_low"]
        if va_range > 1e-9:
            swing_va_position[i] = (cur_close - prof["va_low"]) / va_range
        elif not np.isnan(prof["va_low"]):
            swing_va_position[i] = 0.5

    return {
        "swing_poc_dist": swing_poc_dist,
        "swing_lvn_dist": swing_lvn_dist,
        "swing_hvn_dist": swing_hvn_dist,
        "swing_va_position": swing_va_position,
        "breakout_direction": breakout_dir,
        "bars_since_breakout": bars_since,
    }


def compute_volume_profile_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Volume profile features from pre-aggregated bar data.

    Expects bars with columns: ts_event, open, high, low, close, volume.
    If bars also include `vap_prices` and `vap_volumes` (from bars.py),
    profiles are built from true executed traded volume-at-price.
    Otherwise falls back to bar close + bar volume approximation.
    """
    if len(bars) == 0:
        return _empty_result()

    select_cols = ["ts_event", "open", "high", "low", "close", "volume"]
    if "vap_prices" in bars.columns and "vap_volumes" in bars.columns:
        select_cols.extend(["vap_prices", "vap_volumes"])
    bars = bars.select(select_cols).sort("ts_event")

    # True Range for ATR (14-bar rolling)
    bars = bars.with_columns([
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_date"),
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ).alias("_true_range"),
    ])
    bars = bars.with_columns(
        pl.col("_true_range").rolling_mean(window_size=14, min_samples=1).alias("_atr"),
    )

    bars_df = bars

    if len(bars_df) == 0:
        return _empty_result()

    prices_per_bar, sizes_per_bar = _build_bar_vap_inputs(bars_df)
    n = len(bars_df)

    # --- Session profile (expanding window — causal, no lookahead) ---
    # At each bar, profile uses only bars from session start up to current bar.
    date_list = bars_df["_date"].to_list()
    dates = sorted(set(date_list))
    day_start_idx = {}
    for d in dates:
        day_idxs = [i for i, day in enumerate(date_list) if day == d]
        if day_idxs:
            day_start_idx[d] = day_idxs[0]

    session_poc = np.full(n, np.nan)
    session_va_high = np.full(n, np.nan)
    session_va_low = np.full(n, np.nan)
    session_skew = np.full(n, np.nan)
    session_kurtosis = np.full(n, np.nan)
    session_hvn = np.zeros(n, dtype=np.int32)

    for i in range(n):
        d = date_list[i]
        start = day_start_idx.get(d, i)
        prices, sizes = _concat_bar_vap_window(
            prices_per_bar, sizes_per_bar, start, i + 1,
        )
        if len(prices) == 0:
            continue
        prof = _compute_profile(prices, sizes)
        session_poc[i] = prof["poc"]
        session_va_high[i] = prof["va_high"]
        session_va_low[i] = prof["va_low"]
        session_skew[i] = prof["skew"]
        session_kurtosis[i] = prof["kurtosis"]
        session_hvn[i] = prof["hvn_count"]

    close = bars_df["close"].to_numpy()
    high = bars_df["high"].to_numpy()
    low = bars_df["low"].to_numpy()
    atr = bars_df["_atr"].to_numpy()
    safe_atr = np.where(np.isnan(atr) | (atr < 1e-9), 1e-9, atr)

    # Session features
    poc_distance_raw = close - session_poc
    poc_distance = poc_distance_raw / safe_atr
    va_width = (session_va_high - session_va_low) / np.where(close == 0, 1e-9, close)
    va_range = session_va_high - session_va_low
    safe_va_range = np.where(va_range < 1e-9, 1e-9, va_range)
    position_in_va = np.where(
        va_range < 1e-9, 0.5,
        (close - session_va_low) / safe_va_range,
    )

    # --- Rolling profile (24-bar window, bar-level data) ---
    rolling_window = 24
    rolling_poc_arr = np.full(n, np.nan)
    rolling_va_high_arr = np.full(n, np.nan)
    rolling_va_low_arr = np.full(n, np.nan)
    rolling_skew_arr = np.full(n, np.nan)
    rolling_hvn_arr = np.full(n, 0)
    rolling_lvn_arr = np.full(n, 0)
    vol_concentration_arr = np.full(n, np.nan)
    dist_nearest_hvn_arr = np.full(n, np.nan)
    dist_nearest_lvn_arr = np.full(n, np.nan)
    at_hvn_arr = np.full(n, 0.0)
    at_lvn_arr = np.full(n, 0.0)
    hvn_lvn_ratio_arr = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - rolling_window + 1)
        window_prices, window_sizes = _concat_bar_vap_window(
            prices_per_bar,
            sizes_per_bar,
            start,
            i + 1,
        )

        if len(window_prices) == 0 or np.sum(window_sizes) == 0:
            continue

        prof = _compute_profile(window_prices, window_sizes)
        rolling_poc_arr[i] = prof["poc"]
        rolling_va_high_arr[i] = prof["va_high"]
        rolling_va_low_arr[i] = prof["va_low"]
        rolling_skew_arr[i] = prof["skew"]
        rolling_hvn_arr[i] = prof["hvn_count"]
        rolling_lvn_arr[i] = prof["lvn_count"]

        cur_close = close[i]
        cur_atr = safe_atr[i]

        # Distance to nearest HVN — magnets where price consolidates
        hvn_prices = prof["hvn_prices"]
        if len(hvn_prices) > 0:
            hvn_dists = np.abs(cur_close - hvn_prices)
            min_hvn_dist = hvn_dists.min()
            dist_nearest_hvn_arr[i] = min_hvn_dist / cur_atr
            at_hvn_arr[i] = 1.0 if min_hvn_dist < VP_PROXIMITY_ATR_FRACTION * cur_atr else 0.0

        # Distance to nearest LVN — thin zones where price slips fast
        lvn_prices = prof["lvn_prices"]
        if len(lvn_prices) > 0:
            lvn_dists = np.abs(cur_close - lvn_prices)
            min_lvn_dist = lvn_dists.min()
            dist_nearest_lvn_arr[i] = min_lvn_dist / cur_atr
            at_lvn_arr[i] = 1.0 if min_lvn_dist < VP_PROXIMITY_ATR_FRACTION * cur_atr else 0.0

        # HVN/LVN ratio: many balance areas = rotational, few = trending
        hvn_c = prof["hvn_count"]
        lvn_c = prof["lvn_count"]
        hvn_lvn_ratio_arr[i] = hvn_c / (lvn_c + 1)

        if not np.isnan(prof["poc"]):
            half_atr = cur_atr * VP_PROXIMITY_ATR_FRACTION
            mask = np.abs(window_prices - prof["poc"]) <= half_atr
            conc = window_sizes[mask].sum() / window_sizes.sum() if window_sizes.sum() > 0 else 0.0
            vol_concentration_arr[i] = conc

    # Rolling POC distance and slope
    rolling_poc_distance = (close - rolling_poc_arr) / safe_atr

    poc_slope_6 = np.full(n, np.nan)
    for i in range(6, n):
        if not np.isnan(rolling_poc_arr[i]) and not np.isnan(rolling_poc_arr[i - 6]):
            poc_slope_6[i] = (rolling_poc_arr[i] - rolling_poc_arr[i - 6]) / safe_atr[i]

    # Rolling VA position
    rolling_va_range = rolling_va_high_arr - rolling_va_low_arr
    safe_rolling_va_range = np.where(
        np.isnan(rolling_va_range) | (rolling_va_range < 1e-9), 1e-9, rolling_va_range,
    )
    rolling_va_position = np.where(
        np.isnan(rolling_va_range) | (rolling_va_range < 1e-9), 0.5,
        (close - rolling_va_low_arr) / safe_rolling_va_range,
    )

    # --- Swing-anchored volume profile ---
    swing = _compute_swing_vp_features(
        high, low, close, safe_atr, prices_per_bar, sizes_per_bar,
    )

    # Build result DataFrame
    int_cols = ("hvn_count", "rolling_hvn_count", "rolling_lvn_count")
    result = pl.DataFrame({
        "ts_event": bars_df["ts_event"],
        "poc_price": session_poc,
        "poc_distance": poc_distance,
        "poc_distance_raw": poc_distance_raw,
        "va_high": session_va_high,
        "va_low": session_va_low,
        "va_width": va_width,
        "position_in_va": position_in_va,
        "vp_skew": session_skew,
        "vp_kurtosis": session_kurtosis,
        "hvn_count": session_hvn.astype(np.int32),
        "rolling_poc": rolling_poc_arr,
        "rolling_poc_distance": rolling_poc_distance,
        "poc_slope_6": poc_slope_6,
        "rolling_va_high": rolling_va_high_arr,
        "rolling_va_low": rolling_va_low_arr,
        "rolling_va_position": rolling_va_position,
        "vol_concentration": vol_concentration_arr,
        "rolling_vp_skew": rolling_skew_arr,
        "rolling_hvn_count": rolling_hvn_arr.astype(np.int32),
        "rolling_lvn_count": rolling_lvn_arr.astype(np.int32),
        "dist_nearest_hvn": dist_nearest_hvn_arr,
        "dist_nearest_lvn": dist_nearest_lvn_arr,
        "at_hvn": at_hvn_arr,
        "at_lvn": at_lvn_arr,
        "hvn_lvn_ratio": hvn_lvn_ratio_arr,
        "swing_poc_dist": swing["swing_poc_dist"],
        "swing_lvn_dist": swing["swing_lvn_dist"],
        "swing_hvn_dist": swing["swing_hvn_dist"],
        "swing_va_position": swing["swing_va_position"],
        "breakout_direction": swing["breakout_direction"],
        "bars_since_breakout": swing["bars_since_breakout"],
    })

    float_cols = [c for c in result.columns if c != "ts_event" and c not in int_cols]
    result = result.with_columns([pl.col(c).cast(pl.Float64) for c in float_cols])

    return result


def _empty_result() -> pl.DataFrame:
    schema = {
        "ts_event": pl.Datetime("ns", "UTC"),
        "poc_price": pl.Float64, "poc_distance": pl.Float64,
        "poc_distance_raw": pl.Float64,
        "va_high": pl.Float64, "va_low": pl.Float64,
        "va_width": pl.Float64, "position_in_va": pl.Float64,
        "vp_skew": pl.Float64, "vp_kurtosis": pl.Float64,
        "hvn_count": pl.Int32,
        "rolling_poc": pl.Float64, "rolling_poc_distance": pl.Float64,
        "poc_slope_6": pl.Float64,
        "rolling_va_high": pl.Float64, "rolling_va_low": pl.Float64,
        "rolling_va_position": pl.Float64,
        "vol_concentration": pl.Float64,
        "rolling_vp_skew": pl.Float64,
        "rolling_hvn_count": pl.Int32,
        "rolling_lvn_count": pl.Int32,
        "dist_nearest_hvn": pl.Float64,
        "dist_nearest_lvn": pl.Float64,
        "at_hvn": pl.Float64,
        "at_lvn": pl.Float64,
        "hvn_lvn_ratio": pl.Float64,
        "swing_poc_dist": pl.Float64,
        "swing_lvn_dist": pl.Float64,
        "swing_hvn_dist": pl.Float64,
        "swing_va_position": pl.Float64,
        "breakout_direction": pl.Float64,
        "bars_since_breakout": pl.Float64,
    }
    return pl.DataFrame(schema=schema)
