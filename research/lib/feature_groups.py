"""A/B experiment feature group definitions and filtering.

Defines which features belong to Group A (OHLCV-only) vs Group B (all features).
Group A includes only features derivable from open/high/low/close/volume/timestamp.
Group B includes everything (OHLCV + MBP1 microstructure features).

Any feature NOT in OHLCV_FEATURE_COLUMNS is automatically treated as MBP1-derived
and excluded from Group A. This is the conservative default — new features must be
explicitly added to the OHLCV set to appear in Group A.
"""

from __future__ import annotations

import polars as pl

# Valid feature group names for mission config
VALID_FEATURE_GROUPS = frozenset({"ohlcv", "all"})

# ── Group A: OHLCV-only features (56 columns) ───────────────────────────────
# These features are derivable purely from OHLCV bars + timestamps.
# Sources: momentum, ohlcv_indicators, statistical, opening_range,
#          and the OHLCV-derived subset of pipeline features.

OHLCV_FEATURE_COLUMNS = frozenset({
    # --- Momentum module (15 features) ---
    "body_ratio",
    "close_position",
    "high_low_range",
    "lower_wick_ratio",
    "momentum_volume",
    "range_ma5",
    "return_1bar",
    "return_5bar",
    "return_12bar",
    "upper_wick_ratio",
    "volume",
    "volume_ma5",
    "volume_ratio",
    "vwap_deviation",
    "vwap_deviation_ma5",

    # --- OHLCV indicators module (20 features) ---
    "sma_ratio_8",
    "sma_ratio_21",
    "sma_ratio_50",
    "sma_ratio_200",
    "ema_ratio_8",
    "ema_ratio_21",
    "ema_ratio_50",
    "rsi_14",
    "atr_norm_14",
    "bb_bandwidth_20",
    "bb_pctb_20",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "stoch_k_14",
    "stoch_d_14",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "obv_slope_14",

    # --- Statistical module (5 features) ---
    "log_return",
    "fracdiff_close",
    "yz_volatility",
    "vol_zscore",
    "vwap_dev_zscore",

    # --- Opening range module (4 features) ---
    "or_width",
    "position_in_or",
    "or_broken_up",
    "or_broken_down",

    # --- Pipeline module: OHLCV-derived subset (12 features) ---
    # Regime (uses range_ma5 and return_1bar — both OHLCV)
    "regime_vol_relative",
    "regime_autocorr",
    # Time-of-day (uses ts_event only)
    "minutes_since_open",
    "session_progress",
    "is_power_hour",
    "is_london_session",
    # Previous session (uses OHLCV only)
    "gap_open",
    "dist_prev_high",
    "dist_prev_low",
    # Accumulation (uses high_low_range — OHLCV)
    "range_compression",
    "range_compression_z",
    "volatility_compression_1bar",
})


def filter_feature_group(
    df: pl.DataFrame,
    group: str,
    *,
    keep_non_features: bool = True,
) -> pl.DataFrame:
    """Filter a feature matrix to only include columns for the specified group.

    Args:
        df: Full feature matrix from build_feature_matrix / load_cached_matrix.
        group: Feature group name — "ohlcv" for Group A, "all" for Group B.
        keep_non_features: If True, retain non-feature columns (ts_event, labels,
            close, bar metadata). Default True.

    Returns:
        Filtered DataFrame with only the allowed feature columns (plus non-features
        if keep_non_features is True).

    Raises:
        ValueError: If group is not a valid feature group name.
    """
    if group not in VALID_FEATURE_GROUPS:
        raise ValueError(
            f"Unknown feature_group '{group}'. Valid: {sorted(VALID_FEATURE_GROUPS)}"
        )

    # "all" means no filtering
    if group == "all":
        return df

    # For "ohlcv", keep only OHLCV feature columns + non-feature columns
    from src.framework.features_canonical.builder import NON_FEATURE_COLUMNS

    non_feature_set = set(NON_FEATURE_COLUMNS)
    cols_to_keep = []
    for col in df.columns:
        if col in non_feature_set:
            if keep_non_features:
                cols_to_keep.append(col)
        elif col in OHLCV_FEATURE_COLUMNS:
            cols_to_keep.append(col)
        # else: MBP1 feature → drop

    return df.select(cols_to_keep)
