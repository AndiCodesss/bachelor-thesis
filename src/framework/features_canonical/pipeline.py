"""Interaction, regime, time-of-day, and previous session features."""
import polars as pl

from src.framework.data.constants import (
    RTH_START_TIME,
    RTH_END_TIME,
    REGIME_VOL_LOOKBACK,
    REGIME_AUTOCORR_WINDOW,
    REGIME_AUTOCORR_MIN_SAMPLES,
    RTH_TOTAL_MINUTES,
    RTH_OPEN_HOUR,
    RTH_OPEN_MINUTE,
    POWER_HOUR_END_HOUR,
    POWER_HOUR_END_MINUTE,
    ACCUM_RANGE_WINDOW,
    RANGE_COMPRESSION_WINDOW,
    RANGE_COMPRESSION_Z_WINDOW,
    RECOIL_STALL_VOL_COMPRESSION,
)

# Columns this module requires from the joined feature matrix
REQUIRED_COLUMNS = [
    "range_ma5",
    "return_1bar",
    "order_flow_imbalance",
    "spread_bps",
    "book_imbalance",
    "volume_delta",
]


def compute_pipeline_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute interaction and regime features on the joined feature matrix.

    Unlike other feature modules that operate on raw ticks, this creates
    multiplicative cross-terms between features from different modules.

    Args:
        df: DataFrame with all base features already computed and joined

    Returns:
        DataFrame with additional interaction and regime columns added
    """
    # Regime vol relative: current volatility / median(volatility, lookback)
    # range_ma5 is smoothed bar range — proxy for realized volatility
    df = df.with_columns(
        (pl.col("range_ma5")
         / pl.col("range_ma5").rolling_median(window_size=REGIME_VOL_LOOKBACK, min_samples=1))
        .alias("regime_vol_relative")
    )

    # Regime autocorrelation: rolling correlation of returns with lagged returns
    # Positive = trending, negative = mean-reverting/choppy
    df = df.with_columns(
        pl.col("return_1bar").shift(1).alias("_return_lag1")
    )
    df = df.with_columns(
        pl.rolling_corr(
            pl.col("return_1bar"),
            pl.col("_return_lag1"),
            window_size=REGIME_AUTOCORR_WINDOW,
            min_samples=REGIME_AUTOCORR_MIN_SAMPLES,
        ).alias("regime_autocorr")
    )
    df = df.drop("_return_lag1")

    # Interaction cross-terms
    # ix_ofi_x_vol: flow matters more in high vol
    # ix_ofi_x_spread: flow fighting wide spreads implies urgency
    # ix_bbo_imb_x_vdelta: agreement between LOB shape and trade execution
    df = df.with_columns([
        (pl.col("order_flow_imbalance") * pl.col("range_ma5")).alias("ix_ofi_x_vol"),
        (pl.col("order_flow_imbalance") * pl.col("spread_bps")).alias("ix_ofi_x_spread"),
        (pl.col("book_imbalance") * pl.col("volume_delta")).alias("ix_bbo_imb_x_vdelta"),
    ])

    # --- Time-of-day features (includes is_london_session) ---
    if "ts_event" in df.columns:
        df = _add_time_features(df)

    # --- Previous session features ---
    if "ts_event" in df.columns and "close" in df.columns and "high" in df.columns:
        df = _add_prev_session_features(df)

    # --- Previous session VP features ---
    if all(c in df.columns for c in ["poc_price", "va_high", "va_low", "close", "range_ma5"]):
        df = _add_prev_session_vp_features(df)

    # --- Orderflow + auction context features ---
    # Cached at build time so downstream ML runners don't have to recompute.
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        df = _add_orderflow_context_features(df)

    # --- Accumulation detection + micro-stall features ---
    if "high_low_range" in df.columns:
        df = _add_accumulation_features(df)

    # --- Failed auction at VA edges ---
    if all(c in df.columns for c in ["prev_day_vah", "prev_day_val", "high", "low", "close"]):
        df = _add_failed_auction_features(df)

    # --- Squeeze detection (trapped trader covering) ---
    if all(c in df.columns for c in ["high_low_range", "volume_imbalance"]):
        df = _add_squeeze_features(df)

    return df


def _add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add time-of-day features using US/Eastern timezone."""
    # Convert UTC ts_event to US/Eastern for time-of-day computation
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").alias("_ts_eastern"),
    )

    # minutes_since_open: minutes elapsed since 09:30 ET
    # Cast to Int32 first — dt.hour()/minute() return UInt8 which overflows on multiplication
    df = df.with_columns(
        (pl.col("_ts_eastern").dt.hour().cast(pl.Int32) * 60
         + pl.col("_ts_eastern").dt.minute().cast(pl.Int32)
         - RTH_OPEN_HOUR * 60 - RTH_OPEN_MINUTE)
        .cast(pl.Float64)
        .alias("minutes_since_open"),
    )

    # session_progress: 0.0 at open, 1.0 at close (09:30 to 16:00 ET = 390 min)
    df = df.with_columns(
        (pl.col("minutes_since_open") / RTH_TOTAL_MINUTES).alias("session_progress"),
    )

    # is_power_hour: 1.0 if within first 90 minutes of RTH (09:30-11:00 ET)
    df = df.with_columns(
        pl.when(
            (pl.col("minutes_since_open") >= 0)
            & (pl.col("minutes_since_open") < 90)
        )
        .then(1.0)
        .otherwise(0.0)
        .alias("is_power_hour"),
    )

    # is_london_session: 1.0 if bar is from London session (before RTH open)
    df = df.with_columns(
        pl.when(pl.col("minutes_since_open") < 0)
        .then(1.0)
        .otherwise(0.0)
        .alias("is_london_session"),
    )

    df = df.drop("_ts_eastern")
    return df


def _add_prev_session_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add previous session high/low/close and derived distance features."""
    # Extract trading date from ts_event in US/Eastern
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_session_date"),
    )

    # Per-session aggregates: high, low, close (last bar's close)
    session_agg = df.group_by("_session_date").agg([
        pl.col("high").max().alias("_session_high"),
        pl.col("low").min().alias("_session_low"),
        pl.col("close").last().alias("_session_close"),
        pl.col("open").first().alias("_session_open"),
    ]).sort("_session_date")

    # Shift by 1 session to get previous day values (no lookahead)
    session_agg = session_agg.with_columns([
        pl.col("_session_high").shift(1).alias("prev_day_high"),
        pl.col("_session_low").shift(1).alias("prev_day_low"),
        pl.col("_session_close").shift(1).alias("prev_day_close"),
    ])

    # Gap open: (today's open - prev_day_close) / prev_day_close
    session_agg = session_agg.with_columns(
        pl.when(pl.col("prev_day_close").is_not_null() & (pl.col("prev_day_close").abs() > 1e-9))
        .then((pl.col("_session_open") - pl.col("prev_day_close")) / pl.col("prev_day_close"))
        .otherwise(None)
        .alias("gap_open"),
    )

    session_agg = session_agg.select([
        "_session_date", "prev_day_high", "prev_day_low", "prev_day_close", "gap_open",
    ])

    # Join back to bar-level data (forward-fills prev_day values for entire session)
    df = df.join(session_agg, on="_session_date", how="left")

    # Distance to previous day levels, normalized by ATR
    # Use range_ma5 as ATR proxy (already computed by momentum module)
    safe_atr = pl.when(pl.col("range_ma5").is_not_null() & (pl.col("range_ma5").abs() > 1e-9)).then(
        pl.col("range_ma5")
    ).otherwise(1e-9)

    df = df.with_columns([
        ((pl.col("close") - pl.col("prev_day_high")) / safe_atr).alias("dist_prev_high"),
        ((pl.col("close") - pl.col("prev_day_low")) / safe_atr).alias("dist_prev_low"),
    ])

    df = df.drop("_session_date")
    return df


def _add_prev_session_vp_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add previous session volume profile levels and distance features."""
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_vp_session_date"),
    )

    # Per-session: last bar's VP levels
    session_vp = df.group_by("_vp_session_date").agg([
        pl.col("poc_price").last().alias("prev_day_poc"),
        pl.col("va_high").last().alias("prev_day_vah"),
        pl.col("va_low").last().alias("prev_day_val"),
    ]).sort("_vp_session_date")

    # Shift by 1 session (no lookahead)
    session_vp = session_vp.with_columns([
        pl.col("prev_day_poc").shift(1),
        pl.col("prev_day_vah").shift(1),
        pl.col("prev_day_val").shift(1),
    ])

    df = df.join(session_vp, on="_vp_session_date", how="left")

    # Distance features normalized by absolute ATR (range_ma5 is pct-based, convert)
    abs_atr = pl.col("close") * pl.col("range_ma5")
    safe_atr = pl.when(abs_atr.is_not_null() & (abs_atr > 1e-9)).then(abs_atr).otherwise(1.0)

    safe_va_width = pl.when(
        (pl.col("prev_day_vah") - pl.col("prev_day_val")).abs() > 1e-9
    ).then(
        pl.col("prev_day_vah") - pl.col("prev_day_val")
    ).otherwise(1e-9)

    df = df.with_columns([
        ((pl.col("close") - pl.col("prev_day_poc")) / safe_atr).alias("dist_prev_poc"),
        ((pl.col("close") - pl.col("prev_day_vah")) / safe_atr).alias("dist_prev_vah"),
        ((pl.col("close") - pl.col("prev_day_val")) / safe_atr).alias("dist_prev_val"),
        ((pl.col("close") - pl.col("prev_day_val")) / safe_va_width).alias("prev_day_va_position"),
    ])

    df = df.drop("_vp_session_date")
    return df


def _add_accumulation_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add accumulation detection and micro-stall confirmation features.

    Requires high_low_range from momentum, absorption_signal from orderflow,
    and recoil_50pct from microstructure in the joined matrix.
    """
    hlr = pl.col("high_low_range")

    # range_compression: coefficient of variation of bar range over 6 bars
    # Low CV = tight contraction (energy building before breakout)
    rolling_mean_6 = hlr.rolling_mean(window_size=RANGE_COMPRESSION_WINDOW, min_samples=1)
    rolling_std_6 = hlr.rolling_std(window_size=RANGE_COMPRESSION_WINDOW, min_samples=2)
    safe_mean = pl.when(rolling_mean_6.abs() > 1e-9).then(rolling_mean_6).otherwise(1e-9)

    df = df.with_columns([
        (rolling_std_6 / safe_mean).alias("range_compression"),
        (hlr / rolling_mean_6.replace(0, 1e-9)).alias("volatility_compression_1bar"),
    ])

    # range_compression_z: z-score of range_compression over 24 bars
    # Negative = unusually tight range compression
    rc = pl.col("range_compression")
    rc_mean = rc.rolling_mean(window_size=RANGE_COMPRESSION_Z_WINDOW, min_samples=2)
    rc_std = rc.rolling_std(window_size=RANGE_COMPRESSION_Z_WINDOW, min_samples=2)
    safe_rc_std = pl.when(rc_std.is_not_null() & (rc_std > 1e-9)).then(rc_std).otherwise(1e-9)

    df = df.with_columns(
        ((rc - rc_mean) / safe_rc_std).alias("range_compression_z"),
    )

    # Rolling 50th percentile of range over 12 bars (for contraction check)
    range_median_12 = hlr.rolling_median(window_size=ACCUM_RANGE_WINDOW, min_samples=1)

    # post_absorption_contraction: absorption in last 3 bars AND tight range
    if "absorption_signal" in df.columns:
        # Any absorption in last 3 bars (sum of shifted values > 0)
        recent_absorption = pl.max_horizontal(
            pl.col("absorption_signal"),
            pl.col("absorption_signal").shift(1).fill_null(0.0),
            pl.col("absorption_signal").shift(2).fill_null(0.0),
        )
        df = df.with_columns(
            pl.when(
                (recent_absorption > 0) & (hlr < range_median_12)
            ).then(1.0).otherwise(0.0).alias("post_absorption_contraction"),
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("post_absorption_contraction"))

    # post_recoil_stall: Valentini "Flick Back" stall detection (simplified B1 logic)
    # Detects when price recoils 50% AND volatility compresses (stall before continuation)
    if "recoil_50pct" in df.columns and "volatility_compression_1bar" in df.columns:
        df = df.with_columns(
            pl.when(
                (pl.col("recoil_50pct") == 1.0)
                & (pl.col("volatility_compression_1bar") < RECOIL_STALL_VOL_COMPRESSION)
            ).then(1.0).otherwise(0.0).alias("post_recoil_stall"),
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("post_recoil_stall"))

    return df


def _add_orderflow_context_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add advanced orderflow/AMT features used by event-driven ensembles.

    Features:
    - ofi_impulse_z: z-scored OFI shock over rolling 50 bars
    - whale_imbalance_ratio_30: whale directional ratio in [-1, 1] approx
    - whale_participation_rate_30: whale share of bar volume
    - whale_ofi_alignment: signed alignment between whale flow and OFI
    - value_acceptance_rate_8: rolling share of bars accepted inside value
    - va_rejection_score: directional rejection at VA edges
    - price_discovery_score: directional discovery outside value area
    - structure_break_score: break-of-structure with flow confirmation
    """
    def _expr(df_in: pl.DataFrame, name: str, default: float) -> pl.Expr:
        if name in df_in.columns:
            return pl.col(name).cast(pl.Float64)
        return pl.lit(float(default)).cast(pl.Float64)

    def _tanh(expr: pl.Expr) -> pl.Expr:
        # tanh(x) = 2 / (1 + exp(-2x)) - 1
        return (2.0 / (1.0 + (-2.0 * expr).exp())) - 1.0

    def _clip_01(expr: pl.Expr) -> pl.Expr:
        return pl.max_horizontal(pl.min_horizontal(expr, pl.lit(1.0)), pl.lit(0.0))

    ofi = _expr(df, "order_flow_imbalance", 0.0)
    vol_imb = _expr(df, "volume_imbalance", 0.0)
    va_pos = _expr(df, "rolling_va_position", 0.5)
    whale_buy = _expr(df, "whale_buy_volume_30", 0.0)
    whale_sell = _expr(df, "whale_sell_volume_30", 0.0)
    volume = _expr(df, "volume", 0.0)

    df = df.with_columns([
        ofi.alias("_ofi_raw"),
        vol_imb.alias("_vol_imb_raw"),
        va_pos.alias("_va_pos_raw"),
        whale_buy.alias("_whale_buy_raw"),
        whale_sell.alias("_whale_sell_raw"),
        volume.alias("_volume_raw"),
    ])

    df = df.with_columns([
        pl.col("_ofi_raw").abs().rolling_mean(window_size=50, min_samples=1).alias("_ofi_abs_ma50"),
        pl.col("_ofi_raw").rolling_mean(window_size=50, min_samples=2).alias("_ofi_mean50"),
        pl.col("_ofi_raw").rolling_std(window_size=50, min_samples=2).alias("_ofi_std50"),
        (pl.col("_whale_buy_raw") - pl.col("_whale_sell_raw")).alias("_whale_delta_raw"),
        (pl.col("_whale_buy_raw") + pl.col("_whale_sell_raw")).alias("_whale_total_raw"),
    ])

    df = df.with_columns([
        pl.when(pl.col("_ofi_abs_ma50") > 1e-9)
        .then(pl.col("_ofi_raw") / pl.col("_ofi_abs_ma50"))
        .otherwise(0.0)
        .alias("_ofi_norm"),
        pl.when(pl.col("_ofi_std50") > 1e-9)
        .then((pl.col("_ofi_raw") - pl.col("_ofi_mean50")) / pl.col("_ofi_std50"))
        .otherwise(0.0)
        .alias("ofi_impulse_z"),
        (pl.col("_whale_delta_raw") / (pl.col("_whale_total_raw") + 1.0)).alias("whale_imbalance_ratio_30"),
        (pl.col("_whale_total_raw") / (pl.col("_volume_raw") + 1.0)).alias("whale_participation_rate_30"),
        pl.col("_whale_delta_raw").abs().rolling_mean(window_size=50, min_samples=1).alias("_whale_abs_ma50"),
        ((pl.col("_va_pos_raw") >= 0.0) & (pl.col("_va_pos_raw") <= 1.0))
        .cast(pl.Float64)
        .rolling_mean(window_size=8, min_samples=1)
        .alias("value_acceptance_rate_8"),
    ])

    df = df.with_columns([
        pl.when(pl.col("_whale_abs_ma50") > 1e-9)
        .then(pl.col("_whale_delta_raw") / pl.col("_whale_abs_ma50"))
        .otherwise(0.0)
        .alias("_whale_norm"),
    ])

    bar_range = pl.max_horizontal(
        pl.col("high").cast(pl.Float64) - pl.col("low").cast(pl.Float64),
        pl.lit(1e-9),
    )
    lower_wick = _clip_01(
        (pl.min_horizontal(pl.col("open").cast(pl.Float64), pl.col("close").cast(pl.Float64)) - pl.col("low").cast(pl.Float64))
        / bar_range,
    )
    upper_wick = _clip_01(
        (pl.col("high").cast(pl.Float64) - pl.max_horizontal(pl.col("open").cast(pl.Float64), pl.col("close").cast(pl.Float64)))
        / bar_range,
    )
    edge = 0.20
    near_val = _clip_01((edge - pl.col("_va_pos_raw")) / edge)
    near_vah = _clip_01((pl.col("_va_pos_raw") - (1.0 - edge)) / edge)
    flow_combo = 0.5 * pl.col("_vol_imb_raw") + 0.5 * _tanh(pl.col("_ofi_norm"))

    df = df.with_columns([
        (_tanh(pl.col("_whale_norm")) * _tanh(pl.col("_ofi_norm"))).alias("whale_ofi_alignment"),
        flow_combo.alias("_flow_combo"),
        lower_wick.alias("_lower_wick"),
        upper_wick.alias("_upper_wick"),
        near_val.alias("_near_val"),
        near_vah.alias("_near_vah"),
        pl.col("high").cast(pl.Float64).shift(1).rolling_max(window_size=12, min_samples=12).alias("_prev_high_12"),
        pl.col("low").cast(pl.Float64).shift(1).rolling_min(window_size=12, min_samples=12).alias("_prev_low_12"),
    ])

    above_va = pl.when(pl.col("_va_pos_raw") > 1.0).then(pl.col("_va_pos_raw") - 1.0).otherwise(0.0)
    below_va = pl.when(pl.col("_va_pos_raw") < 0.0).then(-pl.col("_va_pos_raw")).otherwise(0.0)
    pos_flow = pl.max_horizontal(pl.col("_flow_combo"), pl.lit(0.0))
    neg_flow = pl.max_horizontal(-pl.col("_flow_combo"), pl.lit(0.0))

    df = df.with_columns([
        (
            pl.col("_near_val") * pl.col("_lower_wick") * pos_flow
            - pl.col("_near_vah") * pl.col("_upper_wick") * neg_flow
        ).alias("va_rejection_score"),
        ((above_va - below_va) * pl.col("_flow_combo")).alias("price_discovery_score"),
        (
            pl.when(pl.col("_prev_high_12").is_not_null() & (pl.col("close").cast(pl.Float64) > pl.col("_prev_high_12")))
            .then(pos_flow)
            .otherwise(0.0)
            - pl.when(pl.col("_prev_low_12").is_not_null() & (pl.col("close").cast(pl.Float64) < pl.col("_prev_low_12")))
            .then(neg_flow)
            .otherwise(0.0)
        ).alias("structure_break_score"),
    ])

    return df.drop([
        "_ofi_raw",
        "_vol_imb_raw",
        "_va_pos_raw",
        "_whale_buy_raw",
        "_whale_sell_raw",
        "_volume_raw",
        "_ofi_abs_ma50",
        "_ofi_mean50",
        "_ofi_std50",
        "_whale_delta_raw",
        "_whale_total_raw",
        "_ofi_norm",
        "_whale_abs_ma50",
        "_whale_norm",
        "_flow_combo",
        "_lower_wick",
        "_upper_wick",
        "_near_val",
        "_near_vah",
        "_prev_high_12",
        "_prev_low_12",
    ])


def _add_failed_auction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Detect failed auctions at previous day's value area edges.

    Bull failed auction: price probes below prev_day_val, fails to hold,
    closes back above with absorption or whale confirmation.
    Bear failed auction: mirror at prev_day_vah.
    """
    has_absorption = "absorption_signal" in df.columns
    has_whale_buy = "whale_buy_volume_30" in df.columns
    has_whale_sell = "whale_sell_volume_30" in df.columns

    # Guard: prev_day values must be non-null
    has_prev = pl.col("prev_day_val").is_not_null() & pl.col("prev_day_vah").is_not_null()

    # Bull: low probes below VAL + close recovers above VAL
    probe_below_val = has_prev & (pl.col("low") < pl.col("prev_day_val"))
    close_above_val = pl.col("close") > pl.col("prev_day_val")

    # Bear: high probes above VAH + close recovers below VAH
    probe_above_vah = has_prev & (pl.col("high") > pl.col("prev_day_vah"))
    close_below_vah = pl.col("close") < pl.col("prev_day_vah")

    # Flow confirmation: absorption OR whale activity
    bull_confirm = pl.lit(False)
    if has_absorption:
        bull_confirm = bull_confirm | (pl.col("absorption_signal") > 0)
    if has_whale_buy:
        bull_confirm = bull_confirm | (pl.col("whale_buy_volume_30") > 0)

    bear_confirm = pl.lit(False)
    if has_absorption:
        bear_confirm = bear_confirm | (pl.col("absorption_signal") > 0)
    if has_whale_sell:
        bear_confirm = bear_confirm | (pl.col("whale_sell_volume_30") > 0)

    df = df.with_columns([
        pl.when(probe_below_val & close_above_val & bull_confirm)
        .then(1.0).otherwise(0.0).alias("failed_auction_bull"),

        pl.when(probe_above_vah & close_below_vah & bear_confirm)
        .then(1.0).otherwise(0.0).alias("failed_auction_bear"),
    ])

    # Score: directional composite (+1 bull, -1 bear, 0 none)
    df = df.with_columns(
        (pl.col("failed_auction_bull") - pl.col("failed_auction_bear"))
        .alias("failed_auction_score"),
    )

    return df


def _add_squeeze_features(df: pl.DataFrame) -> pl.DataFrame:
    """Detect squeeze (trapped trader covering) patterns.

    Squeeze = recent absorption + range expansion + strong directional flow
    + CVD acceleration. Identifies trapped traders being forced out.
    """
    hlr = pl.col("high_low_range")
    range_median_6 = hlr.rolling_median(window_size=6, min_samples=1)

    # Recent absorption: current or 1-2 bars ago
    if "absorption_signal" in df.columns:
        recent_absorption = pl.max_horizontal(
            pl.col("absorption_signal"),
            pl.col("absorption_signal").shift(1).fill_null(0.0),
            pl.col("absorption_signal").shift(2).fill_null(0.0),
        ) > 0
    else:
        recent_absorption = pl.lit(False)

    # Range expansion: current range > 2× 6-bar median
    range_expanding = hlr > (2.0 * range_median_6)

    # Strong directional flow (volume_imbalance typically [-0.3, 0.3] on vol bars)
    vol_imb = pl.col("volume_imbalance")
    strong_buy_flow = vol_imb > 0.15
    strong_sell_flow = vol_imb < -0.15

    # CVD acceleration (optional)
    if "cvd_accel_3" in df.columns:
        cvd_accel_bull = pl.col("cvd_accel_3").fill_null(0.0) > 0
        cvd_accel_bear = pl.col("cvd_accel_3").fill_null(0.0) < 0
    else:
        cvd_accel_bull = pl.lit(True)
        cvd_accel_bear = pl.lit(True)

    df = df.with_columns([
        pl.when(recent_absorption & range_expanding & strong_buy_flow & cvd_accel_bull)
        .then(1.0).otherwise(0.0).alias("squeeze_bull"),

        pl.when(recent_absorption & range_expanding & strong_sell_flow & cvd_accel_bear)
        .then(1.0).otherwise(0.0).alias("squeeze_bear"),
    ])

    df = df.with_columns(
        (pl.col("squeeze_bull") - pl.col("squeeze_bear"))
        .alias("squeeze_score"),
    )

    return df


def recompute_cross_session_features(df: pl.DataFrame) -> pl.DataFrame:
    """Recompute features that need multi-day context after concatenating cached data.

    Per-file caching means prev_day_* features are always null in cache.
    Call this after pl.concat() of multiple cached days.
    Recomputes: prev_day VP, failed_auction (depends on prev_day VP).
    """
    # Drop stale prev_day and failed_auction columns so we can recompute
    drop_cols = [
        c for c in df.columns
        if c.startswith("prev_day_") or c.startswith("dist_prev_")
        or c.startswith("failed_auction")
    ]
    df = df.drop(drop_cols)

    # Recompute prev_day VP
    if all(c in df.columns for c in ["poc_price", "va_high", "va_low", "close", "range_ma5"]):
        df = _add_prev_session_vp_features(df)

    # Recompute failed auction with fresh prev_day values
    if all(c in df.columns for c in ["prev_day_vah", "prev_day_val", "high", "low", "close"]):
        df = _add_failed_auction_features(df)

    return df
