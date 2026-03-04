"""OHLCV technical indicators from pre-aggregated bar data.

Standard industry indicators computed from open/high/low/close/volume.
All outputs are stationary (ratios, bounded oscillators, or normalized values)
to be valid ML features.

Indicators (20 features):
    SMA ratios (4):  sma_ratio_8, sma_ratio_21, sma_ratio_50, sma_ratio_200
    EMA ratios (3):  ema_ratio_8, ema_ratio_21, ema_ratio_50
    RSI (1):         rsi_14
    ATR (1):         atr_norm_14
    Bollinger (2):   bb_bandwidth_20, bb_pctb_20
    MACD (3):        macd_norm, macd_signal_norm, macd_hist_norm
    Stochastic (2):  stoch_k_14, stoch_d_14
    ADX (3):         adx_14, plus_di_14, minus_di_14
    OBV (1):         obv_slope_14
"""

import polars as pl

# All output feature column names (excluding ts_event)
OHLCV_INDICATOR_COLUMNS = [
    "sma_ratio_8", "sma_ratio_21", "sma_ratio_50", "sma_ratio_200",
    "ema_ratio_8", "ema_ratio_21", "ema_ratio_50",
    "rsi_14",
    "atr_norm_14",
    "bb_bandwidth_20", "bb_pctb_20",
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "stoch_k_14", "stoch_d_14",
    "adx_14", "plus_di_14", "minus_di_14",
    "obv_slope_14",
]

# SMA lookback periods
SMA_PERIODS = (8, 21, 50, 200)

# EMA lookback periods
EMA_PERIODS = (8, 21, 50)

# RSI / ATR / ADX Wilder smoothing period
WILDER_PERIOD = 14

# Bollinger Bands parameters
BB_PERIOD = 20
BB_STD_MULT = 2

# MACD parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Stochastic parameters
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

# OBV slope lookback
OBV_SLOPE_PERIOD = 14

# Epsilon to avoid division by zero
_EPS = 1e-10


def compute_ohlcv_indicators(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute standard technical indicators from OHLCV bar data.

    Expects bars with columns: ts_event, open, high, low, close, volume.
    Returns DataFrame with ts_event + 20 indicator columns.
    """
    if len(bars) == 0:
        return pl.DataFrame(schema=_empty_schema())

    bars = bars.select(["ts_event", "open", "high", "low", "close", "volume"]).sort("ts_event")

    # --- SMA ratios: close / SMA(N) - 1 ---
    bars = bars.with_columns([
        (pl.col("close") / pl.col("close").rolling_mean(window_size=n, min_samples=n) - 1)
        .alias(f"sma_ratio_{n}")
        for n in SMA_PERIODS
    ])

    # --- EMA ratios: close / EMA(N) - 1 ---
    bars = bars.with_columns([
        (pl.col("close") / pl.col("close").ewm_mean(span=n, adjust=False, min_samples=n) - 1)
        .alias(f"ema_ratio_{n}")
        for n in EMA_PERIODS
    ])

    # --- RSI (Wilder's smoothing, period=14) ---
    bars = bars.with_columns(
        (pl.col("close") - pl.col("close").shift(1)).alias("_change")
    )
    bars = bars.with_columns([
        pl.col("_change").clip(lower_bound=0).alias("_gain"),
        (-pl.col("_change")).clip(lower_bound=0).alias("_loss"),
    ])
    bars = bars.with_columns([
        pl.col("_gain").ewm_mean(
            alpha=1.0 / WILDER_PERIOD, adjust=False, min_samples=WILDER_PERIOD,
        ).alias("_avg_gain"),
        pl.col("_loss").ewm_mean(
            alpha=1.0 / WILDER_PERIOD, adjust=False, min_samples=WILDER_PERIOD,
        ).alias("_avg_loss"),
    ])
    bars = bars.with_columns(
        (100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / (pl.col("_avg_loss") + _EPS)))
        .alias("rsi_14")
    )

    # --- True Range (shared by ATR and ADX) ---
    bars = bars.with_columns(
        pl.col("close").shift(1).alias("_prev_close")
    )
    bars = bars.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("_prev_close")).abs(),
            (pl.col("low") - pl.col("_prev_close")).abs(),
        ).alias("_tr")
    )

    # --- ATR (normalized by close, Wilder's smoothing) ---
    bars = bars.with_columns(
        (pl.col("_tr").ewm_mean(
            alpha=1.0 / WILDER_PERIOD, adjust=False, min_samples=WILDER_PERIOD,
        ) / (pl.col("close") + _EPS))
        .alias("atr_norm_14")
    )

    # --- Bollinger Bands (20, 2std) ---
    bars = bars.with_columns([
        pl.col("close").rolling_mean(window_size=BB_PERIOD, min_samples=BB_PERIOD).alias("_bb_sma"),
        pl.col("close").rolling_std(window_size=BB_PERIOD, min_samples=BB_PERIOD).alias("_bb_std"),
    ])
    bars = bars.with_columns([
        (pl.col("_bb_sma") + BB_STD_MULT * pl.col("_bb_std")).alias("_bb_upper"),
        (pl.col("_bb_sma") - BB_STD_MULT * pl.col("_bb_std")).alias("_bb_lower"),
    ])
    bars = bars.with_columns([
        (BB_STD_MULT * 2 * pl.col("_bb_std") / (pl.col("_bb_sma") + _EPS))
        .alias("bb_bandwidth_20"),
        ((pl.col("close") - pl.col("_bb_lower")) / (pl.col("_bb_upper") - pl.col("_bb_lower") + _EPS))
        .alias("bb_pctb_20"),
    ])

    # --- MACD (12/26/9, normalized by close) ---
    bars = bars.with_columns(
        (pl.col("close").ewm_mean(span=MACD_FAST, adjust=False, min_samples=MACD_FAST)
         - pl.col("close").ewm_mean(span=MACD_SLOW, adjust=False, min_samples=MACD_SLOW))
        .alias("_macd_line")
    )
    bars = bars.with_columns(
        pl.col("_macd_line").ewm_mean(span=MACD_SIGNAL, adjust=False, min_samples=MACD_SIGNAL)
        .alias("_macd_signal")
    )
    bars = bars.with_columns([
        (pl.col("_macd_line") / (pl.col("close") + _EPS)).alias("macd_norm"),
        (pl.col("_macd_signal") / (pl.col("close") + _EPS)).alias("macd_signal_norm"),
        ((pl.col("_macd_line") - pl.col("_macd_signal")) / (pl.col("close") + _EPS))
        .alias("macd_hist_norm"),
    ])

    # --- Stochastic Oscillator (14/3) ---
    bars = bars.with_columns([
        pl.col("low").rolling_min(window_size=STOCH_K_PERIOD, min_samples=STOCH_K_PERIOD)
        .alias("_lowest_low"),
        pl.col("high").rolling_max(window_size=STOCH_K_PERIOD, min_samples=STOCH_K_PERIOD)
        .alias("_highest_high"),
    ])
    bars = bars.with_columns([
        (pl.col("_highest_high") - pl.col("_lowest_low")).alias("_stoch_range"),
    ])
    bars = bars.with_columns(
        pl.when(pl.col("_stoch_range").abs() <= _EPS)
        .then(50.0)
        .otherwise(
            ((pl.col("close") - pl.col("_lowest_low"))
             / (pl.col("_stoch_range") + _EPS) * 100)
            .clip(lower_bound=0.0, upper_bound=100.0)
        )
        .alias("stoch_k_14")
    )
    bars = bars.with_columns(
        pl.col("stoch_k_14").rolling_mean(window_size=STOCH_D_PERIOD, min_samples=STOCH_D_PERIOD)
        .alias("stoch_d_14")
    )

    # --- ADX (Average Directional Index, period=14) ---
    bars = bars.with_columns([
        (pl.col("high") - pl.col("high").shift(1)).alias("_up_move"),
        (pl.col("low").shift(1) - pl.col("low")).alias("_down_move"),
    ])
    bars = bars.with_columns([
        pl.when((pl.col("_up_move") > pl.col("_down_move")) & (pl.col("_up_move") > 0))
        .then(pl.col("_up_move"))
        .otherwise(0.0)
        .alias("_plus_dm"),
        pl.when((pl.col("_down_move") > pl.col("_up_move")) & (pl.col("_down_move") > 0))
        .then(pl.col("_down_move"))
        .otherwise(0.0)
        .alias("_minus_dm"),
    ])
    bars = bars.with_columns([
        pl.col("_plus_dm").ewm_mean(
            alpha=1.0 / WILDER_PERIOD, adjust=False, min_samples=WILDER_PERIOD,
        ).alias("_smooth_plus_dm"),
        pl.col("_minus_dm").ewm_mean(
            alpha=1.0 / WILDER_PERIOD, adjust=False, min_samples=WILDER_PERIOD,
        ).alias("_smooth_minus_dm"),
        pl.col("_tr").ewm_mean(
            alpha=1.0 / WILDER_PERIOD, adjust=False, min_samples=WILDER_PERIOD,
        ).alias("_smooth_tr"),
    ])
    bars = bars.with_columns([
        (pl.col("_smooth_plus_dm") / (pl.col("_smooth_tr") + _EPS) * 100).alias("plus_di_14"),
        (pl.col("_smooth_minus_dm") / (pl.col("_smooth_tr") + _EPS) * 100).alias("minus_di_14"),
    ])
    bars = bars.with_columns(
        ((pl.col("plus_di_14") - pl.col("minus_di_14")).abs()
         / (pl.col("plus_di_14") + pl.col("minus_di_14") + _EPS) * 100)
        .alias("_dx")
    )
    bars = bars.with_columns(
        pl.col("_dx").ewm_mean(
            alpha=1.0 / WILDER_PERIOD, adjust=False, min_samples=WILDER_PERIOD,
        ).alias("adx_14")
    )

    # --- OBV slope (rate of change over 14 bars) ---
    bars = bars.with_columns(
        pl.when(pl.col("close") > pl.col("close").shift(1))
        .then(pl.col("volume").cast(pl.Float64))
        .when(pl.col("close") < pl.col("close").shift(1))
        .then(-pl.col("volume").cast(pl.Float64))
        .otherwise(0.0)
        .alias("_signed_vol")
    )
    bars = bars.with_columns(
        pl.col("_signed_vol").cum_sum().alias("_obv")
    )
    bars = bars.with_columns(
        ((pl.col("_obv") - pl.col("_obv").shift(OBV_SLOPE_PERIOD))
         / (pl.col("_obv").abs() + pl.col("_obv").shift(OBV_SLOPE_PERIOD).abs() + _EPS))
        .alias("obv_slope_14")
    )

    # --- Select output columns ---
    return bars.select(["ts_event"] + OHLCV_INDICATOR_COLUMNS)


def _empty_schema() -> dict:
    """Schema for empty DataFrame output."""
    schema = {"ts_event": pl.Datetime("ns", "UTC")}
    for col in OHLCV_INDICATOR_COLUMNS:
        schema[col] = pl.Float64
    return schema
