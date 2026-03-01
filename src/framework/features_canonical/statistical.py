"""Statistical features: fractional differentiation, Yang-Zhang volatility, VWAP deviation."""
import polars as pl
import numpy as np

# Fractional differentiation params
FRACDIFF_D = 0.4
FRACDIFF_WINDOW = 50

# Yang-Zhang volatility rolling window
YZ_WINDOW = 20

# Z-score lookbacks
VOL_ZSCORE_WINDOW = 50
VWAP_ZSCORE_WINDOW = 24


def _fracdiff_weights(d: float, window: int) -> np.ndarray:
    """Compute fractional differentiation weights using Hosking (1981) method.

    w[0] = 1, w[k] = w[k-1] * (d - k + 1) / k
    """
    w = np.zeros(window)
    w[0] = 1.0
    for k in range(1, window):
        w[k] = w[k - 1] * (d - k + 1) / k
    return w


def _apply_fracdiff(prices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply fractional differentiation as a convolution.

    fracdiff[t] = sum(w[k] * price[t-k]) for k=0..window-1
    First (window-1) values are NaN (warmup).
    """
    n = len(prices)
    window = len(weights)
    result = np.full(n, np.nan)
    for t in range(window - 1, n):
        result[t] = np.dot(weights, prices[t - window + 1:t + 1][::-1])
    return result


def compute_statistical_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Statistical features from pre-aggregated bar data.

    Expects bars with columns: ts_event, open, high, low, close, volume, vwap.
    """
    if len(bars) == 0:
        return pl.DataFrame(schema={
            "ts_event": pl.Datetime("ns", "UTC"),
            "log_return": pl.Float64,
            "fracdiff_close": pl.Float64,
            "yz_volatility": pl.Float64,
            "vol_zscore": pl.Float64,
            "vwap_deviation": pl.Float64,
            "vwap_dev_zscore": pl.Float64,
        })

    bars = bars.select(["ts_event", "open", "high", "low", "close", "volume", "vwap"]).sort("ts_event")

    # Use vwap from bars directly
    bars = bars.rename({"vwap": "_bar_vwap"})

    # --- Log return ---
    bars = bars.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return"),
    )

    # --- Yang-Zhang volatility components ---
    bars = bars.with_columns(
        (pl.col("open").log() - pl.col("close").shift(1).log()).alias("_o_ret"),
    )
    bars = bars.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("_cc_ret"),
    )

    bars = bars.with_columns([
        (pl.col("high").log() - pl.col("open").log()).alias("_h"),
        (pl.col("low").log() - pl.col("open").log()).alias("_l"),
        (pl.col("close").log() - pl.col("open").log()).alias("_c"),
    ])

    bars = bars.with_columns(
        (pl.col("_h") * (pl.col("_h") - pl.col("_c")) +
         pl.col("_l") * (pl.col("_l") - pl.col("_c"))).alias("_rs_bar"),
    )

    n = YZ_WINDOW
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    bars = bars.with_columns([
        pl.col("_o_ret").rolling_var(window_size=n, min_samples=n).alias("_var_overnight"),
        pl.col("_cc_ret").rolling_var(window_size=n, min_samples=n).alias("_var_cc"),
        pl.col("_rs_bar").rolling_mean(window_size=n, min_samples=n).alias("_mean_rs"),
    ])

    bars = bars.with_columns(
        (pl.col("_var_overnight") + k * pl.col("_var_cc") + (1 - k) * pl.col("_mean_rs"))
        .alias("_yz_var"),
    )

    bars = bars.with_columns(
        pl.when(pl.col("_yz_var").is_not_null() & (pl.col("_yz_var") > 0))
        .then(pl.col("_yz_var").sqrt())
        .otherwise(pl.lit(0.0))
        .alias("yz_volatility"),
    )

    # --- Vol z-score ---
    bars = bars.with_columns([
        pl.col("yz_volatility").rolling_mean(window_size=VOL_ZSCORE_WINDOW, min_samples=1).alias("_vol_mean"),
        pl.col("yz_volatility").rolling_std(window_size=VOL_ZSCORE_WINDOW, min_samples=2).alias("_vol_std"),
    ])

    bars = bars.with_columns(
        pl.when(pl.col("_vol_std") > 1e-12)
        .then((pl.col("yz_volatility") - pl.col("_vol_mean")) / pl.col("_vol_std"))
        .otherwise(pl.lit(0.0))
        .alias("vol_zscore"),
    )

    # --- Session VWAP deviation ---
    bars = bars.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_date"),
    )

    bars = bars.with_columns([
        (pl.col("_bar_vwap") * pl.col("volume").cast(pl.Float64))
        .cum_sum()
        .over("_date")
        .alias("_cum_pv"),
        pl.col("volume").cast(pl.Float64)
        .cum_sum()
        .over("_date")
        .alias("_cum_vol"),
    ])

    bars = bars.with_columns(
        (pl.col("_cum_pv") / pl.col("_cum_vol")).alias("_session_vwap"),
    )

    bars = bars.with_columns(
        (pl.col("close") - pl.col("_session_vwap")).alias("vwap_deviation"),
    )

    # vwap_dev_zscore
    bars = bars.with_columns([
        pl.col("vwap_deviation").rolling_mean(window_size=VWAP_ZSCORE_WINDOW, min_samples=1).alias("_vdev_mean"),
        pl.col("vwap_deviation").rolling_std(window_size=VWAP_ZSCORE_WINDOW, min_samples=2).alias("_vdev_std"),
    ])

    bars = bars.with_columns(
        pl.when(pl.col("_vdev_std") > 1e-12)
        .then((pl.col("vwap_deviation") - pl.col("_vdev_mean")) / pl.col("_vdev_std"))
        .otherwise(pl.lit(0.0))
        .alias("vwap_dev_zscore"),
    )

    # --- Fracdiff (must collect for numpy convolution) ---
    weights = _fracdiff_weights(FRACDIFF_D, FRACDIFF_WINDOW)
    close_arr = bars["close"].to_numpy().astype(np.float64)
    fracdiff_arr = _apply_fracdiff(close_arr, weights)

    bars = bars.with_columns(
        pl.Series("fracdiff_close", fracdiff_arr, dtype=pl.Float64),
    )

    # Drop all temp columns, keep only ts_event + feature columns
    temp_cols = [c for c in bars.columns if c.startswith("_")]
    extra_cols = ["open", "high", "low", "close", "volume"]
    drop_cols = temp_cols + [c for c in extra_cols if c in bars.columns]
    bars = bars.drop(drop_cols)

    return bars
