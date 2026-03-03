"""Opening Range features: OR width, position, breakout flags."""
import polars as pl

from src.framework.data.constants import OR_PERIOD_MINUTES


def compute_opening_range_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute Opening Range context features per bar.

    The Opening Range (OR) is the high/low of the first OR_PERIOD_MINUTES of each
    RTH session. Bars during the OR formation period get null (no lookahead).
    Bars after the OR period get the completed OR levels.

    Args:
        bars: DataFrame with ts_event (UTC), open, high, low, close columns.
    """
    if len(bars) == 0:
        return _empty_result()

    df = bars.select(["ts_event", "open", "high", "low", "close"]).sort("ts_event")

    # Convert to Eastern for session date + time-of-day
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").alias("_ts_et"),
    )

    df = df.with_columns([
        pl.col("_ts_et").dt.date().alias("_date"),
        (pl.col("_ts_et").dt.hour().cast(pl.Int32) * 60
         + pl.col("_ts_et").dt.minute().cast(pl.Int32))
        .alias("_min_of_day"),
    ])

    # Minutes since RTH open (09:30 = minute 570)
    rth_start_min = 9 * 60 + 30
    df = df.with_columns(
        (pl.col("_min_of_day") - rth_start_min).alias("_min_since_open"),
    )

    # Mark bars in the OR period: [0, OR_PERIOD_MINUTES)
    df = df.with_columns(
        ((pl.col("_min_since_open") >= 0)
         & (pl.col("_min_since_open") < OR_PERIOD_MINUTES))
        .alias("_in_or"),
    )

    # Compute OR high and low per session from OR-period bars only
    or_stats = (
        df.filter(pl.col("_in_or"))
        .group_by("_date")
        .agg([
            pl.col("high").max().alias("_or_high"),
            pl.col("low").min().alias("_or_low"),
        ])
    )

    df = df.join(or_stats, on="_date", how="left")

    # Null out OR values until OR is complete.
    # This includes pre-open ETH bars (minutes_since_open < 0) to avoid
    # leaking 09:30-10:00 OR stats into earlier bars.
    or_not_ready = pl.col("_min_since_open") < OR_PERIOD_MINUTES
    df = df.with_columns([
        pl.when(or_not_ready)
        .then(None)
        .otherwise(pl.col("_or_high"))
        .alias("_or_high"),

        pl.when(or_not_ready)
        .then(None)
        .otherwise(pl.col("_or_low"))
        .alias("_or_low"),
    ])

    # Derived features (null during OR period, populated after)
    df = df.with_columns([
        # "Narrow OR = directional day; wide OR = rotational day"
        ((pl.col("_or_high") - pl.col("_or_low"))
         / (pl.col("close") + 1e-9))
        .alias("or_width"),

        # "0=at OR low, 1=at OR high, >1=broken up, <0=broken down"
        ((pl.col("close") - pl.col("_or_low"))
         / (pl.col("_or_high") - pl.col("_or_low") + 1e-9))
        .alias("position_in_or"),

        # "Breakout above OR high = trend day candidate"
        pl.when(or_not_ready)
        .then(None)
        .when(pl.col("close") > pl.col("_or_high"))
        .then(1.0).otherwise(0.0)
        .alias("or_broken_up"),

        # "Breakout below OR low = trend day candidate"
        pl.when(or_not_ready)
        .then(None)
        .when(pl.col("close") < pl.col("_or_low"))
        .then(1.0).otherwise(0.0)
        .alias("or_broken_down"),
    ])

    output_cols = [
        "ts_event",
        "or_width",
        "position_in_or",
        "or_broken_up",
        "or_broken_down",
    ]

    return df.select(output_cols)


def _empty_result() -> pl.DataFrame:
    schema = {
        "ts_event": pl.Datetime("ns", "UTC"),
        "or_width": pl.Float64,
        "position_in_or": pl.Float64,
        "or_broken_up": pl.Float64,
        "or_broken_down": pl.Float64,
    }
    return pl.DataFrame(schema=schema)
