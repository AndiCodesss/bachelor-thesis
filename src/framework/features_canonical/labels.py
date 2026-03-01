"""Forward return labels for ML training."""
import polars as pl


def compute_labels(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute forward-looking return labels from pre-aggregated bar data.

    These are TARGET LABELS (y) for the ML model. They look into the future
    by definition and are only used during training, never at prediction time.

    Expects bars with columns: ts_event, close.
    """
    if len(bars) == 0:
        return pl.DataFrame(schema={
            "ts_event": pl.Datetime("ns", "UTC"),
            "close": pl.Float64,
            "fwd_return_1bar": pl.Float64,
            "fwd_return_3bar": pl.Float64,
            "fwd_return_5bar": pl.Float64,
            "fwd_return_6bar": pl.Float64,
            "fwd_return_10bar": pl.Float64,
            "fwd_return_12bar": pl.Float64,
            "label_1bar": pl.UInt8,
            "label_3bar": pl.UInt8,
            "label_5bar": pl.UInt8,
        })

    bars = bars.select(["ts_event", "close"]).sort("ts_event")

    # Compute forward returns by shifting close price backward (looking ahead)
    bars = bars.with_columns([
        ((pl.col("close").shift(-1) - pl.col("close")) / pl.col("close")).alias("fwd_return_1bar"),
        ((pl.col("close").shift(-3) - pl.col("close")) / pl.col("close")).alias("fwd_return_3bar"),
        ((pl.col("close").shift(-5) - pl.col("close")) / pl.col("close")).alias("fwd_return_5bar"),
        ((pl.col("close").shift(-6) - pl.col("close")) / pl.col("close")).alias("fwd_return_6bar"),
        ((pl.col("close").shift(-10) - pl.col("close")) / pl.col("close")).alias("fwd_return_10bar"),
        ((pl.col("close").shift(-12) - pl.col("close")) / pl.col("close")).alias("fwd_return_12bar"),
    ])

    # Null out forward returns that cross day boundaries (overnight gap poison)
    bars = bars.with_columns(
        pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.date().alias("_date"),
    )
    bars = bars.with_columns(
        pl.col("_date").count().over("_date").alias("_day_len"),
        pl.col("_date").cum_count().over("_date").alias("_day_pos"),
    )
    bars = bars.with_columns(
        (pl.col("_day_len") - pl.col("_day_pos")).alias("_bars_remaining"),
    )

    bars = bars.with_columns([
        pl.when(pl.col("_bars_remaining") >= 1).then(pl.col("fwd_return_1bar")).otherwise(None).alias("fwd_return_1bar"),
        pl.when(pl.col("_bars_remaining") >= 3).then(pl.col("fwd_return_3bar")).otherwise(None).alias("fwd_return_3bar"),
        pl.when(pl.col("_bars_remaining") >= 5).then(pl.col("fwd_return_5bar")).otherwise(None).alias("fwd_return_5bar"),
        pl.when(pl.col("_bars_remaining") >= 6).then(pl.col("fwd_return_6bar")).otherwise(None).alias("fwd_return_6bar"),
        pl.when(pl.col("_bars_remaining") >= 10).then(pl.col("fwd_return_10bar")).otherwise(None).alias("fwd_return_10bar"),
        pl.when(pl.col("_bars_remaining") >= 12).then(pl.col("fwd_return_12bar")).otherwise(None).alias("fwd_return_12bar"),
    ])

    bars = bars.drop(["_date", "_day_len", "_day_pos", "_bars_remaining"])

    # Classification labels (binary: up=1, down=0)
    bars = bars.with_columns([
        (pl.col("fwd_return_1bar") > 0).cast(pl.UInt8).alias("label_1bar"),
        (pl.col("fwd_return_3bar") > 0).cast(pl.UInt8).alias("label_3bar"),
        (pl.col("fwd_return_5bar") > 0).cast(pl.UInt8).alias("label_5bar"),
    ])

    return bars
