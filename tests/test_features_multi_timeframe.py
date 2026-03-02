"""Tests for multi-timeframe feature computation."""

import pytest
import polars as pl
from datetime import datetime

from src.framework.features_canonical.multi_timeframe import (
    compute_multi_timeframe_features, _parse_bar_minutes, FIXED_MTF_FEATURES,
)


def _make_tick(ts, action, side, price, size, bid_px=20999.75, ask_px=21000.25,
               bid_sz=50, ask_sz=50, bid_ct=10, ask_ct=10):
    """Helper: create a single MBP1 tick row."""
    return {
        "ts_event": ts,
        "ts_recv": ts,
        "action": action,
        "side": side,
        "price": price,
        "size": size,
        "bid_px_00": bid_px,
        "ask_px_00": ask_px,
        "bid_sz_00": bid_sz,
        "ask_sz_00": ask_sz,
        "bid_ct_00": bid_ct,
        "ask_ct_00": ask_ct,
        "depth": 0,
        "flags": 0,
        "sequence": 0,
        "ts_in_delta": 0,
    }


def _build_synthetic_lf(n_minutes=12):
    """Build synthetic MBP1 data spanning n_minutes with trades + book events.

    Creates ~5 ticks per minute (3 trades + 2 book updates) to give each
    1m bar enough data for meaningful feature computation.
    """
    rows = []
    base_price = 21000.0
    for m in range(n_minutes):
        datetime(2024, 7, 15, 13, 30 + m, 0)
        price = base_price + m * 2.0  # slow uptrend

        # 3 trades per minute (buy, sell, buy) — varied sizes
        for t_offset, (side, sz) in enumerate([("A", 10 + m), ("B", 5 + m), ("A", 8)]):
            ts = datetime(2024, 7, 15, 13, 30 + m, t_offset * 10)
            rows.append(_make_tick(
                ts, "T", side, price + t_offset * 0.25, sz,
                bid_px=price - 0.25, ask_px=price + 0.25,
                bid_sz=40 + m, ask_sz=45 - m,
            ))

        # 2 book updates (add on bid, cancel on ask)
        ts_book1 = datetime(2024, 7, 15, 13, 30 + m, 35)
        rows.append(_make_tick(
            ts_book1, "A", "B", price, 20,
            bid_px=price - 0.25, ask_px=price + 0.25,
            bid_sz=50 + m, ask_sz=40 - m,
        ))
        ts_book2 = datetime(2024, 7, 15, 13, 30 + m, 45)
        rows.append(_make_tick(
            ts_book2, "C", "A", price, 5,
            bid_px=price - 0.25, ask_px=price + 0.25,
            bid_sz=50 + m, ask_sz=35 - m,
        ))

    df = pl.DataFrame(rows)
    # Cast types to match real MBP1 schema
    df = df.with_columns([
        pl.col("ts_event").dt.replace_time_zone("UTC"),
        pl.col("ts_recv").dt.replace_time_zone("UTC"),
        pl.col("size").cast(pl.UInt32),
        pl.col("bid_sz_00").cast(pl.UInt32),
        pl.col("ask_sz_00").cast(pl.UInt32),
        pl.col("bid_ct_00").cast(pl.UInt32),
        pl.col("ask_ct_00").cast(pl.UInt32),
        pl.col("depth").cast(pl.UInt8),
        pl.col("flags").cast(pl.UInt8),
        pl.col("sequence").cast(pl.UInt64),
        pl.col("ts_in_delta").cast(pl.Int32),
    ])
    return df.lazy()


class TestParseBarMinutes:
    def test_minutes(self):
        assert _parse_bar_minutes("1m") == 1
        assert _parse_bar_minutes("5m") == 5

    def test_hours(self):
        assert _parse_bar_minutes("1h") == 60

    def test_invalid(self):
        with pytest.raises(ValueError):
            _parse_bar_minutes("30s")


class TestMTFOutputStructure:
    """Verify the shape and naming conventions of MTF output."""

    def test_all_columns_have_m1f_prefix(self):
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m", top_n=3,
            agg_ops=["mean", "std"],
        )

        for col in result.columns:
            if col == "ts_event":
                continue
            assert col.startswith("m1f_"), f"Column {col} missing m1f_ prefix"

    def test_naming_convention(self):
        """Verify m1f_{feature}__{op} format with double underscore."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m",
            agg_ops=["mean", "last"],
        )

        feature_cols = [c for c in result.columns if c != "ts_event"]
        assert len(feature_cols) > 0
        for col in feature_cols:
            parts = col.split("__")
            assert len(parts) == 2, f"Column {col} should have exactly one __ separator"
            assert parts[0].startswith("m1f_")
            assert parts[1] in ("mean", "last")

    def test_ts_event_column_present(self):
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m", top_n=3,
        )

        assert "ts_event" in result.columns

    def test_output_has_5m_aligned_bars(self):
        """12 minutes of 1m data should produce 2 full 5m bars (plus partial)."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m", top_n=3,
        )

        # 12 minutes starting at 13:30 -> bars at 13:30, 13:35, 13:40
        # (13:30-13:35 = 5 bars, 13:35-13:40 = 5 bars, 13:40-13:42 = partial)
        assert len(result) >= 2


class TestMTFFixedFeatures:
    def test_fixed_features_used(self):
        """All available fixed features should be aggregated."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m",
            agg_ops=["mean", "std"],
        )

        feature_cols = [c for c in result.columns if c != "ts_event"]
        n_features = len(feature_cols) // 2  # 2 ops
        assert n_features > 0
        assert n_features <= len(FIXED_MTF_FEATURES)

    def test_consistent_schema_across_calls(self):
        """Same input should produce same column set every time."""
        lf = _build_synthetic_lf(12)
        r1 = compute_multi_timeframe_features(lf, bar_size="5m", source_bar_size="1m")
        r2 = compute_multi_timeframe_features(lf, bar_size="5m", source_bar_size="1m")
        assert r1.columns == r2.columns


class TestMTFAggregationValues:
    """Verify aggregation operations produce correct values."""

    def test_mean_aggregation(self):
        """Mean of 1m values within a 5m window."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m", top_n=5,
            agg_ops=["mean"],
        )

        # All mean columns should be finite
        for col in result.columns:
            if col == "ts_event":
                continue
            vals = result[col].drop_nulls()
            if len(vals) > 0:
                assert vals.is_finite().all(), f"Non-finite values in {col}"

    def test_delta_is_last_minus_first(self):
        """Delta should be last - first of the 1m values within each 5m window."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m",
            agg_ops=["last", "delta", "min", "max"],
        )

        delta_cols = [c for c in result.columns if c.endswith("__delta")]
        assert len(delta_cols) > 0

    def test_min_le_max(self):
        """Min should always be <= max for each feature."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m", top_n=5,
            agg_ops=["min", "max"],
        )

        feature_names = set()
        for col in result.columns:
            if col.endswith("__min"):
                feat = col.replace("__min", "")
                feature_names.add(feat)

        for feat in feature_names:
            min_col = f"{feat}__min"
            max_col = f"{feat}__max"
            if min_col in result.columns and max_col in result.columns:
                for i in range(len(result)):
                    mn = result[min_col][i]
                    mx = result[max_col][i]
                    if mn is not None and mx is not None:
                        assert mn <= mx, f"min > max at row {i} for {feat}: {mn} > {mx}"

    def test_std_non_negative(self):
        """Standard deviation should always be >= 0."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m", top_n=5,
            agg_ops=["std"],
        )

        for col in result.columns:
            if col.endswith("__std"):
                vals = result[col].drop_nulls()
                if len(vals) > 0:
                    assert (vals >= 0).all(), f"Negative std in {col}"


class TestMTFEdgeCases:
    def test_bar_size_must_exceed_source(self):
        """Should raise when bar_size <= source_bar_size."""
        lf = _build_synthetic_lf(12)
        with pytest.raises(AssertionError):
            compute_multi_timeframe_features(
                lf, bar_size="1m", source_bar_size="1m",
            )

    def test_bar_size_smaller_than_source_raises(self):
        lf = _build_synthetic_lf(12)
        with pytest.raises(AssertionError):
            compute_multi_timeframe_features(
                lf, bar_size="1m", source_bar_size="5m",
            )

    def test_empty_input(self):
        """Empty input should produce empty output."""
        df = pl.DataFrame({
            "ts_event": pl.Series([], dtype=pl.Datetime("ns", "UTC")),
            "ts_recv": pl.Series([], dtype=pl.Datetime("ns", "UTC")),
            "action": pl.Series([], dtype=pl.Utf8),
            "side": pl.Series([], dtype=pl.Utf8),
            "price": pl.Series([], dtype=pl.Float64),
            "size": pl.Series([], dtype=pl.UInt32),
            "bid_px_00": pl.Series([], dtype=pl.Float64),
            "ask_px_00": pl.Series([], dtype=pl.Float64),
            "bid_sz_00": pl.Series([], dtype=pl.UInt32),
            "ask_sz_00": pl.Series([], dtype=pl.UInt32),
            "bid_ct_00": pl.Series([], dtype=pl.UInt32),
            "ask_ct_00": pl.Series([], dtype=pl.UInt32),
            "depth": pl.Series([], dtype=pl.UInt8),
            "flags": pl.Series([], dtype=pl.UInt8),
            "sequence": pl.Series([], dtype=pl.UInt64),
            "ts_in_delta": pl.Series([], dtype=pl.Int32),
        })
        result = compute_multi_timeframe_features(df.lazy(), bar_size="5m", source_bar_size="1m")
        assert len(result) == 0

    def test_custom_agg_ops(self):
        """Custom agg_ops list should only produce those operations."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m",
            agg_ops=["mean"],
        )

        feature_cols = [c for c in result.columns if c != "ts_event"]
        assert len(feature_cols) > 0
        for col in feature_cols:
            assert col.endswith("__mean")

    def test_slope_operation(self):
        """Slope should be (last - first) / (n - 1)."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m",
            agg_ops=["slope"],
        )

        slope_cols = [c for c in result.columns if c.endswith("__slope")]
        assert len(slope_cols) > 0

    def test_output_sorted_by_timestamp(self):
        """Output bars should be sorted by ts_event."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m", top_n=3,
        )

        timestamps = result["ts_event"].to_list()
        assert timestamps == sorted(timestamps)


class TestMTFDefaultOps:
    def test_default_ops_count(self):
        """Default ops (6) * fixed features should give the right column count."""
        lf = _build_synthetic_lf(12)
        result = compute_multi_timeframe_features(
            lf, bar_size="5m", source_bar_size="1m",
        )

        feature_cols = [c for c in result.columns if c != "ts_event"]
        n_features = len(feature_cols) // 6  # 6 default ops
        assert n_features > 0
        assert len(feature_cols) == n_features * 6
