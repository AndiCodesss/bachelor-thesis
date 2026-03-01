"""Tests for data loading functions."""

import pytest
import polars as pl
from pathlib import Path
from src.framework.data.loader import get_parquet_files, load_split, filter_rth, validate_data
from src.framework.data.constants import SPLITS, TRAIN_FOLDERS, VALIDATE_FOLDERS, TEST_FOLDERS


def test_get_parquet_files_train():
    """Verify get_parquet_files returns non-empty list for train split."""
    files = get_parquet_files("train")
    assert len(files) > 0, "Expected non-empty list of parquet files"
    assert all(f.suffix == ".parquet" for f in files), "All files should be .parquet"
    assert all(f.exists() for f in files), "All files should exist"


def test_get_parquet_files_invalid_split():
    """Verify get_parquet_files raises on invalid split name."""
    with pytest.raises(ValueError, match="Invalid split"):
        get_parquet_files("invalid_split")


def test_load_split_lazy():
    """Load train split lazily and verify it returns LazyFrame."""
    lf = load_split("train", lazy=True)
    assert isinstance(lf, pl.LazyFrame), "Should return LazyFrame when lazy=True"


def test_load_split_schema():
    """Verify loaded data has expected columns."""
    # Use single file for faster testing
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))

    # Collect a small sample to check schema
    df = lf.head(10).collect()

    expected_columns = [
        "ts_event",
        "ts_recv",
        "price",
        "size",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
        "bid_ct_00",
        "ask_ct_00",
        "action",
        "side",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' not found in schema"


def test_filter_rth():
    """Load 1 day, filter RTH, verify all timestamps within 09:30-16:00 ET."""
    from datetime import time

    # Load just one file for faster testing
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))

    # Apply RTH filter
    lf_rth = filter_rth(lf)

    # Collect and verify
    df = lf_rth.with_columns([
        pl.col("ts_event")
        .dt.convert_time_zone("US/Eastern")
        .alias("ts_eastern")
    ]).collect()

    assert len(df) > 0, "RTH filtered data should not be empty"

    # Check all times are within RTH
    times = df["ts_eastern"].dt.time()
    min_time = times.min()
    max_time = times.max()

    assert min_time >= time(9, 30, 0), f"Min time {min_time} before RTH start"
    assert max_time < time(16, 0, 0), f"Max time {max_time} after RTH end"


def test_validate_data():
    """Run validation and verify stats dict has expected keys."""
    # Use single file for faster testing
    files = get_parquet_files("train")
    lf = pl.scan_parquet(str(files[0]))

    # Take a small subset for fast testing
    lf_sample = lf.head(10000)

    stats = validate_data(lf_sample)

    # Verify expected keys
    assert "row_count" in stats
    assert "date_range" in stats
    assert "null_counts" in stats

    # Verify types and values
    assert stats["row_count"] > 0
    assert "start" in stats["date_range"]
    assert "end" in stats["date_range"]
    assert stats["date_range"]["start"] is not None
    assert stats["date_range"]["end"] is not None


def test_no_date_overlap():
    """Verify TRAIN/VALIDATE/TEST folder lists have zero overlap."""
    train_set = set(TRAIN_FOLDERS)
    validate_set = set(VALIDATE_FOLDERS)
    test_set = set(TEST_FOLDERS)

    # Check no overlap between any pair of splits
    assert len(train_set & validate_set) == 0, "TRAIN and VALIDATE folders overlap"
    assert len(train_set & test_set) == 0, "TRAIN and TEST folders overlap"
    assert len(validate_set & test_set) == 0, "VALIDATE and TEST folders overlap"

    # Verify all splits are non-empty
    assert len(train_set) > 0, "TRAIN folders empty"
    assert len(validate_set) > 0, "VALIDATE folders empty"
    assert len(test_set) > 0, "TEST folders empty"


def test_filter_rth_synthetic():
    """Test RTH filter with synthetic known-answer data."""
    from datetime import datetime

    # Create synthetic data with known UTC timestamps
    # Use summer date (EDT, UTC-4) for consistent timezone behavior
    # RTH is 09:30-16:00 ET
    # In EDT (summer), ET = UTC - 4 hours
    # So RTH in UTC is 13:30-20:00

    synthetic_data = pl.DataFrame({
        "ts_event": [
            # 13:29 UTC = 09:29 EDT -> EXCLUDED (before RTH)
            datetime(2024, 7, 15, 13, 29, 0),
            # 13:30 UTC = 09:30 EDT -> INCLUDED (RTH start)
            datetime(2024, 7, 15, 13, 30, 0),
            # 19:59 UTC = 15:59 EDT -> INCLUDED (before RTH end)
            datetime(2024, 7, 15, 19, 59, 0),
            # 20:00 UTC = 16:00 EDT -> EXCLUDED (at RTH end)
            datetime(2024, 7, 15, 20, 0, 0),
            # 20:01 UTC = 16:01 EDT -> EXCLUDED (after RTH)
            datetime(2024, 7, 15, 20, 1, 0),
        ],
        "price": [100.0, 100.25, 100.50, 100.75, 101.0],
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC")
    )

    # Convert to LazyFrame and apply filter
    lf = synthetic_data.lazy()
    lf_filtered = filter_rth(lf)
    df_result = lf_filtered.collect()

    # Should have exactly 2 rows (indices 1 and 2)
    assert len(df_result) == 2, f"Expected 2 rows, got {len(df_result)}"

    # Verify the correct rows survived
    expected_prices = [100.25, 100.50]
    actual_prices = df_result["price"].to_list()
    assert actual_prices == expected_prices, f"Expected prices {expected_prices}, got {actual_prices}"


def test_validate_data_synthetic():
    """Test validate_data with synthetic known-answer data."""

    # Create synthetic data with known null patterns
    synthetic_data = pl.DataFrame({
        "ts_event": [i for i in range(10)],  # 10 rows, no nulls
        "price": [100.0, None, 100.5, None, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5],  # 2 nulls
        "bid_px_00": [99.0, 99.25, None, 99.75, 100.0, 100.25, 100.5, 100.75, 101.0, 101.25],  # 1 null
        "ask_px_00": [100.0, 100.25, 100.5, 100.75, 101.0, 101.25, 101.5, 101.75, 102.0, 102.25],  # 0 nulls
    })

    # Convert to LazyFrame
    lf = synthetic_data.lazy()

    # Call validate_data
    stats = validate_data(lf)

    # Assert exact known values
    assert stats["row_count"] == 10, f"Expected row_count=10, got {stats['row_count']}"
    assert stats["null_counts"]["ts_event"] == 0, f"Expected 0 ts_event nulls, got {stats['null_counts']['ts_event']}"
    assert stats["null_counts"]["price"] == 2, f"Expected 2 price nulls, got {stats['null_counts']['price']}"
    assert stats["null_counts"]["bid_px_00"] == 1, f"Expected 1 bid nulls, got {stats['null_counts']['bid_px_00']}"
    assert stats["null_counts"]["ask_px_00"] == 0, f"Expected 0 ask nulls, got {stats['null_counts']['ask_px_00']}"

    # Verify date range is populated
    assert stats["date_range"]["start"] is not None
    assert stats["date_range"]["end"] is not None
