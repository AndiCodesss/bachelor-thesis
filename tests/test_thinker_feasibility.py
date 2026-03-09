from __future__ import annotations

import polars as pl
import pytest

from research.lib.thinker_feasibility import (
    ThinkerFeasibilityError,
    assess_entry_condition_feasibility,
    format_feasibility_error,
    normalize_entry_conditions,
)


def test_normalize_entry_conditions_requires_numeric_param_keys():
    with pytest.raises(ValueError, match="numeric params_template"):
        normalize_entry_conditions(
            [{"feature": "volume_ratio", "op": ">", "param_key": "threshold", "role": "primary"}],
            params_template={"threshold": "x"},
        )


def test_assess_entry_condition_feasibility_detects_dead_primary_feature():
    report = assess_entry_condition_feasibility(
        entry_conditions=[
            {"feature": "prev_day_va_position", "op": ">", "param_key": "va_pos_min", "role": "primary"}
        ],
        params_template={"va_pos_min": 1.0},
        selected_bar_configs=["tick_610"],
        validation_sample_cache={
            "tick_610": [
                (
                    "sample_day",
                    pl.DataFrame({"prev_day_va_position": [None, None, None]}),
                )
            ]
        },
    )
    assert report["bar_results"][0]["status"] == "dead_feature_primary"


def test_assess_entry_condition_feasibility_detects_zero_signal_combination():
    report = assess_entry_condition_feasibility(
        entry_conditions=[
            {"feature": "volume_ratio", "op": ">", "param_key": "vol_ratio_min", "role": "primary"},
            {"feature": "close_position", "op": ">", "param_key": "close_pos_min", "role": "primary"},
        ],
        params_template={"vol_ratio_min": 2.0, "close_pos_min": 0.95},
        selected_bar_configs=["tick_610"],
        validation_sample_cache={
            "tick_610": [
                (
                    "sample_day",
                    pl.DataFrame(
                        {
                            "volume_ratio": [0.9, 1.0, 1.1, 1.2],
                            "close_position": [0.1, 0.2, 0.3, 0.4],
                        }
                    ),
                )
            ]
        },
    )
    assert report["bar_results"][0]["status"] == "zero_signal"
    assert "Combined entry_conditions pass-through is 0/4 bars" in format_feasibility_error(report)


def test_assess_entry_condition_feasibility_accepts_sparse_but_nonzero_conditions():
    report = assess_entry_condition_feasibility(
        entry_conditions=[
            {"feature": "volume_ratio", "op": ">", "param_key": "vol_ratio_min", "role": "primary"},
        ],
        params_template={"vol_ratio_min": 1.25},
        selected_bar_configs=["tick_610"],
        validation_sample_cache={
            "tick_610": [
                (
                    "sample_day",
                    pl.DataFrame({"volume_ratio": [1.0] * 98 + [1.2, 1.3]}),
                )
            ]
        },
    )
    assert report["bar_results"][0]["status"] == "ok"


def test_thinker_feasibility_error_carries_report_and_brief():
    exc = ThinkerFeasibilityError("bad", report={"bar_results": []}, brief={"hypothesis_id": "h1"})
    assert exc.report == {"bar_results": []}
    assert exc.brief["hypothesis_id"] == "h1"
