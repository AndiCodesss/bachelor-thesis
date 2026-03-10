from __future__ import annotations

import json

import polars as pl
import pytest

from research.lib.thinker_feasibility import (
    ThinkerFeasibilityError,
    assess_entry_condition_feasibility,
    format_feasibility_error,
    normalize_entry_conditions,
    repair_thinker_brief_for_feasibility,
)


def test_normalize_entry_conditions_requires_numeric_param_keys():
    with pytest.raises(ValueError, match="numeric params_template"):
        normalize_entry_conditions(
            [{"feature": "volume_ratio", "op": ">", "param_key": "threshold", "role": "primary"}],
            params_template={"threshold": "x"},
        )


def test_normalize_entry_conditions_limits_primary_and_confirmation_counts():
    with pytest.raises(ValueError, match="at most 2 primary conditions"):
        normalize_entry_conditions(
            [
                {"feature": "a", "op": ">", "param_key": "a_min", "role": "primary"},
                {"feature": "b", "op": ">", "param_key": "b_min", "role": "primary"},
                {"feature": "c", "op": ">", "param_key": "c_min", "role": "primary"},
            ],
            params_template={"a_min": 1.0, "b_min": 1.0, "c_min": 1.0},
        )

    with pytest.raises(ValueError, match="at most 1 confirmation condition"):
        normalize_entry_conditions(
            [
                {"feature": "a", "op": ">", "param_key": "a_min", "role": "primary"},
                {"feature": "b", "op": ">", "param_key": "b_min", "role": "confirmation"},
                {"feature": "c", "op": ">", "param_key": "c_min", "role": "confirmation"},
            ],
            params_template={"a_min": 1.0, "b_min": 1.0, "c_min": 1.0},
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
    assert "_mask" not in report["bar_results"][0]["condition_rows"][0]
    json.dumps(report)
    assert "Combined entry_conditions pass-through is 0/4 bars" in format_feasibility_error(report)


def test_assess_entry_condition_feasibility_scores_each_sample_independently():
    report = assess_entry_condition_feasibility(
        entry_conditions=[
            {"feature": "volume_ratio", "op": ">", "param_key": "vol_ratio_min", "role": "primary"},
        ],
        params_template={"vol_ratio_min": 1.25},
        selected_bar_configs=["tick_610"],
        validation_sample_cache={
            "tick_610": [
                (
                    "day_one",
                    pl.DataFrame({"volume_ratio": [1.0, 1.1, 1.2]}),
                ),
                (
                    "day_two",
                    pl.DataFrame({"volume_ratio": [1.0] * 98 + [1.3, 1.4]}),
                ),
            ]
        },
    )

    assert [(row["sample_label"], row["status"]) for row in report["bar_results"]] == [
        ("day_one", "zero_signal"),
        ("day_two", "ok"),
    ]


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


def test_repair_thinker_brief_for_feasibility_relaxes_zero_signal_threshold():
    brief = {
        "params_template": {"vol_ratio_min": 2.0, "pt_ticks": 40, "sl_ticks": 20},
        "entry_conditions": [
            {"feature": "volume_ratio", "op": ">", "param_key": "vol_ratio_min", "role": "primary"},
        ],
    }
    sample_cache = {
        "tick_610": [
            (
                "sample_day",
                pl.DataFrame({"volume_ratio": [1.0] * 98 + [1.2, 1.3]}),
            )
        ]
    }
    report = assess_entry_condition_feasibility(
        entry_conditions=brief["entry_conditions"],
        params_template=brief["params_template"],
        selected_bar_configs=["tick_610"],
        validation_sample_cache=sample_cache,
    )

    repaired, actions = repair_thinker_brief_for_feasibility(brief, report)

    assert actions
    assert repaired is not None
    assert repaired["params_template"]["vol_ratio_min"] < 2.0
    repaired_report = assess_entry_condition_feasibility(
        entry_conditions=repaired["entry_conditions"],
        params_template=repaired["params_template"],
        selected_bar_configs=["tick_610"],
        validation_sample_cache=sample_cache,
    )
    assert repaired_report["bar_results"][0]["status"] == "ok"


def test_repair_thinker_brief_for_feasibility_tightens_over_signal_threshold():
    brief = {
        "params_template": {"delta_min": 0.0, "pt_ticks": 40, "sl_ticks": 20},
        "entry_conditions": [
            {"feature": "delta_heat", "op": ">", "param_key": "delta_min", "role": "primary"},
        ],
    }
    sample_cache = {
        "tick_610": [
            (
                "sample_day",
                pl.DataFrame({"delta_heat": [float(i) for i in range(100)]}),
            )
        ]
    }
    report = assess_entry_condition_feasibility(
        entry_conditions=brief["entry_conditions"],
        params_template=brief["params_template"],
        selected_bar_configs=["tick_610"],
        validation_sample_cache=sample_cache,
    )

    repaired, actions = repair_thinker_brief_for_feasibility(brief, report)

    assert report["bar_results"][0]["status"] == "over_signal"
    assert actions
    assert repaired is not None
    assert repaired["params_template"]["delta_min"] > 90.0
    repaired_report = assess_entry_condition_feasibility(
        entry_conditions=repaired["entry_conditions"],
        params_template=repaired["params_template"],
        selected_bar_configs=["tick_610"],
        validation_sample_cache=sample_cache,
    )
    assert repaired_report["bar_results"][0]["status"] == "ok"


def test_thinker_feasibility_error_carries_report_and_brief():
    exc = ThinkerFeasibilityError("bad", report={"bar_results": []}, brief={"hypothesis_id": "h1"})
    assert exc.report == {"bar_results": []}
    assert exc.brief["hypothesis_id"] == "h1"
