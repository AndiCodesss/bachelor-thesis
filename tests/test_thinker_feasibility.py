from __future__ import annotations

import json

import polars as pl
import pytest

from research.lib.thinker_feasibility import (
    ThinkerFeasibilityError,
    assess_entry_condition_feasibility,
    format_feasibility_error,
    is_context_dependent_feature,
    normalize_entry_conditions,
    repair_thinker_brief_for_feasibility,
    summarize_cross_sample_conflicts,
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


def test_normalize_entry_conditions_can_coerce_recoverable_thinker_schema_slop():
    out = normalize_entry_conditions(
        [
            {"feature": "prev_day_va_position", "op": ">=", "param_key": "va_min", "role": "location_filter"},
            {"feature": "trade_intensity", "op": ">", "param_key": "ti_min", "role": "primary"},
            {"feature": "cvd_price_divergence_3", "op": "bool_true", "role": "trigger_filter"},
            {"feature": "volume_ratio", "op": ">", "param_key": "vol_min", "role": "setup_filter"},
        ],
        params_template={"va_min": 0.5, "ti_min": 1.0, "vol_min": 1.1},
        coerce_schema_slop=True,
    )
    assert [(row["feature"], row["role"]) for row in out] == [
        ("prev_day_va_position", "primary"),
        ("trade_intensity", "primary"),
        ("cvd_price_divergence_3", "confirmation"),
    ]


def test_is_context_dependent_feature_matches_prev_session_columns():
    assert is_context_dependent_feature("prev_day_va_position") is True
    assert is_context_dependent_feature("dist_prev_vah") is True
    assert is_context_dependent_feature("volume_ratio") is False


def test_assess_entry_condition_feasibility_marks_context_unavailable_for_prev_session_feature():
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
    assert report["bar_results"][0]["status"] == "context_unavailable"
    assert "context-dependent feature(s) unavailable" in report["bar_results"][0]["error"]


def test_assess_entry_condition_feasibility_detects_dead_primary_feature():
    report = assess_entry_condition_feasibility(
        entry_conditions=[
            {"feature": "stale_anchor", "op": ">", "param_key": "anchor_min", "role": "primary"}
        ],
        params_template={"anchor_min": 1.0},
        selected_bar_configs=["tick_610"],
        validation_sample_cache={
            "tick_610": [
                (
                    "sample_day",
                    pl.DataFrame({"stale_anchor": [None, None, None]}),
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


def test_repair_thinker_brief_for_feasibility_never_drops_primary_conditions():
    brief = {
        "params_template": {"anchor_min": 1.0, "confirm_min": 0.5, "pt_ticks": 40, "sl_ticks": 20},
        "entry_conditions": [
            {"feature": "stale_anchor", "op": ">", "param_key": "anchor_min", "role": "primary"},
            {"feature": "volume_ratio", "op": ">", "param_key": "confirm_min", "role": "confirmation"},
        ],
    }
    report = assess_entry_condition_feasibility(
        entry_conditions=brief["entry_conditions"],
        params_template=brief["params_template"],
        selected_bar_configs=["tick_610"],
        validation_sample_cache={
            "tick_610": [
                (
                    "sample_day",
                    pl.DataFrame(
                        {
                            "stale_anchor": [None, None, None],
                            "volume_ratio": [0.6, 0.7, 0.8],
                        }
                    ),
                )
            ]
        },
    )

    repaired, actions = repair_thinker_brief_for_feasibility(brief, report)

    assert report["bar_results"][0]["status"] == "dead_feature_primary"
    assert repaired is None
    assert actions == []


def test_repair_thinker_brief_for_feasibility_refuses_cross_sample_threshold_conflicts():
    brief = {
        "params_template": {"position_in_va_min": 0.724548, "pt_ticks": 40, "sl_ticks": 20},
        "entry_conditions": [
            {"feature": "position_in_va", "op": ">=", "param_key": "position_in_va_min", "role": "primary"},
        ],
    }
    report = {
        "bar_results": [
            {
                "status": "over_signal",
                "bar_config": "tick_610",
                "sample_label": "nq_2023-03-06",
                "nonzero": 55,
                "total": 523,
                "signal_rate_pct": 10.52,
                "condition_rows": [
                    {
                        "column": "position_in_va",
                        "operator": ">=",
                        "role": "primary",
                        "param_key": "position_in_va_min",
                        "threshold": 0.724548,
                        "p10": 0.0924,
                        "p50": 0.5726,
                        "p90": 1.4852,
                        "pass_rate_pct": 42.8,
                    }
                ],
            },
            {
                "status": "zero_signal",
                "bar_config": "tick_610",
                "sample_label": "nq_2024-06-13",
                "nonzero": 0,
                "total": 539,
                "signal_rate_pct": 0.0,
                "condition_rows": [
                    {
                        "column": "position_in_va",
                        "operator": ">=",
                        "role": "primary",
                        "param_key": "position_in_va_min",
                        "threshold": 0.724548,
                        "p10": -0.6538,
                        "p50": 0.0944,
                        "p90": 0.7245,
                        "pass_rate_pct": 0.0,
                    }
                ],
            },
        ]
    }

    repaired, actions = repair_thinker_brief_for_feasibility(brief, report)

    assert repaired is None
    assert actions == []


def test_thinker_feasibility_error_carries_report_and_brief():
    exc = ThinkerFeasibilityError("bad", report={"bar_results": []}, brief={"hypothesis_id": "h1"})
    assert exc.report == {"bar_results": []}
    assert exc.brief["hypothesis_id"] == "h1"


def test_summarize_cross_sample_conflicts_highlights_distribution_drift():
    report = {
        "bar_results": [
            {
                "status": "over_signal",
                "sample_label": "nq_2023-03-06",
                "condition_rows": [
                    {
                        "column": "prev_day_va_position",
                        "operator": ">=",
                        "role": "primary",
                        "param_key": "va_position_min",
                        "threshold": 1.385153,
                        "p10": 1.3261,
                        "p50": 1.5651,
                        "p90": 1.9812,
                        "pass_rate_pct": 19.12,
                    }
                ],
            },
            {
                "status": "zero_signal",
                "sample_label": "nq_2023-08-08",
                "condition_rows": [
                    {
                        "column": "prev_day_va_position",
                        "operator": ">=",
                        "role": "primary",
                        "param_key": "va_position_min",
                        "threshold": 1.385153,
                        "p10": -2.4855,
                        "p50": -1.6756,
                        "p90": -0.3554,
                        "pass_rate_pct": 0.0,
                    }
                ],
            },
        ]
    }

    notes = summarize_cross_sample_conflicts(report)

    assert len(notes) == 1
    assert "sample p10 range -2.4855..1.3261" in notes[0]
    assert "p90 range -0.3554..1.9812" in notes[0]
    assert "likely blocks on nq_2023-08-08" in notes[0]
    assert "still dense on nq_2023-03-06" in notes[0]


def test_format_feasibility_error_includes_cross_sample_conflicts_and_cycle_note():
    report = {
        "repair_cycle_detected": True,
        "bar_results": [
            {
                "status": "over_signal",
                "bar_config": "tick_610",
                "sample_label": "nq_2023-03-06",
                "nonzero": 55,
                "total": 523,
                "signal_rate_pct": 10.52,
                "condition_rows": [
                    {
                        "column": "position_in_va",
                        "operator": ">=",
                        "role": "primary",
                        "param_key": "position_in_va_min",
                        "threshold": 0.724548,
                        "p10": 0.0924,
                        "p50": 0.5726,
                        "p90": 1.4852,
                        "pass_rate_pct": 42.8,
                    }
                ],
            },
            {
                "status": "zero_signal",
                "bar_config": "tick_610",
                "sample_label": "nq_2024-06-13",
                "nonzero": 0,
                "total": 539,
                "signal_rate_pct": 0.0,
                "condition_rows": [
                    {
                        "column": "position_in_va",
                        "operator": ">=",
                        "role": "primary",
                        "param_key": "position_in_va_min",
                        "threshold": 0.724548,
                        "p10": -0.6538,
                        "p50": 0.0944,
                        "p90": 0.7245,
                        "pass_rate_pct": 0.0,
                    }
                ],
            },
        ],
    }

    message = format_feasibility_error(report)

    assert "Cross-sample distribution drift:" in message
    assert "position_in_va >= 0.724548" in message
    assert "Auto-repair oscillated between incompatible thresholds" in message


def test_format_feasibility_error_includes_cross_sample_repair_conflict_note():
    report = {
        "repair_conflict_detected": True,
        "bar_results": [
            {
                "status": "over_signal",
                "bar_config": "tick_610",
                "sample_label": "nq_2023-03-06",
                "nonzero": 55,
                "total": 523,
                "signal_rate_pct": 10.52,
                "condition_rows": [
                    {
                        "column": "position_in_va",
                        "operator": ">=",
                        "role": "primary",
                        "param_key": "position_in_va_min",
                        "threshold": 0.724548,
                        "p10": 0.0924,
                        "p50": 0.5726,
                        "p90": 1.4852,
                        "pass_rate_pct": 42.8,
                    }
                ],
            },
            {
                "status": "zero_signal",
                "bar_config": "tick_610",
                "sample_label": "nq_2024-06-13",
                "nonzero": 0,
                "total": 539,
                "signal_rate_pct": 0.0,
                "condition_rows": [
                    {
                        "column": "position_in_va",
                        "operator": ">=",
                        "role": "primary",
                        "param_key": "position_in_va_min",
                        "threshold": 0.724548,
                        "p10": -0.6538,
                        "p50": 0.0944,
                        "p90": 0.7245,
                        "pass_rate_pct": 0.0,
                    }
                ],
            },
        ],
    }

    message = format_feasibility_error(report)

    assert "Cross-sample repair conflicts:" in message
    assert "too tight on some samples and too loose on others" in message
    assert "Cross-sample feasibility constraints conflict across validation samples" in message
