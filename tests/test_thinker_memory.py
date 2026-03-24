from __future__ import annotations

from pathlib import Path

import numpy as np

from research.lib.thinker_memory import (
    append_thinker_attempt,
    derive_primary_feature_cooldowns,
    format_thinker_memory_context,
    primary_feature_cooldown_key,
    read_thinker_memory,
)


def test_append_thinker_attempt_keeps_last_three_rows(tmp_path: Path):
    path = tmp_path / "thinker_memory_A.json"
    lock_path = tmp_path / "thinker_memory_A.lock"

    for iteration in range(1, 6):
        append_thinker_attempt(
            path=path,
            lock_path=lock_path,
            lane_id="A",
            attempt={
                "iteration": iteration,
                "theme_tag": f"theme_{iteration}",
                "status_label": "REJECTED",
                "summary": f"attempt {iteration}",
            },
            window_size=3,
        )

    payload = read_thinker_memory(
        path=path,
        lock_path=lock_path,
        lane_id="A",
        window_size=3,
    )
    assert [row["iteration"] for row in payload["recent_attempts"]] == [3, 4, 5]


def test_format_thinker_memory_context_is_compact_and_includes_params():
    text = format_thinker_memory_context(
        {
            "recent_attempts": [
                {
                    "iteration": 4,
                    "theme_tag": "vol_compression",
                    "mechanism": "fade failed compression that expands without sustained activity",
                    "status_label": "REJECTED (zero_signal)",
                    "summary": "0/523 bars on tick_610.",
                    "conditions_label": "Blocking",
                    "highlighted_conditions": [
                        {
                            "column": "squeeze_score",
                            "operator": ">",
                            "threshold": 0.5,
                            "pass_rate_pct": 0.0,
                        }
                    ],
                    "offending_params": {"squeeze_threshold": 0.5},
                }
            ]
        }
    )
    assert "RECENT_ATTEMPT_MEMORY" in text
    assert (
        "[iter 4] theme=vol_compression | mechanism=fade failed compression that expands without sustained activity "
        "-> REJECTED (zero_signal): 0/523 bars on tick_610."
    ) in text
    assert "Blocking: squeeze_score > 0.5 (0.0% pass)" in text
    assert "Params: squeeze_threshold=0.5" in text


def test_append_thinker_attempt_strips_private_fields_and_numpy_values(tmp_path: Path):
    path = tmp_path / "thinker_memory_B.json"
    lock_path = tmp_path / "thinker_memory_B.lock"

    append_thinker_attempt(
        path=path,
        lock_path=lock_path,
        lane_id="B",
        attempt={
            "iteration": 1,
            "theme_tag": "amt_value_area",
            "status_label": "REJECTED (zero_signal)",
            "summary": "0/5609 bars on tick_610.",
            "_debug_mask": np.array([True, False, True]),
            "highlighted_conditions": [
                {
                    "column": "high_low_vol_ratio",
                    "operator": ">=",
                    "threshold": np.float64(0.25),
                    "pass_rate_pct": np.float64(0.0),
                    "_mask": np.array([False, False, False]),
                }
            ],
        },
        window_size=3,
    )

    payload = read_thinker_memory(
        path=path,
        lock_path=lock_path,
        lane_id="B",
        window_size=3,
    )
    row = payload["recent_attempts"][0]
    assert "_debug_mask" not in row
    assert row["highlighted_conditions"][0]["threshold"] == 0.25
    assert row["highlighted_conditions"][0]["pass_rate_pct"] == 0.0
    assert "_mask" not in row["highlighted_conditions"][0]


def test_primary_feature_cooldown_key_groups_related_value_area_features():
    assert primary_feature_cooldown_key("prev_day_va_position") == "value_area_position"
    assert primary_feature_cooldown_key("position_in_va") == "value_area_position"
    assert primary_feature_cooldown_key("dist_prev_val") == "previous_session_value_reference"


def test_thinker_memory_derives_and_formats_primary_feature_cooldowns():
    payload = {
        "recent_attempts": [
            {
                "iteration": 5,
                "theme_tag": "amt_value_area",
                "status_label": "REJECTED (zero_signal)",
                "failure_type": "zero_signal",
                "summary": "0/539 bars on tick_610.",
                "cross_sample_conflicts": [
                    "prev_day_va_position between [0.7, 1.2] conflicts across validation samples."
                ],
                "highlighted_conditions": [
                    {
                        "column": "prev_day_va_position",
                        "role": "primary",
                        "severity": "blocks_all",
                        "pass_rate_pct": 0.0,
                    }
                ],
            },
            {
                "iteration": 6,
                "theme_tag": "amt_value_area",
                "status_label": "REJECTED (zero_signal)",
                "failure_type": "zero_signal",
                "summary": "0/523 bars on tick_610.",
                "cross_sample_conflicts": [
                    "position_in_va > 1.35 conflicts across validation samples."
                ],
                "highlighted_conditions": [
                    {
                        "column": "position_in_va",
                        "role": "primary",
                        "severity": "blocks_all",
                        "pass_rate_pct": 0.0,
                    }
                ],
            },
        ]
    }

    cooldowns = derive_primary_feature_cooldowns(payload)

    assert cooldowns == [
        {
            "family_key": "value_area_position",
            "feature": "prev_day_va_position",
            "score": 4,
            "reason": "cross-sample feasibility conflict",
        }
    ]

    text = format_thinker_memory_context(payload)
    assert "RECENT_PRIMARY_FEATURE_COOLDOWN:" in text
    assert "Avoid reusing `prev_day_va_position` as a primary gate next iteration" in text
