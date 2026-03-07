from __future__ import annotations

import json
from pathlib import Path

from research.lib.learning_scorecard import (
    empty_bar_entry,
    empty_scorecard,
    empty_theme_entry,
    format_learning_context,
    laplace_rate,
    normalize_theme_tag,
    read_learning_scorecard,
    rebuild_learning_scorecard,
    resolve_focus_anchors,
    update_learning_scorecard,
)


def test_laplace_rate_zero():
    assert laplace_rate(0, 0) == 1 / 2


def test_laplace_rate_normal():
    assert laplace_rate(5, 12) == 6 / 14


def test_laplace_rate_perfect_not_one():
    assert laplace_rate(1, 1) == 2 / 3


def test_empty_scorecard_has_expected_shape():
    scorecard = empty_scorecard()
    assert scorecard["schema_version"] == "1.0"
    assert scorecard["theme_stats"] == {}
    assert scorecard["bar_config_affinity"] == {}
    assert scorecard["near_misses"] == []
    assert scorecard["low_sample_themes"] == []


def test_empty_theme_entry_has_smoothed_defaults():
    entry = empty_theme_entry()
    assert entry["attempts"] == 0
    assert entry["search_passes"] == 0
    assert entry["search_rate"] == laplace_rate(0, 0)
    assert entry["selection_attempts"] == 0
    assert entry["selection_passes"] == 0
    assert entry["selection_rate"] == laplace_rate(0, 0)
    assert entry["fail_counts"] == {}


def test_empty_bar_entry_has_smoothed_defaults():
    entry = empty_bar_entry()
    assert entry["attempts"] == 0
    assert entry["search_passes"] == 0
    assert entry["search_rate"] == laplace_rate(0, 0)
    assert entry["selection_passes"] == 0
    assert entry["selection_rate"] == laplace_rate(0, 0)


def test_resolve_focus_anchors_uses_current_focus():
    mission = {
        "current_focus": [
            "AMT value area rejection and rotation (VAH/VAL/POC mean-reversion)",
            "Sequential state machine patterns: arm on condition A, fire on condition B",
        ]
    }
    assert resolve_focus_anchors(mission) == ["amt_value_area", "sequential_state_machine"]


def test_normalize_theme_tag_keeps_dynamic_tags():
    assert normalize_theme_tag("amt_value_area") == "amt_value_area"
    assert normalize_theme_tag("something-random") == "something_random"
    assert normalize_theme_tag("very specific opening drive failure fade") == "very_specific_opening_drive"
    assert normalize_theme_tag("") == "other"
    assert normalize_theme_tag(None) == "other"


def test_update_learning_scorecard_from_task_updates_bar_affinity_and_fail_counts(tmp_path: Path):
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_lock = tmp_path / "scorecard.lock"

    task = {
        "strategy_name": "alpha_amt",
        "theme_tag": "amt_value_area",
        "bar_config": "tick_610",
        "search_result": {
            "verdict": "FAIL",
            "metrics": {"sharpe_ratio": 1.32, "trade_count": 24},
            "gauntlet": {
                "overall_verdict": "FAIL",
                "walk_forward": {"verdict": "FAIL"},
                "shuffle_test": {"verdict": "PASS"},
            },
            "advanced_validation_gates": {"enabled": False, "checks": {}},
        },
        "selection_result": None,
        "details": {},
    }
    update_learning_scorecard(
        scorecard_path,
        scorecard_lock,
        task=task,
    )
    scorecard = read_learning_scorecard(scorecard_path, scorecard_lock)

    bar = scorecard["bar_config_affinity"]["amt_value_area"]["tick_610"]
    assert bar["attempts"] == 1
    assert bar["search_passes"] == 0
    assert bar["search_rate"] == laplace_rate(0, 1)
    assert bar["selection_passes"] == 0

    theme = scorecard["theme_stats"]["amt_value_area"]
    assert theme["fail_counts"] == {"walk_forward": 1}

    near_miss = scorecard["near_misses"][0]
    assert near_miss["strategy"] == "alpha_amt"
    assert near_miss["theme"] == "amt_value_area"
    assert near_miss["failed_checks"] == ["walk_forward"]


def test_update_learning_scorecard_records_selection_near_miss(tmp_path: Path):
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_lock = tmp_path / "scorecard.lock"

    task = {
        "strategy_name": "alpha_selection",
        "theme_tag": "orderflow_divergence",
        "bar_config": "tick_610",
        "search_result": {
            "verdict": "PASS",
            "metrics": {"sharpe_ratio": 1.30, "trade_count": 32},
            "gauntlet": {"overall_verdict": "PASS"},
            "advanced_validation_gates": {"enabled": False, "checks": {}},
        },
        "selection_result": {
            "verdict": "FAIL",
            "metrics": {"sharpe_ratio": 1.10, "trade_count": 21},
            "gauntlet": {
                "overall_verdict": "FAIL",
                "walk_forward": {"verdict": "FAIL"},
            },
            "advanced_validation_gates": {"enabled": False, "checks": {}},
        },
        "details": {},
    }

    update_learning_scorecard(
        scorecard_path,
        scorecard_lock,
        task=task,
    )
    scorecard = read_learning_scorecard(scorecard_path, scorecard_lock)

    near_miss = scorecard["near_misses"][0]
    assert near_miss["strategy"] == "alpha_selection"
    assert near_miss["theme"] == "orderflow_divergence"
    assert near_miss["bar_config"] == "tick_610"
    assert near_miss["split"] == "selection"
    assert near_miss["failed_checks"] == ["walk_forward"]


def test_update_learning_scorecard_from_handoff_updates_family_stats(tmp_path: Path):
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_lock = tmp_path / "scorecard.lock"

    handoff = {
        "handoff_type": "validation_request",
        "payload": {"theme_tag": "amt_value_area"},
        "result": {
            "pass_count": 1,
            "tasks": [
                {"selection_attempted": True, "selection_verdict": "PASS"},
                {"selection_attempted": False, "selection_verdict": ""},
            ],
        },
    }
    update_learning_scorecard(
        scorecard_path,
        scorecard_lock,
        handoff=handoff,
    )
    scorecard = read_learning_scorecard(scorecard_path, scorecard_lock)
    theme = scorecard["theme_stats"]["amt_value_area"]
    assert theme["attempts"] == 1
    assert theme["search_passes"] == 1
    assert theme["selection_attempts"] == 1
    assert theme["selection_passes"] == 1
    assert theme["selection_rate"] == laplace_rate(1, 1)


def test_read_learning_scorecard_recomputes_low_sample_themes(tmp_path: Path):
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_lock = tmp_path / "scorecard.lock"
    scorecard_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "theme_stats": {
                    "amt_value_area": {"attempts": 3, "search_passes": 1, "selection_attempts": 1, "selection_passes": 0, "fail_counts": {}},
                    "volatility_compression": {"attempts": 1, "search_passes": 0, "selection_attempts": 0, "selection_passes": 0, "fail_counts": {}},
                },
                "bar_config_affinity": {},
                "near_misses": [],
                "low_sample_themes": [],
            }
        ),
        encoding="utf-8",
    )

    scorecard = read_learning_scorecard(scorecard_path, scorecard_lock)
    assert scorecard["low_sample_themes"] == ["volatility_compression"]


def test_rebuild_learning_scorecard_uses_experiments_and_handoffs(tmp_path: Path):
    experiments_path = tmp_path / "research_experiments.jsonl"
    handoffs_path = tmp_path / "handoffs.json"
    handoffs_lock = tmp_path / "handoffs.lock"
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_lock = tmp_path / "scorecard.lock"

    experiments_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "task_result",
                        "strategy_name": "alpha_amt",
                        "theme_tag": "amt_value_area",
                        "bar_config": "tick_610",
                        "verdict": "PASS",
                        "metrics": {"sharpe_ratio": 1.5, "trade_count": 30},
                        "gauntlet": {"overall_verdict": "PASS"},
                        "search_result": {
                            "verdict": "PASS",
                            "metrics": {"sharpe_ratio": 1.5, "trade_count": 30},
                            "gauntlet": {"overall_verdict": "PASS"},
                            "advanced_validation_gates": {"enabled": False, "checks": {}},
                        },
                        "selection_result": {"verdict": "FAIL", "metrics": {"sharpe_ratio": 0.4}, "gauntlet": {"overall_verdict": "FAIL", "walk_forward": {"verdict": "FAIL"}}},
                    }
                ),
                json.dumps(
                    {
                        "event": "task_result",
                        "strategy_name": "alpha_vol",
                        "theme_tag": "volatility_compression",
                        "bar_config": "time_1m",
                        "verdict": "FAIL",
                        "search_result": {
                            "verdict": "FAIL",
                            "metrics": {"sharpe_ratio": 0.8, "trade_count": 12},
                            "gauntlet": {"overall_verdict": "FAIL", "trade_count": {"verdict": "FAIL"}},
                            "advanced_validation_gates": {"enabled": False, "checks": {}},
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "validation_handoff_completed",
                        "handoff_id": "h_amt",
                        "theme_tag": "amt_value_area",
                        "pass_count": 1,
                        "selection_attempted": True,
                        "selection_passed": False,
                    }
                ),
                json.dumps(
                    {
                        "event": "validation_handoff_completed",
                        "handoff_id": "h_vol",
                        "theme_tag": "volatility_compression",
                        "pass_count": 0,
                        "selection_attempted": False,
                        "selection_passed": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    handoffs_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "pending": [],
                "completed": [
                    {
                        "handoff_id": "h_amt",
                        "handoff_type": "validation_request",
                        "payload": {"theme_tag": "amt_value_area"},
                        "result": {
                            "pass_count": 1,
                            "tasks": [
                                {"selection_attempted": True, "selection_verdict": "FAIL"},
                            ],
                        },
                    },
                    {
                        "handoff_id": "h_vol",
                        "handoff_type": "validation_request",
                        "payload": {"theme_tag": "volatility_compression"},
                        "result": {
                            "pass_count": 0,
                            "tasks": [
                                {"selection_attempted": False, "selection_verdict": ""},
                            ],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rebuild_learning_scorecard(
        experiments_path=experiments_path,
        handoffs_path=handoffs_path,
        handoffs_lock=handoffs_lock,
        scorecard_path=scorecard_path,
        scorecard_lock=scorecard_lock,
    )
    scorecard = read_learning_scorecard(scorecard_path, scorecard_lock)

    assert scorecard["theme_stats"]["amt_value_area"]["attempts"] == 1
    assert scorecard["theme_stats"]["amt_value_area"]["search_passes"] == 1
    assert scorecard["theme_stats"]["amt_value_area"]["selection_attempts"] == 1
    assert scorecard["theme_stats"]["amt_value_area"]["selection_passes"] == 0
    assert scorecard["bar_config_affinity"]["amt_value_area"]["tick_610"]["selection_passes"] == 0
    assert scorecard["near_misses"][0]["strategy"] == "alpha_vol"
    assert scorecard["low_sample_themes"] == ["amt_value_area", "volatility_compression"]


def test_rebuild_learning_scorecard_recovers_family_stats_from_experiment_log_only(tmp_path: Path):
    experiments_path = tmp_path / "research_experiments.jsonl"
    handoffs_path = tmp_path / "handoffs.json"
    handoffs_lock = tmp_path / "handoffs.lock"
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_lock = tmp_path / "scorecard.lock"

    experiments_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "task_result",
                        "strategy_name": "alpha_amt",
                        "theme_tag": "amt_value_area",
                        "bar_config": "tick_610",
                        "search_result": {
                            "verdict": "PASS",
                            "metrics": {"sharpe_ratio": 1.4, "trade_count": 28},
                            "gauntlet": {"overall_verdict": "PASS"},
                            "advanced_validation_gates": {"enabled": False, "checks": {}},
                        },
                        "selection_result": {
                            "verdict": "PASS",
                            "metrics": {"sharpe_ratio": 1.1, "trade_count": 20},
                            "gauntlet": {"overall_verdict": "PASS"},
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "validation_handoff_completed",
                        "handoff_id": "h_001",
                        "theme_tag": "amt_value_area",
                        "pass_count": 1,
                        "selection_attempted": True,
                        "selection_passed": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    handoffs_path.write_text(
        json.dumps({"schema_version": "1.0", "pending": [], "completed": []}),
        encoding="utf-8",
    )

    rebuild_learning_scorecard(
        experiments_path=experiments_path,
        handoffs_path=handoffs_path,
        handoffs_lock=handoffs_lock,
        scorecard_path=scorecard_path,
        scorecard_lock=scorecard_lock,
    )

    scorecard = read_learning_scorecard(scorecard_path, scorecard_lock)
    theme = scorecard["theme_stats"]["amt_value_area"]
    assert theme["attempts"] == 1
    assert theme["search_passes"] == 1
    assert theme["selection_attempts"] == 1
    assert theme["selection_passes"] == 1


def test_format_learning_context_is_compact_and_ranked():
    scorecard = {
        "schema_version": "1.0",
        "rebuilt_at": "2026-03-07T14:00:00+00:00",
        "theme_stats": {
            "amt_value_area": {
                "attempts": 12,
                "search_passes": 5,
                "search_rate": 0.43,
                "selection_attempts": 5,
                "selection_passes": 2,
                "selection_rate": 0.43,
                "fail_counts": {"alpha_decay": 3, "trade_count": 2},
            },
            "state_machine": {
                "attempts": 4,
                "search_passes": 1,
                "search_rate": 0.33,
                "selection_attempts": 1,
                "selection_passes": 0,
                "selection_rate": 0.33,
                "fail_counts": {"shuffle_test": 3},
            },
        },
        "bar_config_affinity": {
            "amt_value_area": {
                "tick_610": {"attempts": 8, "search_passes": 4, "search_rate": 0.50, "selection_passes": 2, "selection_rate": 0.30}
            }
        },
        "near_misses": [
            {
                "strategy": "ib_fade_vol_filter_03",
                "theme": "amt_value_area",
                "bar_config": "tick_610",
                "sharpe": 1.32,
                "failed_checks": ["walk_forward"],
                "split": "search",
            }
        ],
        "low_sample_themes": ["volatility_compression"],
    }
    context = format_learning_context(scorecard)
    assert "LEARNING_SCORECARD:" in context
    assert "amt_value_area: 12 attempts" in context
    assert "amt_value_area + tick_610" in context
    assert "alpha_decay (3)" in context
    assert "ib_fade_vol_filter_03" in context
    assert "Low-sample themes: volatility_compression" in context
    assert "EXPLORE a weakly sampled or newly justified one" in context
