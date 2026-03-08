from __future__ import annotations

import polars as pl

from research.lib.feature_surface import (
    build_feature_surface,
    describe_referenced_columns,
    format_feature_surface_context,
    format_referenced_surface_warnings,
)


def test_build_feature_surface_classifies_dead_sparse_and_variable_features():
    sparse_values = [0.0] * 999 + [1.0]
    active_values = [float((idx % 10) + 1) for idx in range(1000)]
    surface = build_feature_surface(
        samples_by_bar_config={
            "tick_610": [
                (
                    "sample_a",
                    pl.DataFrame(
                        {
                            "ts_event": list(range(1000)),
                            "dead_feature": [0.0] * 1000,
                            "sparse_feature": sparse_values,
                            "active_feature": active_values,
                        },
                    ),
                ),
            ],
        },
    )

    bar_surface = surface["by_bar_config"]["tick_610"]
    assert bar_surface["total_rows"] == 1000
    assert any(row["name"] == "dead_feature" for row in bar_surface["dead_features"])
    assert any(row["name"] == "sparse_feature" for row in bar_surface["sparse_features"])
    assert "active_feature" not in {row["name"] for row in bar_surface["dead_features"]}
    assert "active_feature" not in {row["name"] for row in bar_surface["sparse_features"]}


def test_format_feature_surface_context_is_compact_and_readable():
    surface = {
        "by_bar_config": {
            "tick_610": {
                "sample_count": 2,
                "sample_labels": ["a", "b"],
                "total_rows": 1200,
                "dead_features": [{"name": "squeeze_score", "nonzero_rate": 0.0}],
                "sparse_features": [{"name": "failed_auction_bull", "nonzero_rate": 0.001}],
            },
        },
    }

    context = format_feature_surface_context(surface, selected_bar_configs=["tick_610"])
    assert "FEATURE_SURFACE_INTELLIGENCE:" in context
    assert "tick_610: 1200 sampled rows across 2 file(s)" in context
    assert "squeeze_score (0.00% non-zero)" in context
    assert "failed_auction_bull (0.10% non-zero)" in context


def test_format_referenced_surface_warnings_only_flags_referenced_risky_features():
    surface = {
        "by_bar_config": {
            "tick_610": {
                "feature_stats": {
                    "squeeze_score": {"kind": "constant_zero", "nonzero_rate": 0.0},
                    "close_position": {"kind": "variable", "nonzero_rate": 1.0},
                    "failed_auction_bull": {"kind": "binary_sparse", "nonzero_rate": 0.001},
                },
            },
        },
    }

    warning = format_referenced_surface_warnings(
        surface=surface,
        selected_bar_configs=["tick_610"],
        text_fragments=[
            'Use "squeeze_score" with "close_position" for confirmation.',
            'Ignore "failed_auction_bull" until price confirms.',
        ],
    )
    assert "REFERENCED_FEATURE_SURFACE_WARNINGS:" in warning
    assert "squeeze_score" in warning
    assert "failed_auction_bull" in warning
    assert "close_position" not in warning


def test_format_referenced_surface_warnings_matches_bare_feature_names():
    surface = {
        "by_bar_config": {
            "tick_610": {
                "feature_stats": {
                    "squeeze_score": {"kind": "constant_zero", "nonzero_rate": 0.0},
                    "price_velocity_z": {"kind": "variable", "nonzero_rate": 1.0},
                },
            },
        },
    }

    warning = format_referenced_surface_warnings(
        surface=surface,
        selected_bar_configs=["tick_610"],
        text_fragments=[
            "Use squeeze_score only after price_velocity_z confirms a breakout.",
        ],
    )
    assert "squeeze_score" in warning
    assert "price_velocity_z" not in warning


def test_describe_referenced_columns_reports_sparse_boolean_and_percentiles():
    df = pl.DataFrame(
        {
            "absorption_signal": [True, False, False, False],
            "close_position": [0.1, 0.4, 0.6, 0.9],
        },
    )

    description = describe_referenced_columns(
        df=df,
        code='signal = df["absorption_signal"] & (df["close_position"] > 0.8)',
    )
    assert "absorption_signal" in description
    assert "25.00%" in description
    assert "close_position" in description
    assert "p50" in description
