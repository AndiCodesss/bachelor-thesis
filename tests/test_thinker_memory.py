from __future__ import annotations

from pathlib import Path

from research.lib.thinker_memory import (
    append_thinker_attempt,
    format_thinker_memory_context,
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
    assert "[iter 4] vol_compression -> REJECTED (zero_signal): 0/523 bars on tick_610." in text
    assert "Blocking: squeeze_score > 0.5 (0.0% pass)" in text
    assert "Params: squeeze_threshold=0.5" in text
