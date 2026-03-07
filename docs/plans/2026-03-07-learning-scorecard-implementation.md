# Learning Scorecard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a derived learning scorecard that tracks theme performance, bar config affinity, failure modes, and near misses — making autonomous hypothesis generation informed and adaptive.

**Architecture:** A single new module `research/lib/learning_scorecard.py` owns all scorecard logic. The validator (research.py) writes to the scorecard after persisted task/handoff completion. The orchestrator (llm_orchestrator.py) reads it into a compact prompt block for the thinker. The scorecard is a rebuildable cache, not sacred state.

**Tech Stack:** Python, JSON, portalocker (already in use), existing `update_json_file`/`read_json_file` from `research.lib.coordination`.

**Design doc:** `docs/plans/2026-03-07-learning-scorecard-design.md`

---

### Task 1: Core Scorecard Module — Data Model and Smoothing

**Files:**
- Create: `research/lib/learning_scorecard.py`
- Test: `tests/test_learning_scorecard.py`

**Step 1: Write the failing tests**

```python
"""Tests for research.lib.learning_scorecard."""

from research.lib.learning_scorecard import (
    laplace_rate,
    empty_scorecard,
    empty_theme_entry,
    empty_bar_entry,
)


def test_laplace_rate_zero():
    assert laplace_rate(0, 0) == 1 / 2  # (0+1)/(0+2)


def test_laplace_rate_normal():
    assert laplace_rate(5, 12) == 6 / 14  # (5+1)/(12+2)


def test_laplace_rate_perfect():
    # 1/1 should NOT be 1.0
    assert laplace_rate(1, 1) == 2 / 3


def test_empty_scorecard_has_schema():
    sc = empty_scorecard()
    assert sc["schema_version"] == "1.0"
    assert sc["theme_stats"] == {}
    assert sc["bar_config_affinity"] == {}
    assert sc["near_misses"] == []
    assert sc["underexplored_themes"] == []


def test_empty_theme_entry():
    entry = empty_theme_entry()
    assert entry["attempts"] == 0
    assert entry["search_passes"] == 0
    assert entry["search_rate"] == laplace_rate(0, 0)
    assert entry["selection_attempts"] == 0
    assert entry["selection_passes"] == 0
    assert entry["selection_rate"] == laplace_rate(0, 0)
    assert entry["fail_counts"] == {}


def test_empty_bar_entry():
    entry = empty_bar_entry()
    assert entry["attempts"] == 0
    assert entry["search_passes"] == 0
    assert entry["selection_passes"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_learning_scorecard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'research.lib.learning_scorecard'`

**Step 3: Write minimal implementation**

```python
"""Learning scorecard — derived intelligence for the autonomy loop.

This module is a rebuildable cache. Delete learning_scorecard.json and it
regenerates from experiment logs and handoffs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def laplace_rate(passes: int, attempts: int) -> float:
    """Laplace-smoothed success rate: (passes+1)/(attempts+2)."""
    return (passes + 1) / (attempts + 2)


def empty_scorecard() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "rebuilt_at": None,
        "theme_stats": {},
        "bar_config_affinity": {},
        "near_misses": [],
        "underexplored_themes": [],
    }


def empty_theme_entry() -> dict[str, Any]:
    return {
        "attempts": 0,
        "search_passes": 0,
        "search_rate": laplace_rate(0, 0),
        "selection_attempts": 0,
        "selection_passes": 0,
        "selection_rate": laplace_rate(0, 0),
        "fail_counts": {},
    }


def empty_bar_entry() -> dict[str, Any]:
    return {
        "attempts": 0,
        "search_passes": 0,
        "search_rate": laplace_rate(0, 0),
        "selection_passes": 0,
        "selection_rate": laplace_rate(0, 0),
    }
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_learning_scorecard.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add research/lib/learning_scorecard.py tests/test_learning_scorecard.py
git commit -m "feat: add learning scorecard data model and smoothing"
```

---

### Task 2: Theme Vocabulary — Slug Resolution

**Files:**
- Modify: `research/lib/learning_scorecard.py`
- Test: `tests/test_learning_scorecard.py`

**Step 1: Write the failing tests**

```python
from research.lib.learning_scorecard import resolve_theme_universe, normalize_theme_tag


def test_resolve_theme_universe_from_explicit_tags():
    mission = {"theme_tags": ["amt_value_area", "state_machine", "breakout_fade"]}
    assert resolve_theme_universe(mission) == ["amt_value_area", "state_machine", "breakout_fade"]


def test_resolve_theme_universe_from_current_focus():
    mission = {
        "current_focus": [
            "AMT value area rejection and rotation (VAH/VAL/POC mean-reversion)",
            "Sequential state machine patterns: arm on condition A, fire on condition B",
        ]
    }
    result = resolve_theme_universe(mission)
    assert result == ["amt_value_area", "sequential_state_machine"]


def test_resolve_theme_universe_empty():
    assert resolve_theme_universe({}) == []


def test_normalize_theme_tag_exact_match():
    universe = ["amt_value_area", "state_machine"]
    assert normalize_theme_tag("amt_value_area", universe) == "amt_value_area"


def test_normalize_theme_tag_unknown():
    universe = ["amt_value_area"]
    assert normalize_theme_tag("something_random", universe) == "other"


def test_normalize_theme_tag_empty():
    universe = ["amt_value_area"]
    assert normalize_theme_tag("", universe) == "other"
    assert normalize_theme_tag(None, universe) == "other"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_learning_scorecard.py -v -k theme`
Expected: FAIL — `ImportError`

**Step 3: Write minimal implementation**

Add to `research/lib/learning_scorecard.py`:

```python
import re

def _slugify(text: str) -> str:
    """Convert free-form text to a stable slug.

    Takes the first few meaningful words before punctuation/parens.
    """
    # Strip parenthetical/colon suffixes
    text = re.split(r"[:(]", text)[0].strip()
    # Take first 4 words max
    words = text.lower().split()[:4]
    slug = "_".join(re.sub(r"[^a-z0-9]", "", w) for w in words if re.sub(r"[^a-z0-9]", "", w))
    return slug or "other"


def resolve_theme_universe(mission: dict[str, Any]) -> list[str]:
    """Resolve the controlled theme vocabulary from mission config.

    Priority: explicit theme_tags > slugged current_focus.
    """
    explicit = mission.get("theme_tags")
    if isinstance(explicit, list) and explicit:
        return [str(t).strip() for t in explicit if str(t).strip()]

    focus = mission.get("current_focus")
    if isinstance(focus, list) and focus:
        return [_slugify(str(f)) for f in focus if str(f).strip()]

    return []


def normalize_theme_tag(tag: str | None, universe: list[str]) -> str:
    """Normalize a thinker-provided theme tag against the known universe."""
    if not tag or not isinstance(tag, str):
        return "other"
    clean = tag.strip().lower().replace("-", "_").replace(" ", "_")
    clean = re.sub(r"[^a-z0-9_]", "", clean)
    if clean in universe:
        return clean
    return "other"
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_learning_scorecard.py -v -k theme`
Expected: PASS

**Step 5: Commit**

```bash
git add research/lib/learning_scorecard.py tests/test_learning_scorecard.py
git commit -m "feat: add theme vocabulary resolution and normalization"
```

---

### Task 3: Scorecard Update from Persisted Task

**Files:**
- Modify: `research/lib/learning_scorecard.py`
- Test: `tests/test_learning_scorecard.py`

**Step 1: Write the failing tests**

```python
import json
from pathlib import Path
from research.lib.learning_scorecard import (
    update_scorecard_from_task,
    read_scorecard,
    laplace_rate,
)


def test_update_from_task_bar_affinity(tmp_path):
    sc_path = tmp_path / "scorecard.json"
    lock_path = tmp_path / "scorecard.lock"

    task = {
        "theme_tag": "amt_value_area",
        "bar_config": "tick_610",
        "verdict": "PASS",
        "split": "train",
        "details": {
            "metrics": {"sharpe_ratio": 1.5, "trade_count": 40},
            "gauntlet": {"overall_verdict": "PASS"},
        },
    }
    update_scorecard_from_task(sc_path, lock_path, task=task, theme_universe=["amt_value_area"])
    sc = read_scorecard(sc_path, lock_path)

    bar = sc["bar_config_affinity"]["amt_value_area"]["tick_610"]
    assert bar["attempts"] == 1
    assert bar["search_passes"] == 1
    assert bar["search_rate"] == laplace_rate(1, 1)
    assert bar["selection_passes"] == 0


def test_update_from_task_fail_counts(tmp_path):
    sc_path = tmp_path / "scorecard.json"
    lock_path = tmp_path / "scorecard.lock"

    task = {
        "theme_tag": "amt_value_area",
        "bar_config": "tick_610",
        "verdict": "FAIL",
        "split": "train",
        "details": {
            "metrics": {"sharpe_ratio": 0.5, "trade_count": 10},
            "gauntlet": {
                "overall_verdict": "FAIL",
                "alpha_decay": {"verdict": "FAIL", "msg": "declining"},
                "shuffle_test": {"verdict": "PASS", "msg": "ok"},
            },
        },
    }
    update_scorecard_from_task(sc_path, lock_path, task=task, theme_universe=["amt_value_area"])
    sc = read_scorecard(sc_path, lock_path)

    # fail_counts should be in theme_stats
    assert sc["theme_stats"]["amt_value_area"]["fail_counts"]["alpha_decay"] == 1
    assert "shuffle_test" not in sc["theme_stats"]["amt_value_area"]["fail_counts"]


def test_update_from_task_near_miss(tmp_path):
    sc_path = tmp_path / "scorecard.json"
    lock_path = tmp_path / "scorecard.lock"

    task = {
        "strategy_name": "ib_fade_03",
        "theme_tag": "amt_value_area",
        "bar_config": "tick_610",
        "verdict": "FAIL",
        "split": "train",
        "details": {
            "metrics": {"sharpe_ratio": 1.3, "trade_count": 30},
            "gauntlet": {
                "overall_verdict": "FAIL",
                "walk_forward": {"verdict": "FAIL", "msg": "not enough"},
                "trade_count": {"verdict": "PASS", "msg": "ok"},
            },
        },
    }
    update_scorecard_from_task(sc_path, lock_path, task=task, theme_universe=["amt_value_area"])
    sc = read_scorecard(sc_path, lock_path)

    assert len(sc["near_misses"]) == 1
    nm = sc["near_misses"][0]
    assert nm["strategy"] == "ib_fade_03"
    assert nm["sharpe"] == 1.3
    assert nm["failed_checks"] == ["walk_forward"]


def test_near_miss_cap_and_dedup(tmp_path):
    sc_path = tmp_path / "scorecard.json"
    lock_path = tmp_path / "scorecard.lock"
    universe = ["amt_value_area"]

    # Add 12 near misses (cap is 10)
    for i in range(12):
        task = {
            "strategy_name": f"strat_{i:02d}",
            "theme_tag": "amt_value_area",
            "bar_config": "tick_610",
            "verdict": "FAIL",
            "split": "train",
            "details": {
                "metrics": {"sharpe_ratio": 0.5 + i * 0.1, "trade_count": 30},
                "gauntlet": {
                    "overall_verdict": "FAIL",
                    "shuffle_test": {"verdict": "FAIL", "msg": "random"},
                },
            },
        }
        update_scorecard_from_task(sc_path, lock_path, task=task, theme_universe=universe)

    sc = read_scorecard(sc_path, lock_path)
    assert len(sc["near_misses"]) == 10
    # Should keep highest Sharpe
    sharpes = [nm["sharpe"] for nm in sc["near_misses"]]
    assert sharpes == sorted(sharpes, reverse=True)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_learning_scorecard.py -v -k "update_from_task or near_miss"`
Expected: FAIL — `ImportError`

**Step 3: Write minimal implementation**

Add to `research/lib/learning_scorecard.py`:

```python
from pathlib import Path

from research.lib.coordination import read_json_file, update_json_file

_MAX_NEAR_MISSES = 10
_SCORECARD_DEFAULT = empty_scorecard


def read_scorecard(scorecard_path: Path, lock_path: Path) -> dict[str, Any]:
    """Read scorecard from disk, returning empty scorecard if missing."""
    try:
        return read_json_file(
            json_path=scorecard_path,
            lock_path=lock_path,
            default_payload=_SCORECARD_DEFAULT(),
        )
    except Exception:
        return _SCORECARD_DEFAULT()


def _extract_failed_checks(gauntlet: dict[str, Any]) -> list[str]:
    """Extract names of failed gauntlet checks."""
    failed: list[str] = []
    for name, payload in gauntlet.items():
        if name in {"overall_verdict", "pass_count", "total_tests"}:
            continue
        if isinstance(payload, dict) and str(payload.get("verdict", "")).upper() == "FAIL":
            failed.append(str(name))
    return failed


def _is_search_split(split: str) -> bool:
    return split in {"train", "search"}


def _is_selection_split(split: str) -> bool:
    return split in {"validate", "selection"}


def update_scorecard_from_task(
    scorecard_path: Path,
    lock_path: Path,
    *,
    task: dict[str, Any],
    theme_universe: list[str],
) -> None:
    """Update scorecard bar_config_affinity, fail_counts, and near_misses from a completed task."""
    theme = normalize_theme_tag(task.get("theme_tag"), theme_universe)
    bar_config = str(task.get("bar_config", ""))
    verdict = str(task.get("verdict", ""))
    split = str(task.get("split", ""))
    details = task.get("details") if isinstance(task.get("details"), dict) else {}
    metrics = details.get("metrics") if isinstance(details.get("metrics"), dict) else {}
    gauntlet = details.get("gauntlet") if isinstance(details.get("gauntlet"), dict) else {}
    sharpe = metrics.get("sharpe_ratio")
    strategy_name = str(task.get("strategy_name", ""))
    failed_checks = _extract_failed_checks(gauntlet)
    is_search = _is_search_split(split)
    is_selection = _is_selection_split(split)
    passed = verdict == "PASS"

    def _update(sc: dict[str, Any]) -> dict[str, Any]:
        # --- bar_config_affinity ---
        bca = sc.setdefault("bar_config_affinity", {})
        theme_bars = bca.setdefault(theme, {})
        bar = theme_bars.setdefault(bar_config, empty_bar_entry())
        bar["attempts"] += 1
        if passed and is_search:
            bar["search_passes"] += 1
        if passed and is_selection:
            bar["selection_passes"] += 1
        bar["search_rate"] = laplace_rate(bar["search_passes"], bar["attempts"])
        bar["selection_rate"] = laplace_rate(bar["selection_passes"], bar["attempts"])

        # --- theme_stats.fail_counts ---
        ts = sc.setdefault("theme_stats", {}).setdefault(theme, empty_theme_entry())
        if not passed and failed_checks:
            fc = ts.setdefault("fail_counts", {})
            for check in failed_checks:
                fc[check] = fc.get(check, 0) + 1

        # --- near_misses ---
        if (
            sharpe is not None
            and float(sharpe) > 0
            and failed_checks
            and not passed
        ):
            near_misses = sc.setdefault("near_misses", [])
            dedup_key = (strategy_name, bar_config, split)
            near_misses = [
                nm for nm in near_misses
                if (nm.get("strategy"), nm.get("bar_config"), nm.get("split")) != dedup_key
            ]
            near_misses.append({
                "strategy": strategy_name,
                "theme": theme,
                "bar_config": bar_config,
                "sharpe": float(sharpe),
                "failed_checks": failed_checks,
                "split": split,
            })
            near_misses.sort(key=lambda x: x.get("sharpe", 0), reverse=True)
            sc["near_misses"] = near_misses[:_MAX_NEAR_MISSES]

        return sc

    update_json_file(
        json_path=scorecard_path,
        lock_path=lock_path,
        default_payload=_SCORECARD_DEFAULT(),
        update_fn=_update,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_learning_scorecard.py -v -k "update_from_task or near_miss"`
Expected: PASS

**Step 5: Commit**

```bash
git add research/lib/learning_scorecard.py tests/test_learning_scorecard.py
git commit -m "feat: add scorecard task-level updates (bar affinity, failures, near misses)"
```

---

### Task 4: Scorecard Update from Resolved Handoff

**Files:**
- Modify: `research/lib/learning_scorecard.py`
- Test: `tests/test_learning_scorecard.py`

**Step 1: Write the failing tests**

```python
from research.lib.learning_scorecard import update_scorecard_from_handoff, read_scorecard, laplace_rate


def test_update_from_handoff_search_pass(tmp_path):
    sc_path = tmp_path / "scorecard.json"
    lock_path = tmp_path / "scorecard.lock"

    handoff = {
        "payload": {
            "hypothesis_id": "h_001",
            "strategy_name": "ib_fade_01",
            "theme_tag": "amt_value_area",
        },
        "result": {
            "overall_verdict": "PASS",
            "pass_count": 3,
            "fail_count": 0,
            "task_count": 3,
            "split": "train",
        },
    }
    update_scorecard_from_handoff(sc_path, lock_path, handoff=handoff, theme_universe=["amt_value_area"])
    sc = read_scorecard(sc_path, lock_path)

    ts = sc["theme_stats"]["amt_value_area"]
    assert ts["attempts"] == 1
    assert ts["search_passes"] == 1
    assert ts["search_rate"] == laplace_rate(1, 1)
    assert ts["selection_attempts"] == 0


def test_update_from_handoff_selection_fail(tmp_path):
    sc_path = tmp_path / "scorecard.json"
    lock_path = tmp_path / "scorecard.lock"

    handoff = {
        "payload": {
            "hypothesis_id": "h_002",
            "strategy_name": "ib_fade_02",
            "theme_tag": "amt_value_area",
        },
        "result": {
            "overall_verdict": "FAIL",
            "pass_count": 1,
            "fail_count": 2,
            "task_count": 3,
            "split": "validate",
        },
    }
    update_scorecard_from_handoff(sc_path, lock_path, handoff=handoff, theme_universe=["amt_value_area"])
    sc = read_scorecard(sc_path, lock_path)

    ts = sc["theme_stats"]["amt_value_area"]
    assert ts["attempts"] == 1
    assert ts["selection_attempts"] == 1
    assert ts["selection_passes"] == 0
    assert ts["search_passes"] == 0


def test_update_from_handoff_underexplored(tmp_path):
    sc_path = tmp_path / "scorecard.json"
    lock_path = tmp_path / "scorecard.lock"
    universe = ["amt_value_area", "state_machine", "volatility_compression"]

    handoff = {
        "payload": {"hypothesis_id": "h_003", "theme_tag": "amt_value_area"},
        "result": {"overall_verdict": "PASS", "split": "train", "pass_count": 1, "task_count": 1},
    }
    for i in range(4):
        handoff["payload"]["hypothesis_id"] = f"h_{i:03d}"
        update_scorecard_from_handoff(sc_path, lock_path, handoff=handoff, theme_universe=universe)

    sc = read_scorecard(sc_path, lock_path)
    # state_machine and volatility_compression have 0 attempts => underexplored
    assert "state_machine" in sc["underexplored_themes"]
    assert "volatility_compression" in sc["underexplored_themes"]
    assert "amt_value_area" not in sc["underexplored_themes"]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_learning_scorecard.py -v -k handoff`
Expected: FAIL — `ImportError`

**Step 3: Write minimal implementation**

Add to `research/lib/learning_scorecard.py`:

```python
_UNDEREXPLORED_THRESHOLD = 3


def _recompute_underexplored(
    theme_stats: dict[str, Any],
    theme_universe: list[str],
) -> list[str]:
    """Themes from the universe with fewer than _UNDEREXPLORED_THRESHOLD attempts."""
    return [
        theme for theme in theme_universe
        if theme_stats.get(theme, {}).get("attempts", 0) < _UNDEREXPLORED_THRESHOLD
    ]


def update_scorecard_from_handoff(
    scorecard_path: Path,
    lock_path: Path,
    *,
    handoff: dict[str, Any],
    theme_universe: list[str],
) -> None:
    """Update scorecard theme_stats (family-level) from a resolved handoff."""
    payload = handoff.get("payload") if isinstance(handoff.get("payload"), dict) else {}
    result = handoff.get("result") if isinstance(handoff.get("result"), dict) else {}
    theme = normalize_theme_tag(payload.get("theme_tag"), theme_universe)
    overall_verdict = str(result.get("overall_verdict", ""))
    split = str(result.get("split", ""))
    is_search = _is_search_split(split)
    is_selection = _is_selection_split(split)
    passed = overall_verdict == "PASS"

    def _update(sc: dict[str, Any]) -> dict[str, Any]:
        ts = sc.setdefault("theme_stats", {}).setdefault(theme, empty_theme_entry())
        ts["attempts"] += 1

        if is_search and passed:
            ts["search_passes"] += 1
        if is_selection:
            ts["selection_attempts"] += 1
            if passed:
                ts["selection_passes"] += 1

        ts["search_rate"] = laplace_rate(ts["search_passes"], ts["attempts"])
        ts["selection_rate"] = laplace_rate(
            ts["selection_passes"],
            ts["selection_attempts"],
        )

        sc["underexplored_themes"] = _recompute_underexplored(
            sc.get("theme_stats", {}), theme_universe,
        )
        return sc

    update_json_file(
        json_path=scorecard_path,
        lock_path=lock_path,
        default_payload=_SCORECARD_DEFAULT(),
        update_fn=_update,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_learning_scorecard.py -v -k handoff`
Expected: PASS

**Step 5: Commit**

```bash
git add research/lib/learning_scorecard.py tests/test_learning_scorecard.py
git commit -m "feat: add scorecard handoff-level updates (theme stats, underexplored)"
```

---

### Task 5: Rebuild Scorecard from Logs

**Files:**
- Modify: `research/lib/learning_scorecard.py`
- Test: `tests/test_learning_scorecard.py`

**Step 1: Write the failing test**

```python
import json
from research.lib.learning_scorecard import rebuild_learning_scorecard, read_scorecard


def test_rebuild_from_logs(tmp_path):
    sc_path = tmp_path / "scorecard.json"
    sc_lock = tmp_path / "scorecard.lock"
    exp_path = tmp_path / "experiments.jsonl"
    handoffs_path = tmp_path / "handoffs.json"
    handoffs_lock = tmp_path / "handoffs.lock"

    # Write experiment log entries
    entries = [
        {
            "event": "task_result",
            "strategy_name": "strat_a",
            "bar_config": "tick_610",
            "verdict": "PASS",
            "split": "train",
            "theme_tag": "amt_value_area",
            "metrics": {"sharpe_ratio": 1.8, "trade_count": 50},
            "gauntlet": {"overall_verdict": "PASS"},
        },
        {
            "event": "task_result",
            "strategy_name": "strat_b",
            "bar_config": "volume_2000",
            "verdict": "FAIL",
            "split": "train",
            "theme_tag": "amt_value_area",
            "metrics": {"sharpe_ratio": 0.9, "trade_count": 15},
            "gauntlet": {
                "overall_verdict": "FAIL",
                "alpha_decay": {"verdict": "FAIL"},
            },
        },
    ]
    with open(exp_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    # Write handoffs
    handoffs = {
        "schema_version": "1.0",
        "pending": [],
        "completed": [
            {
                "handoff_type": "validation_request",
                "payload": {
                    "hypothesis_id": "h_001",
                    "strategy_name": "strat_a",
                    "theme_tag": "amt_value_area",
                },
                "result": {
                    "overall_verdict": "PASS",
                    "split": "train",
                    "pass_count": 2,
                    "task_count": 2,
                },
            },
        ],
    }
    with open(handoffs_path, "w") as f:
        json.dump(handoffs, f)

    universe = ["amt_value_area", "state_machine"]
    rebuild_learning_scorecard(
        experiments_path=exp_path,
        handoffs_path=handoffs_path,
        handoffs_lock=handoffs_lock,
        scorecard_path=sc_path,
        scorecard_lock=sc_lock,
        theme_universe=universe,
    )

    sc = read_scorecard(sc_path, sc_lock)

    # Theme stats from handoff
    assert sc["theme_stats"]["amt_value_area"]["attempts"] == 1
    assert sc["theme_stats"]["amt_value_area"]["search_passes"] == 1

    # Bar affinity from task results
    assert sc["bar_config_affinity"]["amt_value_area"]["tick_610"]["attempts"] == 1
    assert sc["bar_config_affinity"]["amt_value_area"]["tick_610"]["search_passes"] == 1

    # Fail counts from task results
    assert sc["theme_stats"]["amt_value_area"]["fail_counts"]["alpha_decay"] == 1

    # Near miss (strat_b: positive sharpe, failed)
    assert len(sc["near_misses"]) == 1
    assert sc["near_misses"][0]["strategy"] == "strat_b"

    # Underexplored
    assert "state_machine" in sc["underexplored_themes"]

    # rebuilt_at is set
    assert sc["rebuilt_at"] is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_learning_scorecard.py::test_rebuild_from_logs -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `research/lib/learning_scorecard.py`:

```python
import json

from research.lib.atomic_io import atomic_json_write


def rebuild_learning_scorecard(
    *,
    experiments_path: Path,
    handoffs_path: Path,
    handoffs_lock: Path,
    scorecard_path: Path,
    scorecard_lock: Path,
    theme_universe: list[str],
) -> None:
    """Rebuild scorecard from scratch by replaying all experiment logs and handoffs."""
    # Start fresh
    sc = empty_scorecard()
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(scorecard_path, sc)

    # Replay task results from experiment log
    if experiments_path.exists():
        with open(experiments_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                if row.get("event") != "task_result":
                    continue
                task = {
                    "strategy_name": row.get("strategy_name", ""),
                    "theme_tag": row.get("theme_tag", ""),
                    "bar_config": row.get("bar_config", ""),
                    "verdict": row.get("verdict", ""),
                    "split": row.get("split", ""),
                    "details": {
                        "metrics": row.get("metrics", {}),
                        "gauntlet": row.get("gauntlet", {}),
                    },
                }
                update_scorecard_from_task(
                    scorecard_path, scorecard_lock,
                    task=task, theme_universe=theme_universe,
                )

    # Replay handoff results
    try:
        handoffs = read_json_file(
            json_path=handoffs_path,
            lock_path=handoffs_lock,
            default_payload={"schema_version": "1.0", "pending": [], "completed": []},
        )
    except Exception:
        handoffs = {"completed": []}

    for row in handoffs.get("completed", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("handoff_type", "")) != "validation_request":
            continue
        update_scorecard_from_handoff(
            scorecard_path, scorecard_lock,
            handoff=row, theme_universe=theme_universe,
        )

    # Stamp rebuilt_at
    update_json_file(
        json_path=scorecard_path,
        lock_path=scorecard_lock,
        default_payload=_SCORECARD_DEFAULT(),
        update_fn=lambda sc: {**sc, "rebuilt_at": datetime.now(timezone.utc).isoformat()},
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_learning_scorecard.py::test_rebuild_from_logs -v`
Expected: PASS

**Step 5: Commit**

```bash
git add research/lib/learning_scorecard.py tests/test_learning_scorecard.py
git commit -m "feat: add scorecard rebuild from experiment logs and handoffs"
```

---

### Task 6: Format Scorecard for Thinker Prompt

**Files:**
- Modify: `research/lib/learning_scorecard.py`
- Test: `tests/test_learning_scorecard.py`

**Step 1: Write the failing test**

```python
from research.lib.learning_scorecard import format_scorecard_prompt, empty_scorecard, laplace_rate


def test_format_empty_scorecard():
    result = format_scorecard_prompt(empty_scorecard())
    assert result == ""


def test_format_scorecard_with_data():
    sc = {
        "schema_version": "1.0",
        "theme_stats": {
            "amt_value_area": {
                "attempts": 12,
                "search_passes": 5,
                "search_rate": laplace_rate(5, 12),
                "selection_attempts": 5,
                "selection_passes": 2,
                "selection_rate": laplace_rate(2, 5),
                "fail_counts": {"alpha_decay": 3, "trade_count": 2},
            },
        },
        "bar_config_affinity": {
            "amt_value_area": {
                "tick_610": {
                    "attempts": 8,
                    "search_passes": 4,
                    "search_rate": laplace_rate(4, 8),
                    "selection_passes": 2,
                    "selection_rate": laplace_rate(2, 8),
                },
            },
        },
        "near_misses": [
            {
                "strategy": "ib_fade_03",
                "theme": "amt_value_area",
                "bar_config": "tick_610",
                "sharpe": 1.32,
                "failed_checks": ["walk_forward"],
                "split": "search",
            },
        ],
        "underexplored_themes": ["volatility_compression"],
    }
    result = format_scorecard_prompt(sc)
    assert "LEARNING_SCORECARD:" in result
    assert "amt_value_area" in result
    assert "12 attempts" in result
    assert "ib_fade_03" in result
    assert "volatility_compression" in result
    assert "EXPLOIT" in result
    assert "EXPLORE" in result
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_learning_scorecard.py -v -k format`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `research/lib/learning_scorecard.py`:

```python
_TOP_THEMES = 3
_TOP_BAR_AFFINITIES = 3
_TOP_FAILURE_SUMMARIES = 3
_TOP_NEAR_MISSES = 5


def format_scorecard_prompt(scorecard: dict[str, Any]) -> str:
    """Format scorecard as a compact text block for the thinker prompt."""
    theme_stats = scorecard.get("theme_stats", {})
    bar_affinity = scorecard.get("bar_config_affinity", {})
    near_misses = scorecard.get("near_misses", [])
    underexplored = scorecard.get("underexplored_themes", [])

    if not theme_stats and not near_misses and not underexplored:
        return ""

    lines: list[str] = ["LEARNING_SCORECARD:", ""]

    # --- Theme performance ---
    if theme_stats:
        lines.append("Theme performance (family-level, smoothed rates):")
        sorted_themes = sorted(theme_stats.items(), key=lambda x: x[1].get("attempts", 0), reverse=True)
        for theme, stats in sorted_themes[:_TOP_THEMES]:
            search_pct = int(stats.get("search_rate", 0) * 100)
            sel_pct = int(stats.get("selection_rate", 0) * 100)
            sel_n = stats.get("selection_attempts", 0)
            sel_part = f" | selection {sel_pct}% (n={sel_n})" if sel_n > 0 else ""
            lines.append(f"  {theme}: {stats.get('attempts', 0)} attempts | search {search_pct}%{sel_part}")
        lines.append("")

    # --- Bar config affinity ---
    bar_rows: list[tuple[str, str, dict[str, Any]]] = []
    for theme, bars in bar_affinity.items():
        for bar, stats in bars.items():
            if stats.get("attempts", 0) > 0:
                bar_rows.append((theme, bar, stats))
    bar_rows.sort(key=lambda x: x[2].get("search_rate", 0), reverse=True)
    if bar_rows:
        lines.append("Bar config affinity (top patterns):")
        for theme, bar, stats in bar_rows[:_TOP_BAR_AFFINITIES]:
            search_pct = int(stats.get("search_rate", 0) * 100)
            sel_pct = int(stats.get("selection_rate", 0) * 100)
            lines.append(f"  {theme} + {bar}: search {search_pct}%, selection {sel_pct}%")
        lines.append("")

    # --- Failure modes ---
    failure_rows: list[tuple[str, list[tuple[str, int]]]] = []
    for theme, stats in theme_stats.items():
        fc = stats.get("fail_counts", {})
        if fc:
            sorted_fc = sorted(fc.items(), key=lambda x: x[1], reverse=True)
            failure_rows.append((theme, sorted_fc))
    if failure_rows:
        lines.append("Dominant failure modes:")
        for theme, checks in failure_rows[:_TOP_FAILURE_SUMMARIES]:
            parts = ", ".join(f"{name} ({count})" for name, count in checks[:3])
            lines.append(f"  {theme}: {parts}")
        lines.append("")

    # --- Near misses ---
    if near_misses:
        lines.append("Near misses (positive Sharpe, failed gauntlet):")
        for nm in near_misses[:_TOP_NEAR_MISSES]:
            checks = ",".join(nm.get("failed_checks", []))
            lines.append(
                f"  {nm['strategy']} [{nm['bar_config']}] "
                f"sharpe={nm['sharpe']:.2f} failed=[{checks}] ({nm.get('split', '')})"
            )
        lines.append("")

    # --- Underexplored ---
    if underexplored:
        lines.append(f"Underexplored themes: {', '.join(underexplored)}")
        lines.append("")

    lines.append(
        "Use this scorecard to decide whether this lane should EXPLOIT a promising\n"
        "direction or EXPLORE an under-tested one. Justify your choice."
    )

    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_learning_scorecard.py -v -k format`
Expected: PASS

**Step 5: Commit**

```bash
git add research/lib/learning_scorecard.py tests/test_learning_scorecard.py
git commit -m "feat: add scorecard prompt formatting with caps"
```

---

### Task 7: Integrate Theme Tag into Orchestrator

**Files:**
- Modify: `scripts/llm_orchestrator.py:811-899` (`_normalize_thinker_brief`)
- Modify: `scripts/llm_orchestrator.py:1635-1660` (`_build_task`)
- Modify: `scripts/llm_orchestrator.py:2082-2098` (thinker schema hint)
- Modify: `configs/missions/alpha-discovery.yaml`
- Modify: `.claude/agents/quant-thinker.md`

**Step 1: Add `theme_tags` to mission YAML**

In `configs/missions/alpha-discovery.yaml`, add after `current_focus`:

```yaml
theme_tags:
  - amt_value_area
  - ib_breakout
  - state_machine
  - orderflow_divergence
  - volatility_compression
```

**Step 2: Normalize `theme_tag` in thinker output**

In `scripts/llm_orchestrator.py`, modify `_normalize_thinker_brief` (~line 888). After the existing `validation_focus` field, add:

```python
    # --- theme_tag normalization ---
    raw_theme = str(payload.get("theme_tag", "")).strip()
    theme_universe = resolve_theme_universe(mission_bar_configs_or_mission)
    # NOTE: theme_universe will be passed from the caller; for now normalize against empty
    # The caller provides it. See integration below.
```

The function signature gains a `mission` parameter (or `theme_universe` list). Add `theme_tag` to the returned dict at line 898.

**Step 3: Propagate `theme_tag` into task dict**

In `_build_task` (~line 1654), inside the `"source"` dict, add:

```python
    "theme_tag": str(kwargs.get("theme_tag", "")),
```

And add `theme_tag` as a top-level field in the task dict (not nested in source) so the validator can read it directly.

**Step 4: Update thinker schema hint**

At line ~2086, update the schema_hint string to include `theme_tag`:

```python
schema_hint=(
    "keys: hypothesis_id, theme_tag, strategy_name_hint, thesis, bar_configs, params_template, "
    "entry_logic, exit_logic, risk_controls, anti_lookahead_checks, validation_focus"
),
```

**Step 5: Update quant-thinker agent**

In `.claude/agents/quant-thinker.md`, add to the non-negotiables section:

```markdown
- Include a `theme_tag` field in your output JSON. Pick from the mission's allowed tags. This tracks which research direction each hypothesis belongs to.
```

**Step 6: Commit**

```bash
git add scripts/llm_orchestrator.py configs/missions/alpha-discovery.yaml .claude/agents/quant-thinker.md
git commit -m "feat: propagate theme_tag from thinker through task pipeline"
```

---

### Task 8: Inject Scorecard into Thinker Prompt

**Files:**
- Modify: `scripts/llm_orchestrator.py:1365-1439` (`_build_thinker_user_prompt`)
- Modify: `scripts/llm_orchestrator.py:2057-2075` (main loop, read scorecard)

**Step 1: Add scorecard read to main loop**

At ~line 2062, after `feedback_items = _build_merged_feedback_items(...)`, add:

```python
        scorecard_context = ""
        try:
            from research.lib.learning_scorecard import read_scorecard, format_scorecard_prompt
            sc = read_scorecard(state_paths["scorecard"], state_paths["scorecard_lock"])
            scorecard_context = format_scorecard_prompt(sc)
        except Exception:
            pass
```

The `state_paths` dict needs `"scorecard"` and `"scorecard_lock"` entries pointing to `research/.state/learning_scorecard.json` and `.lock`.

**Step 2: Pass scorecard context to prompt builder**

Add `scorecard_context: str = ""` parameter to `_build_thinker_user_prompt`. Inject it after the results table (line ~1392):

```python
    if scorecard_context:
        prompt += f"\n{scorecard_context}\n\n"
```

Before the "Design exactly one hypothesis" instruction.

**Step 3: Wire it up at the call site**

At line ~2069, pass the new parameter:

```python
            thinker_user_prompt = _build_thinker_user_prompt(
                mission=active_mission,
                existing_strategies=existing,
                feedback_items=feedback_items,
                runtime_context=runtime_context,
                feature_knowledge=feature_knowledge,
                scorecard_context=scorecard_context,
            )
```

**Step 4: Commit**

```bash
git add scripts/llm_orchestrator.py
git commit -m "feat: inject learning scorecard into thinker prompt"
```

---

### Task 9: Integrate Scorecard Writes into Validator

**Files:**
- Modify: `scripts/research.py:1557-1597` (after complete_task and handoff finalization)
- Modify: `scripts/research.py:1-40` (imports)

**Step 1: Add imports**

At the top of `research.py`, add:

```python
from research.lib.learning_scorecard import (
    update_scorecard_from_task,
    update_scorecard_from_handoff,
    rebuild_learning_scorecard,
    resolve_theme_universe,
)
```

**Step 2: Add scorecard paths to state_paths dict**

Find where `state_paths` is constructed and add:

```python
    "scorecard": state_dir / "learning_scorecard.json",
    "scorecard_lock": state_dir / "learning_scorecard.lock",
```

**Step 3: Rebuild on startup if missing or fresh-state**

Before the main loop, add:

```python
    theme_universe = resolve_theme_universe(mission)
    if not state_paths["scorecard"].exists():
        rebuild_learning_scorecard(
            experiments_path=experiments_path,
            handoffs_path=state_paths["handoffs"],
            handoffs_lock=state_paths["handoffs_lock"],
            scorecard_path=state_paths["scorecard"],
            scorecard_lock=state_paths["scorecard_lock"],
            theme_universe=theme_universe,
        )
```

**Step 4: Update after complete_task (~line 1564)**

After `complete_task()` returns the completed task dict, add:

```python
                completed_row = complete_task(...)
                try:
                    update_scorecard_from_task(
                        state_paths["scorecard"],
                        state_paths["scorecard_lock"],
                        task=completed_row,
                        theme_universe=theme_universe,
                    )
                except Exception:
                    print("WARN: scorecard task update failed", file=sys.stderr)
                    traceback.print_exc()
```

**Step 5: Update after handoff finalization (~line 1570)**

After `_finalize_ready_validation_handoffs()` returns resolved handoffs:

```python
                for handoff in resolved_handoffs:
                    try:
                        update_scorecard_from_handoff(
                            state_paths["scorecard"],
                            state_paths["scorecard_lock"],
                            handoff=handoff,
                            theme_universe=theme_universe,
                        )
                    except Exception:
                        print("WARN: scorecard handoff update failed", file=sys.stderr)
                        traceback.print_exc()
```

**Step 6: Commit**

```bash
git add scripts/research.py
git commit -m "feat: wire scorecard updates into validator loop"
```

---

### Task 10: Propagate theme_tag Through Experiment Logging

**Files:**
- Modify: `scripts/research.py` — wherever `log_experiment` is called with `task_result` events, include `theme_tag` from the task.

**Step 1: Find experiment logging call sites**

Search for `log_experiment` calls that log `task_result` events. These need to include `"theme_tag": task.get("theme_tag", "")` so that `rebuild_learning_scorecard` can find it.

The task dict already carries `theme_tag` (added in Task 7). Ensure it appears in the experiment log record.

**Step 2: Add theme_tag to log records**

In each `log_experiment({"event": "task_result", ...})` call in `research.py`, add:

```python
"theme_tag": str(claimed.get("theme_tag", "")),
```

**Step 3: Commit**

```bash
git add scripts/research.py
git commit -m "feat: include theme_tag in experiment log records for rebuild"
```

---

### Task 11: Full Integration Test

**Files:**
- Test: `tests/test_learning_scorecard.py`

**Step 1: Write end-to-end test**

```python
def test_full_cycle(tmp_path):
    """Simulate: task completes -> handoff resolves -> scorecard reads -> prompt formats."""
    from research.lib.learning_scorecard import (
        update_scorecard_from_task,
        update_scorecard_from_handoff,
        read_scorecard,
        format_scorecard_prompt,
    )

    sc_path = tmp_path / "scorecard.json"
    sc_lock = tmp_path / "scorecard.lock"
    universe = ["amt_value_area", "state_machine", "volatility_compression"]

    # 1. Three tasks complete for one hypothesis
    for bar in ["tick_610", "volume_2000", "time_1m"]:
        task = {
            "strategy_name": "ib_fade_01",
            "theme_tag": "amt_value_area",
            "bar_config": bar,
            "verdict": "PASS" if bar == "tick_610" else "FAIL",
            "split": "train",
            "details": {
                "metrics": {"sharpe_ratio": 1.5 if bar == "tick_610" else 0.3},
                "gauntlet": {
                    "overall_verdict": "PASS" if bar == "tick_610" else "FAIL",
                    **({"alpha_decay": {"verdict": "FAIL"}} if bar != "tick_610" else {}),
                },
            },
        }
        update_scorecard_from_task(sc_path, sc_lock, task=task, theme_universe=universe)

    # 2. Handoff resolves
    handoff = {
        "payload": {"hypothesis_id": "h_001", "theme_tag": "amt_value_area"},
        "result": {
            "overall_verdict": "PASS",
            "split": "train",
            "pass_count": 1,
            "fail_count": 2,
            "task_count": 3,
        },
    }
    update_scorecard_from_handoff(sc_path, sc_lock, handoff=handoff, theme_universe=universe)

    # 3. Read and format
    sc = read_scorecard(sc_path, sc_lock)
    prompt = format_scorecard_prompt(sc)

    # Verify scorecard state
    assert sc["theme_stats"]["amt_value_area"]["attempts"] == 1
    assert sc["theme_stats"]["amt_value_area"]["search_passes"] == 1
    assert sc["bar_config_affinity"]["amt_value_area"]["tick_610"]["search_passes"] == 1
    assert sc["theme_stats"]["amt_value_area"]["fail_counts"]["alpha_decay"] == 2
    assert "state_machine" in sc["underexplored_themes"]
    assert "volatility_compression" in sc["underexplored_themes"]

    # Verify prompt is non-empty and contains key sections
    assert "LEARNING_SCORECARD:" in prompt
    assert "amt_value_area" in prompt
    assert "Underexplored" in prompt
```

**Step 2: Run test**

Run: `uv run pytest tests/test_learning_scorecard.py::test_full_cycle -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `uv run pytest -q`
Expected: all tests PASS, no regressions

**Step 4: Commit**

```bash
git add tests/test_learning_scorecard.py
git commit -m "test: add full-cycle integration test for learning scorecard"
```

---

### Task 12: Add scorecard_lock to orchestrator state_paths

**Files:**
- Modify: `scripts/llm_orchestrator.py` — find where `state_paths` dict is built, add scorecard entries.

**Step 1: Locate state_paths construction**

Search for `state_paths` dict literal in `llm_orchestrator.py`. Add:

```python
    "scorecard": state_dir / "learning_scorecard.json",
    "scorecard_lock": state_dir / "learning_scorecard.lock",
```

**Step 2: Commit**

```bash
git add scripts/llm_orchestrator.py
git commit -m "feat: add scorecard paths to orchestrator state_paths"
```
