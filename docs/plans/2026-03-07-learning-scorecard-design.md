# Learning Scorecard Design

Derived intelligence layer for the autonomy loop. Makes the thinker aware of what has been tried, what works, what fails, and what remains unexplored.

## Principles

- **Derived cache, not sacred state.** Rebuildable from experiment logs + handoffs at any time. Delete it and it regenerates.
- **Validator owns writes.** Updates happen in `research.py` after persisted task completion and handoff finalization. The orchestrator only reads.
- **Mission YAML stays immutable.** The scorecard does not mutate `current_focus` or any mission field. Human intent and machine learning are separate layers.
- **Structured codes, not NLP.** Failure modes use gauntlet check names. Theme tags use a controlled vocabulary. No fuzzy clustering.
- **Compact prompt injection.** The thinker sees a summary, not the full JSON.

## Schema: `research/.state/learning_scorecard.json`

```json
{
  "schema_version": "1.0",
  "rebuilt_at": "2026-03-07T14:00:00Z",
  "theme_stats": {
    "amt_value_area": {
      "attempts": 12,
      "search_passes": 5,
      "search_rate": 0.43,
      "selection_attempts": 5,
      "selection_passes": 2,
      "selection_rate": 0.43,
      "fail_counts": {
        "alpha_decay": 3,
        "shuffle_test": 1,
        "trade_count": 2,
        "walk_forward": 1
      }
    }
  },
  "bar_config_affinity": {
    "amt_value_area": {
      "tick_610": {
        "attempts": 8,
        "search_passes": 4,
        "search_rate": 0.50,
        "selection_passes": 2,
        "selection_rate": 0.30
      }
    }
  },
  "near_misses": [
    {
      "strategy": "ib_fade_vol_filter_03",
      "theme": "amt_value_area",
      "bar_config": "tick_610",
      "sharpe": 1.32,
      "failed_checks": ["walk_forward"],
      "split": "search"
    }
  ],
  "underexplored_themes": ["volatility_compression"]
}
```

### Counting rules

- **`theme_stats`**: Family-level. One count per hypothesis family (keyed by `hypothesis_id` or resolved handoff), not per bar-config task. Multi-bar hypotheses count as one attempt.
  - `attempts`: number of hypothesis families attempted for this theme.
  - `search_passes`: families where at least one bar config passed search gauntlet.
  - `search_rate`: Laplace-smoothed `(search_passes + 1) / (attempts + 2)`.
  - `selection_attempts`: families that reached selection gate.
  - `selection_passes`: families that passed selection gate.
  - `selection_rate`: Laplace-smoothed `(selection_passes + 1) / (selection_attempts + 2)`.
  - `fail_counts`: per gauntlet check name, count of task-level failures across all families. Aggregated from per-task `failed_checks`.

- **`bar_config_affinity`**: Task-level. One count per bar-config task result.
  - Keyed by `theme -> bar_config`.
  - Tracks `attempts`, `search_passes`, `search_rate`, `selection_passes`, `selection_rate` independently.

- **`near_misses`**: Top 10 by Sharpe, deduped by `(strategy_name, bar_config, split)`. Only strategies with positive Sharpe that failed at least one gauntlet check.

- **`underexplored_themes`**: Themes from the allowed theme universe (mission `theme_tags` or slugged `current_focus`) with fewer than 3 family-level attempts.

### Smoothing

All rates use Laplace smoothing: `(passes + 1) / (attempts + 2)`.

This prevents `1/1 = 100%` from being treated as strong evidence.

## Theme Tags

### Thinker output change

One new required field in the thinker JSON output:

```json
{
  "hypothesis_id": "h_042",
  "theme_tag": "amt_value_area",
  "strategy_name_hint": "...",
  "thesis": "...",
  ...
}
```

### Vocabulary

Controlled list derived from the mission. Resolution order:

1. **Explicit**: mission YAML field `theme_tags` (list of slug strings), if present.
2. **Fallback**: slugged `current_focus` bullets (`"AMT value area rejection and rotation"` -> `"amt_value_area"`).
3. **Unknown**: any tag not in the vocabulary gets bucketed into `"other"`.

The orchestrator normalizes the thinker's output to match the vocabulary before propagating the tag into the task payload.

### Propagation

`theme_tag` flows: thinker output -> task payload -> experiment log -> scorecard.

The tag is written into the task dict by the orchestrator at enqueue time. The validator reads it from the task when updating the scorecard.

## Write Path (Validator)

Updates happen in `research.py` at two points, consuming **persisted state only** (not in-memory execution variables):

### After `complete_task()` (~line 1557)

Read the completed task row from the queue file. Extract:
- `theme_tag`, `bar_config`, `verdict`, `failed_checks` from the persisted task.
- Update `bar_config_affinity` (task-level stats).
- Update `theme_stats.fail_counts` (task-level failure modes).
- Update `near_misses` if positive Sharpe + failed checks.

### After `_finalize_ready_validation_handoffs()` (~line 1565)

Read the resolved handoff result. Extract:
- `theme_tag` from handoff payload.
- `overall_verdict`, `pass_count`, `fail_count` from handoff result.
- Whether this was search or selection split.
- Update `theme_stats` family-level counts (attempts, search_passes, selection_passes).
- Update `bar_config_affinity` selection-level stats from per-bar results in the handoff.

### Function signature

```python
def update_learning_scorecard(
    scorecard_path: Path,
    scorecard_lock: Path,
    *,
    task: dict | None = None,
    handoff: dict | None = None,
    theme_universe: list[str],
) -> None:
```

Exactly one of `task` or `handoff` is provided per call. `theme_universe` is the allowed tag list from the mission.

## Read Path (Orchestrator)

One new function in `llm_orchestrator.py`:

```python
def _build_learning_context(
    scorecard_path: Path,
    scorecard_lock: Path,
) -> str:
```

Returns a compact text block for the thinker prompt. Injected after the results table, before the "Design exactly one hypothesis" instruction.

### Prompt format

```
LEARNING_SCORECARD:

Theme performance (family-level, smoothed rates):
  amt_value_area: 12 attempts | search 43% | selection 21%
  state_machine: 4 attempts | search 20% | selection 0%
  breakout_fade: 8 attempts | search 38% | selection 14%

Bar config affinity (top patterns):
  amt_value_area + tick_610: search 50%, selection 30%
  breakout_fade + volume_2000: search 40%, selection 0%

Dominant failure modes:
  amt_value_area: alpha_decay (3), trade_count (2)
  state_machine: shuffle_test (3), walk_forward (2)

Near misses (positive Sharpe, failed gauntlet):
  ib_fade_vol_filter_03 [tick_610] sharpe=1.32 failed=[walk_forward] (search)
  state_seq_probe_07 [volume_2000] sharpe=1.10 failed=[trade_count] (search)

Underexplored themes: volatility_compression, orderflow_divergence

Use this scorecard to decide whether this lane should EXPLOIT a promising
direction or EXPLORE an under-tested one. Justify your choice.
```

### Caps

- Top 3 theme rows by attempt count.
- Top 3 bar config affinity rows by search rate.
- Top 3 failure mode summaries.
- Top 5 near misses by Sharpe.
- All underexplored themes.

## Rebuild Path

```python
def rebuild_learning_scorecard(
    experiments_path: Path,
    handoffs_path: Path,
    handoffs_lock: Path,
    scorecard_path: Path,
    scorecard_lock: Path,
    theme_universe: list[str],
) -> None:
```

Scans the full experiment JSONL + all completed handoffs. Recomputes every stat from scratch.

### Triggers

- Scorecard file missing at validator startup.
- Explicit maintenance command.
- `--fresh-state` flag (not `--no-resume`).

## File Ownership

| File | Owner | Reads | Writes |
|---|---|---|---|
| `alpha-discovery.yaml` | Human | orchestrator, validator | never |
| `research_experiments.jsonl` | Validator | orchestrator (feedback), rebuild | validator |
| `handoffs.json` | Validator | orchestrator (feedback), rebuild | validator |
| `learning_scorecard.json` | Validator | orchestrator (thinker prompt) | validator |
| Thinker `theme_tag` field | Orchestrator | validator (via task payload) | orchestrator |

## What This Does NOT Do

- No NLP clustering or embeddings.
- No thinker-written lessons or playbook entries.
- No mission YAML mutation.
- No cross-lane coordination (lanes read the same scorecard, make independent exploit/explore decisions).
- No vector similarity search.

## Changes by File

### New files
- `research/lib/learning_scorecard.py` — `update_learning_scorecard()`, `rebuild_learning_scorecard()`, `read_learning_scorecard()`, smoothing math, near-miss dedup logic.

### Modified files
- `scripts/research.py` — call `update_learning_scorecard()` after `complete_task()` and after `_finalize_ready_validation_handoffs()`. Call `rebuild_learning_scorecard()` on startup if file missing or `--fresh-state`.
- `scripts/llm_orchestrator.py` — add `_build_learning_context()`, inject into `_build_thinker_user_prompt()`. Normalize `theme_tag` from thinker output. Propagate tag into task payload.
- `configs/missions/alpha-discovery.yaml` — add optional `theme_tags` field.
- `.claude/agents/quant-thinker.md` — document `theme_tag` field requirement.
