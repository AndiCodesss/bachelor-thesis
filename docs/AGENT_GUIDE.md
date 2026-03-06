# Agent Guide

## Objective

Agents can freely research alpha while framework integrity and holdout discipline remain enforced.

## Hard Boundaries

1. Do not edit `src/framework/`.
2. Do not access `test` split in research mode.
3. Do not manually shift strategy signals.
4. Do not bypass contract checks, logging, or validation.
5. Do not write candidates from non-validator roles.

## Creative Scope

Agents are expected to be creative in:

1. Strategy logic in `research/signals/`.
2. Feature confluence and event definitions.
3. Bar-config choice and parameter exploration.
4. Model variants and quality gates in `research/`.

## Workspace

- Signals: `research/signals/`
- Runtime state: `research/.state/`
- Candidates: `research/candidates/` (created at runtime)
- Event log: `results/logs/research_experiments.jsonl` (created at runtime)

## Signal Contract

Each strategy file must expose:

```python
def generate_signal(df: pl.DataFrame, params: dict) -> np.ndarray
```

Rules:

- output shape: `(len(df),)`
- values: `-1, 0, 1`
- no NaN
- no lookahead

## Research Loop

Run:

```bash
uv run python scripts/research.py \
  --mission configs/missions/alpha-discovery.yaml \
  --auto-mode \
  --worker-agent validator
```

What happens:

1. set mode to `RESEARCH`
2. verify framework lock
3. run signal contract tests
4. initialize queue/handoffs/budget state files
5. bootstrap tasks from `research/signals/` when queue is empty (unless `--no-bootstrap`)
6. claim pending tasks by priority
7. run the strategy on the mission `search_split` and write search artifacts/feedback
8. if search passes, rerun the exact same strategy on `selection_split` before candidate promotion
9. complete task with search feedback plus final selection-gated verdict/details
10. write run summary under `results/runs/<run_id>/summary.json`

## Full Autonomy (Generator + Validator)

Run both processes:

```bash
set -a && source .env && set +a
claude auth login  # one-time setup for Claude Max subscription

# Recommended: launch both in tmux with one command + live dashboard
uv run python scripts/launch_autonomy.py --lane-count 2
# By default Ctrl+C shuts down both workers gracefully (use --keep-running to detach).
# Use --fresh-state only when you explicitly want a new runtime state.

# Process 1: LLM generator (writes research/signals + enqueues tasks)
uv run python scripts/llm_orchestrator.py \
  --mission configs/missions/alpha-discovery.yaml \
  --lane A \
  --resume

# Process 2: validator worker (claims tasks and evaluates)
uv run python scripts/research.py \
  --mission configs/missions/alpha-discovery.yaml \
  --auto-mode \
  --worker-agent validator \
  --resume
```

`scripts/llm_orchestrator.py` uses a staged LLM chain:
1. `feedback_analyst` (summarizes recent validator outcomes),
2. `quant_thinker` (proposes one structured hypothesis),
3. `coder` (implements only that hypothesis as signal code).

The generator calls Claude through the local `claude` CLI (`provider: claude_cli`)
rather than direct API billing.

It uses lock-safe queue updates and writes audit logs to
`results/logs/llm_orchestrator*.jsonl`.
Role models and temperatures are configured in
`configs/agents/llm_orchestrator.yaml`.
Stage retries, JSON-repair attempts, and backoff windows are also configured in
that same file under `runtime`.

## Per-Task Protocol

Each task should contain:

1. `strategy_name`
2. `split` (research-safe, usually `validate`)
3. `bar_config` (e.g., `tick_610`, `volume_2000`, `time_1m`)
4. `params`
5. optional risk/backtest controls (`stop_loss`, `profit_target`, `exit_bars`, `max_daily_loss`)

Task verdicts:

1. `PASS`: validation criteria satisfied
2. `FAIL`: ran successfully but did not meet criteria
3. `ERROR`: runtime/contract/processing failure
4. `NEEDS_WORK` / `ABANDON`: optional strategic labels when applicable

## Candidate Promotion

Candidates that pass the validation gauntlet are written as immutable JSON artifacts to `research/candidates/`. Promotion to the holdout test set verifies artifact hashes and lock provenance before granting access.
