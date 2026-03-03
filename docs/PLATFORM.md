# Platform Architecture

## Layers

- `src/framework/`: immutable evaluation layer (hash-locked via `configs/framework_lock.json`)
  - `backtest/`: engine, metrics, validation gauntlet, adaptive transaction costs
  - `data/`: constants, splits, loader with execution-mode firewall
  - `features_canonical/`: 17 feature modules
  - `validation/`: robustness metrics, alpha decay modeling, factor attribution
  - `security/`: framework lock build/verify
  - `api.py`: stable public API surface
- `research/`: agent workspace for signals, experiments, candidates, and state
  - `signals/`: strategy signal generators
  - `lib/`: atomic IO, coordination, budget, trial counter, candidates

## Execution Modes

`src/framework/data/loader.py` enforces mode:

- `ExecutionMode.RESEARCH`: `get_parquet_files("test")` is forbidden.
- `ExecutionMode.PROMOTION`: holdout access allowed.

Entrypoints set mode exactly once per process.

## Framework Lock

- Manifest: `configs/framework_lock.json` (28 locked files)
- Build: `scripts/framework/build_lock.py`
- Verify: `scripts/framework/verify_lock.py`

Research and promotion entrypoints verify lock before running.

## Autonomous Roles

- `scripts/llm_orchestrator.py`: Claude Code CLI-driven generator role. Produces/updates
  `research/signals/*.py` and enqueues validation tasks.
  Internal roles: `feedback_analyst -> quant_thinker -> coder`.
- `scripts/research.py --worker-agent validator`: execution role. Claims tasks,
  runs backtest + gauntlet, writes verdicts/candidates.

This separation keeps generation and evaluation responsibilities isolated.

## Runtime Coordination

`research/lib/coordination.py`:

- sidecar file locks (`*.lock`) via `portalocker`
- atomic JSON writes (temp + fsync + os.replace + parent fsync)
- task claim/complete state transitions
- lease + heartbeat renewal
- watchdog timeout reclaim/fail

`research/lib/budget.py` persists mission budget across restarts.

## Candidate Immutability

`research/lib/candidates.py`:

- only `agent_name="validator"` can write candidates
- write-once semantics (no overwrite)
- file made read-only on write (best effort)

`research/lib/promotion.py` verifies candidate artifact hashes and lock provenance before promotion.
