# NQ Alpha — Fortress Framework

## Architecture

- Immutable evaluation framework: `src/framework/` (28 files, SHA-256 locked)
- Mutable research workspace: `research/`
- Lock manifest: `configs/framework_lock.json`

## Entry Point

- `scripts/research.py` — Autonomous research loop (holdout blocked)

## Data

- Instrument: NQ E-mini Nasdaq-100 futures
- Source: Databento MBP1 (Level-1 quotes + trades, nanosecond precision)
- Path: set `NQ_DATA_PATH` env var

## Splits (enforced at runtime)

- TRAIN: Oct 2022 -- Aug 2024 (23 month folders)
- VALIDATE: Sep 2024 -- Mar 2025 (7 month folders)
- TEST: Apr 2025 -- Feb 2026 (blocked in research mode)

## Transaction Costs

- Commission: $4.50 RT
- Slippage: 1 tick/side ($5.00)
- Total: $14.50 RT (or adaptive via `CostModel`)

## Commands

```bash
uv sync                                                    # Install deps
uv run pytest -q                                           # Run tests
uv run python scripts/framework/verify_lock.py \
  --manifest configs/framework_lock.json --mode error      # Lock check
uv run python scripts/research.py \
  --mission configs/missions/alpha-discovery.yaml           # Research loop
```

## Key Modules

- Stable API: `src/framework/api.py`
- Backtest engine: `src/framework/backtest/engine.py`
- Validation gauntlet: `src/framework/backtest/validators.py`
- Canonical features (221): `src/framework/features_canonical/`
- Constants: `src/framework/data/constants.py`
- Signal workspace: `research/signals/`
