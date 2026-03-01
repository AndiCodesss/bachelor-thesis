# NQ Alpha — Fortress Framework

## Architecture

- Immutable evaluation framework: `src/framework/` (13 files, SHA-256 locked)
- Mutable research workspace: `research/`
- Lock manifest: `configs/framework_lock.json`

## Entry Point

- `scripts/research.py` — Autonomous research loop (holdout blocked)

## Data

- Instrument: NQ E-mini Nasdaq-100 futures
- Source: Databento MBP1 (Level-1 quotes + trades, nanosecond precision)
- Path: set `NQ_DATA_PATH` env var

## Splits (enforced at runtime)

- TRAIN: Mar 2023 -- Aug 2024 (6 regime folders)
- VALIDATE: Feb 2025 -- Jun 2025 (5 month folders)
- TEST: Jul 2025 -- Jan 2026 (blocked in research mode)

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
