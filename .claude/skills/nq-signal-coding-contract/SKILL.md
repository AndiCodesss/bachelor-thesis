---
name: nq-signal-coding-contract
description: Use when the autonomy coder must convert a structured NQ hypothesis handoff into a valid signal module that obeys the repo contract, helper API, and anti-lookahead rules.
---

# NQ Signal Coding Contract

Use this skill during the coder stage when implementing one signal module from a structured handoff.

## Goal

Produce one valid repo signal module without inventing extra strategy logic.

## Required Shape

The module must define:

- `DEFAULT_PARAMS`
- `STRATEGY_METADATA`
- `generate_signal(df, params)`

The final returned signal must:

- have length `len(df)`
- be dtype `np.int8`
- contain only `-1`, `0`, `1`
- be deterministic
- avoid lookahead entirely

## Imports

Keep imports minimal and within the repo contract:

- `from __future__ import annotations`
- `typing`
- `numpy as np`
- `polars as pl`
- `from research.signals import safe_f64_col, session_start_mask, signal_from_conditions`

Do not add I/O, network, subprocess, or filesystem imports.

## Helper Discipline

Prefer repo helpers over ad hoc dataframe extraction.

Use:

- `safe_f64_col(df, "...", fill=0.0)`
- `session_start_mask(df)`
- `signal_from_conditions(long_cond, short_cond)`

Avoid:

- `df["col"].to_numpy()`
- `np.nan_to_num(..., copy=False)`
- mutating Polars-backed arrays unless you created an explicit copy

## Logic Discipline

- Implement the handoff exactly.
- The engine handles exits; `generate_signal` is for entries.
- Reset sequential state at session boundaries.
- Prefer precomputed features over recomputing indicators.
- Never use `shift(-1)` or any future-bar dependency.

## Final Check

Before returning:

1. confirm the code uses only allowed imports
2. confirm helper usage is safe
3. confirm no lookahead
4. confirm the final array is `np.int8`
5. confirm the output is only `-1`, `0`, `1`

Return only the required JSON object with `strategy_name`, `bar_configs`, `params`, and `code`.
