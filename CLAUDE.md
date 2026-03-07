# NQ Alpha Discovery

This repository runs an autonomous intraday alpha-discovery loop for NQ futures.

## Core Workflow

- `scripts/llm_orchestrator.py` generates hypotheses and signal modules in `research/signals/`
- `scripts/research.py` validates them on `search_split`, then gates them on `selection_split`
- `scripts/promote.py` is the holdout / promotion path

## Research Contract

- `train` is the search split
- `validate` is the untouched selection gate
- `test` is blocked until promotion
- do not contaminate `validate` or `test`

## Signal Contract

Every generated module must expose:

```python
def generate_signal(df: pl.DataFrame, params: dict[str, Any]) -> np.ndarray
```

Requirements:

- output length == `len(df)`
- dtype `np.int8`
- values strictly in `{-1, 0, 1}`
- no lookahead
- deterministic
- no I/O or networking

Use the helper imports from `research.signals` instead of ad hoc dataframe-to-numpy handling.

## NotebookLM

- each orchestrator lane gets its own persistent notebook
- fresh lane notebooks must be seeded before the hypothesis is accepted
- prefer `--deep-research` for fresh notebooks or new directions
- ask for high-quality trusted sources: exchange/operator docs, academic papers, serious market-structure research, broker/execution research, and technical references
- avoid low-signal forum chatter, script marketplaces, and recycled summaries

## Important Files

- mission: `configs/missions/alpha-discovery.yaml`
- agent runtime config: `configs/agents/llm_orchestrator.yaml`
- feature catalog: `research/feature_catalog.md`
- agent guide: `docs/AGENT_GUIDE.md`

## Hard Boundaries

- do not edit `src/framework/` from research roles
- do not access `test` in research mode
- do not bypass queue, logging, validation, or candidate-writing rules
