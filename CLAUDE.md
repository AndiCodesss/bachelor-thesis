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
- NotebookLM is optional, not a blocker: at most one `--research` query and three total notebook queries per iteration
- `--deep-research` is disabled in the autonomy loop because it is too slow
- ask for high-quality trusted sources: exchange/operator docs, academic papers, serious market-structure research, broker/execution research, and technical references
- avoid low-signal forum chatter, script marketplaces, and recycled summaries
- the thinker agent preloads the project skill `notebook-alpha-research` to decide how to query NotebookLM and turn research into one hypothesis

## Important Files

- mission: `configs/missions/alpha-discovery.yaml`
- agent runtime config: `configs/agents/llm_orchestrator.yaml`
- feature catalog: `research/feature_catalog.md`
- agent guide: `docs/AGENT_GUIDE.md`
- Claude project skills: `.claude/skills/<skill-name>/SKILL.md`

## Hard Boundaries

- do not edit `src/framework/` from research roles
- do not access `test` in research mode
- do not bypass queue, logging, validation, or candidate-writing rules
