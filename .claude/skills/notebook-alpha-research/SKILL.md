---
name: notebook-alpha-research
description: Use when the NQ autonomy thinker must choose a research direction, query NotebookLM, run deep research, judge evidence quality, or convert NotebookLM findings into one falsifiable intraday alpha hypothesis.
---

# Notebook Alpha Research

Use this skill during the thinker stage when the task involves NotebookLM-backed research for one NQ intraday alpha direction.

## Goal

Turn NotebookLM research into one concrete hypothesis, not a vague theme list.

## Workflow

1. Choose one direction from the mission, recent failures, and notebook state.
2. Ask one pointed `--research` question if the notebook needs new sources.
3. Use plain notebook queries only for narrow follow-ups on that same direction.
4. Do not use `--deep-research` in the autonomy loop; the runtime disables it because it is too slow for iteration-time use.
5. Ask explicitly for high-quality trusted sources:
   - exchange or operator documentation
   - academic papers
   - serious market-structure or execution research
   - broker or execution studies
   - technical references with real methodology
6. Avoid spending time on generic strategy summaries. If the research comes back shallow, pivot the hypothesis, not the query budget.
7. Extract only the evidence that changes the trade design:
   - structural level or regime condition
   - orderflow / footprint confirmation
   - expected failure mode
   - causal rationale
8. Convert the evidence into one falsifiable hypothesis with:
   - setup
   - trigger
   - invalidation
   - risk asymmetry
   - specific repo features that can express it

## Query Patterns

Prefer one pointed question over broad brainstorming.

Good patterns:

- "For NQ RTH, what market-structure evidence supports fading failed auction probes back into value after value-area rejection? Prefer exchange docs, academic microstructure research, and serious execution studies."
- "What evidence suggests absorption plus unfinished auction behavior around prior-session value extremes leads to short-horizon mean reversion in equity index futures? Prefer primary or technical sources."
- "What conditions make opening-drive continuation robust versus fragile in NQ intraday trading? Prefer methodology-rich sources."

Weak patterns:

- "Find good NQ strategies."
- "What indicators work for scalping?"
- "Give me some trading ideas."

## Evidence Standard

Do not trust a direction just because multiple sources repeat it.

Stronger evidence:

- mechanism-level explanation
- data or methodology
- microstructure / execution framing
- concrete conditions under which the pattern fails

Weaker evidence:

- generic retail indicator advice
- recycled listicles
- script marketplace explanations
- source text with no mechanism or methodology

## Output Standard

Finish with one hypothesis only.

That hypothesis should be:

- sparse
- causal
- expressible with repo features
- robust enough to justify coding
- assigned to exactly one concise `theme_tag` in snake_case
- reuse a current-focus anchor when it fits; otherwise mint a more precise tag if the evidence clearly justifies it

If research does not justify a strong idea, use the notebook you already have and finalize the best falsifiable hypothesis you can defend within the iteration budget.
