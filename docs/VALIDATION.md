# Validation Methodology

## Split Discipline

- Research runs only on `train`/`validate`.
- Holdout `test` is blocked at API level in research mode.
- Promotion is the first stage allowed to access `test`.

## Robustness Metrics

`src/framework/validation/robustness.py`:

- `deflated_sharpe_ratio(returns, n_trials)` — corrects for multiple testing
- `estimate_pbo(score_matrix)` — probability of backtest overfitting
- `adversarial_validation_report(train_df, holdout_df, feature_cols, seed)` — regime drift detection

## Alpha Decay

`src/framework/validation/alpha_decay.py`:

- `fit_alpha_decay(trades)` — exponential fit on rolling Sharpe
- Verdicts: STABLE, DECAYING, NOISY, DEAD, INSUFFICIENT_DATA
- Half-life metric quantifies signal degradation rate

## Factor Attribution

`src/framework/validation/factor_attribution.py`:

- `factor_attribution(trades, bars)` — OLS decomposition into market, volatility, momentum factors
- `compute_factor_returns(bars)` — daily factor return series
- Verdicts: PURE_ALPHA, ALPHA_WITH_BETA_EXPOSURE, FACTOR_EXPOSED, INCONCLUSIVE, INSUFFICIENT_DATA
- Uses Holm-Bonferroni-adjusted p-values before assigning exposure verdicts
- Isolates true alpha from common factor exposure

## Adaptive Transaction Costs

`src/framework/backtest/costs.py`:

- `CostModel` — parameterized cost function (spread, volatility, volume, time-of-day)
- `compute_adaptive_costs(trades, bars)` — per-trade cost estimation
- Replaces flat $14.50 with market-condition-aware costs

## Trial Counting

`research/lib/trial_counter.py`:

- Tracks cumulative hypotheses tested for Deflated Sharpe correction
- Atomic persistence with portalocker sidecar locks
- Syncs with `results/logs/research_experiments.jsonl` (monotonic, never decreases)
- `estimate_effective_trials(...)` computes correlation-adjusted effective trials
  via a conservative sqrt-family aggregation to reduce sweep overcounting

Discovery missions can optionally enforce these stronger checks at research time
via `mission.advanced_validation` gates (DSR floor, allowed alpha-decay verdicts,
and allowed factor-attribution verdicts).

## Validation Gauntlet (7 tests)

`src/framework/backtest/validators.py`:

1. **Shuffle test** — random signal permutation baseline
2. **Walk-forward** — rolling window out-of-sample
3. **Regime stability** — performance across market regimes
4. **Signal perturbation** — degradation when a small fraction of live signals are flipped
5. **Cost sensitivity** — survival under 1.5x transaction costs
6. **Alpha decay** — signal half-life estimation
7. **Trade count** — minimum sample size threshold

## Promotion Gates

`research/lib/promotion_gates.py` runs month-based walk-forward evaluation with:

- rolling 12-month train / 2-month test folds (2-month step, non-overlapping tests)
- optional embargo months and day-level purge at fold boundaries
- fold-level Sharpe tracking and positive-fold ratio checks
- aggregate Sharpe on combined out-of-sample daily returns
- Deflated Sharpe adjustment using tracked trial counts
- optional final lockbox gate on untouched trailing months

`scripts/promote.py` performs artifact verification and evaluates these gates for
candidate promotion decisions.

## Integrity Verification

```bash
uv run python scripts/framework/verify_lock.py --manifest configs/framework_lock.json --mode error
```
