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
- Verdicts: PURE_ALPHA, FACTOR_EXPOSED, INSUFFICIENT_DATA
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

## Validation Gauntlet (7 tests)

`src/framework/backtest/validators.py`:

1. **Shuffle test** — random signal permutation baseline
2. **Walk-forward** — rolling window out-of-sample
3. **Regime stability** — performance across market regimes
4. **Parameter sensitivity** — degradation under perturbation
5. **Cost sensitivity** — survival under 2x transaction costs
6. **Alpha decay** — signal half-life estimation
7. **Trade count** — minimum sample size threshold

## Integrity Verification

```bash
uv run python scripts/framework/verify_lock.py --manifest configs/framework_lock.json --mode error
```
