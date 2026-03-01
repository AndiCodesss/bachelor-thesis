# Do Microstructure Features Help LLMs Find Better Alpha?

## Overview

This repository contains the **alpha-discovery platform** for an A/B experiment
on NQ E-mini futures. An autonomous LLM agent writes Python signal functions,
submits them to a locked evaluation engine, reads the results, and iterates —
searching for robust intraday trading signals without human intervention.

The central question: does giving the agent microstructure features derived from
Level-1 market data (order flow imbalance, micro-price, cumulative volume delta)
lead to measurably better signals than restricting it to standard OHLCV bars?

## System Architecture

```mermaid
flowchart LR
    %% Define styles for light mode / dark mode safety
    style Workspace fill:#fbfbfb,stroke:#333,stroke-dasharray: 5 5,stroke-width:2px,color:#000
    style Framework fill:#f4f6f7,stroke:#2b5c8f,stroke-width:2px,color:#000
    style Agent fill:#d4e6f1,stroke:#1f618d,stroke-width:2px,color:#000
    style SignalFunction fill:#d5f5e3,stroke:#1e8449,stroke-width:2px,color:#000
    style DataLayer fill:#fcf3cf,stroke:#b7950b,stroke-width:2px,color:#000
    style BacktestEngine fill:#fadbd8,stroke:#922b21,stroke-width:2px,color:#000
    style ValidationGauntlet fill:#e8daef,stroke:#5b2c6f,stroke-width:2px,color:#000
    style Results fill:#f9e79f,stroke:#d4ac0d,stroke-width:2px,color:#000

    subgraph Workspace ["MUTABLE Workspace"]
        direction TB
        Agent["LLM Agent<br>Iteratively writes<br>hypotheses"]
        SignalFunction("Signal Function<br>Returns {-1, 0, +1}<br>array per bar<br>for feature matrix")
        Agent -- "Writes/Updates" --> SignalFunction
    end

    subgraph Framework ["IMMUTABLE Engine (SHA-256 locked)"]
        direction TB
        DataLayer[("Data Layer<br>1. Loads MBP1 Data<br>2. Builds Bars<br>3. Computes Feats<br>4. Blocks Test Split")]

        BacktestEngine["Backtest Engine<br>1. Bar-by-bar<br>2. Entry: Next Open<br>3. PT/SL: High/Low<br>4. Costs: $14.50 RT"]

        ValidationGauntlet["Validation Gauntlet<br>1. Shuffle Test<br>2. Walk-forward<br>3. Regime Stability<br>4. Param Sensitivity<br>5. Cost Sensitivity<br>6. Time Decay<br>7. Sample Size"]

        Results["Metrics Output<br>Sharpe, PF,<br>MDD, Def. Sharpe"]

        DataLayer -- "Feeds DF<br>(Features)" --> BacktestEngine
        BacktestEngine -- "Equity Curve<br>& Trade Log" --> ValidationGauntlet
        ValidationGauntlet -- "7-Test Verdict<br>& Output" --> Results
    end

    SignalFunction -.-> |"Reads DF"| DataLayer
    SignalFunction --> |"Submits Signal"| BacktestEngine
    Results --> |"Verdict /<br>Feedback Loop"| Agent
```

## A/B Experiment Design

The experiment compares two groups that differ only in the input feature space
available to the LLM agent:

|                             | Group A (OHLCV)                 | Group B (OHLCV + MBP1)                 |
| --------------------------- | ------------------------------- | -------------------------------------- |
| **Price bars**              | Open, High, Low, Close, Volume  | Same                                   |
| **Technical indicators**    | EMAs, RSI, ATR, Bollinger Bands | Same                                   |
| **Microstructure features** | —                               | OFI, Micro-Price, CVD, Queue Imbalance |
| **Evaluation engine**       | Identical                       | Identical                              |
| **Transaction costs**       | $14.50 RT                       | $14.50 RT                              |

Same agent, same engine, same rules — only the feature space changes.

## Repository Structure

```
.
├── src/framework/                        # Immutable evaluation layer (13 hash-locked files)
│   ├── api.py                            # Stable public API surface
│   ├── backtest/
│   │   ├── engine.py                     # Bar-by-bar backtest with PT/SL/time-stop
│   │   ├── metrics.py                    # 16 financial metrics (Sharpe, Sortino, PF, MDD, …)
│   │   ├── validators.py                 # 7-test validation gauntlet
│   │   └── costs.py                      # Adaptive transaction cost model
│   ├── data/
│   │   ├── loader.py                     # Parquet loader with ExecutionMode firewall
│   │   ├── constants.py                  # Tick size, costs, splits, thresholds
│   │   ├── bars.py                       # Time and tick bar aggregation
│   │   ├── volume_bars.py                # Volume bar aggregation
│   │   └── splits.py                     # Train / validate / test date ranges
│   ├── features_canonical/               # 17 modules → 221 features
│   │   ├── orderflow.py                  # OFI, buy/sell pressure, volume imbalance
│   │   ├── book.py                       # Depth imbalance, bid-ask spread, book skew
│   │   ├── microstructure.py             # Tape speed, whip bars, recoil, VPIN
│   │   ├── microstructure_v2.py          # Trade clustering, size anomalies
│   │   ├── aggressor.py                  # CVD, CVD divergence, extreme aggression
│   │   ├── momentum.py                   # Returns, EMAs, RSI, rate-of-change
│   │   ├── toxicity.py                   # Flow toxicity, adverse selection
│   │   ├── statistical.py                # Skew, kurtosis, entropy, Hurst exponent
│   │   ├── volume_profile.py             # POC distance, VA position, HVN/LVN
│   │   ├── footprint.py                  # Delta intensity, stacked imbalances
│   │   ├── opening_range.py              # OR high/low, range width, breakout signals
│   │   ├── scalping.py                   # Trap/break setups, absorption, tape speed
│   │   ├── pipeline.py                   # Regime detection, time features, interactions
│   │   ├── multi_timeframe.py            # 5 min aggregates of top-20 1 min features
│   │   ├── ohlcv_indicators.py            # SMA, EMA, RSI, ATR, Bollinger, MACD, Stochastic, ADX, OBV
│   │   ├── labels.py                     # Forward returns (ML targets)
│   │   └── builder.py                    # Feature matrix construction and caching
│   ├── validation/
│   │   ├── robustness.py                 # Deflated Sharpe, PBO, adversarial validation
│   │   ├── alpha_decay.py                # Temporal decay (STABLE / DECAYING / NOISY / DEAD)
│   │   └── factor_attribution.py         # Pure alpha vs. factor exposure (OLS)
│   └── security/
│       └── framework_lock.py             # SHA-256 manifest verification
│
├── research/                             # Mutable workspace (LLM agent operates here)
│   ├── signals/
│   │   └── example_ema_turn.py           # Reference signal implementation
│   └── lib/
│       ├── atomic_io.py                  # Thread-safe atomic JSON writes
│       ├── candidates.py                 # Write-once candidate artifacts
│       ├── coordination.py               # File locks, task state, heartbeat
│       ├── experiments.py                # Experiment logging
│       ├── budget.py                     # Mission budget persistence
│       ├── trial_counter.py              # Trial count for deflated Sharpe
│       └── promotion.py                  # Candidate artifact verification
│
├── configs/
│   ├── framework_lock.json               # SHA-256 manifest of locked files
│   ├── missions/
│   │   └── alpha-discovery.yaml          # Research mission specification
│   └── modern_meta.yaml                  # Strategy config (bar types, risk params)
│
├── scripts/
│   ├── research.py                       # Autonomous research loop entrypoint
│   └── framework/
│       ├── build_lock.py                 # Build framework integrity manifest
│       ├── verify_lock.py                # Verify integrity before runs
│       └── set_readonly.py               # Set framework files read-only
│
├── tests/                                # 39 test files covering all modules
└── docs/                                 # Architecture, validation, agent guide
```

## Signal Contract

Every signal the LLM agent produces must conform to a strict interface:

```python
def generate_signal(df: pl.DataFrame, params: dict) -> np.ndarray:
    """
    Args:
        df: Feature matrix with 221 columns (OHLCV + canonical features)
        params: Strategy parameters

    Returns:
        Array of {-1, 0, 1} (short, flat, long), same length as df, no NaNs
    """
```

Execution uses `entry_on_next_open=True`: a signal at bar T triggers entry at
bar T+1 open, preventing lookahead bias. Intra-bar profit-target and stop-loss
use high/low for worst-case fills.

## Anti-Overfitting Controls

| Control                | Implementation                                               |
| ---------------------- | ------------------------------------------------------------ |
| **Framework lock**     | SHA-256 manifest of 13 core files; verified before every run |
| **Split firewall**     | `ExecutionMode.RESEARCH` blocks all access to the test split |
| **Deflated Sharpe**    | Corrects observed Sharpe for number of hypotheses tested     |
| **Shuffle test**       | Signal must beat 100 random permutations (p < 0.05)          |
| **Walk-forward**       | Rolling out-of-sample windows must show positive performance |
| **Regime stability**   | Must work across 4+ distinct market regimes                  |
| **Cost sensitivity**   | Must remain profitable at 2x transaction costs ($29 RT)      |
| **Alpha decay**        | Exponential fit on rolling Sharpe; half-life > 60 days       |
| **Factor attribution** | OLS decomposition isolates pure alpha from factor exposure   |

## Data

Raw data is sourced from [Databento](https://databento.com) MBP1 (Market-by-Price
Level 1) for NQ E-mini Nasdaq-100 futures — best bid/ask prices and sizes plus
individual trades with aggressor-side flags at nanosecond precision in Parquet
format.

| Split    | Period              | Purpose                             |
| -------- | ------------------- | ----------------------------------- |
| Train    | Mar 2023 – Aug 2024 | Feature engineering, model training |
| Validate | Feb 2025 – Jun 2025 | Signal evaluation, gauntlet testing |
| Test     | Jul 2025 – Jan 2026 | Holdout (blocked in research mode)  |

Data is not included. Set `NQ_DATA_PATH` to your raw data directory.

## Setup

```bash
uv sync                                                                         # install dependencies
uv run pytest -q                                                                # run tests
uv run python scripts/framework/verify_lock.py --manifest configs/framework_lock.json --mode error   # verify integrity
uv run python scripts/research.py --mission configs/missions/alpha-discovery.yaml --max-experiments 100  # research loop
```

## Documentation

| File                  | Contents                                                                   |
| --------------------- | -------------------------------------------------------------------------- |
| `docs/PLATFORM.md`    | Layer boundaries, execution modes, framework lock, runtime coordination    |
| `docs/VALIDATION.md`  | Gauntlet details, robustness metrics, alpha decay, factor attribution      |
| `docs/AGENT_GUIDE.md` | Agent constraints, creative scope, signal contract, research loop protocol |
