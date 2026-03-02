"""Stable framework API surface for research and promotion entrypoints."""

from __future__ import annotations

from src.framework.backtest.costs import CostModel, compute_adaptive_costs
from src.framework.backtest.engine import run_backtest
from src.framework.backtest.metrics import compute_metrics
from src.framework.backtest.validators import run_validation_gauntlet
from src.framework.data.constants import (
    CACHE_ROOT,
    DATA_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    SPLITS,
    TEST_FOLDERS,
    TICK_SIZE,
    TICK_VALUE,
    TOTAL_COST_RT,
    TRAIN_FOLDERS,
    VALIDATE_FOLDERS,
)
from src.framework.data.loader import get_parquet_files
from src.framework.data.loader import ExecutionMode, get_execution_mode, set_execution_mode
from src.framework.data.splits import get_split_dates
from src.framework.features_canonical.builder import build_feature_matrix, load_cached_matrix
from src.framework.validation.alpha_decay import fit_alpha_decay
from src.framework.validation.factor_attribution import (
    compute_factor_returns,
    factor_attribution,
)
from src.framework.validation.robustness import (
    adversarial_validation_report,
    deflated_sharpe_ratio,
    estimate_pbo,
)


def get_split_files(split: str):
    """Compatibility alias for split file retrieval."""
    return get_parquet_files(split)


__all__ = [
    "CostModel",
    "compute_adaptive_costs",
    "run_backtest",
    "compute_metrics",
    "run_validation_gauntlet",
    "fit_alpha_decay",
    "compute_factor_returns",
    "factor_attribution",
    "deflated_sharpe_ratio",
    "estimate_pbo",
    "adversarial_validation_report",
    "load_cached_matrix",
    "build_feature_matrix",
    "get_split_files",
    "set_execution_mode",
    "get_execution_mode",
    "ExecutionMode",
    "get_split_dates",
    "PROJECT_ROOT",
    "DATA_PATH",
    "CACHE_ROOT",
    "RESULTS_DIR",
    "TICK_SIZE",
    "TICK_VALUE",
    "TOTAL_COST_RT",
    "SEED",
    "SPLITS",
    "TRAIN_FOLDERS",
    "VALIDATE_FOLDERS",
    "TEST_FOLDERS",
]
