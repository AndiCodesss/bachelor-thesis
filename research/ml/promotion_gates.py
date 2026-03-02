"""Walk-forward promotion validation and gate evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import math
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import numpy as np

from src.framework.validation.robustness import deflated_sharpe_ratio


_MONTH_RE = re.compile(r"nq_(\d{4}-\d{2})-\d{2}\.parquet$")
_DATE_RE = re.compile(r"nq_(\d{4})-(\d{2})-(\d{2})\.parquet$")


@dataclass(frozen=True)
class WalkForwardFold:
    """One walk-forward fold with fixed train/test month partitions."""

    index: int
    train_months: tuple[str, ...]
    test_months: tuple[str, ...]
    train_files: tuple[Path, ...]
    test_files: tuple[Path, ...]


@dataclass(frozen=True)
class FoldResult:
    """Per-fold evaluation metrics."""

    fold_index: int
    train_months: tuple[str, ...]
    test_months: tuple[str, ...]
    trade_count: int
    net_pnl: float
    sharpe: float


@dataclass(frozen=True)
class WFAResult:
    """Aggregate walk-forward evaluation output."""

    folds: tuple[FoldResult, ...]
    fold_sharpes: tuple[float, ...]
    pct_positive: float
    aggregate_sharpe: float
    aggregate_trade_count: int
    aggregate_net_pnl: float
    raw_trial_count: int
    effective_trial_count: int
    deflated_sharpe: float
    expected_max_sharpe_null: float
    dsr_probability: float


FoldEvaluator = Callable[[WalkForwardFold, dict[str, Any]], dict[str, Any]]


def _float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)


def _int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return int(default)


def _extract_month(file_path: Path | str) -> str:
    name = Path(file_path).name
    match = _MONTH_RE.search(name)
    if match is None:
        raise ValueError(f"Cannot parse month from parquet filename: {name}")
    return match.group(1)


def _extract_date(file_path: Path | str) -> date:
    name = Path(file_path).name
    match = _DATE_RE.search(name)
    if match is None:
        raise ValueError(f"Cannot parse date from parquet filename: {name}")
    y, m, d = match.groups()
    return date(int(y), int(m), int(d))


def _annualized_sharpe(values: Sequence[float], annualization: float = 252.0) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float((float(np.mean(arr)) / std) * math.sqrt(float(annualization)))


class WalkForwardValidator:
    """Rolling month-based walk-forward validator.

    Uses sequential monthly folds where each fold trains on trailing
    `train_months` and tests on the next `test_months`.
    """

    def __init__(
        self,
        all_files: Sequence[Path | str],
        *,
        train_months: int = 12,
        test_months: int = 2,
        step_months: int = 2,
        embargo_months: int = 0,
        purge_days: int = 0,
        allow_partial_last_test: bool = True,
    ):
        if train_months < 1:
            raise ValueError("train_months must be >= 1")
        if test_months < 1:
            raise ValueError("test_months must be >= 1")
        if step_months < 1:
            raise ValueError("step_months must be >= 1")
        if embargo_months < 0:
            raise ValueError("embargo_months must be >= 0")
        if purge_days < 0:
            raise ValueError("purge_days must be >= 0")
        if step_months != test_months:
            raise ValueError(
                "step_months must equal test_months to keep test folds non-overlapping",
            )

        self.train_months = int(train_months)
        self.test_months = int(test_months)
        self.step_months = int(step_months)
        self.embargo_months = int(embargo_months)
        self.purge_days = int(purge_days)
        self.allow_partial_last_test = bool(allow_partial_last_test)

        resolved_files = tuple(sorted({Path(p).resolve() for p in all_files}))
        if not resolved_files:
            raise ValueError("all_files is empty")

        self.all_files = resolved_files
        self.folds = self._build_folds()
        if not self.folds:
            raise ValueError("No walk-forward folds could be built from provided files")

    def _build_folds(self) -> tuple[WalkForwardFold, ...]:
        files_by_month: dict[str, list[Path]] = defaultdict(list)
        for path in self.all_files:
            files_by_month[_extract_month(path)].append(path)

        months = sorted(files_by_month)
        folds: list[WalkForwardFold] = []

        first_test_start = self.train_months + self.embargo_months
        for test_start in range(first_test_start, len(months), self.step_months):
            test_end = test_start + self.test_months
            test_month_block = months[test_start:test_end]
            if not test_month_block:
                continue
            if len(test_month_block) < self.test_months and not self.allow_partial_last_test:
                break

            train_end = test_start - self.embargo_months
            train_start = train_end - self.train_months
            if train_start < 0:
                continue

            train_month_block = months[train_start:train_end]
            if len(train_month_block) < self.train_months:
                continue

            train_files = [
                f
                for month in train_month_block
                for f in sorted(files_by_month[month])
            ]
            test_files = [
                f
                for month in test_month_block
                for f in sorted(files_by_month[month])
            ]

            if not train_files or not test_files:
                continue

            # Purge recent training observations close to test start.
            if self.purge_days > 0:
                first_test_date = min(_extract_date(f) for f in test_files)
                cutoff = first_test_date - timedelta(days=self.purge_days)
                train_files = [f for f in train_files if _extract_date(f) < cutoff]

            if not train_files:
                continue

            folds.append(
                WalkForwardFold(
                    index=len(folds) + 1,
                    train_months=tuple(train_month_block),
                    test_months=tuple(test_month_block),
                    train_files=tuple(train_files),
                    test_files=tuple(test_files),
                )
            )

        return tuple(folds)

    def evaluate(
        self,
        evaluator: FoldEvaluator,
        params: dict[str, Any] | None = None,
        *,
        raw_trial_count: int,
        effective_trial_count: int | None = None,
    ) -> WFAResult:
        """Run evaluator on each fold and aggregate the walk-forward result."""
        p = params or {}
        fold_results: list[FoldResult] = []
        combined_daily_returns: list[float] = []

        for fold in self.folds:
            metrics = evaluator(fold, p)
            sharpe = _float(metrics.get("sharpe_ratio", metrics.get("sharpe", 0.0)))
            trade_count = _int(metrics.get("trade_count", 0))
            net_pnl = _float(metrics.get("net_pnl", 0.0))

            returns = metrics.get("daily_returns", [])
            if returns is not None:
                combined_daily_returns.extend([_float(v) for v in returns])

            fold_results.append(
                FoldResult(
                    fold_index=fold.index,
                    train_months=fold.train_months,
                    test_months=fold.test_months,
                    trade_count=trade_count,
                    net_pnl=net_pnl,
                    sharpe=sharpe,
                )
            )

        fold_sharpes = tuple(r.sharpe for r in fold_results)
        pct_positive = (
            float(sum(1 for s in fold_sharpes if s > 0.0) / len(fold_sharpes))
            if fold_sharpes
            else 0.0
        )

        aggregate_sharpe = _annualized_sharpe(combined_daily_returns)

        raw_trials = max(1, int(raw_trial_count))
        eff_trials = raw_trials if effective_trial_count is None else max(1, int(effective_trial_count))
        eff_trials = min(eff_trials, raw_trials)

        dsr_payload = deflated_sharpe_ratio(
            np.asarray(combined_daily_returns, dtype=np.float64),
            n_trials=eff_trials,
        )
        if bool(dsr_payload.get("available", False)):
            sharpe_observed = _float(dsr_payload.get("sharpe", 0.0))
            sr_null = _float(dsr_payload.get("expected_max_sharpe_null", 0.0))
            deflated_sharpe = sharpe_observed - sr_null
            dsr_probability = _float(dsr_payload.get("dsr", 0.0))
        else:
            sr_null = 0.0
            deflated_sharpe = 0.0
            dsr_probability = 0.0

        return WFAResult(
            folds=tuple(fold_results),
            fold_sharpes=fold_sharpes,
            pct_positive=pct_positive,
            aggregate_sharpe=float(aggregate_sharpe),
            aggregate_trade_count=int(sum(r.trade_count for r in fold_results)),
            aggregate_net_pnl=float(sum(r.net_pnl for r in fold_results)),
            raw_trial_count=raw_trials,
            effective_trial_count=eff_trials,
            deflated_sharpe=float(deflated_sharpe),
            expected_max_sharpe_null=float(sr_null),
            dsr_probability=float(dsr_probability),
        )


def evaluate_promotion_gates(
    *,
    wfa_result: WFAResult,
    thresholds: dict[str, Any],
    lockbox_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate promotion gates using walk-forward metrics (+ optional lockbox)."""
    min_fold_count = _int(thresholds.get("min_fold_count", 1))
    min_positive_ratio = _float(thresholds.get("min_positive_fold_ratio", 0.60))
    min_aggregate_sharpe = _float(thresholds.get("min_aggregate_sharpe", 0.50))
    min_deflated_sharpe = _float(thresholds.get("min_deflated_sharpe", 0.0))

    checks = {
        "min_fold_count": {
            "passed": len(wfa_result.folds) >= min_fold_count,
            "value": int(len(wfa_result.folds)),
            "min_required": int(min_fold_count),
        },
        "min_positive_fold_ratio": {
            "passed": wfa_result.pct_positive >= min_positive_ratio,
            "value": float(wfa_result.pct_positive),
            "min_required": float(min_positive_ratio),
        },
        "min_aggregate_sharpe": {
            "passed": wfa_result.aggregate_sharpe >= min_aggregate_sharpe,
            "value": float(wfa_result.aggregate_sharpe),
            "min_required": float(min_aggregate_sharpe),
        },
        "min_deflated_sharpe": {
            "passed": wfa_result.deflated_sharpe > min_deflated_sharpe,
            "value": float(wfa_result.deflated_sharpe),
            "min_required": float(min_deflated_sharpe),
        },
    }

    metrics: dict[str, Any] = {
        "fold_count": int(len(wfa_result.folds)),
        "fold_sharpes": [float(s) for s in wfa_result.fold_sharpes],
        "pct_positive_folds": float(wfa_result.pct_positive),
        "aggregate_sharpe": float(wfa_result.aggregate_sharpe),
        "aggregate_trade_count": int(wfa_result.aggregate_trade_count),
        "aggregate_net_pnl": float(wfa_result.aggregate_net_pnl),
        "raw_trial_count": int(wfa_result.raw_trial_count),
        "effective_trial_count": int(wfa_result.effective_trial_count),
        "deflated_sharpe": float(wfa_result.deflated_sharpe),
        "expected_max_sharpe_null": float(wfa_result.expected_max_sharpe_null),
        "dsr_probability": float(wfa_result.dsr_probability),
        "folds": [
            {
                "fold_index": int(fr.fold_index),
                "train_months": list(fr.train_months),
                "test_months": list(fr.test_months),
                "trade_count": int(fr.trade_count),
                "net_pnl": float(fr.net_pnl),
                "sharpe": float(fr.sharpe),
            }
            for fr in wfa_result.folds
        ],
    }

    if lockbox_metrics is not None:
        lockbox_sharpe = _float(lockbox_metrics.get("sharpe", 0.0))
        lockbox_net_pnl = _float(lockbox_metrics.get("net_pnl", 0.0))
        lockbox_trades = _int(lockbox_metrics.get("trade_count", 0))

        min_lockbox_sharpe = _float(thresholds.get("min_lockbox_sharpe", 0.0))
        min_lockbox_net_pnl = _float(thresholds.get("min_lockbox_net_pnl", 0.0))
        min_lockbox_trades = _int(thresholds.get("min_lockbox_trade_count", 1))

        checks["min_lockbox_sharpe"] = {
            "passed": lockbox_sharpe >= min_lockbox_sharpe,
            "value": lockbox_sharpe,
            "min_required": min_lockbox_sharpe,
        }
        checks["min_lockbox_net_pnl"] = {
            "passed": lockbox_net_pnl >= min_lockbox_net_pnl,
            "value": lockbox_net_pnl,
            "min_required": min_lockbox_net_pnl,
        }
        checks["min_lockbox_trade_count"] = {
            "passed": lockbox_trades >= min_lockbox_trades,
            "value": lockbox_trades,
            "min_required": min_lockbox_trades,
        }

        metrics["lockbox"] = {
            "months": [str(m) for m in lockbox_metrics.get("months", [])],
            "trade_count": lockbox_trades,
            "net_pnl": lockbox_net_pnl,
            "sharpe": lockbox_sharpe,
            "sharpe_2x_cost": _float(lockbox_metrics.get("sharpe_2x_cost", 0.0)),
            "avg_trades_per_day": _float(lockbox_metrics.get("avg_trades_per_day", 0.0)),
            "max_trades_per_day": _int(lockbox_metrics.get("max_trades_per_day", 0)),
        }

    passed = all(bool(c["passed"]) for c in checks.values())
    return {
        "passed": bool(passed),
        "checks": checks,
        "metrics": metrics,
    }


__all__ = [
    "FoldResult",
    "WFAResult",
    "WalkForwardFold",
    "WalkForwardValidator",
    "evaluate_promotion_gates",
]

