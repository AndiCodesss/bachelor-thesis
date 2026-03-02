"""Tests for walk-forward promotion gate framework."""

from __future__ import annotations

from pathlib import Path

from research.ml.promotion_gates import WalkForwardValidator, evaluate_promotion_gates


def _make_month_files(tmp_path: Path, start_year: int, start_month: int, n_months: int) -> list[Path]:
    out: list[Path] = []
    year = start_year
    month = start_month
    for i in range(n_months):
        folder = tmp_path / f"m{i:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"nq_{year:04d}-{month:02d}-01.parquet"
        file_path.write_text("", encoding="utf-8")
        out.append(file_path)

        month += 1
        if month > 12:
            month = 1
            year += 1
    return out


def test_walk_forward_builds_non_overlapping_folds(tmp_path: Path) -> None:
    files = _make_month_files(tmp_path, 2023, 1, 24)
    validator = WalkForwardValidator(
        files,
        train_months=12,
        test_months=2,
        step_months=2,
    )

    # 24 months with 12m warmup and 2m step => 6 folds
    assert len(validator.folds) == 6

    first = validator.folds[0]
    assert first.train_months == (
        "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
        "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12",
    )
    assert first.test_months == ("2024-01", "2024-02")

    test_months = [m for fold in validator.folds for m in fold.test_months]
    assert len(test_months) == len(set(test_months)), "Test months must not overlap across folds"


def test_walk_forward_gates_pass_with_robust_results(tmp_path: Path) -> None:
    files = _make_month_files(tmp_path, 2023, 1, 24)
    validator = WalkForwardValidator(files, train_months=12, test_months=2, step_months=2)

    sharpe_by_fold = {
        1: 1.1,
        2: 0.9,
        3: 0.2,
        4: 0.7,
        5: -0.1,
        6: 0.8,
    }

    def evaluator(fold, _params):
        daily = [2.0] * 20 + [-1.0] * 10  # positive mean / finite variance
        return {
            "trade_count": 40,
            "net_pnl": 1000.0 + (fold.index * 50.0),
            "sharpe_ratio": sharpe_by_fold[fold.index],
            "daily_returns": daily,
        }

    wfa = validator.evaluate(evaluator, {}, raw_trial_count=120, effective_trial_count=40)

    out = evaluate_promotion_gates(
        wfa_result=wfa,
        thresholds={
            "min_fold_count": 6,
            "min_positive_fold_ratio": 0.60,
            "min_aggregate_sharpe": 0.50,
            "min_deflated_sharpe": 0.0,
        },
    )

    assert out["passed"] is True
    assert out["checks"]["min_positive_fold_ratio"]["passed"] is True
    assert out["checks"]["min_aggregate_sharpe"]["passed"] is True
    assert out["checks"]["min_deflated_sharpe"]["passed"] is True


def test_walk_forward_gates_fail_when_results_are_fragile(tmp_path: Path) -> None:
    files = _make_month_files(tmp_path, 2023, 1, 24)
    validator = WalkForwardValidator(files, train_months=12, test_months=2, step_months=2)

    sharpe_by_fold = {
        1: -1.0,
        2: -0.5,
        3: 0.1,
        4: -0.2,
        5: -0.7,
        6: -0.3,
    }

    def evaluator(fold, _params):
        daily = [-2.0] * 20 + [1.0] * 10  # negative mean
        return {
            "trade_count": 18,
            "net_pnl": -500.0 - (fold.index * 20.0),
            "sharpe_ratio": sharpe_by_fold[fold.index],
            "daily_returns": daily,
        }

    wfa = validator.evaluate(evaluator, {}, raw_trial_count=120, effective_trial_count=60)

    out = evaluate_promotion_gates(
        wfa_result=wfa,
        thresholds={
            "min_fold_count": 6,
            "min_positive_fold_ratio": 0.60,
            "min_aggregate_sharpe": 0.50,
            "min_deflated_sharpe": 0.0,
        },
    )

    assert out["passed"] is False
    assert out["checks"]["min_positive_fold_ratio"]["passed"] is False
    assert out["checks"]["min_aggregate_sharpe"]["passed"] is False


def test_walk_forward_embargo_and_purge_reduce_training_window(tmp_path: Path) -> None:
    files = _make_month_files(tmp_path, 2023, 1, 24)
    validator = WalkForwardValidator(
        files,
        train_months=12,
        test_months=2,
        step_months=2,
        embargo_months=1,
        purge_days=32,
    )

    first = validator.folds[0]
    # With 1-month embargo, first test starts at 2024-02 and training ends at 2024-01 exclusive.
    assert first.test_months == ("2024-02", "2024-03")
    # Purge drops the most recent train file near test boundary.
    assert "2024-01" not in first.train_months
    # train_months still records pre-purge month span, but file list is purged.
    train_files_text = [p.name for p in first.train_files]
    assert "nq_2024-01-01.parquet" not in train_files_text


def test_walk_forward_lockbox_checks_applied() -> None:
    class _Stub:
        folds = ()
        fold_sharpes = ()
        pct_positive = 1.0
        aggregate_sharpe = 1.0
        aggregate_trade_count = 100
        aggregate_net_pnl = 2000.0
        raw_trial_count = 100
        effective_trial_count = 10
        deflated_sharpe = 0.2
        expected_max_sharpe_null = 0.1
        dsr_probability = 0.8

    out = evaluate_promotion_gates(
        wfa_result=_Stub(),  # type: ignore[arg-type]
        thresholds={
            "min_fold_count": 0,
            "min_positive_fold_ratio": 0.0,
            "min_aggregate_sharpe": 0.0,
            "min_deflated_sharpe": 0.0,
            "min_lockbox_sharpe": 0.1,
            "min_lockbox_net_pnl": 1.0,
            "min_lockbox_trade_count": 5,
        },
        lockbox_metrics={
            "months": ["2026-02"],
            "trade_count": 4,
            "net_pnl": 10.0,
            "sharpe": 0.2,
        },
    )

    assert out["passed"] is False
    assert out["checks"]["min_lockbox_trade_count"]["passed"] is False
