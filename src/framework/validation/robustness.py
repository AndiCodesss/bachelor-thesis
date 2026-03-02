"""Evaluation-only robustness and anti-overfitting utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    # Acklam inverse normal CDF approximation.
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    return ranks


def _median_impute_standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64, order="C")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    mask = ~np.isfinite(arr)
    if mask.any():
        arr = arr.copy()
        arr[mask] = np.take(med, np.where(mask)[1])
    mu = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    z = (arr - mu) / sd
    return z, med, mu, sd


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int) -> dict[str, Any]:
    vals = np.asarray(returns, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    n = int(len(vals))
    if n < 3:
        return {
            "available": False,
            "reason": "insufficient_samples",
            "sample_count": n,
            "n_trials": int(max(n_trials, 1)),
            "sharpe": 0.0,
            "dsr": 0.0,
        }

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    if std <= 1e-12:
        return {
            "available": False,
            "reason": "near_zero_std",
            "sample_count": n,
            "n_trials": int(max(n_trials, 1)),
            "sharpe": 0.0,
            "dsr": 0.0,
        }

    sr = mean / std
    centered = (vals - mean) / std
    skew = float(np.mean(centered**3))
    kurt = float(np.mean(centered**4))

    sr_var_num = 1.0 - skew * sr + ((kurt - 3.0) / 4.0) * (sr**2)
    sr_var_num = max(sr_var_num, 1e-9)
    sr_std = math.sqrt(sr_var_num / max(n - 1, 1))

    trials = int(max(int(n_trials), 1))
    if trials == 1:
        sr_star = 0.0
    else:
        euler_gamma = 0.5772156649015329
        z1 = _normal_ppf(1.0 - (1.0 / trials))
        z2 = _normal_ppf(1.0 - (1.0 / (trials * math.e)))
        sr_star = sr_std * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)

    z = (sr - sr_star) / max(sr_std, 1e-9)
    dsr = _normal_cdf(z)

    return {
        "available": True,
        "reason": "ok",
        "sample_count": n,
        "n_trials": trials,
        "mean_return": mean,
        "std_return": std,
        "sharpe": float(sr),
        "sharpe_std_error": float(sr_std),
        "skew": skew,
        "kurtosis": kurt,
        "expected_max_sharpe_null": float(sr_star),
        "z_score": float(z),
        "dsr": float(dsr),
    }


def estimate_pbo(score_matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be 2D [candidate, split]")
    n_candidates, n_splits = matrix.shape
    if n_candidates < 2 or n_splits < 3:
        return {
            "available": False,
            "reason": "insufficient_shape",
            "candidate_count": int(n_candidates),
            "split_count": int(n_splits),
            "pbo": 1.0,
        }

    under_median = 0
    lambdas: list[float] = []
    oos_percentiles: list[float] = []
    selected_indices: list[int] = []

    for k in range(n_splits):
        oos = matrix[:, k]
        is_mask = np.ones(n_splits, dtype=bool)
        is_mask[k] = False
        ins = matrix[:, is_mask]
        ins_score = np.nanmean(ins, axis=1)
        if not np.any(np.isfinite(ins_score)) or not np.any(np.isfinite(oos)):
            continue

        best_idx = int(np.nanargmax(ins_score))
        selected_indices.append(best_idx)

        oos_ranks = _rankdata(np.where(np.isfinite(oos), oos, -np.inf))
        pct = float(oos_ranks[best_idx] / float(n_candidates))
        pct = float(np.clip(pct, 1e-6, 1.0 - 1e-6))
        oos_percentiles.append(pct)
        lambdas.append(float(math.log(pct / (1.0 - pct))))

        if pct < 0.5:
            under_median += 1

    effective = len(oos_percentiles)
    if effective == 0:
        return {
            "available": False,
            "reason": "no_effective_splits",
            "candidate_count": int(n_candidates),
            "split_count": int(n_splits),
            "pbo": 1.0,
        }

    pbo = float(under_median / effective)
    return {
        "available": True,
        "reason": "ok",
        "candidate_count": int(n_candidates),
        "split_count": int(n_splits),
        "effective_split_count": int(effective),
        "pbo": pbo,
        "under_median_count": int(under_median),
        "oos_rank_percentiles": [float(v) for v in oos_percentiles],
        "lambda_logit": [float(v) for v in lambdas],
        "selected_candidate_indices": [int(v) for v in selected_indices],
        "lambda_mean": float(np.mean(lambdas)) if lambdas else 0.0,
        "lambda_std": float(np.std(lambdas)) if lambdas else 0.0,
    }


def adversarial_validation_report(
    *,
    train_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    feature_cols: list[str],
    seed: int,
    top_k: int = 20,
) -> dict[str, Any]:
    common = [c for c in feature_cols if (c in train_df.columns and c in holdout_df.columns)]
    if len(common) < 3:
        return {
            "enabled": True,
            "used": False,
            "reason": "insufficient_common_features",
            "feature_count": int(len(common)),
            "auc": 0.5,
            "top_features": [],
        }

    x_train = np.asarray(train_df.select(common).to_numpy(), dtype=np.float64, order="C")
    x_hold = np.asarray(holdout_df.select(common).to_numpy(), dtype=np.float64, order="C")
    if len(x_train) < 200 or len(x_hold) < 200:
        return {
            "enabled": True,
            "used": False,
            "reason": "insufficient_rows",
            "feature_count": int(len(common)),
            "train_rows": int(len(x_train)),
            "holdout_rows": int(len(x_hold)),
            "auc": 0.5,
            "top_features": [],
        }

    x = np.vstack([x_train, x_hold])
    y = np.concatenate(
        [
            np.zeros(len(x_train), dtype=np.int8),
            np.ones(len(x_hold), dtype=np.int8),
        ],
    )
    x_norm, _, _, _ = _median_impute_standardize(x)

    rng = np.random.default_rng(int(seed))
    idx0 = rng.permutation(np.flatnonzero(y == 0))
    idx1 = rng.permutation(np.flatnonzero(y == 1))
    n0_train = max(1, int(0.7 * len(idx0)))
    n1_train = max(1, int(0.7 * len(idx1)))
    idx_fit = np.concatenate([idx0[:n0_train], idx1[:n1_train]])
    idx_val = np.concatenate([idx0[n0_train:], idx1[n1_train:]])
    if len(idx_val) < 50:
        return {
            "enabled": True,
            "used": False,
            "reason": "insufficient_validation_rows",
            "feature_count": int(len(common)),
            "auc": 0.5,
            "top_features": [],
        }

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for adversarial validation") from exc

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=int(seed) + 91,
        n_jobs=1,
    )
    model.fit(x_norm[idx_fit], y[idx_fit].astype(np.int8))
    p_val = np.asarray(model.predict_proba(x_norm[idx_val])[:, 1], dtype=np.float64)
    auc = float(roc_auc_score(y[idx_val], p_val))

    imp = np.asarray(model.feature_importances_, dtype=np.float64)
    m0 = np.nanmean(x_train, axis=0)
    m1 = np.nanmean(x_hold, axis=0)
    v0 = np.nanvar(x_train, axis=0)
    v1 = np.nanvar(x_hold, axis=0)
    pooled = np.sqrt(np.maximum((v0 + v1) * 0.5, 1e-9))
    smd = (m1 - m0) / pooled

    imp_rank = _rankdata(imp)
    smd_rank = _rankdata(np.abs(smd))
    combined_rank = imp_rank + smd_rank
    order = np.argsort(combined_rank)[::-1]
    top_n = int(max(0, min(int(top_k), len(common))))

    top_features: list[dict[str, Any]] = []
    for j in order[:top_n]:
        top_features.append(
            {
                "feature": common[int(j)],
                "importance": float(imp[int(j)]),
                "std_mean_diff": float(smd[int(j)]),
                "combined_rank_score": float(combined_rank[int(j)]),
                "train_mean": float(m0[int(j)]),
                "holdout_mean": float(m1[int(j)]),
            },
        )

    return {
        "enabled": True,
        "used": True,
        "reason": "ok",
        "feature_count": int(len(common)),
        "train_rows": int(len(x_train)),
        "holdout_rows": int(len(x_hold)),
        "auc": auc,
        "model": "random_forest",
        "top_features": top_features,
    }
