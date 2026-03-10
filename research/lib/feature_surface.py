from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import polars as pl

_EPSILON = 1e-12
_SPARSE_NONZERO_RATE = 0.005
_MAX_FORMAT_FEATURES = 8
_PARAM_HINT_PATTERNS = (
    "zscore",
    "ratio",
    "position",
    "width",
    "bandwidth",
    "velocity",
    "intensity",
    "imbalance",
    "distance",
    "progress",
    "wick",
    "delta",
    "cvd",
    "volume",
    "range",
    "pctb",
)

_NUMERIC_DTYPES = {
    pl.Boolean,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype in _NUMERIC_DTYPES


def _to_numeric_array(series: pl.Series) -> np.ndarray:
    if series.dtype == pl.Boolean:
        return np.asarray(series.cast(pl.UInt8).to_numpy(), dtype=np.float64)
    return np.asarray(series.cast(pl.Float64, strict=False).to_numpy(), dtype=np.float64)


def _safe_rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def _is_binary_like(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    unique = np.unique(values)
    return bool(np.all(np.isin(unique, [0.0, 1.0])))


def _feature_kind(*, finite: np.ndarray, nonzero_rate: float) -> str:
    if finite.size == 0:
        return "all_null"
    if np.count_nonzero(np.abs(finite) > _EPSILON) == 0:
        return "constant_zero"
    if _is_binary_like(finite) and nonzero_rate <= _SPARSE_NONZERO_RATE:
        return "binary_sparse"
    if nonzero_rate <= _SPARSE_NONZERO_RATE:
        return "sparse"
    if np.allclose(finite, finite[0], rtol=0.0, atol=_EPSILON):
        return "constant"
    if _is_binary_like(finite):
        return "binary_active"
    return "variable"


def _compute_column_profile(series: pl.Series) -> dict[str, Any] | None:
    if not _is_numeric_dtype(series.dtype):
        return None

    values = _to_numeric_array(series)
    finite = values[np.isfinite(values)]
    finite_count = int(finite.size)
    nonzero_count = int(np.count_nonzero(np.abs(finite) > _EPSILON))
    nonzero_rate = _safe_rate(nonzero_count, finite_count)
    null_count = int(series.null_count())

    return {
        "dtype": str(series.dtype),
        "rows": int(len(series)),
        "null_count": null_count,
        "null_rate": _safe_rate(null_count, len(series)),
        "finite_count": finite_count,
        "nonzero_count": nonzero_count,
        "nonzero_rate": nonzero_rate,
        "kind": _feature_kind(finite=finite, nonzero_rate=nonzero_rate),
        "p10": _safe_quantile(finite, 0.10),
        "p50": _safe_quantile(finite, 0.50),
        "p90": _safe_quantile(finite, 0.90),
    }


def build_feature_surface(
    *,
    samples_by_bar_config: dict[str, list[tuple[str, pl.DataFrame]]],
) -> dict[str, Any]:
    surface: dict[str, Any] = {
        "schema_version": "1.0",
        "sparse_nonzero_rate": _SPARSE_NONZERO_RATE,
        "by_bar_config": {},
    }

    for bar_config, samples in samples_by_bar_config.items():
        if not samples:
            surface["by_bar_config"][bar_config] = {
                "sample_count": 0,
                "sample_labels": [],
                "total_rows": 0,
                "feature_stats": {},
                "dead_features": [],
                "sparse_features": [],
            }
            continue

        sample_labels = [str(label) for label, _ in samples]
        frames = [df for _, df in samples if len(df) > 0]
        if not frames:
            surface["by_bar_config"][bar_config] = {
                "sample_count": len(samples),
                "sample_labels": sample_labels,
                "total_rows": 0,
                "feature_stats": {},
                "dead_features": [],
                "sparse_features": [],
            }
            continue

        sample_profiles_by_column: dict[str, list[dict[str, Any]]] = {}
        for label, frame in samples:
            if len(frame) == 0:
                continue
            for column in frame.columns:
                if column == "ts_event":
                    continue
                profile = _compute_column_profile(frame[column])
                if profile is None:
                    continue
                sample_profiles_by_column.setdefault(str(column), []).append(
                    {
                        "sample_label": str(label),
                        "p10": profile.get("p10"),
                        "p50": profile.get("p50"),
                        "p90": profile.get("p90"),
                    }
                )

        combined = pl.concat(frames, how="vertical_relaxed")
        feature_stats: dict[str, dict[str, Any]] = {}
        for column in combined.columns:
            if column == "ts_event":
                continue
            profile = _compute_column_profile(combined[column])
            if profile is not None:
                sample_profiles = sample_profiles_by_column.get(str(column), [])
                p10_values = [
                    float(row["p10"])
                    for row in sample_profiles
                    if isinstance(row.get("p10"), (int, float))
                ]
                p50_values = [
                    float(row["p50"])
                    for row in sample_profiles
                    if isinstance(row.get("p50"), (int, float))
                ]
                p90_values = [
                    float(row["p90"])
                    for row in sample_profiles
                    if isinstance(row.get("p90"), (int, float))
                ]
                if p10_values:
                    profile["sample_p10_min"] = min(p10_values)
                    profile["sample_p10_max"] = max(p10_values)
                if p50_values:
                    profile["sample_p50_min"] = min(p50_values)
                    profile["sample_p50_max"] = max(p50_values)
                if p90_values:
                    profile["sample_p90_min"] = min(p90_values)
                    profile["sample_p90_max"] = max(p90_values)
                profile["sample_profile_count"] = len(sample_profiles)
                feature_stats[column] = profile

        dead_features = sorted(
            (
                {
                    "name": name,
                    "kind": stats["kind"],
                    "nonzero_rate": float(stats["nonzero_rate"]),
                }
                for name, stats in feature_stats.items()
                if stats["kind"] in {"all_null", "constant_zero"}
            ),
            key=lambda row: (row["nonzero_rate"], row["name"]),
        )
        sparse_features = sorted(
            (
                {
                    "name": name,
                    "kind": stats["kind"],
                    "nonzero_rate": float(stats["nonzero_rate"]),
                }
                for name, stats in feature_stats.items()
                if stats["kind"] in {"binary_sparse", "sparse"}
            ),
            key=lambda row: (row["nonzero_rate"], row["name"]),
        )

        surface["by_bar_config"][bar_config] = {
            "sample_count": len(frames),
            "sample_labels": sample_labels,
            "total_rows": int(sum(len(df) for df in frames)),
            "feature_stats": feature_stats,
            "dead_features": dead_features,
            "sparse_features": sparse_features,
        }

    return surface


def extract_referenced_columns(text: str, available_columns: Iterable[str]) -> list[str]:
    available = set(available_columns)
    referenced = (
        set(re.findall(r'"([a-z][a-z0-9_]{2,})"', text))
        | set(re.findall(r"'([a-z][a-z0-9_]{2,})'", text))
        | set(re.findall(r"\b([a-z][a-z0-9_]{2,})\b", text))
    )
    return sorted(referenced & available)


def format_feature_surface_context(
    surface: dict[str, Any],
    *,
    selected_bar_configs: list[str] | None = None,
    max_features_per_bucket: int = _MAX_FORMAT_FEATURES,
) -> str:
    by_bar_config = surface.get("by_bar_config")
    if not isinstance(by_bar_config, dict) or not by_bar_config:
        return ""

    chosen = selected_bar_configs or sorted(by_bar_config)
    lines = [
        "FEATURE_SURFACE_INTELLIGENCE:",
        "These stats come from representative sample files on the active split and session filter.",
        "Avoid dead or extremely sparse features as PRIMARY entry conditions. Use them only as secondary confirmation if the core setup relies on live, varying features.",
    ]

    for bar_config in chosen:
        payload = by_bar_config.get(bar_config)
        if not isinstance(payload, dict):
            continue
        total_rows = int(payload.get("total_rows", 0) or 0)
        sample_count = int(payload.get("sample_count", 0) or 0)
        labels = payload.get("sample_labels") if isinstance(payload.get("sample_labels"), list) else []
        label_blob = ", ".join(str(v) for v in labels[:3])
        lines.append(
            f"- {bar_config}: {total_rows} sampled rows across {sample_count} file(s)"
            + (f" [{label_blob}]" if label_blob else "")
        )

        dead = payload.get("dead_features") if isinstance(payload.get("dead_features"), list) else []
        sparse = payload.get("sparse_features") if isinstance(payload.get("sparse_features"), list) else []

        if dead:
            dead_blob = ", ".join(
                f"{row['name']} ({100.0 * float(row['nonzero_rate']):.2f}% non-zero)"
                for row in dead[:max_features_per_bucket]
            )
            lines.append(f"  Dead features: {dead_blob}")
        if sparse:
            sparse_blob = ", ".join(
                f"{row['name']} ({100.0 * float(row['nonzero_rate']):.2f}% non-zero)"
                for row in sparse[:max_features_per_bucket]
            )
            lines.append(f"  Sparse features: {sparse_blob}")

    return "\n".join(lines)


def _param_hint_priority(name: str, stats: dict[str, Any]) -> tuple[int, float, str]:
    lowered = str(name).lower()
    score = 0
    if any(pattern in lowered for pattern in _PARAM_HINT_PATTERNS):
        score += 4
    if str(stats.get("kind", "")) == "variable":
        score += 2
    if float(stats.get("null_rate", 1.0) or 1.0) == 0.0:
        score += 1
    p10 = stats.get("p10")
    p90 = stats.get("p90")
    spread = 0.0
    if isinstance(p10, (int, float)) and isinstance(p90, (int, float)):
        spread = abs(float(p90) - float(p10))
        if spread > _EPSILON:
            score += 1
    return (-score, -spread, lowered)


def format_param_feasibility_context(
    surface: dict[str, Any],
    *,
    selected_bar_configs: list[str] | None = None,
    priority_features: list[str] | None = None,
    max_features_per_bar: int = 6,
) -> str:
    by_bar_config = surface.get("by_bar_config")
    if not isinstance(by_bar_config, dict) or not by_bar_config:
        return ""

    chosen = selected_bar_configs or sorted(by_bar_config)
    lines = [
        "PARAM_FEASIBILITY_HINTS:",
        "For first-pass params, keep thresholds near the observed p10-p90 band on the chosen bar_config.",
        "Values outside these bands are intentionally rare and should only be used with very simple conjunctions.",
    ]

    wrote_bar = False
    for bar_config in chosen:
        payload = by_bar_config.get(bar_config)
        if not isinstance(payload, dict):
            continue
        stats = payload.get("feature_stats")
        if not isinstance(stats, dict) or not stats:
            continue
        candidates: list[tuple[str, dict[str, Any]]] = []
        for name, feature_stats in stats.items():
            if not isinstance(feature_stats, dict):
                continue
            kind = str(feature_stats.get("kind", ""))
            if kind not in {"variable", "binary_active"}:
                continue
            p10 = feature_stats.get("p10")
            p50 = feature_stats.get("p50")
            p90 = feature_stats.get("p90")
            if not all(isinstance(value, (int, float)) for value in (p10, p50, p90)):
                continue
            if abs(float(p90) - float(p10)) <= _EPSILON:
                continue
            candidates.append((str(name), feature_stats))
        if not candidates:
            continue

        candidate_map = {name: feature_stats for name, feature_stats in candidates}
        priority_names = [
            str(name)
            for name in (priority_features or [])
            if str(name) in candidate_map
        ]
        candidates.sort(key=lambda item: _param_hint_priority(item[0], item[1]))
        lines.append(f"- {bar_config}:")
        wrote_bar = True
        chosen_names: list[str] = []
        seen_names: set[str] = set()
        for name in priority_names:
            if name in seen_names:
                continue
            chosen_names.append(name)
            seen_names.add(name)
            if len(chosen_names) >= max_features_per_bar:
                break
        if len(chosen_names) < max_features_per_bar:
            for name, _ in candidates:
                if name in seen_names:
                    continue
                chosen_names.append(name)
                seen_names.add(name)
                if len(chosen_names) >= max_features_per_bar:
                    break

        for name in chosen_names:
            feature_stats = candidate_map[name]
            line = (
                "  "
                + f"{name}: p10={float(feature_stats['p10']):.4f} "
                + f"p50={float(feature_stats['p50']):.4f} "
                + f"p90={float(feature_stats['p90']):.4f}"
            )
            sample_profile_count = int(feature_stats.get("sample_profile_count", 0) or 0)
            sample_p10_min = feature_stats.get("sample_p10_min")
            sample_p10_max = feature_stats.get("sample_p10_max")
            sample_p90_min = feature_stats.get("sample_p90_min")
            sample_p90_max = feature_stats.get("sample_p90_max")
            if (
                sample_profile_count > 1
                and all(
                    isinstance(value, (int, float))
                    for value in (sample_p10_min, sample_p10_max, sample_p90_min, sample_p90_max)
                )
            ):
                line += (
                    " | "
                    + f"sample p10 range={float(sample_p10_min):.4f}..{float(sample_p10_max):.4f} "
                    + f"p90 range={float(sample_p90_min):.4f}..{float(sample_p90_max):.4f}"
                )
            lines.append(line)

    return "\n".join(lines) if wrote_bar else ""


def format_referenced_surface_warnings(
    *,
    surface: dict[str, Any],
    selected_bar_configs: list[str],
    text_fragments: list[str],
    max_features_per_bucket: int = _MAX_FORMAT_FEATURES,
) -> str:
    by_bar_config = surface.get("by_bar_config")
    if not isinstance(by_bar_config, dict) or not by_bar_config:
        return ""

    combined_text = "\n".join(str(fragment or "") for fragment in text_fragments)
    warnings: list[str] = []
    for bar_config in selected_bar_configs:
        payload = by_bar_config.get(bar_config)
        if not isinstance(payload, dict):
            continue
        stats = payload.get("feature_stats")
        if not isinstance(stats, dict) or not stats:
            continue
        referenced = extract_referenced_columns(combined_text, stats.keys())
        risky_rows = []
        for name in referenced:
            stat = stats.get(name)
            if not isinstance(stat, dict):
                continue
            kind = str(stat.get("kind", ""))
            if kind not in {"all_null", "constant_zero", "binary_sparse", "sparse"}:
                continue
            risky_rows.append(
                (name, 100.0 * float(stat.get("nonzero_rate", 0.0) or 0.0), kind),
            )
        if not risky_rows:
            continue
        risky_rows.sort(key=lambda row: (row[1], row[0]))
        blob = ", ".join(
            f"{name} ({rate:.2f}% non-zero)"
            for name, rate, _ in risky_rows[:max_features_per_bucket]
        )
        warnings.append(f"- {bar_config}: {blob}")

    if not warnings:
        return ""

    return (
        "REFERENCED_FEATURE_SURFACE_WARNINGS:\n"
        "The current hypothesis references features that are dead or extremely sparse on the active samples.\n"
        "Do not use these as primary entry gates unless the rest of the setup is anchored on features with real variation.\n"
        + "\n".join(warnings)
    )


def describe_referenced_columns(
    *,
    df: pl.DataFrame,
    code: str,
    max_columns: int = 15,
) -> str:
    n = len(df)
    if n == 0:
        return "  (empty sample frame - no statistics available)"

    used_columns = extract_referenced_columns(code, df.columns)
    if not used_columns:
        return "  (no referenced columns found in code - check column name spelling)"

    lines: list[str] = []
    for column in used_columns[:max_columns]:
        profile = _compute_column_profile(df[column])
        if profile is None:
            continue
        try:
            kind = str(profile["kind"])
            if kind == "binary_active":
                lines.append(
                    f"  {column} (bool-like): True={100.0 * float(profile['nonzero_rate']):.2f}%"
                    f" ({int(profile['nonzero_count'])}/{int(profile['finite_count'])} bars)"
                )
            elif kind in {"all_null", "constant_zero", "binary_sparse", "sparse"}:
                lines.append(
                    f"  {column}: {100.0 * float(profile['nonzero_rate']):.2f}% non-zero"
                    f" ({int(profile['nonzero_count'])}/{int(profile['finite_count'])} finite bars)"
                )
            else:
                p10 = profile.get("p10")
                p50 = profile.get("p50")
                p90 = profile.get("p90")
                if p10 is None or p50 is None or p90 is None:
                    lines.append(f"  {column}: (insufficient finite values)")
                else:
                    lines.append(
                        f"  {column}: p10={float(p10):.4f} p50={float(p50):.4f} p90={float(p90):.4f}"
                    )
        except Exception:
            lines.append(f"  {column}: (error computing stats)")

    return "\n".join(lines) if lines else "  (could not compute stats for referenced columns)"


# ---------------------------------------------------------------------------
# Per-condition pass-through diagnostic
# ---------------------------------------------------------------------------

_REVERSE_OPS = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "==": "==", "!=": "!="}
_NP_OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}

# Regex: var = safe_f64_col(df, "col")  or  safe_bool_col / safe_int_col
_RE_COL_ASSIGN = re.compile(
    r"(\w+)\s*=\s*safe_(?:f64|bool|int)_col\s*\(\s*df\s*,\s*[\"'](\w+)[\"']"
)
# var OP params["key"]  or  params.get("key", ...)
_RE_VAR_CMP_PARAM = re.compile(
    r"(\w+)\s*(>=|<=|>|<|==|!=)\s*params(?:\[[\"'](\w+)['\"]\]|\.get\([\"'](\w+)[\"'])"
)
# params["key"] OP var  (reversed)
_RE_PARAM_CMP_VAR = re.compile(
    r"params(?:\[[\"'](\w+)['\"]\]|\.get\([\"'](\w+)[\"'])\s*(>=|<=|>|<|==|!=)\s*(\w+)"
)


def diagnose_condition_passthrough(
    *,
    df: pl.DataFrame,
    code: str,
    params: dict[str, Any],
    max_conditions: int = 20,
) -> str:
    rows = collect_condition_passthrough(
        df=df,
        code=code,
        params=params,
        max_conditions=max_conditions,
        include_mask=True,
    )
    if not rows:
        return ""

    n = int(rows[0]["total_count"])
    lines = ["  Per-condition pass-through (each evaluated independently):"]
    for row in rows:
        pass_pct = float(row["pass_rate_pct"])
        tag = ""
        if str(row.get("severity", "")) == "blocks_all":
            tag = "  <-- BLOCKS ALL"
        elif str(row.get("severity", "")) == "restrictive":
            tag = "  <-- RESTRICTIVE"
        lines.append(
            f"    {row['column']} {row['operator']} {row['threshold']} ({row['param_key']}): "
            f"{pass_pct:.2f}% pass ({int(row['pass_count'])}/{n}){tag}"
        )

    combined_count = int(np.prod([1], dtype=np.int64)[0])
    combined_mask = np.ones(n, dtype=bool)
    for row in rows:
        mask = row.get("_mask")
        if isinstance(mask, np.ndarray):
            combined_mask &= mask
    combined_count = int(np.sum(combined_mask))
    combined_pct = 100.0 * combined_count / n if n > 0 else 0.0
    lines.append(f"    -> Combined (all AND'd): {combined_pct:.2f}% ({combined_count}/{n})")

    return "\n".join(lines)


def collect_condition_passthrough(
    *,
    df: pl.DataFrame,
    code: str,
    params: dict[str, Any],
    max_conditions: int = 20,
    include_mask: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate each threshold condition independently and report pass-through rates.

    Parses safe_*_col assignments and ``var OP params["key"]`` comparisons from
    the generated code, evaluates each on *df*, and returns structured rows.
    """
    n = len(df)
    if n == 0:
        return []

    # Step 1: map local variable names → canonical column names
    var_to_col: dict[str, str] = {}
    for m in _RE_COL_ASSIGN.finditer(code):
        var_to_col[m.group(1)] = m.group(2)

    if not var_to_col:
        return []

    # Step 2: collect (column, operator, param_key, threshold) tuples
    conditions: list[tuple[str, str, str, float]] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(col: str, op: str, param_key: str) -> None:
        threshold = params.get(param_key)
        if threshold is None or not isinstance(threshold, (int, float)):
            return
        key = (col, op, param_key)
        if key in seen:
            return
        seen.add(key)
        conditions.append((col, op, param_key, float(threshold)))

    for m in _RE_VAR_CMP_PARAM.finditer(code):
        var, op, key1, key2 = m.groups()
        col = var_to_col.get(var)
        if col:
            _add(col, op, key1 or key2)

    for m in _RE_PARAM_CMP_VAR.finditer(code):
        key1, key2, op, var = m.groups()
        col = var_to_col.get(var)
        if col:
            _add(col, _REVERSE_OPS[op], key1 or key2)

    if not conditions:
        return []

    rows: list[dict[str, Any]] = []
    for col, op, param_key, threshold in conditions[:max_conditions]:
        if col not in df.columns:
            continue

        try:
            series = df[col]
            values = (
                series.cast(pl.Float64).to_numpy()
                if series.dtype != pl.Float64
                else series.to_numpy()
            )
        except Exception:
            continue

        valid = ~np.isnan(values) if np.issubdtype(values.dtype, np.floating) else np.ones(n, dtype=bool)
        finite_count = int(np.sum(valid))
        op_fn = _NP_OPS.get(op)
        if op_fn is None:
            continue

        mask = op_fn(values, threshold) & valid
        pass_count = int(np.sum(mask))
        pass_pct = 100.0 * pass_count / n

        severity = "normal"
        if pass_pct == 0.0:
            severity = "blocks_all"
        elif pass_pct < 1.0:
            severity = "restrictive"

        row = {
            "column": col,
            "operator": op,
            "param_key": param_key,
            "threshold": float(threshold),
            "pass_count": pass_count,
            "total_count": int(n),
            "finite_count": finite_count,
            "pass_rate_pct": float(pass_pct),
            "severity": severity,
        }
        if include_mask:
            row["_mask"] = mask
        rows.append(row)

    return rows


__all__ = [
    "build_feature_surface",
    "collect_condition_passthrough",
    "describe_referenced_columns",
    "diagnose_condition_passthrough",
    "extract_referenced_columns",
    "format_feature_surface_context",
    "format_param_feasibility_context",
    "format_referenced_surface_warnings",
]
