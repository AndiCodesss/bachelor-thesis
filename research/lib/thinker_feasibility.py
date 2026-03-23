from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import polars as pl

_EPSILON = 1e-12
_SUPPORTED_OPS = {">", ">=", "<", "<=", "between", "bool_true", "bool_false"}
_SUPPORTED_ROLES = {"primary", "confirmation"}
_MAX_CONDITIONS = 3
_MAX_PRIMARY_CONDITIONS = 2
_MAX_CONFIRMATION_CONDITIONS = 1
_PROTECTED_PARAM_PREFIXES = ("sl_ticks", "pt_ticks")
_CONTEXT_DEPENDENT_PREFIXES = ("prev_day_", "dist_prev_")


class ThinkerFeasibilityError(ValueError):
    def __init__(self, message: str, *, report: dict[str, Any], brief: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report
        self.brief = dict(brief or {})


def is_context_dependent_feature(feature: str) -> bool:
    feature_name = str(feature).strip().lower()
    return any(feature_name.startswith(prefix) for prefix in _CONTEXT_DEPENDENT_PREFIXES)


def normalize_entry_conditions(
    raw_conditions: Any,
    *,
    params_template: dict[str, Any],
    max_conditions: int = _MAX_CONDITIONS,
) -> list[dict[str, Any]]:
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("entry_conditions must be a non-empty list")
    if len(raw_conditions) > max_conditions:
        raise ValueError(
            f"entry_conditions supports at most {max_conditions} conditions "
            f"({_MAX_PRIMARY_CONDITIONS} primary + {_MAX_CONFIRMATION_CONDITIONS} confirmation)"
        )

    normalized: list[dict[str, Any]] = []
    primary_count = 0
    confirmation_count = 0
    for index, raw in enumerate(raw_conditions):
        if not isinstance(raw, dict):
            raise ValueError(f"entry_conditions[{index}] must be an object")

        feature = str(raw.get("feature", "")).strip()
        if not feature:
            raise ValueError(f"entry_conditions[{index}].feature is required")

        op = str(raw.get("op", "")).strip()
        if op not in _SUPPORTED_OPS:
            raise ValueError(
                f"entry_conditions[{index}].op must be one of {sorted(_SUPPORTED_OPS)}, got {op!r}"
            )

        role = str(raw.get("role", "primary")).strip().lower() or "primary"
        if role not in _SUPPORTED_ROLES:
            raise ValueError(
                f"entry_conditions[{index}].role must be one of {sorted(_SUPPORTED_ROLES)}, got {role!r}"
            )
        if role == "primary":
            primary_count += 1
            if primary_count > _MAX_PRIMARY_CONDITIONS:
                raise ValueError(
                    f"entry_conditions supports at most {_MAX_PRIMARY_CONDITIONS} primary conditions",
                )
        else:
            confirmation_count += 1
            if confirmation_count > _MAX_CONFIRMATION_CONDITIONS:
                raise ValueError(
                    f"entry_conditions supports at most {_MAX_CONFIRMATION_CONDITIONS} confirmation condition",
                )

        condition: dict[str, Any] = {
            "feature": feature,
            "op": op,
            "role": role,
        }

        if op in {">", ">=", "<", "<="}:
            param_key = str(raw.get("param_key", "")).strip()
            if not param_key:
                raise ValueError(f"entry_conditions[{index}].param_key is required for op {op}")
            value = params_template.get(param_key)
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"entry_conditions[{index}].param_key={param_key!r} must reference a numeric params_template value"
                )
            condition["param_key"] = param_key
        elif op == "between":
            param_key_low = str(raw.get("param_key_low", "")).strip()
            param_key_high = str(raw.get("param_key_high", "")).strip()
            if not param_key_low or not param_key_high:
                raise ValueError(
                    f"entry_conditions[{index}] between op requires param_key_low and param_key_high"
                )
            low = params_template.get(param_key_low)
            high = params_template.get(param_key_high)
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise ValueError(
                    f"entry_conditions[{index}] between keys must reference numeric params_template values"
                )
            if float(low) > float(high):
                raise ValueError(
                    f"entry_conditions[{index}] has param_key_low={param_key_low!r} > param_key_high={param_key_high!r}"
                )
            condition["param_key_low"] = param_key_low
            condition["param_key_high"] = param_key_high

        normalized.append(condition)

    if not normalized:
        raise ValueError("entry_conditions must contain at least one condition")
    return normalized


def _to_float_array(series: pl.Series) -> np.ndarray:
    if series.dtype == pl.Boolean:
        return np.asarray(series.cast(pl.UInt8).to_numpy(), dtype=np.float64)
    return np.asarray(series.cast(pl.Float64, strict=False).to_numpy(), dtype=np.float64)


def _is_binary_like(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    unique = np.unique(values)
    return bool(np.all(np.isin(unique, [0.0, 1.0])))


def _condition_threshold_blob(condition: dict[str, Any], params_template: dict[str, Any]) -> str:
    op = str(condition.get("op", ""))
    if op == "between":
        low_key = str(condition.get("param_key_low", ""))
        high_key = str(condition.get("param_key_high", ""))
        return f"[{params_template.get(low_key)}, {params_template.get(high_key)}]"
    if op in {"bool_true", "bool_false"}:
        return op
    param_key = str(condition.get("param_key", ""))
    return str(params_template.get(param_key))


def _evaluate_condition(
    *,
    df: pl.DataFrame,
    condition: dict[str, Any],
    params_template: dict[str, Any],
) -> dict[str, Any]:
    feature = str(condition["feature"])
    op = str(condition["op"])
    role = str(condition.get("role", "primary"))
    total = len(df)
    mask = np.zeros(total, dtype=bool)

    if feature not in df.columns:
        return {
            "column": feature,
            "operator": op,
            "role": role,
            "param_key": str(condition.get("param_key", "")),
            "pass_count": 0,
            "total_count": total,
            "finite_count": 0,
            "pass_rate_pct": 0.0,
            "severity": "dead_feature",
            "threshold": _condition_threshold_blob(condition, params_template),
            "_mask": mask,
        }

    values = _to_float_array(df[feature])
    finite = np.isfinite(values)
    finite_values = values[finite]
    finite_count = int(finite_values.size)

    if finite_count == 0:
        severity = "context_unavailable" if is_context_dependent_feature(feature) else "dead_feature"
        return {
            "column": feature,
            "operator": op,
            "role": role,
            "param_key": str(condition.get("param_key", "")),
            "pass_count": 0,
            "total_count": total,
            "finite_count": 0,
            "pass_rate_pct": 0.0,
            "severity": severity,
            "threshold": _condition_threshold_blob(condition, params_template),
            "_mask": mask,
        }

    if op in {">", ">=", "<", "<="}:
        param_key = str(condition.get("param_key", ""))
        threshold = float(params_template[param_key])
        if op == ">":
            mask = values > threshold
        elif op == ">=":
            mask = values >= threshold
        elif op == "<":
            mask = values < threshold
        else:
            mask = values <= threshold
    elif op == "between":
        low = float(params_template[str(condition["param_key_low"])])
        high = float(params_template[str(condition["param_key_high"])])
        mask = (values >= low) & (values <= high)
    elif op == "bool_true":
        mask = values > 0.5
    elif op == "bool_false":
        mask = values <= 0.5
    else:  # pragma: no cover - guarded by normalization
        raise ValueError(f"unsupported op {op}")

    mask &= finite
    pass_count = int(np.sum(mask))
    pass_rate_pct = 100.0 * pass_count / total if total > 0 else 0.0
    severity = "normal"
    if pass_rate_pct == 0.0:
        severity = "blocks_all"
    elif pass_rate_pct < 1.0:
        severity = "restrictive"

    row: dict[str, Any] = {
        "column": feature,
        "operator": op,
        "role": role,
        "param_key": str(condition.get("param_key", "")),
        "pass_count": pass_count,
        "total_count": total,
        "finite_count": finite_count,
        "pass_rate_pct": float(pass_rate_pct),
        "severity": severity,
        "threshold": _condition_threshold_blob(condition, params_template),
        "p02": float(np.quantile(finite_values, 0.02)),
        "p10": float(np.quantile(finite_values, 0.10)),
        "p50": float(np.quantile(finite_values, 0.50)),
        "p90": float(np.quantile(finite_values, 0.90)),
        "p98": float(np.quantile(finite_values, 0.98)),
        "is_binary_like": _is_binary_like(finite_values),
        "_mask": mask,
    }
    if op == "between":
        row["param_key_low"] = str(condition.get("param_key_low", ""))
        row["param_key_high"] = str(condition.get("param_key_high", ""))
    return row


def _public_condition_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        key_text = str(key)
        if key_text.startswith("_"):
            continue
        if isinstance(value, np.generic):
            out[key_text] = value.item()
        else:
            out[key_text] = value
    return out


def _condition_identity(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("column", "")).strip(),
        str(row.get("operator", "")).strip(),
        str(row.get("role", "primary")).strip() or "primary",
        str(row.get("param_key", "")).strip(),
        str(row.get("param_key_low", "")).strip(),
        str(row.get("param_key_high", "")).strip(),
    )


def _matching_condition_rows(
    report: dict[str, Any],
    *,
    target_row: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    bar_results = report.get("bar_results")
    if not isinstance(bar_results, list):
        return []

    target_identity = _condition_identity(target_row)
    matches: list[dict[str, Any]] = []
    for bar_result in bar_results:
        if not isinstance(bar_result, dict):
            continue
        status = str(bar_result.get("status", "")).strip()
        if status in {"empty_sample", "context_unavailable"}:
            continue
        sample_label = str(bar_result.get("sample_label", "")).strip()
        condition_rows = bar_result.get("condition_rows")
        if not isinstance(condition_rows, list):
            continue
        for row in condition_rows:
            if not isinstance(row, dict):
                continue
            if _condition_identity(row) != target_identity:
                continue
            enriched = dict(row)
            enriched["sample_label"] = sample_label
            enriched["sample_status"] = status
            matches.append(enriched)
    return matches


def _format_sample_labels(labels: list[str], *, max_samples: int = 2) -> str:
    shown = [label for label in labels if label][:max_samples]
    if not shown:
        return ""
    blob = ", ".join(shown)
    if len(labels) > max_samples:
        blob += ", ..."
    return blob


def summarize_cross_sample_conflicts(
    report: dict[str, Any],
    *,
    focus_rows: list[dict[str, Any]] | None = None,
    max_conditions: int = 2,
    max_samples: int = 2,
) -> list[str]:
    if not isinstance(report, dict):
        return []

    rows = [row for row in (focus_rows or []) if isinstance(row, dict)]
    if not rows:
        relevant = _select_relevant_bar_result(
            [
                row
                for row in (report.get("bar_results") or [])
                if isinstance(row, dict)
            ],
        )
        if not isinstance(relevant, dict):
            return []
        rows = [
            row
            for row in (relevant.get("condition_rows") or [])
            if isinstance(row, dict)
        ]

    notes: list[str] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        identity = _condition_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        matches = _matching_condition_rows(report, target_row=row)
        if len(matches) < 2:
            continue

        p10_values = [
            float(match["p10"])
            for match in matches
            if isinstance(match.get("p10"), (int, float))
        ]
        p90_values = [
            float(match["p90"])
            for match in matches
            if isinstance(match.get("p90"), (int, float))
        ]
        if not p10_values or not p90_values:
            continue

        column = str(row.get("column", "?"))
        operator = str(row.get("operator", "?"))
        threshold = row.get("threshold", "?")
        line = (
            f"{column} {operator} {threshold}: "
            f"sample p10 range {min(p10_values):.4f}..{max(p10_values):.4f}, "
            f"p90 range {min(p90_values):.4f}..{max(p90_values):.4f} "
            f"across {len(matches)} samples"
        )

        threshold_value: float | None = None
        if isinstance(threshold, (int, float)):
            threshold_value = float(threshold)
        else:
            try:
                threshold_value = float(str(threshold))
            except Exception:
                threshold_value = None

        dense_labels = [
            str(match.get("sample_label", ""))
            for match in matches
            if float(match.get("pass_rate_pct", 0.0) or 0.0) > 2.0
        ]
        blocked_labels: list[str] = []
        if threshold_value is not None:
            if operator in {">", ">="}:
                blocked_labels = [
                    str(match.get("sample_label", ""))
                    for match in matches
                    if isinstance(match.get("p90"), (int, float))
                    and float(match["p90"]) < threshold_value - _EPSILON
                ]
            elif operator in {"<", "<="}:
                blocked_labels = [
                    str(match.get("sample_label", ""))
                    for match in matches
                    if isinstance(match.get("p10"), (int, float))
                    and float(match["p10"]) > threshold_value + _EPSILON
                ]

        blocked_blob = _format_sample_labels(blocked_labels, max_samples=max_samples)
        dense_blob = _format_sample_labels(dense_labels, max_samples=max_samples)
        if blocked_blob:
            line += f"; likely blocks on {blocked_blob}"
        if dense_blob:
            line += f"; still dense on {dense_blob}"

        notes.append(line)
        if len(notes) >= max_conditions:
            break

    return notes


def detect_cross_sample_repair_conflicts(
    report: dict[str, Any],
    *,
    focus_rows: list[dict[str, Any]] | None = None,
    max_conditions: int = 2,
    max_samples: int = 2,
) -> list[str]:
    if not isinstance(report, dict):
        return []

    rows = [row for row in (focus_rows or []) if isinstance(row, dict)]
    focus_keys = {_condition_identity(row) for row in rows}
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for result in report.get("bar_results", []):
        if not isinstance(result, dict):
            continue
        status = str(result.get("status", "")).strip()
        if status not in {"zero_signal", "over_signal"}:
            continue
        sample_label = str(result.get("sample_label", "")).strip()
        for row in result.get("condition_rows", []):
            if not isinstance(row, dict):
                continue
            pass_rate = float(row.get("pass_rate_pct", 0.0) or 0.0)
            severity = str(row.get("severity", "")).strip()
            if status == "over_signal" and pass_rate <= 2.0:
                continue
            if status == "zero_signal" and severity not in {"blocks_all", "restrictive"} and pass_rate > 1.0:
                continue
            identity = _condition_identity(row)
            if focus_keys and identity not in focus_keys:
                continue
            bucket = grouped.setdefault(
                identity,
                {
                    "column": str(row.get("column", "?")).strip(),
                    "operator": str(row.get("operator", "?")).strip(),
                    "threshold": row.get("threshold", "?"),
                    "statuses": set(),
                    "zero_samples": [],
                    "dense_samples": [],
                },
            )
            bucket["statuses"].add(status)
            if sample_label:
                if status == "zero_signal":
                    bucket["zero_samples"].append(sample_label)
                elif status == "over_signal":
                    bucket["dense_samples"].append(sample_label)

    notes: list[str] = []
    for bucket in grouped.values():
        statuses = set(bucket["statuses"])
        if not {"zero_signal", "over_signal"} <= statuses:
            continue
        line = (
            f"{bucket['column']} {bucket['operator']} {bucket['threshold']}: "
            "the same threshold is too tight on some samples and too loose on others"
        )
        zero_blob = _format_sample_labels(bucket["zero_samples"], max_samples=max_samples)
        dense_blob = _format_sample_labels(bucket["dense_samples"], max_samples=max_samples)
        if zero_blob:
            line += f"; zero-signal on {zero_blob}"
        if dense_blob:
            line += f"; over-signal on {dense_blob}"
        notes.append(line)
        if len(notes) >= max_conditions:
            break

    return notes


def _format_condition_line(row: dict[str, Any]) -> str:
    column = str(row.get("column", "?"))
    operator = str(row.get("operator", "?"))
    threshold = row.get("threshold", "?")
    pass_rate = float(row.get("pass_rate_pct", 0.0) or 0.0)
    pass_count = int(row.get("pass_count", 0) or 0)
    total_count = int(row.get("total_count", 0) or 0)
    severity = str(row.get("severity", "normal"))
    tag = ""
    if severity == "dead_feature":
        tag = "  <-- DEAD"
    elif severity == "context_unavailable":
        tag = "  <-- CONTEXT UNAVAILABLE"
    elif severity == "blocks_all":
        tag = "  <-- BLOCKS ALL"
    elif severity == "restrictive":
        tag = "  <-- RESTRICTIVE"

    if row.get("is_binary_like"):
        return (
            f"{column} ({operator}): {pass_rate:.2f}% pass ({pass_count}/{total_count}){tag}"
        )
    return (
        f"{column} {operator} {threshold}: {pass_rate:.2f}% pass ({pass_count}/{total_count})"
        f" [p10={float(row.get('p10', 0.0)):.4f} p50={float(row.get('p50', 0.0)):.4f} p90={float(row.get('p90', 0.0)):.4f}]"
        f"{tag}"
    )


def _combined_mask(rows: list[dict[str, Any]], total_count: int) -> np.ndarray:
    mask = np.ones(total_count, dtype=bool)
    for row in rows:
        row_mask = row.get("_mask")
        if isinstance(row_mask, np.ndarray):
            mask &= row_mask
    return mask


def _failing_bar_result_priority(row: dict[str, Any]) -> tuple[Any, ...]:
    status = str(row.get("status", "")).strip()
    signal_rate_pct = float(row.get("signal_rate_pct", 0.0) or 0.0)
    status_rank = {
        "dead_feature_primary": 0,
        "zero_signal": 1,
        "over_signal": 2,
        "context_unavailable": 3,
    }.get(status, 3)
    severity_rank = -signal_rate_pct if status == "over_signal" else 0.0
    return (
        status_rank,
        severity_rank,
        str(row.get("bar_config", "")),
        str(row.get("sample_label", "")),
    )


def _select_relevant_bar_result(bar_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    failing_rows = [
        row
        for row in bar_results
        if isinstance(row, dict) and str(row.get("status", "")) not in {"ok", "empty_sample", "context_unavailable"}
    ]
    if not failing_rows:
        return None
    return min(failing_rows, key=_failing_bar_result_priority)


def assess_entry_condition_feasibility(
    *,
    entry_conditions: list[dict[str, Any]],
    params_template: dict[str, Any],
    selected_bar_configs: list[str],
    validation_sample_cache: dict[str, list[tuple[str, pl.DataFrame]]],
    over_signal_threshold_pct: float = 2.0,
) -> dict[str, Any]:
    report: dict[str, Any] = {"bar_results": []}

    for bar_config in selected_bar_configs:
        samples = validation_sample_cache.get(bar_config) or []
        if not samples:
            report["bar_results"].append(
                {
                    "bar_config": bar_config,
                    "sample_label": "",
                    "status": "empty_sample",
                    "error": "empty sample frame",
                }
            )
            continue

        for sample_label, sample_df in samples:
            label = str(sample_label)
            if len(sample_df) == 0:
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": label,
                        "status": "empty_sample",
                        "error": "empty sample frame",
                    }
                )
                continue

            rows = [
                _evaluate_condition(
                    df=sample_df,
                    condition=condition,
                    params_template=params_template,
                )
                for condition in entry_conditions
            ]
            public_rows = [_public_condition_row(row) for row in rows]
            total_count = len(sample_df)
            combined_mask = _combined_mask(rows, total_count)
            combined_count = int(np.sum(combined_mask))
            combined_rate_pct = 100.0 * combined_count / total_count if total_count > 0 else 0.0

            dead_primary = next(
                (
                    row for row in rows
                    if str(row.get("role", "primary")) == "primary"
                    and str(row.get("severity", "")) == "dead_feature"
                ),
                None,
            )
            context_unavailable = [
                row
                for row in rows
                if str(row.get("severity", "")).strip() == "context_unavailable"
            ]
            if context_unavailable:
                columns = ", ".join(str(row.get("column", "?")) for row in context_unavailable[:3])
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": label,
                        "status": "context_unavailable",
                        "error": (
                            f"context-dependent feature(s) unavailable on {bar_config} ({label}): {columns}"
                        ),
                        "nonzero": 0,
                        "total": total_count,
                        "signal_rate_pct": 0.0,
                        "condition_rows": public_rows,
                    }
                )
                continue
            if dead_primary is not None:
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": label,
                        "status": "dead_feature_primary",
                        "error": (
                            f"primary feature {dead_primary['column']} has no finite values on {bar_config} "
                            f"({label})"
                        ),
                        "nonzero": 0,
                        "total": total_count,
                        "signal_rate_pct": 0.0,
                        "condition_rows": public_rows,
                    }
                )
                continue

            if combined_count == 0:
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": label,
                        "status": "zero_signal",
                        "nonzero": 0,
                        "total": total_count,
                        "signal_rate_pct": 0.0,
                        "condition_rows": public_rows,
                    }
                )
                continue

            if combined_rate_pct > float(over_signal_threshold_pct):
                report["bar_results"].append(
                    {
                        "bar_config": bar_config,
                        "sample_label": label,
                        "status": "over_signal",
                        "nonzero": combined_count,
                        "total": total_count,
                        "signal_rate_pct": combined_rate_pct,
                        "condition_rows": public_rows,
                    }
                )
                continue

            report["bar_results"].append(
                {
                    "bar_config": bar_config,
                    "sample_label": label,
                    "status": "ok",
                    "nonzero": combined_count,
                    "total": total_count,
                    "signal_rate_pct": combined_rate_pct,
                    "condition_rows": public_rows,
                }
            )

    return report


def _is_protected_param_key(param_key: str) -> bool:
    key = str(param_key).strip()
    return any(key.startswith(prefix) for prefix in _PROTECTED_PARAM_PREFIXES)


def _clone_brief(brief: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(brief)


def _find_condition_index(
    entry_conditions: list[dict[str, Any]],
    *,
    row: dict[str, Any],
) -> int | None:
    column = str(row.get("column", "")).strip()
    operator = str(row.get("operator", "")).strip()
    role = str(row.get("role", "primary")).strip() or "primary"
    param_key = str(row.get("param_key", "")).strip()
    param_key_low = str(row.get("param_key_low", "")).strip()
    param_key_high = str(row.get("param_key_high", "")).strip()
    for idx, condition in enumerate(entry_conditions):
        if not isinstance(condition, dict):
            continue
        if str(condition.get("feature", "")).strip() != column:
            continue
        if str(condition.get("op", "")).strip() != operator:
            continue
        if str(condition.get("role", "primary")).strip() != role:
            continue
        if operator == "between":
            if (
                str(condition.get("param_key_low", "")).strip() == param_key_low
                and str(condition.get("param_key_high", "")).strip() == param_key_high
            ):
                return idx
            continue
        if str(condition.get("param_key", "")).strip() == param_key:
            return idx
    return None


def _drop_condition(
    brief: dict[str, Any],
    *,
    row: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    entry_conditions = brief.get("entry_conditions")
    if not isinstance(entry_conditions, list):
        return None, None

    idx = _find_condition_index(entry_conditions, row=row)
    if idx is None:
        return None, None

    role = str(row.get("role", "primary")).strip() or "primary"
    if role == "primary":
        return None, None
    repaired = _clone_brief(brief)
    repaired_conditions = list(repaired.get("entry_conditions", []))
    if len(repaired_conditions) <= 1:
        return None, None

    removed = repaired_conditions.pop(idx)
    repaired["entry_conditions"] = repaired_conditions
    column = str(removed.get("feature", row.get("column", "?")))
    action = f"dropped {role} condition {column} {row.get('operator', '?')}"
    return repaired, action


def _round_param_value(value: float) -> float:
    return float(np.round(float(value), 6))


def _loosen_upper_bound(current: float, *, p10: float, p50: float, p90: float) -> float | None:
    for candidate in (p90, p50, p10):
        if np.isfinite(candidate) and candidate < current - _EPSILON:
            return _round_param_value(candidate)
    return None


def _tighten_upper_bound(
    current: float,
    *,
    p90: float,
    p98: float,
    inclusive: bool,
) -> float | None:
    if not (np.isfinite(current) and np.isfinite(p90) and np.isfinite(p98)):
        return None
    candidate = p98
    if candidate <= current + _EPSILON:
        spread = max(float(p98 - p90) * 0.25, abs(float(p98)) * 0.02, 1e-6)
        candidate = current + spread
    if inclusive:
        candidate = np.nextafter(candidate, np.inf)
    if candidate <= current + _EPSILON:
        return None
    return _round_param_value(candidate)


def _loosen_lower_bound(current: float, *, p10: float, p50: float, p90: float) -> float | None:
    for candidate in (p10, p50, p90):
        if np.isfinite(candidate) and candidate > current + _EPSILON:
            return _round_param_value(candidate)
    return None


def _tighten_lower_bound(
    current: float,
    *,
    p02: float,
    p10: float,
    inclusive: bool,
) -> float | None:
    if not (np.isfinite(current) and np.isfinite(p02) and np.isfinite(p10)):
        return None
    candidate = p02
    if candidate >= current - _EPSILON:
        spread = max(float(p10 - p02) * 0.25, abs(float(p02)) * 0.02, 1e-6)
        candidate = current - spread
    if inclusive:
        candidate = np.nextafter(candidate, -np.inf)
    if candidate >= current - _EPSILON:
        return None
    return _round_param_value(candidate)


def _repair_scalar_param(
    brief: dict[str, Any],
    *,
    row: dict[str, Any],
    failure_type: str,
) -> tuple[dict[str, Any] | None, str | None]:
    param_key = str(row.get("param_key", "")).strip()
    if not param_key or _is_protected_param_key(param_key):
        return None, None

    params_template = brief.get("params_template")
    if not isinstance(params_template, dict):
        return None, None

    current_raw = params_template.get(param_key)
    if not isinstance(current_raw, (int, float)):
        return None, None
    current = float(current_raw)

    operator = str(row.get("operator", "")).strip()
    p02 = float(row.get("p02", np.nan))
    p10 = float(row.get("p10", np.nan))
    p50 = float(row.get("p50", np.nan))
    p90 = float(row.get("p90", np.nan))
    p98 = float(row.get("p98", np.nan))

    candidate: float | None = None
    if failure_type == "zero_signal":
        if operator in {">", ">="}:
            candidate = _loosen_upper_bound(current, p10=p10, p50=p50, p90=p90)
        elif operator in {"<", "<="}:
            candidate = _loosen_lower_bound(current, p10=p10, p50=p50, p90=p90)
    elif failure_type == "over_signal":
        if operator == ">":
            candidate = _tighten_upper_bound(current, p90=p90, p98=p98, inclusive=False)
        elif operator == ">=":
            candidate = _tighten_upper_bound(current, p90=p90, p98=p98, inclusive=True)
        elif operator == "<":
            candidate = _tighten_lower_bound(current, p02=p02, p10=p10, inclusive=False)
        elif operator == "<=":
            candidate = _tighten_lower_bound(current, p02=p02, p10=p10, inclusive=True)

    if candidate is None or abs(candidate - current) <= _EPSILON:
        return None, None

    repaired = _clone_brief(brief)
    repaired_params = dict(repaired.get("params_template", {}))
    repaired_params[param_key] = candidate
    repaired["params_template"] = repaired_params
    action = f"set {param_key}={candidate} for {row.get('column', '?')} {operator}"
    return repaired, action


def _repair_between_param(
    brief: dict[str, Any],
    *,
    row: dict[str, Any],
    failure_type: str,
) -> tuple[dict[str, Any] | None, str | None]:
    low_key = str(row.get("param_key_low", "")).strip()
    high_key = str(row.get("param_key_high", "")).strip()
    if not low_key or not high_key:
        return None, None
    if _is_protected_param_key(low_key) or _is_protected_param_key(high_key):
        return None, None

    params_template = brief.get("params_template")
    if not isinstance(params_template, dict):
        return None, None
    current_low = params_template.get(low_key)
    current_high = params_template.get(high_key)
    if not isinstance(current_low, (int, float)) or not isinstance(current_high, (int, float)):
        return None, None

    p10 = float(row.get("p10", np.nan))
    p50 = float(row.get("p50", np.nan))
    p90 = float(row.get("p90", np.nan))
    if not (np.isfinite(p10) and np.isfinite(p50) and np.isfinite(p90)):
        return None, None

    if failure_type == "zero_signal":
        new_low = min(float(current_low), p10)
        new_high = max(float(current_high), p90)
    elif failure_type == "over_signal":
        new_low = max(float(current_low), (p10 + p50) / 2.0)
        new_high = min(float(current_high), (p50 + p90) / 2.0)
        if new_low >= new_high:
            center = p50
            half_width = max((p90 - p10) / 8.0, 1e-6)
            new_low = center - half_width
            new_high = center + half_width
    else:
        return None, None

    new_low = _round_param_value(new_low)
    new_high = _round_param_value(new_high)
    if new_low >= new_high:
        return None, None
    if (
        abs(new_low - float(current_low)) <= _EPSILON
        and abs(new_high - float(current_high)) <= _EPSILON
    ):
        return None, None

    repaired = _clone_brief(brief)
    repaired_params = dict(repaired.get("params_template", {}))
    repaired_params[low_key] = new_low
    repaired_params[high_key] = new_high
    repaired["params_template"] = repaired_params
    action = f"set [{low_key}, {high_key}]=[{new_low}, {new_high}] for {row.get('column', '?')} between"
    return repaired, action


def _repair_condition_row(
    brief: dict[str, Any],
    *,
    row: dict[str, Any],
    condition_rows: list[dict[str, Any]],
    failure_type: str,
) -> tuple[dict[str, Any] | None, str | None]:
    operator = str(row.get("operator", "")).strip()
    severity = str(row.get("severity", "")).strip()
    role = str(row.get("role", "primary")).strip() or "primary"

    if severity == "context_unavailable":
        return None, None
    if severity == "dead_feature":
        if role == "confirmation":
            return _drop_condition(brief, row=row)
        return None, None

    if operator == "between":
        repaired, action = _repair_between_param(brief, row=row, failure_type=failure_type)
        if repaired is not None:
            return repaired, action
    elif operator in {">", ">=", "<", "<="}:
        repaired, action = _repair_scalar_param(brief, row=row, failure_type=failure_type)
        if repaired is not None:
            return repaired, action

    if failure_type == "zero_signal" and role == "confirmation":
        return _drop_condition(brief, row=row)
    return None, None


def repair_thinker_brief_for_feasibility(
    brief: dict[str, Any],
    report: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(brief, dict) or not isinstance(report, dict):
        return None, []

    bar_results = report.get("bar_results")
    if not isinstance(bar_results, list):
        return None, []

    relevant = _select_relevant_bar_result(
        [row for row in bar_results if isinstance(row, dict)],
    )
    if not isinstance(relevant, dict):
        return None, []

    failure_type = str(relevant.get("status", "runtime_error")).strip() or "runtime_error"
    condition_rows = relevant.get("condition_rows") if isinstance(relevant.get("condition_rows"), list) else []
    if not condition_rows:
        return None, []

    if detect_cross_sample_repair_conflicts(report, focus_rows=condition_rows):
        return None, []

    if failure_type == "over_signal":
        ranked_rows = sorted(
            (row for row in condition_rows if isinstance(row, dict)),
            key=lambda row: (
                0 if str(row.get("role", "")).strip() == "primary" else 1,
                -float(row.get("pass_rate_pct", 0.0) or 0.0),
                str(row.get("param_key", "")),
            ),
        )
    else:
        ranked_rows = sorted(
            (row for row in condition_rows if isinstance(row, dict)),
            key=lambda row: (
                0 if str(row.get("severity", "")).strip() in {"dead_feature", "blocks_all"} else 1,
                0 if str(row.get("role", "")).strip() == "confirmation" else 1,
                float(row.get("pass_rate_pct", 0.0) or 0.0),
                str(row.get("param_key", "")),
            ),
        )

    for row in ranked_rows:
        repaired, action = _repair_condition_row(
            brief,
            row=row,
            condition_rows=condition_rows,
            failure_type=failure_type,
        )
        if repaired is not None and action:
            return repaired, [action]

    return None, []


def format_feasibility_error(report: dict[str, Any]) -> str:
    bar_results = report.get("bar_results") if isinstance(report, dict) else None
    if not isinstance(bar_results, list) or not bar_results:
        return "Thinker feasibility failed unexpectedly. Return corrected JSON only."

    relevant = _select_relevant_bar_result(
        [row for row in bar_results if isinstance(row, dict)],
    )
    if not isinstance(relevant, dict):
        return "Thinker feasibility failed unexpectedly. Return corrected JSON only."

    status = str(relevant.get("status", "runtime_error"))
    bar_config = str(relevant.get("bar_config", "?"))
    sample_label = str(relevant.get("sample_label", "")).strip()
    condition_rows = relevant.get("condition_rows") if isinstance(relevant.get("condition_rows"), list) else []
    lines = [f"THINKER_FEASIBILITY on {bar_config} ({sample_label or 'sample'}):"]
    if status == "dead_feature_primary":
        lines.append(str(relevant.get("error", "a primary feature has no usable data")))
        lines.append("Replace the dead primary feature or demote it to confirmation.")
    elif status == "context_unavailable":
        lines.append(str(relevant.get("error", "context-dependent features are unavailable on this sample")))
        lines.append("Keep the structural hypothesis intact and evaluate it on samples with valid prior-session context.")
    elif status == "zero_signal":
        total = int(relevant.get("total", 0) or 0)
        lines.append(
            f"Combined entry_conditions pass-through is 0/{total} bars (0.00%). "
            "Relax or replace the most restrictive primary condition(s)."
        )
    elif status == "over_signal":
        nonzero = int(relevant.get("nonzero", 0) or 0)
        total = int(relevant.get("total", 0) or 0)
        rate = float(relevant.get("signal_rate_pct", 0.0) or 0.0)
        lines.append(
            f"Combined entry_conditions pass-through is {nonzero}/{total} bars ({rate:.2f}%), which is too dense. "
            "Tighten the loosest primary filter(s) or simplify the bar selection."
        )
    else:
        lines.append(str(relevant.get("error", status)))

    if condition_rows:
        lines.append("Per-condition pass-through (independent):")
        for row in condition_rows[:5]:
            if isinstance(row, dict):
                lines.append(f"- {_format_condition_line(row)}")

    conflict_notes = summarize_cross_sample_conflicts(
        report,
        focus_rows=[row for row in condition_rows if isinstance(row, dict)],
    )
    if conflict_notes:
        lines.append("Cross-sample distribution drift:")
        for note in conflict_notes:
            lines.append(f"- {note}")

    repair_conflicts = detect_cross_sample_repair_conflicts(
        report,
        focus_rows=[row for row in condition_rows if isinstance(row, dict)],
    )
    if repair_conflicts:
        lines.append("Cross-sample repair conflicts:")
        for note in repair_conflicts:
            lines.append(f"- {note}")

    if bool(report.get("repair_conflict_detected", False)):
        lines.append(
            "Cross-sample feasibility constraints conflict across validation samples. "
            "Redesign the setup instead of nudging the same threshold.",
        )

    if bool(report.get("repair_cycle_detected", False)):
        lines.append(
            "Auto-repair oscillated between incompatible thresholds across sampled days. "
            "Redesign the setup instead of nudging the same threshold.",
        )

    lines.append("Return corrected JSON only.")
    return "\n".join(lines)


__all__ = [
    "ThinkerFeasibilityError",
    "assess_entry_condition_feasibility",
    "detect_cross_sample_repair_conflicts",
    "format_feasibility_error",
    "is_context_dependent_feature",
    "normalize_entry_conditions",
    "repair_thinker_brief_for_feasibility",
    "summarize_cross_sample_conflicts",
]
