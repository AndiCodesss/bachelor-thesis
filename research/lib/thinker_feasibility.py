from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

_EPSILON = 1e-12
_SUPPORTED_OPS = {">", ">=", "<", "<=", "between", "bool_true", "bool_false"}
_SUPPORTED_ROLES = {"primary", "confirmation"}
_MAX_CONDITIONS = 6


class ThinkerFeasibilityError(ValueError):
    def __init__(self, message: str, *, report: dict[str, Any], brief: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report
        self.brief = dict(brief or {})


def normalize_entry_conditions(
    raw_conditions: Any,
    *,
    params_template: dict[str, Any],
    max_conditions: int = _MAX_CONDITIONS,
) -> list[dict[str, Any]]:
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("entry_conditions must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_conditions[:max_conditions]):
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
        "p10": float(np.quantile(finite_values, 0.10)),
        "p50": float(np.quantile(finite_values, 0.50)),
        "p90": float(np.quantile(finite_values, 0.90)),
        "is_binary_like": _is_binary_like(finite_values),
        "_mask": mask,
    }
    if op == "between":
        row["param_key_low"] = str(condition.get("param_key_low", ""))
        row["param_key_high"] = str(condition.get("param_key_high", ""))
    return row


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
        frames = [df for _, df in samples if len(df) > 0]
        if not frames:
            report["bar_results"].append(
                {
                    "bar_config": bar_config,
                    "sample_label": "",
                    "status": "empty_sample",
                    "error": "empty sample frame",
                }
            )
            continue

        combined = pl.concat(frames, how="vertical_relaxed")
        labels = [str(label) for label, _ in samples[:3]]
        sample_label = ", ".join(labels)
        rows = [_evaluate_condition(df=combined, condition=condition, params_template=params_template) for condition in entry_conditions]
        total_count = len(combined)
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
        if dead_primary is not None:
            report["bar_results"].append(
                {
                    "bar_config": bar_config,
                    "sample_label": sample_label,
                    "status": "dead_feature_primary",
                    "error": (
                        f"primary feature {dead_primary['column']} has no finite values on {bar_config} "
                        f"({sample_label})"
                    ),
                    "nonzero": 0,
                    "total": total_count,
                    "signal_rate_pct": 0.0,
                    "condition_rows": rows,
                }
            )
            continue

        if combined_count == 0:
            report["bar_results"].append(
                {
                    "bar_config": bar_config,
                    "sample_label": sample_label,
                    "status": "zero_signal",
                    "nonzero": 0,
                    "total": total_count,
                    "signal_rate_pct": 0.0,
                    "condition_rows": rows,
                }
            )
            continue

        if combined_rate_pct > float(over_signal_threshold_pct):
            report["bar_results"].append(
                {
                    "bar_config": bar_config,
                    "sample_label": sample_label,
                    "status": "over_signal",
                    "nonzero": combined_count,
                    "total": total_count,
                    "signal_rate_pct": combined_rate_pct,
                    "condition_rows": rows,
                }
            )
            continue

        report["bar_results"].append(
            {
                "bar_config": bar_config,
                "sample_label": sample_label,
                "status": "ok",
                "nonzero": combined_count,
                "total": total_count,
                "signal_rate_pct": combined_rate_pct,
                "condition_rows": rows,
            }
        )

    return report


def format_feasibility_error(report: dict[str, Any]) -> str:
    bar_results = report.get("bar_results") if isinstance(report, dict) else None
    if not isinstance(bar_results, list) or not bar_results:
        return "Thinker feasibility failed unexpectedly. Return corrected JSON only."

    relevant = next(
        (
            row
            for row in bar_results
            if isinstance(row, dict) and str(row.get("status", "")) not in {"ok", "empty_sample"}
        ),
        None,
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

    lines.append("Return corrected JSON only.")
    return "\n".join(lines)


__all__ = [
    "ThinkerFeasibilityError",
    "assess_entry_condition_feasibility",
    "format_feasibility_error",
    "normalize_entry_conditions",
]
