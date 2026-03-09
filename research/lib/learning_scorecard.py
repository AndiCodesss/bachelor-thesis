"""Derived learning scorecard for autonomy research runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from research.lib.coordination import read_json_file_if_exists, update_json_file
from research.lib.setup_key import task_setup_identity

SCHEMA_VERSION = "1.0"
OTHER_THEME_TAG = "other"
LOW_SAMPLE_ATTEMPTS = 3
NEAR_MISS_LIMIT = 10


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def laplace_rate(passes: int, attempts: int) -> float:
    """Laplace-smoothed success rate: (passes + 1) / (attempts + 2)."""
    p = max(0, int(passes))
    a = max(0, int(attempts))
    return float((p + 1) / (a + 2))


def empty_theme_entry() -> dict[str, Any]:
    return {
        "attempts": 0,
        "search_passes": 0,
        "search_rate": laplace_rate(0, 0),
        "selection_attempts": 0,
        "selection_passes": 0,
        "selection_rate": laplace_rate(0, 0),
        "fail_counts": {},
    }


def empty_bar_entry() -> dict[str, Any]:
    return {
        "attempts": 0,
        "search_passes": 0,
        "search_rate": laplace_rate(0, 0),
        "selection_attempts": 0,
        "selection_passes": 0,
        "selection_rate": laplace_rate(0, 0),
    }


def empty_setup_entry() -> dict[str, Any]:
    return {
        "label": "",
        "attempts": 0,
        "edge_passes": 0,
        "search_passes": 0,
        "selection_attempts": 0,
        "selection_passes": 0,
        "fail_counts": {},
        "recent_outcomes": [],
        "updated_at": None,
    }


def empty_scorecard() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "rebuilt_at": None,
        "setup_stats": {},
        "theme_stats": {},
        "bar_config_affinity": {},
        "near_misses": [],
        "low_sample_themes": [],
    }


def _slugify(text: str) -> str:
    clean = re.sub(r"[^a-z0-9_]+", "_", str(text).strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean


def _focus_slug(text: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return OTHER_THEME_TAG
    head = re.split(r"[:(]", raw, maxsplit=1)[0]
    head = head.replace("→", " ").replace("->", " ")
    tokens = re.findall(r"[a-z0-9]+", head)
    stopwords = {"and", "or", "the", "a", "an", "of", "for", "to", "on", "in"}
    generic_suffixes = {
        "pattern",
        "patterns",
        "rejection",
        "rotation",
        "rotations",
        "reversal",
        "reversals",
        "fade",
        "fades",
        "failed",
        "ignition",
    }
    filtered = [tok for tok in tokens if tok not in stopwords]
    while len(filtered) > 2 and filtered and filtered[-1] in generic_suffixes:
        filtered.pop()
    chosen = filtered[:3] or tokens[:3]
    return "_".join(chosen) if chosen else OTHER_THEME_TAG


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_focus_anchors(mission: dict[str, Any]) -> list[str]:
    focus = mission.get("current_focus")
    if isinstance(focus, list) and focus:
        return _unique_preserve_order([_focus_slug(item) for item in focus if str(item).strip()])
    return []


def normalize_theme_tag(tag: str | None) -> str:
    if tag is None:
        return OTHER_THEME_TAG
    clean = _slugify(str(tag).replace("-", "_").replace(" ", "_"))
    parts = [part for part in clean.split("_") if part]
    if len(parts) > 4:
        clean = "_".join(parts[:4])
    if not clean:
        return OTHER_THEME_TAG
    return clean


def _empty_or_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _refresh_theme_entry(entry: dict[str, Any]) -> dict[str, Any]:
    entry["attempts"] = _empty_or_int(entry.get("attempts"))
    entry["search_passes"] = _empty_or_int(entry.get("search_passes"))
    entry["selection_attempts"] = _empty_or_int(entry.get("selection_attempts"))
    entry["selection_passes"] = _empty_or_int(entry.get("selection_passes"))
    raw_fails = entry.get("fail_counts") if isinstance(entry.get("fail_counts"), dict) else {}
    entry["fail_counts"] = {str(k): _empty_or_int(v) for k, v in raw_fails.items() if str(k).strip()}
    entry["search_rate"] = laplace_rate(entry["search_passes"], entry["attempts"])
    entry["selection_rate"] = laplace_rate(entry["selection_passes"], entry["selection_attempts"])
    return entry


def _refresh_setup_entry(entry: dict[str, Any]) -> dict[str, Any]:
    entry["label"] = str(entry.get("label", "")).strip()
    entry["attempts"] = _empty_or_int(entry.get("attempts"))
    entry["edge_passes"] = _empty_or_int(entry.get("edge_passes"))
    entry["search_passes"] = _empty_or_int(entry.get("search_passes"))
    entry["selection_attempts"] = _empty_or_int(entry.get("selection_attempts"))
    entry["selection_passes"] = _empty_or_int(entry.get("selection_passes"))
    raw_fails = entry.get("fail_counts") if isinstance(entry.get("fail_counts"), dict) else {}
    entry["fail_counts"] = {str(k): _empty_or_int(v) for k, v in raw_fails.items() if str(k).strip()}
    recent = entry.get("recent_outcomes") if isinstance(entry.get("recent_outcomes"), list) else []
    normalized_recent: list[dict[str, Any]] = []
    for raw in recent[-5:]:
        if not isinstance(raw, dict):
            continue
        normalized_recent.append(
            {
                "strategy_name": str(raw.get("strategy_name", "")).strip(),
                "bar_config": str(raw.get("bar_config", "")).strip(),
                "edge_status": str(raw.get("edge_status", "")).strip(),
                "edge_events": _optional_int(raw.get("edge_events")),
                "positive_horizons": _optional_int(raw.get("positive_horizons")),
                "horizon_count": _optional_int(raw.get("horizon_count")),
                "best_horizon_bars": _optional_int(raw.get("best_horizon_bars")),
                "best_avg_trade_pnl": _optional_float(raw.get("best_avg_trade_pnl")),
                "best_long_avg_trade_pnl": _optional_float(raw.get("best_long_avg_trade_pnl")),
                "best_short_avg_trade_pnl": _optional_float(raw.get("best_short_avg_trade_pnl")),
                "search_verdict": str(raw.get("search_verdict", "")).strip(),
                "final_verdict": str(raw.get("final_verdict", "")).strip(),
                "failure_codes": [str(code) for code in (raw.get("failure_codes") or []) if str(code).strip()],
                "completed_at": str(raw.get("completed_at", "")).strip(),
            }
        )
    entry["recent_outcomes"] = normalized_recent
    entry["updated_at"] = str(entry.get("updated_at") or "").strip() or None
    entry["edge_rate"] = laplace_rate(entry["edge_passes"], entry["attempts"])
    entry["search_rate"] = laplace_rate(entry["search_passes"], entry["attempts"])
    entry["selection_rate"] = laplace_rate(entry["selection_passes"], entry["selection_attempts"])
    return entry


def _refresh_bar_entry(entry: dict[str, Any]) -> dict[str, Any]:
    entry["attempts"] = _empty_or_int(entry.get("attempts"))
    entry["search_passes"] = _empty_or_int(entry.get("search_passes"))
    entry["selection_attempts"] = _empty_or_int(entry.get("selection_attempts", entry.get("attempts")))
    entry["selection_passes"] = _empty_or_int(entry.get("selection_passes"))
    entry["search_rate"] = laplace_rate(entry["search_passes"], entry["attempts"])
    entry["selection_rate"] = laplace_rate(entry["selection_passes"], entry["selection_attempts"])
    return entry


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _compute_low_sample_themes(theme_stats: dict[str, Any]) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for theme, raw_entry in theme_stats.items():
        if not isinstance(raw_entry, dict):
            continue
        attempts = _empty_or_int(raw_entry.get("attempts"))
        if 0 < attempts < LOW_SAMPLE_ATTEMPTS:
            ranked.append((attempts, str(theme)))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [theme for _attempts, theme in ranked]


def _sanitize_scorecard(payload: dict[str, Any] | None) -> dict[str, Any]:
    scorecard = empty_scorecard()
    if isinstance(payload, dict):
        if payload.get("rebuilt_at") is not None:
            scorecard["rebuilt_at"] = payload.get("rebuilt_at")

        raw_setup_stats = payload.get("setup_stats") if isinstance(payload.get("setup_stats"), dict) else {}
        for setup_key, raw_entry in raw_setup_stats.items():
            key = str(setup_key).strip()
            if not key:
                continue
            entry = empty_setup_entry()
            if isinstance(raw_entry, dict):
                entry.update(raw_entry)
            scorecard["setup_stats"][key] = _refresh_setup_entry(entry)

        raw_theme_stats = payload.get("theme_stats") if isinstance(payload.get("theme_stats"), dict) else {}
        for theme, raw_entry in raw_theme_stats.items():
            theme_key = normalize_theme_tag(str(theme))
            entry = empty_theme_entry()
            if isinstance(raw_entry, dict):
                entry.update(raw_entry)
            scorecard["theme_stats"][theme_key] = _refresh_theme_entry(entry)

        raw_bar_affinity = payload.get("bar_config_affinity") if isinstance(payload.get("bar_config_affinity"), dict) else {}
        for theme, raw_bars in raw_bar_affinity.items():
            if not isinstance(raw_bars, dict):
                continue
            theme_key = normalize_theme_tag(str(theme))
            theme_bars: dict[str, Any] = {}
            for bar_config, raw_entry in raw_bars.items():
                bar_key = str(bar_config).strip()
                if not bar_key:
                    continue
                entry = empty_bar_entry()
                if isinstance(raw_entry, dict):
                    entry.update(raw_entry)
                theme_bars[bar_key] = _refresh_bar_entry(entry)
            if theme_bars:
                scorecard["bar_config_affinity"][theme_key] = theme_bars

        raw_near_misses = payload.get("near_misses") if isinstance(payload.get("near_misses"), list) else []
        near_misses: list[dict[str, Any]] = []
        for raw in raw_near_misses:
            if not isinstance(raw, dict):
                continue
            strategy = str(raw.get("strategy", "")).strip()
            bar_config = str(raw.get("bar_config", "")).strip()
            split = str(raw.get("split", "search")).strip() or "search"
            theme = normalize_theme_tag(raw.get("theme"))
            try:
                sharpe = float(raw.get("sharpe"))
            except (TypeError, ValueError):
                continue
            failed_checks = [str(v) for v in (raw.get("failed_checks") or []) if str(v).strip()]
            if not strategy or not bar_config or not failed_checks:
                continue
            near_misses.append(
                {
                    "strategy": strategy,
                    "theme": theme,
                    "bar_config": bar_config,
                    "sharpe": sharpe,
                    "failed_checks": failed_checks,
                    "split": split,
                }
            )
        scorecard["near_misses"] = _normalize_near_misses(near_misses)

    scorecard["low_sample_themes"] = _compute_low_sample_themes(scorecard["theme_stats"])
    scorecard["schema_version"] = SCHEMA_VERSION
    return scorecard


def read_learning_scorecard(
    scorecard_path: Path,
    scorecard_lock: Path,
) -> dict[str, Any]:
    payload = read_json_file_if_exists(
        json_path=scorecard_path,
        lock_path=scorecard_lock,
        default_payload=empty_scorecard(),
    )
    return _sanitize_scorecard(payload)


def _ensure_theme(scorecard: dict[str, Any], theme: str) -> dict[str, Any]:
    theme_stats = scorecard.setdefault("theme_stats", {})
    existing = theme_stats.get(theme)
    if not isinstance(existing, dict):
        existing = empty_theme_entry()
        theme_stats[theme] = existing
    return existing


def _ensure_setup(scorecard: dict[str, Any], setup_key: str, setup_label: str) -> dict[str, Any]:
    setup_stats = scorecard.setdefault("setup_stats", {})
    existing = setup_stats.get(setup_key)
    if not isinstance(existing, dict):
        existing = empty_setup_entry()
        setup_stats[setup_key] = existing
    if setup_label and not str(existing.get("label", "")).strip():
        existing["label"] = str(setup_label)
    return existing


def _ensure_bar(scorecard: dict[str, Any], theme: str, bar_config: str) -> dict[str, Any]:
    affinity = scorecard.setdefault("bar_config_affinity", {})
    bars = affinity.get(theme)
    if not isinstance(bars, dict):
        bars = {}
        affinity[theme] = bars
    existing = bars.get(bar_config)
    if not isinstance(existing, dict):
        existing = empty_bar_entry()
        bars[bar_config] = existing
    return existing


def _extract_failed_gauntlet_checks(result: dict[str, Any]) -> list[str]:
    gauntlet = result.get("gauntlet") if isinstance(result.get("gauntlet"), dict) else {}
    failed: list[str] = []
    for name, payload in gauntlet.items():
        if name in {"overall_verdict", "pass_count", "total_tests"}:
            continue
        if isinstance(payload, dict) and str(payload.get("verdict", "")).upper() == "FAIL":
            failed.append(str(name))
    return failed


def _extract_failed_advanced_checks(result: dict[str, Any]) -> list[str]:
    gates = result.get("advanced_validation_gates")
    if not isinstance(gates, dict) or not bool(gates.get("enabled", False)):
        return []
    checks = gates.get("checks") if isinstance(gates.get("checks"), dict) else {}
    name_map = {
        "alpha_decay_verdict": "alpha_decay",
        "factor_verdict": "factor_verdict",
        "min_dsr_probability": "deflated_sharpe",
    }
    failed: list[str] = []
    for name, payload in checks.items():
        if isinstance(payload, dict) and not bool(payload.get("passed", False)):
            failed.append(name_map.get(str(name), str(name)))
    return failed


def _edge_probe_payload(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    edge_probe = result.get("edge_probe")
    return dict(edge_probe) if isinstance(edge_probe, dict) else {}


def _edge_probe_failed(result: dict[str, Any] | None) -> bool:
    edge_probe = _edge_probe_payload(result)
    return bool(edge_probe) and not bool(edge_probe.get("passed", True))


def _best_edge_probe_horizon(edge_probe: dict[str, Any]) -> dict[str, Any]:
    horizon_results = edge_probe.get("horizon_results") if isinstance(edge_probe.get("horizon_results"), list) else []
    best_horizon = _optional_int(edge_probe.get("best_horizon_bars"))
    for row in horizon_results:
        if not isinstance(row, dict):
            continue
        if best_horizon is not None and _optional_int(row.get("horizon_bars")) == best_horizon:
            return dict(row)
    for row in horizon_results:
        if isinstance(row, dict):
            return dict(row)
    return {}


def _edge_probe_outcome_summary(result: dict[str, Any] | None) -> dict[str, Any]:
    edge_probe = _edge_probe_payload(result)
    if not edge_probe:
        return {}
    best_horizon = _best_edge_probe_horizon(edge_probe)
    horizon_results = edge_probe.get("horizon_results") if isinstance(edge_probe.get("horizon_results"), list) else []
    return {
        "edge_events": _optional_int(edge_probe.get("events")),
        "positive_horizons": _optional_int(edge_probe.get("positive_horizons")),
        "horizon_count": sum(1 for row in horizon_results if isinstance(row, dict)),
        "best_horizon_bars": _optional_int(edge_probe.get("best_horizon_bars")),
        "best_avg_trade_pnl": _optional_float(edge_probe.get("best_avg_trade_pnl")),
        "best_long_avg_trade_pnl": _optional_float(best_horizon.get("long_avg_trade_pnl")),
        "best_short_avg_trade_pnl": _optional_float(best_horizon.get("short_avg_trade_pnl")),
    }


def _failure_codes_from_result(result: dict[str, Any] | None) -> list[str]:
    if not isinstance(result, dict):
        return []
    failure_code = str(result.get("failure_code", "")).strip()
    if _edge_probe_failed(result):
        codes: set[str] = set()
        if failure_code:
            codes.add(failure_code)
        edge_probe = _edge_probe_payload(result)
        status = str(edge_probe.get("status", "")).strip()
        if status:
            codes.add(status)
        return sorted(codes)
    codes = set(_extract_failed_gauntlet_checks(result))
    codes.update(_extract_failed_advanced_checks(result))
    if failure_code:
        codes.add(failure_code)
    return sorted(codes)


def _search_result_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    search_result = row.get("search_result")
    if not isinstance(search_result, dict):
        search_result = details.get("search_result")
    if isinstance(search_result, dict):
        return dict(search_result)

    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else details.get("metrics")
    gauntlet = row.get("gauntlet") if isinstance(row.get("gauntlet"), dict) else details.get("gauntlet")
    advanced = (
        row.get("advanced_validation")
        if isinstance(row.get("advanced_validation"), dict)
        else details.get("advanced_validation")
    )
    gates = (
        row.get("advanced_validation_gates")
        if isinstance(row.get("advanced_validation_gates"), dict)
        else details.get("advanced_validation_gates")
    )
    feedback_verdict = details.get("feedback_verdict") if details else row.get("verdict")
    if not any(isinstance(value, dict) for value in (metrics, gauntlet, advanced, gates)) and feedback_verdict is None:
        return None
    return {
        "verdict": str(feedback_verdict or "").upper(),
        "metrics": dict(metrics) if isinstance(metrics, dict) else {},
        "gauntlet": dict(gauntlet) if isinstance(gauntlet, dict) else {},
        "advanced_validation": dict(advanced) if isinstance(advanced, dict) else {},
        "advanced_validation_gates": dict(gates) if isinstance(gates, dict) else {},
    }


def _selection_result_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    selection_result = row.get("selection_result")
    if not isinstance(selection_result, dict):
        selection_result = details.get("selection_result")
    return dict(selection_result) if isinstance(selection_result, dict) else None


def _task_theme_tag(task: dict[str, Any]) -> str:
    theme = task.get("theme_tag")
    if theme is None:
        source = task.get("source") if isinstance(task.get("source"), dict) else {}
        theme = source.get("theme_tag")
    return normalize_theme_tag(theme)


def _metrics_sharpe(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    value = metrics.get("sharpe_ratio")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _increment_fail_counts(theme_entry: dict[str, Any], codes: list[str]) -> None:
    fail_counts = theme_entry.setdefault("fail_counts", {})
    for code in codes:
        if not str(code).strip():
            continue
        key = str(code)
        fail_counts[key] = _empty_or_int(fail_counts.get(key)) + 1


def _normalize_near_misses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        strategy = str(row.get("strategy", "")).strip()
        bar_config = str(row.get("bar_config", "")).strip()
        split = str(row.get("split", "search")).strip() or "search"
        if not strategy or not bar_config:
            continue
        key = (strategy, bar_config, split)
        current = deduped.get(key)
        try:
            sharpe = float(row.get("sharpe"))
        except (TypeError, ValueError):
            continue
        normalized = {
            "strategy": strategy,
            "theme": str(row.get("theme", OTHER_THEME_TAG)).strip() or OTHER_THEME_TAG,
            "bar_config": bar_config,
            "sharpe": sharpe,
            "failed_checks": [str(v) for v in (row.get("failed_checks") or []) if str(v).strip()],
            "split": split,
        }
        if current is None or sharpe > float(current.get("sharpe", float("-inf"))):
            deduped[key] = normalized
    ordered = sorted(deduped.values(), key=lambda row: float(row.get("sharpe", float("-inf"))), reverse=True)
    return ordered[:NEAR_MISS_LIMIT]


def _add_near_miss(scorecard: dict[str, Any], entry: dict[str, Any]) -> None:
    existing = scorecard.get("near_misses") if isinstance(scorecard.get("near_misses"), list) else []
    existing.append(entry)
    scorecard["near_misses"] = _normalize_near_misses(existing)


def _apply_task_update(scorecard: dict[str, Any], task: dict[str, Any]) -> None:
    if not isinstance(task, dict):
        return
    bar_config = str(task.get("bar_config", "")).strip()
    if not bar_config:
        return
    setup_key, setup_label = task_setup_identity(task)
    theme = _task_theme_tag(task)
    setup_entry = _ensure_setup(scorecard, setup_key, setup_label)
    theme_entry = _ensure_theme(scorecard, theme)
    bar_entry = _ensure_bar(scorecard, theme, bar_config)

    setup_entry["attempts"] = _empty_or_int(setup_entry.get("attempts")) + 1
    bar_entry["attempts"] = _empty_or_int(bar_entry.get("attempts")) + 1

    search_result = _search_result_from_row(task)
    selection_result = _selection_result_from_row(task)
    edge_probe = (
        _edge_probe_payload(search_result)
        if isinstance(search_result, dict)
        else {}
    )
    edge_pass = not edge_probe or bool(edge_probe.get("passed", True))
    search_pass = isinstance(search_result, dict) and str(search_result.get("verdict", "")).upper() == "PASS"
    selection_pass = isinstance(selection_result, dict) and str(selection_result.get("verdict", "")).upper() == "PASS"

    if edge_pass:
        setup_entry["edge_passes"] = _empty_or_int(setup_entry.get("edge_passes")) + 1
    if search_pass:
        setup_entry["search_passes"] = _empty_or_int(setup_entry.get("search_passes")) + 1
        bar_entry["search_passes"] = _empty_or_int(bar_entry.get("search_passes")) + 1
    if selection_result is not None:
        setup_entry["selection_attempts"] = _empty_or_int(setup_entry.get("selection_attempts")) + 1
        bar_entry["selection_attempts"] = _empty_or_int(bar_entry.get("selection_attempts")) + 1
    if selection_pass:
        setup_entry["selection_passes"] = _empty_or_int(setup_entry.get("selection_passes")) + 1
        bar_entry["selection_passes"] = _empty_or_int(bar_entry.get("selection_passes")) + 1
    _refresh_setup_entry(setup_entry)
    _refresh_bar_entry(bar_entry)

    failure_codes = set(_failure_codes_from_result(search_result))
    failure_codes.update(_failure_codes_from_result(selection_result))
    if failure_codes:
        _increment_fail_counts(setup_entry, sorted(failure_codes))
        _increment_fail_counts(theme_entry, sorted(failure_codes))
    edge_summary = _edge_probe_outcome_summary(search_result)
    recent = setup_entry.setdefault("recent_outcomes", [])
    recent.append(
        {
            "strategy_name": str(task.get("strategy_name", "")).strip(),
            "bar_config": bar_config,
            "edge_status": str(edge_probe.get("status", "pass" if edge_pass else "unknown")),
            **edge_summary,
            "search_verdict": str(search_result.get("verdict", "")) if isinstance(search_result, dict) else "",
            "final_verdict": str(
                (
                    task.get("details")
                    if isinstance(task.get("details"), dict)
                    else {}
                ).get("final_verdict", task.get("verdict", ""))
            ),
            "failure_codes": sorted(failure_codes),
            "completed_at": str(task.get("completed_at", "")).strip(),
        }
    )
    setup_entry["recent_outcomes"] = recent[-5:]
    setup_entry["updated_at"] = str(task.get("completed_at") or _utc_now())
    _refresh_setup_entry(setup_entry)
    _refresh_theme_entry(theme_entry)

    search_sharpe = _metrics_sharpe(search_result)
    if (
        isinstance(search_result, dict)
        and str(search_result.get("verdict", "")).upper() == "FAIL"
        and search_sharpe is not None
        and search_sharpe > 0.0
    ):
        failed_checks = _failure_codes_from_result(search_result)
        if failed_checks:
            _add_near_miss(
                scorecard,
                {
                    "strategy": str(task.get("strategy_name", "")).strip(),
                    "theme": theme,
                    "bar_config": bar_config,
                    "sharpe": search_sharpe,
                    "failed_checks": failed_checks,
                    "split": "search",
                },
            )

    selection_sharpe = _metrics_sharpe(selection_result)
    if (
        isinstance(selection_result, dict)
        and str(selection_result.get("verdict", "")).upper() == "FAIL"
        and selection_sharpe is not None
        and selection_sharpe > 0.0
    ):
        failed_checks = _failure_codes_from_result(selection_result)
        if failed_checks:
            _add_near_miss(
                scorecard,
                {
                    "strategy": str(task.get("strategy_name", "")).strip(),
                    "theme": theme,
                    "bar_config": bar_config,
                    "sharpe": selection_sharpe,
                    "failed_checks": failed_checks,
                    "split": "selection",
                },
            )


def _apply_handoff_update(scorecard: dict[str, Any], handoff: dict[str, Any]) -> None:
    if not isinstance(handoff, dict):
        return
    if str(handoff.get("handoff_type", "")) != "validation_request":
        return
    payload = handoff.get("payload") if isinstance(handoff.get("payload"), dict) else {}
    result = handoff.get("result") if isinstance(handoff.get("result"), dict) else {}
    theme = normalize_theme_tag(payload.get("theme_tag"))
    theme_entry = _ensure_theme(scorecard, theme)
    theme_entry["attempts"] = _empty_or_int(theme_entry.get("attempts")) + 1

    if _empty_or_int(result.get("pass_count")) > 0:
        theme_entry["search_passes"] = _empty_or_int(theme_entry.get("search_passes")) + 1

    task_rows = result.get("tasks") if isinstance(result.get("tasks"), list) else []
    if task_rows:
        selection_attempt = any(bool(row.get("selection_attempted", False)) for row in task_rows if isinstance(row, dict))
        selection_pass = any(
            str(row.get("selection_verdict", "")).upper() == "PASS"
            for row in task_rows
            if isinstance(row, dict)
        )
    else:
        selection_attempt = bool(result.get("selection_attempted", False))
        selection_pass = bool(result.get("selection_passed", False))
    if selection_attempt:
        theme_entry["selection_attempts"] = _empty_or_int(theme_entry.get("selection_attempts")) + 1
    if selection_pass:
        theme_entry["selection_passes"] = _empty_or_int(theme_entry.get("selection_passes")) + 1
    _refresh_theme_entry(theme_entry)


def _finalize_scorecard(scorecard: dict[str, Any]) -> dict[str, Any]:
    setup_stats = scorecard.get("setup_stats") if isinstance(scorecard.get("setup_stats"), dict) else {}
    for setup_key, entry in list(setup_stats.items()):
        if isinstance(entry, dict):
            setup_stats[setup_key] = _refresh_setup_entry(entry)
    theme_stats = scorecard.get("theme_stats") if isinstance(scorecard.get("theme_stats"), dict) else {}
    for theme, entry in list(theme_stats.items()):
        if isinstance(entry, dict):
            theme_stats[theme] = _refresh_theme_entry(entry)
    bar_affinity = scorecard.get("bar_config_affinity") if isinstance(scorecard.get("bar_config_affinity"), dict) else {}
    for theme, bars in list(bar_affinity.items()):
        if not isinstance(bars, dict):
            bar_affinity[theme] = {}
            continue
        for bar_config, entry in list(bars.items()):
            if isinstance(entry, dict):
                bars[bar_config] = _refresh_bar_entry(entry)
    scorecard["near_misses"] = _normalize_near_misses(
        scorecard.get("near_misses") if isinstance(scorecard.get("near_misses"), list) else [],
    )
    scorecard["low_sample_themes"] = _compute_low_sample_themes(theme_stats)
    scorecard["schema_version"] = SCHEMA_VERSION
    scorecard["rebuilt_at"] = _utc_now()
    return scorecard


def update_learning_scorecard(
    scorecard_path: Path,
    scorecard_lock: Path,
    *,
    task: dict[str, Any] | None = None,
    handoff: dict[str, Any] | None = None,
) -> None:
    if (task is None) == (handoff is None):
        raise ValueError("Provide exactly one of task or handoff")

    def _update(payload: dict[str, Any]) -> dict[str, Any]:
        scorecard = _sanitize_scorecard(payload)
        if task is not None:
            _apply_task_update(scorecard, task)
        else:
            _apply_handoff_update(scorecard, handoff or {})
        return _finalize_scorecard(scorecard)

    update_json_file(
        json_path=scorecard_path,
        lock_path=scorecard_lock,
        default_payload=empty_scorecard(),
        update_fn=_update,
    )


def rebuild_learning_scorecard(
    experiments_path: Path,
    handoffs_path: Path,
    handoffs_lock: Path,
    scorecard_path: Path,
    scorecard_lock: Path,
) -> None:
    scorecard = empty_scorecard()
    seen_handoffs: set[str] = set()

    if experiments_path.exists():
        for raw in experiments_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            event = str(row.get("event", ""))
            if event == "task_result":
                _apply_task_update(scorecard, row)
                continue
            if event == "validation_handoff_completed":
                handoff_id = str(row.get("handoff_id", "")).strip()
                if handoff_id:
                    seen_handoffs.add(handoff_id)
                _apply_handoff_update(
                    scorecard,
                    {
                        "handoff_type": "validation_request",
                        "payload": {"theme_tag": row.get("theme_tag")},
                        "result": {
                            "pass_count": row.get("pass_count", 0),
                            "selection_attempted": row.get("selection_attempted", False),
                            "selection_passed": row.get("selection_passed", False),
                        },
                    },
                )

    handoffs_payload = read_json_file_if_exists(
        json_path=handoffs_path,
        lock_path=handoffs_lock,
        default_payload={"schema_version": SCHEMA_VERSION, "pending": [], "completed": []},
    )
    for row in handoffs_payload.get("completed", []):
        handoff_id = str(row.get("handoff_id", "")).strip()
        if handoff_id and handoff_id in seen_handoffs:
            continue
        _apply_handoff_update(scorecard, row)

    scorecard = _finalize_scorecard(scorecard)
    update_json_file(
        json_path=scorecard_path,
        lock_path=scorecard_lock,
        default_payload=empty_scorecard(),
        update_fn=lambda _payload: scorecard,
    )


def _format_latest_edge_summary(latest: dict[str, Any]) -> str:
    if not isinstance(latest, dict):
        return ""
    parts: list[str] = []
    edge_status = str(latest.get("edge_status", "")).strip()
    if edge_status:
        parts.append(f"last edge={edge_status}")
    edge_events = _optional_int(latest.get("edge_events"))
    if edge_events is not None:
        parts.append(f"events={edge_events}")
    positive_horizons = _optional_int(latest.get("positive_horizons"))
    horizon_count = _optional_int(latest.get("horizon_count"))
    if positive_horizons is not None and horizon_count is not None and horizon_count > 0:
        parts.append(f"pos_horizons={positive_horizons}/{horizon_count}")
    best_horizon_bars = _optional_int(latest.get("best_horizon_bars"))
    best_avg_trade_pnl = _optional_float(latest.get("best_avg_trade_pnl"))
    if best_horizon_bars is not None and best_avg_trade_pnl is not None:
        parts.append(f"best={best_horizon_bars}b avg={best_avg_trade_pnl:.2f}")
    long_avg = _optional_float(latest.get("best_long_avg_trade_pnl"))
    short_avg = _optional_float(latest.get("best_short_avg_trade_pnl"))
    if long_avg is not None or short_avg is not None:
        direction_parts: list[str] = []
        if long_avg is not None:
            direction_parts.append(f"long={long_avg:.2f}")
        if short_avg is not None:
            direction_parts.append(f"short={short_avg:.2f}")
        parts.append(" ".join(direction_parts))
    return " | ".join(parts)


def format_learning_context(scorecard: dict[str, Any]) -> str:
    if not isinstance(scorecard, dict):
        return ""

    setup_stats = scorecard.get("setup_stats") if isinstance(scorecard.get("setup_stats"), dict) else {}
    theme_stats = scorecard.get("theme_stats") if isinstance(scorecard.get("theme_stats"), dict) else {}
    bar_affinity = scorecard.get("bar_config_affinity") if isinstance(scorecard.get("bar_config_affinity"), dict) else {}
    near_misses = scorecard.get("near_misses") if isinstance(scorecard.get("near_misses"), list) else []
    low_sample = [str(v) for v in (scorecard.get("low_sample_themes") or []) if str(v).strip()]

    lines: list[str] = ["LEARNING_SCORECARD:"]
    wrote_section = False

    setup_rows: list[tuple[str, dict[str, Any]]] = []
    for setup_key, entry in setup_stats.items():
        if not isinstance(entry, dict):
            continue
        if _empty_or_int(entry.get("attempts")) <= 0:
            continue
        setup_rows.append((str(setup_key), entry))
    setup_rows.sort(
        key=lambda item: (
            str(item[1].get("updated_at") or ""),
            _empty_or_int(item[1].get("search_passes")),
            _empty_or_int(item[1].get("edge_passes")),
            _empty_or_int(item[1].get("attempts")),
        ),
        reverse=True,
    )
    if setup_rows:
        wrote_section = True
        lines.append("Recent concrete setup outcomes:")
        for _setup_key, entry in setup_rows[:3]:
            label = str(entry.get("label", "")).strip() or "unknown setup"
            recent = entry.get("recent_outcomes") if isinstance(entry.get("recent_outcomes"), list) else []
            latest = recent[-1] if recent else {}
            latest_summary = _format_latest_edge_summary(latest if isinstance(latest, dict) else {})
            selection_attempts = _empty_or_int(entry.get("selection_attempts"))
            selection_blob = (
                f" | selection {_empty_or_int(entry.get('selection_passes'))}/{selection_attempts}"
                if selection_attempts > 0
                else ""
            )
            lines.append(
                "  "
                f"{label}: tries={_empty_or_int(entry.get('attempts'))} | "
                f"edge {_empty_or_int(entry.get('edge_passes'))}/{_empty_or_int(entry.get('attempts'))} | "
                f"search {_empty_or_int(entry.get('search_passes'))}/{_empty_or_int(entry.get('attempts'))}"
                f"{selection_blob}"
                + (f" | {latest_summary}" if latest_summary else "")
            )

    failure_rows: list[tuple[str, str]] = []
    for _setup_key, entry in setup_rows:
        fail_counts = entry.get("fail_counts") if isinstance(entry.get("fail_counts"), dict) else {}
        ranked = sorted(
            ((str(name), _empty_or_int(count)) for name, count in fail_counts.items() if _empty_or_int(count) > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            label = str(entry.get("label", "")).strip() or "unknown setup"
            failure_rows.append((label, ", ".join(f"{name} ({count})" for name, count in ranked[:2])))
    if failure_rows:
        wrote_section = True
        lines.append("Repeated setup failure modes:")
        for label, summary in failure_rows[:3]:
            lines.append(f"  {label}: {summary}")

    theme_rows: list[tuple[str, dict[str, Any]]] = []
    for theme, entry in theme_stats.items():
        if not isinstance(entry, dict):
            continue
        if _empty_or_int(entry.get("attempts")) <= 0:
            continue
        theme_rows.append((str(theme), entry))
    theme_rows.sort(
        key=lambda item: (
            _empty_or_int(item[1].get("selection_passes")),
            _empty_or_int(item[1].get("search_passes")),
            _empty_or_int(item[1].get("attempts")),
        ),
        reverse=True,
    )
    if theme_rows:
        wrote_section = True
        lines.append("Theme performance (secondary audit view):")
        for theme, entry in theme_rows[:2]:
            lines.append(
                "  "
                f"{theme}: "
                f"search {_empty_or_int(entry.get('search_passes'))}/{_empty_or_int(entry.get('attempts'))} | "
                f"selection {_empty_or_int(entry.get('selection_passes'))}/{_empty_or_int(entry.get('selection_attempts'))}"
            )

    affinity_rows: list[tuple[str, str, dict[str, Any]]] = []
    for theme, bars in bar_affinity.items():
        if not isinstance(bars, dict):
            continue
        for bar_config, entry in bars.items():
            if not isinstance(entry, dict):
                continue
            if _empty_or_int(entry.get("attempts")) <= 0:
                continue
            affinity_rows.append((str(theme), str(bar_config), entry))
    affinity_rows.sort(
        key=lambda item: (
            _empty_or_int(item[2].get("selection_passes")),
            _empty_or_int(item[2].get("search_passes")),
            _empty_or_int(item[2].get("attempts")),
        ),
        reverse=True,
    )
    if affinity_rows:
        wrote_section = True
        lines.append("Bar config affinity (secondary):")
        for theme, bar_config, entry in affinity_rows[:2]:
            lines.append(
                "  "
                f"{theme} + {bar_config}: "
                f"search {_empty_or_int(entry.get('search_passes'))}/{_empty_or_int(entry.get('attempts'))} | "
                f"selection {_empty_or_int(entry.get('selection_passes'))}/{_empty_or_int(entry.get('selection_attempts', entry.get('attempts')))}"
            )

    if near_misses:
        wrote_section = True
        lines.append("Near misses (positive Sharpe, failed gauntlet):")
        for row in near_misses[:5]:
            failed_checks = ", ".join(str(v) for v in row.get("failed_checks", [])[:3])
            lines.append(
                "  "
                f"{row.get('strategy')} [{row.get('bar_config')}] "
                f"sharpe={float(row.get('sharpe', 0.0)):.2f} "
                f"failed=[{failed_checks}] ({row.get('split', 'search')})"
            )

    if low_sample:
        wrote_section = True
        lines.append("Low-sample themes: " + ", ".join(low_sample))

    if not wrote_section:
        return ""

    lines.append("Prioritize setups with real raw edge across horizons and avoid repeating setups that keep failing edge probe.")
    return "\n".join(lines)


__all__ = [
    "NEAR_MISS_LIMIT",
    "OTHER_THEME_TAG",
    "SCHEMA_VERSION",
    "LOW_SAMPLE_ATTEMPTS",
    "empty_bar_entry",
    "empty_scorecard",
    "empty_setup_entry",
    "empty_theme_entry",
    "format_learning_context",
    "laplace_rate",
    "normalize_theme_tag",
    "read_learning_scorecard",
    "rebuild_learning_scorecard",
    "resolve_focus_anchors",
    "update_learning_scorecard",
]
