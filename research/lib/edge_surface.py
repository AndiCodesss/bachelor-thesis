"""Conditional entry-edge surface for search-stage discovery."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from src.framework.api import compute_adaptive_costs, compute_metrics
from src.framework.backtest.engine import TRADE_SCHEMA
from src.framework.backtest.metrics import compute_daily_pnl_series

_DEFAULT_EDGE_SURFACE = {
    "enabled": False,
    "horizons": [1, 3, 5, 10, 20, 40, 60, 90],
    "min_events": 60,
    "min_pocket_events": 30,
    "min_positive_horizons": 1,
    "min_avg_trade_pnl": 0.0,
    "min_positive_day_fraction": 0.52,
    "max_day_concentration": 0.40,
    "max_best_pockets": 5,
}
_CORE_ANALYSIS_COLUMNS = (
    "ts_event",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid_price",
    "ask_price",
    "signal",
)
_CONTEXT_ANALYSIS_COLUMNS = (
    "spread_bps",
    "regime_vol_relative",
    "trade_intensity",
    "return_5bar",
    "position_in_va",
)
EDGE_SURFACE_ANALYSIS_COLUMNS = list(_CORE_ANALYSIS_COLUMNS + _CONTEXT_ANALYSIS_COLUMNS)
_NUMERIC_BUCKET_COLUMNS = {
    "vol_bucket": "regime_vol_relative",
    "spread_bucket": "spread_bps",
    "activity_bucket": "trade_intensity",
    "direction_bucket": "return_5bar",
}
_NUMERIC_BUCKET_LABELS = {
    "vol_bucket": ("low", "medium", "high"),
    "spread_bucket": ("tight", "normal", "wide"),
    "activity_bucket": ("quiet", "normal", "active"),
    "direction_bucket": ("mean_revert", "neutral", "trend"),
}


@dataclass(frozen=True)
class EntryEvent:
    signal_index: int
    fill_index: int
    entry_time: Any
    entry_price: float
    direction: int


def normalize_edge_surface_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(_DEFAULT_EDGE_SURFACE)
    if isinstance(payload, dict):
        config.update(payload)

    horizons_raw = config.get("horizons")
    if not isinstance(horizons_raw, list):
        horizons_raw = _DEFAULT_EDGE_SURFACE["horizons"]
    horizons = sorted({max(1, int(value)) for value in horizons_raw if value is not None})

    min_events = max(1, int(config.get("min_events", _DEFAULT_EDGE_SURFACE["min_events"])))
    raw_min_pocket_events = config.get("min_pocket_events", _DEFAULT_EDGE_SURFACE["min_pocket_events"])
    min_pocket_events = max(1, int(raw_min_pocket_events))
    min_pocket_events = min(min_pocket_events, min_events)

    return {
        "enabled": bool(config.get("enabled", False)),
        "horizons": horizons or list(_DEFAULT_EDGE_SURFACE["horizons"]),
        "min_events": min_events,
        "min_pocket_events": min_pocket_events,
        "min_positive_horizons": max(
            1,
            int(config.get("min_positive_horizons", _DEFAULT_EDGE_SURFACE["min_positive_horizons"])),
        ),
        "min_avg_trade_pnl": float(config.get("min_avg_trade_pnl", _DEFAULT_EDGE_SURFACE["min_avg_trade_pnl"])),
        "min_positive_day_fraction": float(
            config.get("min_positive_day_fraction", _DEFAULT_EDGE_SURFACE["min_positive_day_fraction"]),
        ),
        "max_day_concentration": float(
            config.get("max_day_concentration", _DEFAULT_EDGE_SURFACE["max_day_concentration"]),
        ),
        "max_best_pockets": max(
            1,
            int(config.get("max_best_pockets", _DEFAULT_EDGE_SURFACE["max_best_pockets"])),
        ),
    }


def _extract_entry_events(df: pl.DataFrame, *, entry_on_next_open: bool) -> list[EntryEvent]:
    if len(df) == 0:
        return []

    ordered = df.sort("ts_event")
    timestamps = ordered["ts_event"].to_list()
    closes = ordered["close"].to_list()
    opens = ordered["open"].to_list() if "open" in ordered.columns else closes
    signals = ordered["signal"].to_list()

    entries: list[EntryEvent] = []
    position = 0
    pending_direction = 0
    pending_signal_index: int | None = None

    for index in range(len(signals)):
        bar_date = timestamps[index].date()
        has_next_bar = index < len(signals) - 1
        is_last_bar = (not has_next_bar) or (timestamps[index + 1].date() != bar_date)

        signal_to_execute = pending_direction
        signal_index_to_execute = pending_signal_index
        pending_direction = 0
        pending_signal_index = None

        if signal_to_execute != 0 and position == 0 and signal_index_to_execute is not None:
            entries.append(
                EntryEvent(
                    signal_index=int(signal_index_to_execute),
                    fill_index=index,
                    entry_time=timestamps[index],
                    entry_price=float(opens[index]),
                    direction=int(signal_to_execute),
                ),
            )
            position = int(signal_to_execute)

        signal_value = int(signals[index])
        if signal_value != position:
            if position != 0:
                position = 0
            if signal_value != 0:
                if entry_on_next_open:
                    if has_next_bar:
                        pending_direction = signal_value
                        pending_signal_index = index
                elif not is_last_bar:
                    entries.append(
                        EntryEvent(
                            signal_index=index,
                            fill_index=index,
                            entry_time=timestamps[index],
                            entry_price=float(closes[index]),
                            direction=signal_value,
                        ),
                    )
                    position = signal_value

        if is_last_bar and position != 0:
            position = 0

    return entries


def _ensure_required_columns(frame: pl.DataFrame, *, entry_on_next_open: bool) -> None:
    required = {"ts_event", "close", "signal"}
    if entry_on_next_open:
        required.add("open")
    missing = sorted(required - set(frame.columns))
    if missing:
        missing_blob = ", ".join(missing)
        raise ValueError(f"edge_surface requires columns: {missing_blob}")


def _finite_float_array(series: pl.Series) -> np.ndarray:
    values = np.asarray(series.cast(pl.Float64, strict=False).to_numpy(), dtype=np.float64)
    return values[np.isfinite(values)]


def _tercile_thresholds(series: pl.Series) -> tuple[float, float] | None:
    finite = _finite_float_array(series)
    if finite.size < 3:
        return None
    lower = float(np.quantile(finite, 1.0 / 3.0))
    upper = float(np.quantile(finite, 2.0 / 3.0))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if upper <= lower:
        return None
    return lower, upper


def _numeric_bucket_labels(
    series: pl.Series,
    *,
    thresholds: tuple[float, float],
    labels: tuple[str, str, str],
) -> list[str | None]:
    values = np.asarray(series.cast(pl.Float64, strict=False).to_numpy(), dtype=np.float64)
    lower, upper = thresholds
    low_label, mid_label, high_label = labels
    out: list[str | None] = []
    for value in values:
        if not np.isfinite(value):
            out.append(None)
        elif value <= lower:
            out.append(low_label)
        elif value < upper:
            out.append(mid_label)
        else:
            out.append(high_label)
    return out


def _value_area_bucket_labels(series: pl.Series) -> list[str | None]:
    values = np.asarray(series.cast(pl.Float64, strict=False).to_numpy(), dtype=np.float64)
    out: list[str | None] = []
    for value in values:
        if not np.isfinite(value):
            out.append(None)
        elif value < 0.0:
            out.append("below_value")
        elif value > 1.0:
            out.append("above_value")
        else:
            out.append("inside_value")
    return out


def _session_bucket_labels(frame: pl.DataFrame) -> list[str]:
    minute_of_day = frame.select(
        (
            pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.hour().cast(pl.Int32) * 60
            + pl.col("ts_event").dt.convert_time_zone("US/Eastern").dt.minute().cast(pl.Int32)
        ).alias("_minute_of_day")
    )["_minute_of_day"].to_list()
    out: list[str] = []
    for minute in minute_of_day:
        if minute < 180:
            out.append("eth_overnight")
        elif minute < 570:
            out.append("london")
        elif minute < 660:
            out.append("rth_open")
        elif minute < 900:
            out.append("rth_midday")
        elif minute < 960:
            out.append("power_hour")
        else:
            out.append("eth_overnight")
    return out


def _side_label(direction: int) -> str:
    return "long" if int(direction) > 0 else "short"


def _daily_edge_stats(trades: pl.DataFrame) -> tuple[float, float]:
    if len(trades) == 0:
        return 0.0, 1.0
    cost_col = "adaptive_cost_rt" if "adaptive_cost_rt" in trades.columns else None
    daily = compute_daily_pnl_series(trades, cost_override_col=cost_col)
    if len(daily) == 0:
        return 0.0, 1.0
    pnl_col = "net_pnl" if "net_pnl" in daily.columns else daily.columns[-1]
    pnl = daily[pnl_col]
    positive_days = int((pnl > 0).sum())
    total_days = len(daily)
    positive_fraction = float(positive_days / total_days) if total_days else 0.0
    abs_sum = float(pnl.abs().sum())
    if abs_sum <= 0.0:
        concentration = 1.0
    else:
        concentration = float((pnl.abs() / abs_sum).max())
    return positive_fraction, concentration


def _direction_avg_trade_pnl(trades: pl.DataFrame, direction: int) -> float | None:
    subset = trades.filter(pl.col("direction") == int(direction))
    if len(subset) == 0:
        return None
    cost_col = "adaptive_cost_rt" if "adaptive_cost_rt" in subset.columns else None
    metrics = compute_metrics(subset, cost_override_col=cost_col)
    return float(metrics.get("avg_trade_pnl", 0.0))


def _best_horizon_result(horizon_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not horizon_results:
        return {}
    return max(
        horizon_results,
        key=lambda row: (
            bool(row.get("passed", False)),
            float(row.get("avg_trade_pnl", 0.0)),
            float(row.get("net_pnl", 0.0)),
            int(row.get("trade_count", 0)),
            -int(row.get("horizon_bars", 0) or 0),
        ),
    )


def _evaluate_trade_population(
    *,
    trades: pl.DataFrame,
    base_event_count: int,
    min_required_events: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    horizons = list(config["horizons"])
    horizon_results: list[dict[str, Any]] = []
    positive_horizons = 0

    for horizon in horizons:
        subset = trades.filter(pl.col("horizon_bars") == int(horizon)) if len(trades) > 0 else pl.DataFrame()
        if len(subset) == 0:
            horizon_results.append(
                {
                    "horizon_bars": int(horizon),
                    "trade_count": 0,
                    "avg_trade_pnl": 0.0,
                    "net_pnl": 0.0,
                    "sharpe_ratio": 0.0,
                    "positive_day_fraction": 0.0,
                    "max_day_concentration": 1.0,
                    "long_avg_trade_pnl": None,
                    "short_avg_trade_pnl": None,
                    "passed": False,
                },
            )
            continue

        metrics = compute_metrics(
            subset,
            cost_override_col="adaptive_cost_rt" if "adaptive_cost_rt" in subset.columns else None,
        )
        positive_day_fraction, max_day_concentration = _daily_edge_stats(subset)
        trade_count = int(metrics.get("trade_count", 0))
        avg_trade_pnl = float(metrics.get("avg_trade_pnl", 0.0))
        net_pnl = float(metrics.get("net_pnl", 0.0))
        passed = (
            trade_count >= int(min_required_events)
            and avg_trade_pnl > float(config["min_avg_trade_pnl"])
            and net_pnl > 0.0
            and positive_day_fraction >= float(config["min_positive_day_fraction"])
            and max_day_concentration <= float(config["max_day_concentration"])
        )
        if passed:
            positive_horizons += 1
        horizon_results.append(
            {
                "horizon_bars": int(horizon),
                "trade_count": trade_count,
                "avg_trade_pnl": avg_trade_pnl,
                "net_pnl": net_pnl,
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                "positive_day_fraction": positive_day_fraction,
                "max_day_concentration": max_day_concentration,
                "long_avg_trade_pnl": _direction_avg_trade_pnl(subset, 1),
                "short_avg_trade_pnl": _direction_avg_trade_pnl(subset, -1),
                "passed": passed,
            },
        )

    best = _best_horizon_result(horizon_results)
    passed = positive_horizons >= int(config["min_positive_horizons"])
    return {
        "passed": passed,
        "base_event_count": int(base_event_count),
        "min_required_events": int(min_required_events),
        "positive_horizons": int(positive_horizons),
        "min_positive_horizons": int(config["min_positive_horizons"]),
        "best_horizon_bars": _optional_int(best.get("horizon_bars")),
        "best_event_count": _optional_int(best.get("trade_count")),
        "best_avg_trade_pnl": _optional_float(best.get("avg_trade_pnl")),
        "best_positive_day_fraction": _optional_float(best.get("positive_day_fraction")),
        "best_max_day_concentration": _optional_float(best.get("max_day_concentration")),
        "horizon_results": horizon_results,
    }


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _build_context_families(
    analysis_frames: list[pl.DataFrame],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not analysis_frames:
        return {}, []

    combined = pl.concat(analysis_frames, how="vertical_relaxed")
    families: dict[str, Any] = {"session_bucket": {"kind": "session"}}
    omitted: list[dict[str, str]] = []

    for family, column in _NUMERIC_BUCKET_COLUMNS.items():
        if column not in combined.columns:
            omitted.append({"family": family, "reason": "missing_column", "column": column})
            continue
        thresholds = _tercile_thresholds(combined[column])
        if thresholds is None:
            omitted.append({"family": family, "reason": "insufficient_variation", "column": column})
            continue
        families[family] = {
            "kind": "numeric",
            "column": column,
            "thresholds": thresholds,
            "labels": _NUMERIC_BUCKET_LABELS[family],
        }

    if "position_in_va" in combined.columns:
        families["value_area_bucket"] = {"kind": "value_area", "column": "position_in_va"}
    else:
        omitted.append({"family": "value_area_bucket", "reason": "missing_column", "column": "position_in_va"})

    return families, omitted


def _frame_family_labels(frame: pl.DataFrame, families: dict[str, Any]) -> dict[str, list[str | None]]:
    labels: dict[str, list[str | None]] = {}

    for family, spec in families.items():
        kind = spec.get("kind")
        if kind == "session":
            labels[family] = _session_bucket_labels(frame)
        elif kind == "numeric":
            labels[family] = _numeric_bucket_labels(
                frame[spec["column"]],
                thresholds=spec["thresholds"],
                labels=spec["labels"],
            )
        elif kind == "value_area":
            labels[family] = _value_area_bucket_labels(frame[spec["column"]])

    return labels


def _empty_surface(*, enabled: bool, passed: bool, status: str) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "passed": passed,
        "status": status,
        "has_localized_edge": False,
        "global_probe": {},
        "by_side": {},
        "conditional_slices": {},
        "best_pockets": [],
        "omitted_families": [],
    }


def run_edge_surface(
    *,
    analysis_frames: list[pl.DataFrame],
    entry_on_next_open: bool,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = normalize_edge_surface_config(config)
    if not normalized["enabled"]:
        return _empty_surface(enabled=False, passed=True, status="disabled")

    usable_frames = [frame.sort("ts_event") for frame in analysis_frames if isinstance(frame, pl.DataFrame) and len(frame) > 0]
    if not usable_frames:
        result = _empty_surface(enabled=True, passed=False, status="no_data")
        result["global_probe"] = {
            "passed": False,
            "base_event_count": 0,
            "min_required_events": int(normalized["min_events"]),
            "positive_horizons": 0,
            "min_positive_horizons": int(normalized["min_positive_horizons"]),
            "best_horizon_bars": None,
            "best_event_count": None,
            "best_avg_trade_pnl": None,
            "best_positive_day_fraction": None,
            "best_max_day_concentration": None,
            "horizon_results": [],
        }
        return result

    for frame in usable_frames:
        _ensure_required_columns(frame, entry_on_next_open=entry_on_next_open)

    families, omitted_families = _build_context_families(usable_frames)

    base_event_count = 0
    side_counts: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = {family: Counter() for family in families}
    trade_frames: list[pl.DataFrame] = []
    bars_for_costs: list[pl.DataFrame] = []

    for frame in usable_frames:
        timestamps = frame["ts_event"].to_list()
        closes = frame["close"].to_list()
        entries = _extract_entry_events(frame, entry_on_next_open=entry_on_next_open)
        label_lookup = _frame_family_labels(frame, families)

        if entries:
            base_event_count += len(entries)
            bars_for_costs.append(frame.drop("signal"))

        for entry in entries:
            side = _side_label(entry.direction)
            side_counts[side] += 1
            for family, labels in label_lookup.items():
                label = labels[entry.signal_index]
                if label is not None:
                    family_counts[family][label] += 1

        rows: list[dict[str, Any]] = []
        for horizon in normalized["horizons"]:
            for entry in entries:
                exit_index = entry.fill_index + int(horizon) - 1 if entry_on_next_open else entry.fill_index + int(horizon)
                if exit_index >= len(frame):
                    continue
                if timestamps[exit_index].date() != timestamps[entry.fill_index].date():
                    continue
                row = {
                    "entry_time": entry.entry_time,
                    "exit_time": timestamps[exit_index],
                    "entry_price": float(entry.entry_price),
                    "exit_price": float(closes[exit_index]),
                    "direction": int(entry.direction),
                    "size": 1,
                    "horizon_bars": int(horizon),
                    "side_label": _side_label(entry.direction),
                }
                for family, labels in label_lookup.items():
                    row[family] = labels[entry.signal_index]
                rows.append(row)
        if rows:
            trade_frames.append(
                pl.DataFrame(rows).with_columns(
                    [
                        pl.col("entry_time").cast(TRADE_SCHEMA["entry_time"]),
                        pl.col("exit_time").cast(TRADE_SCHEMA["exit_time"]),
                        pl.col("entry_price").cast(TRADE_SCHEMA["entry_price"]),
                        pl.col("exit_price").cast(TRADE_SCHEMA["exit_price"]),
                        pl.col("direction").cast(TRADE_SCHEMA["direction"]),
                        pl.col("size").cast(TRADE_SCHEMA["size"]),
                        pl.col("horizon_bars").cast(pl.Int32),
                        pl.col("side_label").cast(pl.Utf8),
                    ]
                ),
            )

    if base_event_count < int(normalized["min_events"]):
        global_probe = {
            "passed": False,
            "base_event_count": int(base_event_count),
            "min_required_events": int(normalized["min_events"]),
            "positive_horizons": 0,
            "min_positive_horizons": int(normalized["min_positive_horizons"]),
            "best_horizon_bars": None,
            "best_event_count": None,
            "best_avg_trade_pnl": None,
            "best_positive_day_fraction": None,
            "best_max_day_concentration": None,
            "horizon_results": [],
        }
        return {
            "enabled": True,
            "passed": False,
            "status": "insufficient_events",
            "has_localized_edge": False,
            "global_probe": global_probe,
            "by_side": {},
            "conditional_slices": {},
            "best_pockets": [],
            "omitted_families": omitted_families,
            "min_pocket_events": int(normalized["min_pocket_events"]),
        }

    all_trades = pl.concat(trade_frames, how="vertical_relaxed") if trade_frames else pl.DataFrame()
    if len(all_trades) > 0 and bars_for_costs:
        bars_df = pl.concat(bars_for_costs, how="vertical_relaxed").sort("ts_event")
        all_trades = compute_adaptive_costs(all_trades, bars_df)

    global_probe = _evaluate_trade_population(
        trades=all_trades,
        base_event_count=base_event_count,
        min_required_events=int(normalized["min_events"]),
        config=normalized,
    )

    by_side: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        side_summary = _evaluate_trade_population(
            trades=all_trades.filter(pl.col("side_label") == side) if len(all_trades) > 0 else pl.DataFrame(),
            base_event_count=int(side_counts.get(side, 0)),
            min_required_events=int(normalized["min_pocket_events"]),
            config=normalized,
        )
        by_side[side] = side_summary

    conditional_slices: dict[str, list[dict[str, Any]]] = {}
    qualifying_pockets: list[dict[str, Any]] = []
    for family in sorted(family_counts):
        rows: list[dict[str, Any]] = []
        for label in sorted(family_counts[family]):
            base_count = int(family_counts[family][label])
            if base_count <= 0:
                continue
            slice_trades = all_trades.filter(pl.col(family) == label) if len(all_trades) > 0 else pl.DataFrame()
            slice_summary = _evaluate_trade_population(
                trades=slice_trades,
                base_event_count=base_count,
                min_required_events=int(normalized["min_pocket_events"]),
                config=normalized,
            )
            row = {
                "family": family,
                "label": label,
                **slice_summary,
            }
            rows.append(row)
            if bool(slice_summary.get("passed", False)):
                qualifying_pockets.append(
                    {
                        "family": family,
                        "label": label,
                        "base_event_count": base_count,
                        "positive_horizons": int(slice_summary.get("positive_horizons", 0)),
                        "best_horizon_bars": _optional_int(slice_summary.get("best_horizon_bars")),
                        "best_event_count": _optional_int(slice_summary.get("best_event_count")),
                        "best_avg_trade_pnl": _optional_float(slice_summary.get("best_avg_trade_pnl")),
                        "best_positive_day_fraction": _optional_float(
                            slice_summary.get("best_positive_day_fraction"),
                        ),
                        "best_max_day_concentration": _optional_float(
                            slice_summary.get("best_max_day_concentration"),
                        ),
                    },
                )
        if rows:
            conditional_slices[family] = rows

    qualifying_pockets.sort(
        key=lambda row: (
            float(row["best_avg_trade_pnl"]) if row.get("best_avg_trade_pnl") is not None else float("-inf"),
            int(row.get("positive_horizons", 0)),
            int(row.get("best_event_count", 0) or 0),
            str(row.get("family", "")),
            str(row.get("label", "")),
        ),
        reverse=True,
    )
    best_pockets = qualifying_pockets[: int(normalized["max_best_pockets"])]

    has_localized_edge = bool(best_pockets)
    if bool(global_probe.get("passed", False)):
        status = "global_edge"
        passed = True
    elif has_localized_edge:
        status = "localized_edge_only"
        passed = False
    else:
        status = "no_edge"
        passed = False

    return {
        "enabled": True,
        "passed": passed,
        "status": status,
        "has_localized_edge": has_localized_edge,
        "global_probe": global_probe,
        "by_side": by_side,
        "conditional_slices": conditional_slices,
        "best_pockets": best_pockets,
        "omitted_families": omitted_families,
        "min_pocket_events": int(normalized["min_pocket_events"]),
    }


__all__ = [
    "EDGE_SURFACE_ANALYSIS_COLUMNS",
    "normalize_edge_surface_config",
    "run_edge_surface",
]
