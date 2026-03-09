"""Deterministic entry-edge probe before full strategy validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from src.framework.api import compute_adaptive_costs, compute_metrics
from src.framework.backtest.engine import TRADE_SCHEMA
from src.framework.backtest.metrics import compute_daily_pnl_series

_DEFAULT_EDGE_PROBE = {
    "enabled": False,
    "horizons": [1, 3, 5, 10, 20, 40, 60, 90],
    "min_events": 60,
    "min_positive_horizons": 1,
    "min_avg_trade_pnl": 0.0,
    "min_positive_day_fraction": 0.52,
    "max_day_concentration": 0.40,
}


@dataclass(frozen=True)
class EntryEvent:
    fill_index: int
    entry_time: Any
    entry_price: float
    direction: int


def normalize_edge_probe_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(_DEFAULT_EDGE_PROBE)
    if isinstance(payload, dict):
        config.update(payload)

    horizons_raw = config.get("horizons")
    if not isinstance(horizons_raw, list):
        horizons_raw = _DEFAULT_EDGE_PROBE["horizons"]
    horizons = sorted({max(1, int(value)) for value in horizons_raw if value is not None})
    config["horizons"] = horizons or list(_DEFAULT_EDGE_PROBE["horizons"])
    config["enabled"] = bool(config.get("enabled", False))
    config["min_events"] = max(1, int(config.get("min_events", _DEFAULT_EDGE_PROBE["min_events"])))
    config["min_positive_horizons"] = max(
        1,
        int(config.get("min_positive_horizons", _DEFAULT_EDGE_PROBE["min_positive_horizons"])),
    )
    config["min_avg_trade_pnl"] = float(
        config.get("min_avg_trade_pnl", _DEFAULT_EDGE_PROBE["min_avg_trade_pnl"]),
    )
    config["min_positive_day_fraction"] = float(
        config.get("min_positive_day_fraction", _DEFAULT_EDGE_PROBE["min_positive_day_fraction"]),
    )
    config["max_day_concentration"] = float(
        config.get("max_day_concentration", _DEFAULT_EDGE_PROBE["max_day_concentration"]),
    )
    return config


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

    for index in range(len(signals)):
        has_next_bar = index < len(signals) - 1
        is_last_bar = (not has_next_bar) or (timestamps[index + 1].date() != timestamps[index].date())

        signal_to_execute = pending_direction
        pending_direction = 0
        if signal_to_execute != 0 and position == 0:
            entries.append(
                EntryEvent(
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
                elif not is_last_bar:
                    entries.append(
                        EntryEvent(
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


def _synthetic_trades_for_horizon(
    execution_frames: list[pl.DataFrame],
    *,
    horizon_bars: int,
    entry_on_next_open: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    trades_by_frame: list[pl.DataFrame] = []
    bars_for_costs: list[pl.DataFrame] = []

    for frame in execution_frames:
        if len(frame) == 0:
            continue
        ordered = frame.sort("ts_event")
        timestamps = ordered["ts_event"].to_list()
        closes = ordered["close"].to_list()
        entries = _extract_entry_events(ordered, entry_on_next_open=entry_on_next_open)
        if not entries:
            continue

        rows: list[dict[str, Any]] = []
        for entry in entries:
            exit_index = entry.fill_index + horizon_bars - 1 if entry_on_next_open else entry.fill_index + horizon_bars
            if exit_index >= len(ordered):
                continue
            exit_price = float(closes[exit_index])
            rows.append(
                {
                    "entry_time": entry.entry_time,
                    "exit_time": timestamps[exit_index],
                    "entry_price": float(entry.entry_price),
                    "exit_price": exit_price,
                    "direction": int(entry.direction),
                    "size": 1,
                },
            )
        if not rows:
            continue
        trades = pl.DataFrame(rows).cast(TRADE_SCHEMA)
        trades = compute_adaptive_costs(trades, ordered)
        trades_by_frame.append(trades)
        bars_for_costs.append(ordered.select([col for col in ordered.columns if col != "signal"]))

    trades_df = pl.concat(trades_by_frame) if trades_by_frame else pl.DataFrame(schema=TRADE_SCHEMA)
    bars_df = pl.concat(bars_for_costs).sort("ts_event") if bars_for_costs else pl.DataFrame()
    return trades_df, bars_df


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
    if abs_sum <= 0:
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


def run_edge_probe(
    *,
    execution_frames: list[pl.DataFrame],
    entry_on_next_open: bool,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = normalize_edge_probe_config(config)
    if not normalized["enabled"]:
        return {"enabled": False, "passed": True, "status": "disabled", "events": 0, "horizon_results": []}

    if not execution_frames:
        return {"enabled": True, "passed": False, "status": "no_data", "events": 0, "horizon_results": []}

    base_events = sum(len(_extract_entry_events(frame, entry_on_next_open=entry_on_next_open)) for frame in execution_frames)
    if base_events < int(normalized["min_events"]):
        return {
            "enabled": True,
            "passed": False,
            "status": "insufficient_events",
            "events": int(base_events),
            "min_events": int(normalized["min_events"]),
            "horizon_results": [],
        }

    horizon_results: list[dict[str, Any]] = []
    positive_horizons = 0

    for horizon in normalized["horizons"]:
        trades_df, _bars_df = _synthetic_trades_for_horizon(
            execution_frames,
            horizon_bars=int(horizon),
            entry_on_next_open=entry_on_next_open,
        )
        if len(trades_df) == 0:
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
            trades_df,
            cost_override_col="adaptive_cost_rt" if "adaptive_cost_rt" in trades_df.columns else None,
        )
        positive_day_fraction, max_day_concentration = _daily_edge_stats(trades_df)
        avg_trade_pnl = float(metrics.get("avg_trade_pnl", 0.0))
        net_pnl = float(metrics.get("net_pnl", 0.0))
        trade_count = int(metrics.get("trade_count", 0))
        passed = (
            trade_count >= int(normalized["min_events"])
            and avg_trade_pnl > float(normalized["min_avg_trade_pnl"])
            and net_pnl > 0.0
            and positive_day_fraction >= float(normalized["min_positive_day_fraction"])
            and max_day_concentration <= float(normalized["max_day_concentration"])
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
                "long_avg_trade_pnl": _direction_avg_trade_pnl(trades_df, 1),
                "short_avg_trade_pnl": _direction_avg_trade_pnl(trades_df, -1),
                "passed": passed,
            },
        )

    horizon_results.sort(
        key=lambda row: (
            bool(row.get("passed", False)),
            float(row.get("avg_trade_pnl", 0.0)),
            float(row.get("net_pnl", 0.0)),
        ),
        reverse=True,
    )
    best = dict(horizon_results[0]) if horizon_results else None
    passed = positive_horizons >= int(normalized["min_positive_horizons"])
    status = "pass" if passed else "no_raw_edge"
    return {
        "enabled": True,
        "passed": passed,
        "status": status,
        "events": int(base_events),
        "min_events": int(normalized["min_events"]),
        "positive_horizons": int(positive_horizons),
        "min_positive_horizons": int(normalized["min_positive_horizons"]),
        "best_horizon_bars": int(best.get("horizon_bars")) if isinstance(best, dict) and best.get("horizon_bars") is not None else None,
        "best_avg_trade_pnl": float(best.get("avg_trade_pnl", 0.0)) if isinstance(best, dict) else None,
        "best_positive_day_fraction": float(best.get("positive_day_fraction", 0.0)) if isinstance(best, dict) else None,
        "best_max_day_concentration": float(best.get("max_day_concentration", 1.0)) if isinstance(best, dict) else None,
        "horizon_results": horizon_results,
    }


__all__ = [
    "normalize_edge_probe_config",
    "run_edge_probe",
]
