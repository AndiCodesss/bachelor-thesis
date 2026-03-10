from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from research.lib.edge_surface import normalize_edge_surface_config, run_edge_surface


def _analysis_frame(
    *,
    event_returns: list[float],
    vol_regimes: list[float] | None = None,
    include_optional_context: bool = True,
) -> pl.DataFrame:
    closes: list[float] = []
    signals: list[int] = []
    regime_vol: list[float] = []
    spread_bps: list[float] = []
    trade_intensity: list[float] = []
    return_5bar: list[float] = []
    position_in_va: list[float] = []

    for index, move in enumerate(event_returns):
        closes.extend([100.0, 100.0 + float(move)])
        signals.extend([1, 0])
        regime = float(vol_regimes[index]) if vol_regimes is not None else 1.0
        regime_vol.extend([regime, regime])
        spread_bps.extend([0.4 + index * 0.01, 0.4 + index * 0.01])
        trade_intensity.extend([10.0 + index, 10.0 + index])
        return_5bar.extend([move, move])
        position_in_va.extend([0.5, 0.5])

    frame = pl.DataFrame(
        {
            "ts_event": pl.datetime_range(
                datetime(2024, 1, 2, 14, 30),
                datetime(2024, 1, 2, 14, 30) + timedelta(minutes=len(closes) - 1),
                interval="1m",
                eager=True,
            ),
            "close": closes,
            "signal": signals,
        },
    ).with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))

    if include_optional_context:
        frame = frame.with_columns(
            [
                pl.Series("regime_vol_relative", regime_vol),
                pl.Series("spread_bps", spread_bps),
                pl.Series("trade_intensity", trade_intensity),
                pl.Series("return_5bar", return_5bar),
                pl.Series("position_in_va", position_in_va),
            ]
        )

    return frame


def test_normalize_edge_surface_config_defaults_include_long_horizons_and_pocket_floor():
    config = normalize_edge_surface_config(None)

    assert config["enabled"] is False
    assert config["horizons"] == [1, 3, 5, 10, 20, 40, 60, 90]
    assert config["min_events"] == 60
    assert config["min_pocket_events"] == 30


def test_normalize_edge_surface_config_sorts_horizons_and_clamps_pocket_events():
    config = normalize_edge_surface_config(
        {
            "enabled": True,
            "horizons": [90, 5, 5, 0, None, 40, 1],
            "min_events": 12,
            "min_pocket_events": 25,
        }
    )

    assert config["enabled"] is True
    assert config["horizons"] == [1, 5, 40, 90]
    assert config["min_events"] == 12
    assert config["min_pocket_events"] == 12


def test_run_edge_surface_global_edge_passes_when_all_events_are_positive():
    frame = _analysis_frame(
        event_returns=[2.0] * 12,
        vol_regimes=[2.0] * 12,
    )

    result = run_edge_surface(
        analysis_frames=[frame],
        entry_on_next_open=False,
        config={
            "enabled": True,
            "horizons": [1],
            "min_events": 12,
            "min_pocket_events": 6,
            "min_positive_horizons": 1,
            "min_avg_trade_pnl": 0.0,
            "min_positive_day_fraction": 0.5,
            "max_day_concentration": 1.0,
        },
    )

    assert result["passed"] is True
    assert result["status"] == "global_edge"
    assert result["global_probe"]["positive_horizons"] == 1


def test_run_edge_surface_reports_localized_edge_only_for_qualified_single_dimension_pocket():
    frame = _analysis_frame(
        event_returns=[2.0] * 6 + [-2.0] * 6,
        vol_regimes=[6.0] * 6 + [0.1] * 3 + [2.0] * 3,
    )

    result = run_edge_surface(
        analysis_frames=[frame],
        entry_on_next_open=False,
        config={
            "enabled": True,
            "horizons": [1],
            "min_events": 12,
            "min_pocket_events": 6,
            "min_positive_horizons": 1,
            "min_avg_trade_pnl": 0.0,
            "min_positive_day_fraction": 0.5,
            "max_day_concentration": 1.0,
        },
    )

    assert result["passed"] is False
    assert result["status"] == "localized_edge_only"
    assert result["has_localized_edge"] is True
    assert result["global_probe"]["positive_horizons"] == 0
    assert result["best_pockets"][0]["family"] == "vol_bucket"
    assert result["best_pockets"][0]["label"] == "high"
    assert result["best_pockets"][0]["best_event_count"] == 6
    assert result["best_pockets"][0]["best_avg_trade_pnl"] > 0.0


def test_run_edge_surface_suppresses_small_pockets_below_min_pocket_events():
    frame = _analysis_frame(
        event_returns=[2.0] * 6 + [-2.0] * 6,
        vol_regimes=[6.0] * 6 + [0.1] * 3 + [2.0] * 3,
    )

    result = run_edge_surface(
        analysis_frames=[frame],
        entry_on_next_open=False,
        config={
            "enabled": True,
            "horizons": [1],
            "min_events": 12,
            "min_pocket_events": 7,
            "min_positive_horizons": 1,
            "min_avg_trade_pnl": 0.0,
            "min_positive_day_fraction": 0.5,
            "max_day_concentration": 1.0,
        },
    )

    assert result["status"] == "no_edge"
    assert result["has_localized_edge"] is False
    assert result["best_pockets"] == []


def test_run_edge_surface_omits_missing_context_families_without_crashing():
    frame = _analysis_frame(
        event_returns=[1.0, -1.0],
        include_optional_context=False,
    )

    result = run_edge_surface(
        analysis_frames=[frame],
        entry_on_next_open=False,
        config={
            "enabled": True,
            "horizons": [1],
            "min_events": 2,
            "min_pocket_events": 1,
            "min_positive_horizons": 1,
            "min_avg_trade_pnl": 0.0,
            "min_positive_day_fraction": 0.5,
            "max_day_concentration": 1.0,
        },
    )

    omitted = {(row["family"], row["reason"]) for row in result["omitted_families"]}
    assert ("vol_bucket", "missing_column") in omitted
    assert ("spread_bucket", "missing_column") in omitted
    assert ("activity_bucket", "missing_column") in omitted
    assert ("direction_bucket", "missing_column") in omitted
    assert ("value_area_bucket", "missing_column") in omitted
