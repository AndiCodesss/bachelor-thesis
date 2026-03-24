from __future__ import annotations

import pytest

from research.lib.thinker_research_contract import (
    ThinkerResearchContractError,
    normalize_research_brief,
)


def _entry_conditions() -> list[dict[str, object]]:
    return [
        {"feature": "volume_ratio", "op": ">", "param_key": "vol_ratio_min", "role": "primary"},
        {"feature": "trade_intensity", "op": ">", "param_key": "activity_min", "role": "confirmation"},
    ]


def _valid_brief() -> dict[str, object]:
    return {
        "event": "Volume expands into a one-sided push after a quiet ETH rotation and the tape starts to accelerate.",
        "mechanism": "short-term inventory imbalance after quiet-session compression",
        "expected_side": "long",
        "expected_horizon_bars": 5,
        "market_regime": "ETH only, elevated trade_intensity, and relative volatility already lifting from the prior hour.",
        "structural_location": "Price is below prior value and re-tests prev_day_val from underneath while still close to that structural boundary.",
        "micro_trigger": "trade_intensity re-accelerates and volume_ratio expands while the orderflow turns back in favor of buyers.",
        "post_cost_rationale": "The event is sparse, directional, and tied to fast participation so a short 5-bar move can outrun one-turn costs.",
        "falsification": "If volume_ratio expands but trade_intensity does not stay elevated on entry bars, the imbalance thesis is wrong.",
        "novelty_vs_recent_failures": "This is not a generic delta threshold idea; it requires quiet-session compression first and then a specific expansion trigger.",
    }


def _params_template() -> dict[str, float]:
    return {
        "vol_ratio_min": 1.15,
        "activity_min": 1.05,
    }


def test_normalize_research_brief_derives_mechanism_key_and_horizon():
    out = normalize_research_brief(
        _valid_brief(),
        entry_conditions=_entry_conditions(),
        allowed_horizons=[1, 3, 5, 10],
        params_template=_params_template(),
    )
    assert out["expected_horizon_bars"] == 5
    assert out["expected_side"] == "long"
    assert out["mechanism_key"] == "short_term_inventory_imbalance_after_quiet_session_compression"
    assert out["market_regime"].startswith("ETH only")
    assert out["structural_location"].startswith("Price is below prior value")


def test_normalize_research_brief_requires_mission_horizon_membership():
    with pytest.raises(ThinkerResearchContractError, match="expected_horizon_bars must be one of 1, 3, 5, 10"):
        normalize_research_brief(
            {**_valid_brief(), "expected_horizon_bars": 7},
            entry_conditions=_entry_conditions(),
            allowed_horizons=[1, 3, 5, 10],
            params_template=_params_template(),
        )


def test_normalize_research_brief_repairs_featureless_falsification_from_entry_conditions():
    out = normalize_research_brief(
        {**_valid_brief(), "falsification": "If the strategy is not profitable after costs, the idea is wrong over time."},
        entry_conditions=_entry_conditions(),
        allowed_horizons=[1, 3, 5, 10],
        params_template=_params_template(),
    )
    assert "volume_ratio" in out["falsification"]
    assert "trade_intensity" in out["falsification"]
    assert "not profitable" not in out["falsification"].lower()


def test_normalize_research_brief_repairs_tautological_falsification():
    out = normalize_research_brief(
        {
            **_valid_brief(),
            "falsification": "If avg_trade_pnl is not positive then volume_ratio did not really work as a signal here.",
        },
        entry_conditions=_entry_conditions(),
        allowed_horizons=[1, 3, 5, 10],
        params_template=_params_template(),
    )
    assert "avg_trade_pnl" not in out["falsification"].lower()
    assert "volume_ratio" in out["falsification"]


def test_normalize_research_brief_repairs_short_supporting_narrative_fields():
    out = normalize_research_brief(
        {
            **_valid_brief(),
            "post_cost_rationale": "Sparse move.",
            "novelty_vs_recent_failures": "More specific.",
        },
        entry_conditions=_entry_conditions(),
        allowed_horizons=[1, 3, 5, 10],
        params_template=_params_template(),
    )
    assert len(out["post_cost_rationale"]) >= 28
    assert len(out["novelty_vs_recent_failures"]) >= 28
    assert "volume_ratio" in out["novelty_vs_recent_failures"]


def test_normalize_research_brief_accepts_legacy_aliases_and_canonicalizes_output():
    out = normalize_research_brief(
        {
            **_valid_brief(),
            "market_regime": None,
            "structural_location": None,
            "expected_regime": "ETH only, elevated trade_intensity, and relative volatility already lifting from the prior hour.",
            "macro_location": "Price is below prior value and re-tests prev_day_val from underneath while still close to that structural boundary.",
        },
        entry_conditions=_entry_conditions(),
        allowed_horizons=[1, 3, 5, 10],
        params_template=_params_template(),
    )
    assert out["market_regime"].startswith("ETH only")
    assert out["structural_location"].startswith("Price is below prior value")
