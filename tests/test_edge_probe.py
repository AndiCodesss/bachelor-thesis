from __future__ import annotations

from research.lib.edge_probe import normalize_edge_probe_config


def test_normalize_edge_probe_config_defaults_include_long_horizons():
    config = normalize_edge_probe_config(None)

    assert config["enabled"] is False
    assert config["horizons"] == [1, 3, 5, 10, 20, 40, 60, 90]


def test_normalize_edge_probe_config_sorts_deduplicates_and_clamps_horizons():
    config = normalize_edge_probe_config({"enabled": True, "horizons": [90, 5, 5, 0, None, 40, 1]})

    assert config["enabled"] is True
    assert config["horizons"] == [1, 5, 40, 90]
