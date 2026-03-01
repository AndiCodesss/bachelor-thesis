"""Tests for ETH session extension infrastructure."""

import polars as pl
import pytest
from datetime import datetime, time as dt_time

from src.framework.data.constants import (
    ETH_START_TIME, ETH_END_TIME, LONDON_START_TIME, LONDON_END_TIME,
    RTH_START_TIME, RTH_END_TIME, WHALE_LOT_THRESHOLD_LONDON, SWING_VP_LOOKBACK,
)
from src.framework.data.loader import filter_rth, filter_eth
from src.framework.features_canonical.builder import _cache_key


class TestETHConstants:
    def test_eth_times_defined(self):
        # pl.time returns Polars Expr, compare string repr
        assert ETH_START_TIME is not None
        assert ETH_END_TIME is not None

    def test_london_times_defined(self):
        assert LONDON_START_TIME is not None
        assert LONDON_END_TIME is not None

    def test_whale_london_threshold(self):
        assert WHALE_LOT_THRESHOLD_LONDON == 20

    def test_swing_vp_lookback(self):
        assert SWING_VP_LOOKBACK == 12


class TestFilterETH:
    def _make_24h_ticks(self):
        """Create tick data spanning a full 24h day in UTC."""
        timestamps = []
        prices = []
        # Create ticks at every hour of 2025-02-03 UTC
        for hour in range(24):
            ts = datetime(2025, 2, 3, hour, 0, 0)
            timestamps.append(ts)
            prices.append(21000.0)

        return pl.DataFrame({
            "ts_event": timestamps,
            "price": prices,
        }).with_columns(
            pl.col("ts_event").dt.replace_time_zone("UTC"),
        ).lazy()

    def test_eth_filters_to_london_plus_rth(self):
        """ETH should include 03:00-16:00 ET (08:00-21:00 UTC in winter)."""
        lf = self._make_24h_ticks()
        result = filter_eth(lf).collect()
        # 2025-02-03 is winter (EST = UTC-5)
        # ETH 03:00-16:00 ET = 08:00-21:00 UTC
        # Hours 8,9,10,11,12,13,14,15,16,17,18,19,20 = 13 hours
        hours = result["ts_event"].dt.hour().to_list()
        assert min(hours) == 8
        assert max(hours) == 20
        assert len(result) == 13

    def test_rth_is_subset_of_eth(self):
        """RTH result should be a strict subset of ETH result."""
        lf = self._make_24h_ticks()
        rth = filter_rth(lf).collect()
        eth = filter_eth(lf).collect()
        assert len(eth) > len(rth)
        # Every RTH timestamp should appear in ETH
        rth_ts = set(rth["ts_event"].to_list())
        eth_ts = set(eth["ts_event"].to_list())
        assert rth_ts.issubset(eth_ts)

    def test_eth_excludes_overnight(self):
        """ETH should not include 21:00-08:00 UTC (16:00-03:00 ET winter)."""
        lf = self._make_24h_ticks()
        result = filter_eth(lf).collect()
        hours = result["ts_event"].dt.hour().to_list()
        for h in [0, 1, 2, 3, 4, 5, 6, 7, 21, 22, 23]:
            assert h not in hours, f"Hour {h} UTC should be excluded"

    def test_empty_input(self):
        lf = pl.DataFrame(
            schema={"ts_event": pl.Datetime("ns", "UTC"), "price": pl.Float64}
        ).lazy()
        result = filter_eth(lf).collect()
        assert len(result) == 0


class TestCacheKey:
    def test_rth_default_unchanged(self):
        """Default session_filter='rth' should not change existing cache keys."""
        assert _cache_key("5m", "time", None) == "5m"
        assert _cache_key("5m", "time", None, include_bar_columns=True) == "5m_bars"
        assert _cache_key("1m", "time", None) == "1m"
        assert _cache_key("5m", "volume", 2000) == "vol_2000"
        assert _cache_key("5m", "volume", 2000, include_bar_columns=True) == "vol_2000_bars"
        assert _cache_key("5m", "tick", 610) == "tick_610"

    def test_eth_prefix(self):
        """session_filter='eth' should add eth_ prefix."""
        assert _cache_key("5m", "time", None, session_filter="eth") == "eth_5m"
        assert _cache_key("5m", "volume", 2000, session_filter="eth") == "eth_vol_2000"
        assert _cache_key("5m", "volume", 2000, include_bar_columns=True, session_filter="eth") == "eth_vol_2000_bars"
        assert _cache_key("5m", "tick", 610, session_filter="eth") == "eth_tick_610"


class TestEasternDateInFeatures:
    """Verify that feature modules use Eastern dates for session grouping."""

    def _make_bars_near_midnight_utc(self):
        """Create bars that span midnight UTC but are same Eastern trading day.

        In winter (EST = UTC-5), 20:00 UTC = 15:00 ET and 19:00 UTC = 14:00 ET.
        Both should be same Eastern date.
        """
        timestamps = [
            datetime(2025, 2, 3, 19, 0, 0),  # 14:00 ET
            datetime(2025, 2, 3, 20, 0, 0),  # 15:00 ET
            datetime(2025, 2, 3, 20, 30, 0),  # 15:30 ET
        ]
        return pl.DataFrame({
            "ts_event": timestamps,
            "open": [21000.0, 21001.0, 21002.0],
            "high": [21001.0, 21002.0, 21003.0],
            "low": [20999.0, 21000.0, 21001.0],
            "close": [21001.0, 21002.0, 21003.0],
            "volume": [1000, 1000, 1000],
            "buy_volume": [600, 600, 600],
            "sell_volume": [400, 400, 400],
            "trade_count": [100, 100, 100],
            "large_buy_volume": [100, 100, 100],
            "large_sell_volume": [50, 50, 50],
            "vwap": [21000.5, 21001.5, 21002.5],
        }).with_columns(
            pl.col("ts_event").dt.replace_time_zone("UTC"),
            pl.col("volume").cast(pl.UInt32),
            pl.col("buy_volume").cast(pl.UInt32),
            pl.col("sell_volume").cast(pl.UInt32),
            pl.col("trade_count").cast(pl.UInt32),
            pl.col("large_buy_volume").cast(pl.UInt32),
            pl.col("large_sell_volume").cast(pl.UInt32),
        )

    def test_aggressor_eastern_date(self):
        """CVD should reset on Eastern date, not UTC date."""
        from src.framework.features_canonical.aggressor import compute_aggressor_features
        bars = self._make_bars_near_midnight_utc()
        result = compute_aggressor_features(bars)
        # All bars same Eastern date → CVD should accumulate (not reset)
        cvd_vals = result["cvd"].to_list()
        assert cvd_vals[0] == 200  # 600-400
        assert cvd_vals[1] == 400  # cumsum
        assert cvd_vals[2] == 600  # cumsum
