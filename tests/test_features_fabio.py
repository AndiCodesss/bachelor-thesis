"""Tests for Fabio Valentini strategy features: prev_day_vp, failed_auction, squeeze."""

import numpy as np
import polars as pl
from datetime import datetime

from src.framework.features_canonical.pipeline import (
    _add_prev_session_vp_features,
    _add_failed_auction_features,
    _add_squeeze_features,
)
from src.framework.features_canonical.volume_profile import (
    compute_volume_profile_features,
)


def _make_two_session_df():
    """Two-session DataFrame with VP columns, range_ma5, and flow columns."""
    # Day 1: 5 bars, POC=18000, VA=[17998, 18002]
    # Day 2: 5 bars at 18010
    timestamps = []
    closes = []
    highs = []
    lows = []
    opens = []
    poc_prices = []
    va_highs = []
    va_lows = []

    for i in range(5):
        timestamps.append(datetime(2025, 2, 3, 10 + i, 0, 0))
        closes.append(18000.0 + i * 0.5)
        opens.append(18000.0 + i * 0.5 - 0.25)
        highs.append(18000.0 + i * 0.5 + 1.0)
        lows.append(18000.0 + i * 0.5 - 1.0)
        poc_prices.append(18000.0)
        va_highs.append(18002.0)
        va_lows.append(17998.0)

    for i in range(5):
        timestamps.append(datetime(2025, 2, 4, 10 + i, 0, 0))
        closes.append(18010.0 + i * 0.5)
        opens.append(18010.0 + i * 0.5 - 0.25)
        highs.append(18010.0 + i * 0.5 + 1.0)
        lows.append(18010.0 + i * 0.5 - 1.0)
        poc_prices.append(18010.0)
        va_highs.append(18012.0)
        va_lows.append(18008.0)

    return pl.DataFrame({
        "ts_event": timestamps,
        "close": closes,
        "open": opens,
        "high": highs,
        "low": lows,
        "poc_price": poc_prices,
        "va_high": va_highs,
        "va_low": va_lows,
        "range_ma5": [2.0] * 10,
    }).with_columns(
        pl.col("ts_event").dt.replace_time_zone("UTC"),
    )


class TestPrevSessionVPFeatures:
    def test_output_columns(self):
        df = _make_two_session_df()
        result = _add_prev_session_vp_features(df)
        for col in ["prev_day_poc", "prev_day_vah", "prev_day_val",
                     "dist_prev_poc", "dist_prev_vah", "dist_prev_val",
                     "prev_day_va_position"]:
            assert col in result.columns, f"Missing: {col}"

    def test_first_session_null(self):
        """First session has no previous day — values should be null."""
        df = _make_two_session_df()
        result = _add_prev_session_vp_features(df)
        for col in ["prev_day_poc", "prev_day_vah", "prev_day_val"]:
            assert result[col][0] is None, f"{col} should be null on first day"

    def test_second_session_uses_day1_vp(self):
        """Second session should use day 1's last-bar VP levels."""
        df = _make_two_session_df()
        result = _add_prev_session_vp_features(df)
        # Day 1 last bar: poc=18000, va_high=18002, va_low=17998
        assert result["prev_day_poc"][5] == 18000.0
        assert result["prev_day_vah"][5] == 18002.0
        assert result["prev_day_val"][5] == 17998.0

    def test_dist_prev_poc_sign(self):
        """Day 2 close (18010) > day 1 POC (18000) → positive distance."""
        df = _make_two_session_df()
        result = _add_prev_session_vp_features(df)
        assert result["dist_prev_poc"][5] > 0

    def test_va_position_outside_va(self):
        """Day 2 at 18010 is above day 1 VA=[17998,18002] → position > 1."""
        df = _make_two_session_df()
        result = _add_prev_session_vp_features(df)
        assert result["prev_day_va_position"][5] > 1.0

    def test_row_count_preserved(self):
        df = _make_two_session_df()
        result = _add_prev_session_vp_features(df)
        assert len(result) == len(df)


class TestFailedAuctionFeatures:
    def _make_failed_auction_df(self):
        """DataFrame where bar probes below prev_day_val but closes above."""
        return pl.DataFrame({
            "ts_event": [datetime(2025, 2, 4, 10, 0, 0)],
            "open": [18000.0],
            "high": [18001.0],
            "low": [17996.0],       # probes below VAL=17998
            "close": [18000.0],     # recovers above VAL
            "prev_day_vah": [18002.0],
            "prev_day_val": [17998.0],
            "absorption_signal": [1.0],
            "whale_buy_volume_30": [50.0],
            "whale_sell_volume_30": [0.0],
        }).with_columns(
            pl.col("ts_event").dt.replace_time_zone("UTC"),
        )

    def test_bull_failed_auction_fires(self):
        df = self._make_failed_auction_df()
        result = _add_failed_auction_features(df)
        assert result["failed_auction_bull"][0] == 1.0
        assert result["failed_auction_bear"][0] == 0.0

    def test_bear_failed_auction_fires(self):
        """Bar probes above VAH but closes below."""
        df = pl.DataFrame({
            "ts_event": [datetime(2025, 2, 4, 10, 0, 0)],
            "open": [18001.0],
            "high": [18003.0],      # probes above VAH=18002
            "low": [18000.0],
            "close": [18001.0],     # closes below VAH
            "prev_day_vah": [18002.0],
            "prev_day_val": [17998.0],
            "absorption_signal": [1.0],
            "whale_buy_volume_30": [0.0],
            "whale_sell_volume_30": [50.0],
        }).with_columns(
            pl.col("ts_event").dt.replace_time_zone("UTC"),
        )
        result = _add_failed_auction_features(df)
        assert result["failed_auction_bear"][0] == 1.0
        assert result["failed_auction_bull"][0] == 0.0

    def test_no_probe_no_signal(self):
        """Bar stays inside VA → no failed auction."""
        df = pl.DataFrame({
            "ts_event": [datetime(2025, 2, 4, 10, 0, 0)],
            "open": [18000.0],
            "high": [18001.0],
            "low": [17999.0],
            "close": [18000.5],
            "prev_day_vah": [18002.0],
            "prev_day_val": [17998.0],
            "absorption_signal": [1.0],
            "whale_buy_volume_30": [50.0],
            "whale_sell_volume_30": [50.0],
        }).with_columns(
            pl.col("ts_event").dt.replace_time_zone("UTC"),
        )
        result = _add_failed_auction_features(df)
        assert result["failed_auction_bull"][0] == 0.0
        assert result["failed_auction_bear"][0] == 0.0

    def test_score_is_directional(self):
        df = self._make_failed_auction_df()
        result = _add_failed_auction_features(df)
        assert result["failed_auction_score"][0] == 1.0


class TestSqueezeFeatures:
    def _make_squeeze_df(self, n=10):
        """DataFrame with conditions that trigger a bull squeeze."""
        hlr = [2.0] * n
        vol_imb = [0.0] * n
        absorption = [0.0] * n
        cvd_accel = [0.0] * n

        # Bar 5: absorption
        absorption[5] = 1.0
        # Bar 7: range expansion + strong buy flow + CVD accel
        hlr[7] = 10.0  # > 2× median of 2.0
        vol_imb[7] = 0.8  # > 0.6
        cvd_accel[7] = 100.0  # > 0

        return pl.DataFrame({
            "ts_event": [datetime(2025, 2, 3, 10, i, 0) for i in range(n)],
            "high_low_range": hlr,
            "volume_imbalance": vol_imb,
            "absorption_signal": absorption,
            "cvd_accel_3": cvd_accel,
        }).with_columns(
            pl.col("ts_event").dt.replace_time_zone("UTC"),
        )

    def test_squeeze_bull_fires(self):
        df = self._make_squeeze_df()
        result = _add_squeeze_features(df)
        assert result["squeeze_bull"][7] == 1.0

    def test_squeeze_bear_fires(self):
        df = self._make_squeeze_df()
        # Flip flow direction for bear
        df = df.with_columns([
            pl.when(pl.col("volume_imbalance") > 0.5)
            .then(-0.8)
            .otherwise(pl.col("volume_imbalance"))
            .alias("volume_imbalance"),
            pl.when(pl.col("cvd_accel_3") > 0)
            .then(-100.0)
            .otherwise(pl.col("cvd_accel_3"))
            .alias("cvd_accel_3"),
        ])
        result = _add_squeeze_features(df)
        assert result["squeeze_bear"][7] == 1.0

    def test_no_absorption_no_squeeze(self):
        """Without recent absorption, squeeze should not fire."""
        df = self._make_squeeze_df()
        df = df.with_columns(pl.lit(0.0).alias("absorption_signal"))
        result = _add_squeeze_features(df)
        assert result["squeeze_bull"][7] == 0.0


class TestSwingVPFeatures:
    def _make_breakout_bars(self, n_bars=30):
        """Bars with a clear upside breakout in the middle."""
        timestamps = []
        closes = []
        opens = []
        highs = []
        lows = []
        volumes = []
        base = 18000.0

        for i in range(n_bars):
            ts = datetime(2024, 3, 15, 10 + i // 12, (i * 5) % 60, 0)
            timestamps.append(ts)
            if i < 15:
                c = base + np.random.uniform(-1, 1)
            else:
                # Breakout: price jumps above channel
                c = base + 20.0 + (i - 15) * 2.0
            c = round(c / 0.25) * 0.25
            closes.append(c)
            opens.append(c - 0.5)
            highs.append(c + 1.0)
            lows.append(c - 1.0)
            volumes.append(100)

        return pl.DataFrame({
            "ts_event": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }).with_columns(
            pl.col("ts_event").dt.replace_time_zone("UTC"),
            pl.col("volume").cast(pl.UInt32),
        )

    def test_breakout_direction_detected(self):
        np.random.seed(42)
        bars = self._make_breakout_bars()
        result = compute_volume_profile_features(bars)
        bd = result["breakout_direction"].to_numpy()
        # After bar 15 (breakout), direction should be +1 at some point
        assert 1.0 in bd[15:]

    def test_bars_since_breakout_increments(self):
        np.random.seed(42)
        bars = self._make_breakout_bars()
        result = compute_volume_profile_features(bars)
        bsb = result["bars_since_breakout"].to_numpy()
        # Find first non-NaN bars_since_breakout
        valid = bsb[~np.isnan(bsb)]
        if len(valid) > 1:
            # Should be incrementing (at least partly)
            assert valid[-1] > valid[0] or len(valid) == 1

    def test_swing_columns_exist(self):
        np.random.seed(42)
        bars = self._make_breakout_bars()
        result = compute_volume_profile_features(bars)
        for col in ["swing_poc_dist", "swing_lvn_dist", "swing_hvn_dist",
                     "swing_va_position", "breakout_direction", "bars_since_breakout"]:
            assert col in result.columns, f"Missing: {col}"
            assert result[col].dtype == pl.Float64
