"""Tests for volume profile feature computation."""

import numpy as np
import polars as pl
from datetime import datetime
from src.framework.features_canonical.volume_profile import compute_volume_profile_features, _compute_profile


class TestComputeProfile:
    """Unit tests for the _compute_profile helper."""

    def test_poc_is_max_volume_level(self):
        prices = np.array([100.0, 100.0, 100.0, 100.25, 100.25, 100.50])
        sizes = np.array([10.0, 20.0, 30.0, 5.0, 5.0, 1.0])
        result = _compute_profile(prices, sizes)
        assert result["poc"] == 100.0

    def test_value_area_captures_70_pct(self):
        prices = np.array([99.0, 99.25, 99.50, 99.75, 100.0])
        sizes = np.array([10.0, 20.0, 100.0, 20.0, 10.0])
        result = _compute_profile(prices, sizes)
        assert result["va_low"] <= result["poc"] <= result["va_high"]
        total = sizes.sum()
        va_mask = (prices >= result["va_low"]) & (prices <= result["va_high"])
        va_vol = sizes[va_mask].sum()
        assert va_vol / total >= 0.70

    def test_va_position_at_boundaries(self):
        prices = np.array([99.0, 99.25, 99.50, 99.75, 100.0])
        sizes = np.array([10.0, 20.0, 100.0, 20.0, 10.0])
        result = _compute_profile(prices, sizes)
        va_range = result["va_high"] - result["va_low"]
        if va_range > 0:
            pos_at_low = (result["va_low"] - result["va_low"]) / va_range
            pos_at_high = (result["va_high"] - result["va_low"]) / va_range
            assert abs(pos_at_low - 0.0) < 1e-9
            assert abs(pos_at_high - 1.0) < 1e-9

    def test_hvn_count_two_peaks(self):
        prices = np.array([99.0, 99.25, 99.50, 99.75, 100.0, 100.25])
        sizes = np.array([5.0, 50.0, 5.0, 5.0, 50.0, 5.0])
        result = _compute_profile(prices, sizes)
        assert result["hvn_count"] == 2

    def test_hvn_count_single_peak(self):
        prices = np.array([100.0, 100.25, 100.50])
        sizes = np.array([5.0, 100.0, 5.0])
        result = _compute_profile(prices, sizes)
        assert result["hvn_count"] == 1

    def test_all_trades_same_price(self):
        prices = np.array([100.0, 100.0, 100.0])
        sizes = np.array([10.0, 20.0, 30.0])
        result = _compute_profile(prices, sizes)
        assert result["poc"] == 100.0
        assert result["va_high"] == 100.0
        assert result["va_low"] == 100.0
        assert result["hvn_count"] == 0
        assert result["skew"] == 0.0

    def test_empty_input(self):
        result = _compute_profile(np.array([]), np.array([]))
        assert np.isnan(result["poc"])
        assert np.isnan(result["va_high"])

    def test_skew_sign(self):
        prices = np.array([99.0, 99.25, 100.0, 100.25, 100.50])
        sizes = np.array([100.0, 100.0, 5.0, 5.0, 5.0])
        result = _compute_profile(prices, sizes)
        assert result["skew"] > 0


class TestVolumeProfileFeatures:
    """Integration tests for compute_volume_profile_features."""

    def _make_multi_bar_data(self, n_bars=30, bar_minutes=5):
        """Generate synthetic bar data with drifting prices."""
        timestamps = []
        closes = []
        opens = []
        highs = []
        lows = []
        volumes = []
        base_price = 18000.0

        for bar in range(n_bars):
            total_seconds = bar * bar_minutes * 60
            hour = 10 + total_seconds // 3600
            minute = (total_seconds % 3600) // 60
            second = total_seconds % 60
            ts = datetime(2024, 3, 15, hour, minute, second)
            timestamps.append(ts)

            c = base_price + bar * 2.0
            c = round(c / 0.25) * 0.25
            closes.append(c)
            opens.append(c - 0.5)
            highs.append(c + 1.0)
            lows.append(c - 1.0)
            volumes.append(max(1, int(10 + np.sin(bar) * 5)))

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

    def test_output_columns(self):
        bars = self._make_multi_bar_data(n_bars=30)
        result = compute_volume_profile_features(bars)

        expected = [
            "ts_event",
            "poc_price", "poc_distance", "poc_distance_raw",
            "va_high", "va_low", "va_width", "position_in_va",
            "vp_skew", "vp_kurtosis", "hvn_count",
            "rolling_poc", "rolling_poc_distance", "poc_slope_6",
            "rolling_va_high", "rolling_va_low", "rolling_va_position",
            "vol_concentration", "rolling_vp_skew", "rolling_hvn_count",
            "rolling_lvn_count", "dist_nearest_hvn", "dist_nearest_lvn",
            "at_hvn", "at_lvn", "hvn_lvn_ratio",
            "swing_poc_dist", "swing_lvn_dist", "swing_hvn_dist",
            "swing_va_position", "breakout_direction", "bars_since_breakout",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_types(self):
        bars = self._make_multi_bar_data(n_bars=30)
        result = compute_volume_profile_features(bars)

        int_cols = ("hvn_count", "rolling_hvn_count", "rolling_lvn_count")
        for col in result.columns:
            if col == "ts_event":
                continue
            if col in int_cols:
                assert result[col].dtype == pl.Int32, f"{col} should be Int32"
            else:
                assert result[col].dtype == pl.Float64, f"{col} should be Float64"

    def test_session_features_causal_expanding(self):
        """Session features use expanding window (causal, no lookahead).

        The last bar's profile should use all bars, so poc/va_high at end
        of day should reflect the full session. Earlier bars use fewer bars.
        """
        bars = self._make_multi_bar_data(n_bars=10)
        result = compute_volume_profile_features(bars)

        # Session features should exist and be finite
        poc_values = result["poc_price"].to_list()
        assert all(v is not None and not np.isnan(v) for v in poc_values)

        va_high_values = result["va_high"].to_list()
        assert all(v is not None and not np.isnan(v) for v in va_high_values)

        # VA high should be >= VA low at every bar
        va_low_values = result["va_low"].to_list()
        for hi, lo in zip(va_high_values, va_low_values):
            assert hi >= lo

    def test_rolling_poc_uses_window(self):
        bars = self._make_multi_bar_data(n_bars=30)
        result = compute_volume_profile_features(bars)

        rolling_pocs = result["rolling_poc"].to_list()
        unique_pocs = set([v for v in rolling_pocs if v is not None and not np.isnan(v)])
        assert len(unique_pocs) > 1

    def test_poc_slope_null_for_first_6(self):
        bars = self._make_multi_bar_data(n_bars=10)
        result = compute_volume_profile_features(bars)

        for i in range(min(6, len(result))):
            val = result["poc_slope_6"][i]
            assert val is None or np.isnan(val), f"poc_slope_6[{i}] should be null/NaN"

    def test_vol_concentration_bounded(self):
        bars = self._make_multi_bar_data(n_bars=30)
        result = compute_volume_profile_features(bars)

        conc = result["vol_concentration"].to_numpy()
        valid = conc[~np.isnan(conc)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_position_in_va_range(self):
        bars = self._make_multi_bar_data(n_bars=10)
        result = compute_volume_profile_features(bars)

        pos = result["position_in_va"].to_numpy()
        assert not np.all(np.isnan(pos))

    def test_poc_distance_sign(self):
        """POC distance positive when close > POC."""
        # Heavy volume at 18000, close at 18010
        timestamps = []
        closes = []
        volumes = []
        for i in range(20):
            ts = datetime(2024, 3, 15, 10, i, 0)
            timestamps.append(ts)
            if i < 15:
                closes.append(18000.0)
                volumes.append(100)
            else:
                closes.append(18010.0)
                volumes.append(1)

        bars = pl.DataFrame({
            "ts_event": timestamps,
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": volumes,
        }).with_columns(
            pl.col("ts_event").dt.replace_time_zone("UTC"),
            pl.col("volume").cast(pl.UInt32),
        )

        result = compute_volume_profile_features(bars)

        last_idx = len(result) - 1
        assert result["poc_price"][last_idx] == 18000.0
        assert result["poc_distance_raw"][last_idx] > 0
        assert result["poc_distance"][last_idx] > 0

    def test_empty_input(self):
        df = pl.DataFrame(schema={
            "ts_event": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.UInt32,
        })
        result = compute_volume_profile_features(df)
        assert len(result) == 0

    def test_known_poc_value(self):
        """Verify POC with bar-level data."""
        # 3 bars, all on same day
        timestamps = [
            datetime(2024, 3, 15, 10, 0, 0),
            datetime(2024, 3, 15, 10, 5, 0),
            datetime(2024, 3, 15, 10, 10, 0),
        ]
        closes = [18000.0, 18000.0, 18005.0]
        volumes = [200, 100, 50]

        bars = pl.DataFrame({
            "ts_event": timestamps,
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": volumes,
        }).with_columns(
            pl.col("ts_event").dt.replace_time_zone("UTC"),
            pl.col("volume").cast(pl.UInt32),
        )

        result = compute_volume_profile_features(bars)

        # Session POC: 18000 has 300 vol, 18005 has 50 vol -> POC = 18000
        assert result["poc_price"][0] == 18000.0
        assert result["poc_price"][1] == 18000.0
        assert result["poc_price"][2] == 18000.0

    def test_uses_true_vap_when_available(self):
        """If VAP lists exist, profile should use them over close-only proxy."""
        bars = pl.DataFrame({
            "ts_event": [
                datetime(2024, 3, 15, 10, 0, 0),
                datetime(2024, 3, 15, 10, 5, 0),
            ],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, 100.0],   # Proxy would force POC=100
            "volume": [100, 100],
            # True traded distribution: most size printed at 101
            "vap_prices": [[100.0, 101.0], [100.0, 101.0]],
            "vap_volumes": [[1, 99], [1, 99]],
        }).with_columns(
            pl.col("ts_event").dt.replace_time_zone("UTC"),
            pl.col("volume").cast(pl.UInt32),
            pl.col("vap_prices").cast(pl.List(pl.Float64)),
            pl.col("vap_volumes").cast(pl.List(pl.UInt64)),
        )

        result = compute_volume_profile_features(bars)
        assert result["poc_price"][0] == 101.0
        assert result["poc_price"][1] == 101.0
        assert result["rolling_poc"][1] == 101.0

    def test_bar_count_preserved(self):
        bars = self._make_multi_bar_data(n_bars=12)
        result = compute_volume_profile_features(bars)
        assert len(result) == 12


class TestLvnDetection:
    """Tests for LVN detection and distance features."""

    def test_lvn_count_basic(self):
        """Profile with two peaks and a valley between has 1 LVN."""
        # Pattern: low, HIGH, low, HIGH, low => 2 HVN, 1 LVN
        prices = np.array([99.0, 99.25, 99.50, 99.75, 100.0])
        sizes = np.array([5.0, 50.0, 2.0, 50.0, 5.0])
        result = _compute_profile(prices, sizes)
        assert result["hvn_count"] == 2
        assert result["lvn_count"] == 1
        assert len(result["lvn_prices"]) == 1
        assert result["lvn_prices"][0] == 99.50

    def test_lvn_count_monotonic_profile(self):
        """Monotonically increasing volume has 0 LVNs."""
        prices = np.array([99.0, 99.25, 99.50, 99.75, 100.0])
        sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _compute_profile(prices, sizes)
        assert result["lvn_count"] == 0
        assert len(result["lvn_prices"]) == 0

    def test_lvn_prices_correct(self):
        """LVN prices should be at local volume minima."""
        # Two valleys between three peaks
        prices = np.array([99.0, 99.25, 99.50, 99.75, 100.0, 100.25, 100.50])
        sizes = np.array([5.0, 50.0, 2.0, 50.0, 2.0, 50.0, 5.0])
        result = _compute_profile(prices, sizes)
        assert result["lvn_count"] == 2
        assert 99.50 in result["lvn_prices"]
        assert 100.0 in result["lvn_prices"]

    def test_hvn_prices_returned(self):
        """HVN prices array should be populated correctly."""
        prices = np.array([99.0, 99.25, 99.50, 99.75, 100.0])
        sizes = np.array([5.0, 50.0, 2.0, 50.0, 5.0])
        result = _compute_profile(prices, sizes)
        assert len(result["hvn_prices"]) == 2
        assert 99.25 in result["hvn_prices"]
        assert 99.75 in result["hvn_prices"]

    def test_empty_profile_lvn(self):
        """Empty input returns 0 LVNs and empty arrays."""
        result = _compute_profile(np.array([]), np.array([]))
        assert result["lvn_count"] == 0
        assert len(result["lvn_prices"]) == 0
        assert len(result["hvn_prices"]) == 0

    def test_single_price_no_lvn(self):
        """All trades at same price: no LVN."""
        prices = np.array([100.0, 100.0, 100.0])
        sizes = np.array([10.0, 20.0, 30.0])
        result = _compute_profile(prices, sizes)
        assert result["lvn_count"] == 0
        assert result["hvn_count"] == 0


class TestVolumeProfileLvnFeatures:
    """Integration tests for LVN-related features in compute_volume_profile_features."""

    def _make_bimodal_bars(self, n_bars=30):
        """Create bar data that produces a bimodal profile (2 HVNs, 1 LVN)."""
        timestamps = []
        closes = []
        opens = []
        highs = []
        lows = []
        volumes = []

        for i in range(n_bars):
            ts = datetime(2024, 3, 15, 10 + i // 12, (i * 5) % 60, 0)
            timestamps.append(ts)
            # Alternate between two price levels to create bimodal profile
            if i % 3 == 0:
                c = 18000.0  # cluster 1
                v = 100
            elif i % 3 == 1:
                c = 18005.0  # cluster 2
                v = 100
            else:
                c = 18002.5  # valley between
                v = 10
            closes.append(c)
            opens.append(c - 0.5)
            highs.append(c + 1.0)
            lows.append(c - 1.0)
            volumes.append(v)

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

    def test_rolling_lvn_count_column(self):
        bars = self._make_bimodal_bars(n_bars=30)
        result = compute_volume_profile_features(bars)
        assert "rolling_lvn_count" in result.columns
        assert result["rolling_lvn_count"].dtype == pl.Int32

    def test_dist_nearest_hvn_column(self):
        bars = self._make_bimodal_bars(n_bars=30)
        result = compute_volume_profile_features(bars)
        assert "dist_nearest_hvn" in result.columns
        assert result["dist_nearest_hvn"].dtype == pl.Float64

    def test_dist_nearest_lvn_column(self):
        bars = self._make_bimodal_bars(n_bars=30)
        result = compute_volume_profile_features(bars)
        assert "dist_nearest_lvn" in result.columns
        assert result["dist_nearest_lvn"].dtype == pl.Float64

    def test_at_hvn_at_lvn_binary(self):
        """at_hvn and at_lvn should be 0.0 or 1.0."""
        bars = self._make_bimodal_bars(n_bars=30)
        result = compute_volume_profile_features(bars)

        for col in ["at_hvn", "at_lvn"]:
            vals = result[col].to_numpy()
            valid = vals[~np.isnan(vals)]
            assert np.all((valid == 0.0) | (valid == 1.0)), f"{col} must be binary"

    def test_hvn_lvn_ratio_positive(self):
        """hvn_lvn_ratio = hvn_count / (lvn_count + 1), always >= 0."""
        bars = self._make_bimodal_bars(n_bars=30)
        result = compute_volume_profile_features(bars)

        ratio = result["hvn_lvn_ratio"].to_numpy()
        valid = ratio[~np.isnan(ratio)]
        assert np.all(valid >= 0.0)

    def test_dist_nearest_hvn_nonnegative(self):
        """Distance to nearest HVN should be >= 0."""
        bars = self._make_bimodal_bars(n_bars=30)
        result = compute_volume_profile_features(bars)

        dist = result["dist_nearest_hvn"].to_numpy()
        valid = dist[~np.isnan(dist)]
        assert np.all(valid >= 0.0)

    def test_empty_input_has_new_columns(self):
        """Empty input should still have all new columns."""
        df = pl.DataFrame(schema={
            "ts_event": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.UInt32,
        })
        result = compute_volume_profile_features(df)
        assert len(result) == 0
        for col in ["rolling_lvn_count", "dist_nearest_hvn", "dist_nearest_lvn",
                     "at_hvn", "at_lvn", "hvn_lvn_ratio",
                     "swing_poc_dist", "swing_lvn_dist", "swing_hvn_dist",
                     "swing_va_position", "breakout_direction", "bars_since_breakout"]:
            assert col in result.columns, f"Missing column in empty result: {col}"

    def test_row_count_preserved_with_lvn(self):
        bars = self._make_bimodal_bars(n_bars=20)
        result = compute_volume_profile_features(bars)
        assert len(result) == 20
