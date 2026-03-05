# NQ Alpha Feature Catalog
# Compact reference for thinker and coder agents. Grouped by category.
# Format: column_name | formula/computation | interpretation / use in strategy

## OHLCV BASICS
open    | bar open price (pts)            | reference for bar direction; open > close = down bar
high    | bar high price (pts)            | upper extreme; used for stop placement above
low     | bar low price (pts)             | lower extreme; used for stop placement below
close   | bar close price (pts)           | primary signal input; all shifts must use shift(1) for causal use
volume  | total contracts traded in bar   | absolute activity level; compare vs volume_ma5 for anomaly

## MOMENTUM / OHLCV DERIVED
return_1bar         | (close - prev_close) / prev_close         | single-bar return; +0.001 = +0.1%
return_5bar         | sum(return_1bar, 5 bars)                  | short-term momentum; fade extremes on NQ intraday
high_low_range      | (high - low) / close                      | bar volatility proxy; high = expansion, low = compression
range_ma5           | MA_5(high_low_range)                      | smoothed volatility baseline
volume_ma5          | MA_5(volume)                              | volume baseline; use for volume_ratio threshold
volume_ratio        | volume / (volume_ma5 + ε)                 | >1.5 = elevated volume; >2.5 = surge / ignition bar
close_position      | (close - low) / (high - low + ε)         | 0 = closed at low (bearish), 1 = closed at high (bullish)
upper_wick_ratio    | (high - max(open,close)) / range          | rejection at highs; >0.6 = strong selling at top
lower_wick_ratio    | (min(open,close) - low) / range           | rejection at lows; >0.6 = strong buying at bottom
vwap_deviation      | close - session_vwap                      | price premium over VWAP; negative = trading below fair value
vwap_dev_zscore     | (vwap_deviation - μ_24) / σ_24            | standardised distance; <-2 = extended below VWAP = reversion candidate

## STATISTICAL
log_return          | log(close / prev_close)                   | stationary return series; better for z-score comparisons
fracdiff_close      | fractional diff d=0.4 over 50-bar window  | memory-preserving stationarity; useful for ML-style scoring
yz_volatility       | Yang-Zhang realized vol (20-bar rolling)  | robust intraday vol estimator; use for regime filtering
vol_zscore          | (yz_volatility - μ_50) / σ_50             | vol regime; >2 = spike regime, <-1 = compressed regime

## ORDERFLOW
volume_delta        | buy_volume - sell_volume (raw contracts)  | signed aggression; positive = net buying
cvd                 | cumsum(volume_delta), resets each session  | cumulative directional flow; rising trend = sustained buying
order_flow_imbalance| sum(volume_delta, 5 bars)                 | short-term flow persistence; positive = recent buy pressure
volume_imbalance    | (buy_vol - sell_vol) / (buy_vol + sell_vol + 1) | normalised -1 to +1; >0.5 = dominant buying
cvd_price_divergence_3 | 1 if sign(CVD_3bar) ≠ sign(price_slope_3) | divergence: flow opposes price = likely reversal in 1-3 bars
cvd_price_divergence_6 | 1 if sign(CVD_6bar) ≠ sign(price_slope_6) | same over 6 bars; more reliable, fewer signals
absorption_signal   | 1 if volume>p90 AND range<p20 (rolling)   | high vol + small range = institutional absorption; fade signal
absorption_factor   | volume * abs(delta) / (range_ticks + 1)   | continuous absorption strength; higher = more absorption
orderflow_ratio     | max(buy,sell) / (total_vol + 1)           | one-sidedness; >0.8 = institutional sweep / momentum signal
large_trade_ratio   | large_trade_count / (trade_count + 1)     | >0.3 = dominated by large trades = informed flow

## MICROSTRUCTURE
tape_speed          | trade_count / bar_duration_ms             | raw tape speed in trades/ms
tape_speed_z        | (tape_speed - μ_24) / σ_24                | normalised tape speed; >2 = unusually fast tape
tape_speed_spike    | 1 if tape_speed_z > 2.5                   | binary spike flag; precedes breakouts / ignition moves
price_velocity      | abs(close - open) / bar_duration_sec      | urgency of price move in $/sec
price_velocity_z    | (price_velocity - μ_24) / σ_24            | >2.5 = abnormally fast price move
is_whip             | 1 if price_velocity_z > 3.0               | extreme velocity bar; often reverses on next bar
recoil_pct          | abs(close_t - close_t-1) / range_t-1 if prev=whip | 0.4-0.7 = healthy recoil after whip = continuation
recoil_50pct        | 1 if 0.4 < recoil_pct < 0.7              | clean recoil confirmation flag; use in state machine
vpin                | MA_20(abs(buy_vol-sell_vol)/total_vol)     | informed trading probability; >0.6 = elevated toxicity
weighted_book_imbalance | (bid_size*n_bid - ask_size*n_ask) / total | book pressure -1 to +1; positive = more bid depth
mean_trade_interval_us | bar_duration_ns / (trade_count-1) / 1000 | μs between trades; low = clustered arrivals = informed
cancel_trade_ratio  | cancel_count / (trade_count + 1)          | >2 = HFT noise dominates; skip entries in noisy conditions
trade_intensity     | trade_count / (bar_duration_ns / 1e9)     | trades per second; spikes precede directional moves

## FOOTPRINT (DEEP CHART)
stacked_imb_strength | count * direction of stacked imbalances  | positive = buy-side stacked, negative = sell-side stacked
stacked_imb_streak  | bars since sign flip of stacked_imb      | >3 = persistent one-sided stacking = momentum confirmation
fp_unfinished_high  | 1 if bar has poor high (incomplete auction) | auction failed to find sellers; price should return to complete it
fp_unfinished_low   | 1 if bar has poor low (incomplete auction) | auction failed to find buyers; price should return to complete it
bars_since_unfinished_high | bars elapsed since last poor high  | 0 = this bar; use <5 for recency filter in state machine
bars_since_unfinished_low  | bars elapsed since last poor low   | 0 = this bar; use <5 for recency filter in state machine
delta_intensity_z   | (delta_per_sec - μ_24) / σ_24            | normalised delta speed; >2 = aggressive flow, <-2 = aggressive selling
delta_heat          | abs(delta_intensity_z)                    | magnitude only; >2 = extreme flow in either direction
extreme_aggression_high | 1 if sell_vol_at_high > 2x buy_vol_at_high | sellers defending bar high; bearish reversal signal
extreme_aggression_low  | 1 if buy_vol_at_low > 2x sell_vol_at_low  | buyers absorbing bar low; bullish reversal signal
extreme_buy_ratio_high  | buy_vol_at_high / (buy+sell at high + 1)   | 0=all sellers at high, 1=all buyers at high
extreme_buy_ratio_low   | buy_vol_at_low  / (buy+sell at low  + 1)   | 0=all sellers at low,  1=all buyers at low
max_level_vol_ratio | max_level_volume / total_volume           | >0.3 = volume concentrated at single price = strong support/resistance
high_low_vol_ratio  | (vol_at_high + vol_at_low) / total_vol   | >0.4 = activity at extremes = probing / rejection behaviour

## AMT (VOLUME PROFILE & OPENING RANGE)
# See thinker system prompt for full AMT framework description
va_high             | value area high (prior session ~70% vol)  | VAH; price above = out of value; rejection here = short
va_low              | value area low (prior session ~70% vol)   | VAL; price below = out of value; rejection here = long
position_in_va      | (close - va_low) / (va_high - va_low)    | <0=below VA, 0-1=inside VA, >1=above VA; use for VA rejection signal
va_width            | va_high - va_low (points)                 | wide VA = balance day, narrow VA = trending day
poc_price           | price with highest prior-session volume   | POC; magnet for price; fade moves away from POC
poc_distance        | (close - poc_price) / poc_price           | normalised; >0=above POC, <0=below POC; fade extremes
poc_distance_raw    | close - poc_price (points)                | raw distance from POC; use for tick-based threshold
poc_slope_6         | slope of rolling_poc over 6 bars          | positive = POC drifting up = upward value migration
rolling_va_high     | intraday rolling value area high          | intraday VAH equivalent
rolling_va_low      | intraday rolling value area low           | intraday VAL equivalent
rolling_va_position | (close - rolling_va_low) / rolling_va_width | intraday position within rolling VA
rolling_poc         | intraday rolling point of control         | intraday POC
rolling_poc_distance| close - rolling_poc (points)              | distance from intraday POC
or_broken_up        | 1 when price first breaks above OR high  | IB breakout long signal; state machine: arm on this, wait for pullback
or_broken_down      | 1 when price first breaks below OR low   | IB breakout short signal; failed version = fade back inside OR
or_width            | opening range high - low (points)         | <4pts = too narrow (whipsaw); >30pts = too wide (exhaustion)
position_in_or      | (close - or_low) / (or_high - or_low)   | <0=below OR, 0-1=inside OR, >1=above OR
at_hvn              | 1 if price near high volume node          | expect stalling / consolidation; do not initiate trend trade here
at_lvn              | 1 if price near low volume node           | expect fast passage through thin area; momentum trades work here
dist_nearest_hvn    | distance to nearest HVN (points)          | small = near strong support/resistance
dist_nearest_lvn    | distance to nearest LVN (points)          | small = in thin air = price can move quickly
breakout_direction  | 1=up breakout, -1=down breakout, 0=inside | breakout from swing value area
bars_since_breakout | bars since the last breakout occurred     | 0=breakout bar; use <3 for fresh breakout entries
swing_va_position   | position within recent swing value area   | same as position_in_va but swing-scoped
swing_poc_dist      | distance from swing POC (points)          | fade when >N ticks from swing POC
