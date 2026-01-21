# Indicator Catalog - Non-Redundant Design

## Overview

This catalog defines the complete indicator library, organized by unique information content. Each indicator is selected to provide **distinct, non-redundant information** that cannot be derived from other indicators in the set.

### Design Principles

1. **Information Orthogonality:** Each indicator captures unique market dynamics
2. **No Redundancy:** Avoid indicators that are mathematical transformations of each other
3. **Interpretability:** Every indicator has clear, actionable interpretation
4. **Computational Efficiency:** Indicators are designed for batch computation
5. **Parameter Robustness:** Default parameters work across market regimes

---

## Group A: Trend Indicators

### Purpose
Identify the **direction and strength of price movement** over various time horizons. Trend indicators answer: "Is the asset moving up, down, or sideways, and how strongly?"

### Unique Information Content
- Direction of price movement (bullish/bearish/neutral)
- Persistence of directional moves
- Trend strength and momentum
- Multiple timeframe trend alignment

---

### A1. Exponential Moving Average (EMA) Suite

| Property | Value |
|----------|-------|
| **Unique Info** | Smoothed price level with recency weighting; captures trend at specific horizons |
| **Why Not Redundant** | EMAs at different periods capture different trend timeframes; EMA ≠ SMA (recency bias matters) |
| **Parameters** | `periods: [8, 21, 50, 200]` |
| **Output** | Price level (same unit as close price) |
| **Signal** | Price above EMA = bullish; below = bearish |

```python
# Non-redundancy justification:
# - EMA(8): Short-term momentum (1-2 weeks)
# - EMA(21): Intermediate trend (1 month)
# - EMA(50): Medium-term trend (1 quarter)
# - EMA(200): Long-term trend (1 year)
# These capture genuinely different information horizons.
# SMA is NOT included as EMA provides similar info with better responsiveness.
```

---

### A2. MACD (Moving Average Convergence Divergence)

| Property | Value |
|----------|-------|
| **Unique Info** | Trend momentum via EMA convergence/divergence; captures acceleration |
| **Why Not Redundant** | Measures *rate of change* of trend, not trend level; histogram shows momentum of momentum |
| **Parameters** | `fast: 12, slow: 26, signal: 9` |
| **Output** | `{macd: float, signal: float, histogram: float}` |
| **Signal** | Histogram > 0 + rising = accelerating uptrend |

```python
# MACD captures:
# 1. Trend direction (MACD line above/below zero)
# 2. Trend momentum (MACD vs signal line)
# 3. Momentum acceleration (histogram slope)
# This is distinct from raw EMA values.
```

---

### A3. ADX (Average Directional Index)

| Property | Value |
|----------|-------|
| **Unique Info** | Trend *strength* independent of direction |
| **Why Not Redundant** | Only indicator measuring trend intensity; agnostic to up/down |
| **Parameters** | `period: 14` |
| **Output** | `{adx: 0-100, plus_di: float, minus_di: float}` |
| **Signal** | ADX > 25 = trending; < 20 = ranging; DI crossovers for direction |

```python
# ADX answers: "How strongly is price trending?"
# - High ADX (>40): Strong trend (trade with trend)
# - Low ADX (<20): Weak/no trend (mean reversion may work)
# Unique: No other indicator quantifies trend strength this way.
```

---

### A4. Supertrend

| Property | Value |
|----------|-------|
| **Unique Info** | Volatility-adjusted trend with built-in trailing stop levels |
| **Why Not Redundant** | Combines trend direction with ATR-based support/resistance; provides actionable levels |
| **Parameters** | `period: 10, multiplier: 3.0` |
| **Output** | `{direction: 1/-1, level: float}` |
| **Signal** | Price above supertrend = uptrend; crossovers = potential reversals |

```python
# Supertrend is unique because:
# 1. It adapts to volatility (unlike fixed MAs)
# 2. Provides specific price levels (not just direction)
# 3. Acts as dynamic support/resistance
```

---

### A5. Aroon Oscillator

| Property | Value |
|----------|-------|
| **Unique Info** | Time since recent high/low; measures trend "freshness" |
| **Why Not Redundant** | Time-based metric (not price-based); captures how recently trend established |
| **Parameters** | `period: 25` |
| **Output** | `{aroon_up: 0-100, aroon_down: 0-100, oscillator: -100 to 100}` |
| **Signal** | Aroon Up > 70 = recent new highs; strong uptrend |

```python
# Aroon uniquely measures:
# "How many bars since the highest high / lowest low?"
# This time-based information is not captured by price-based indicators.
```

---

## Group B: Momentum Indicators

### Purpose
Measure the **velocity and acceleration of price changes**. Momentum indicators answer: "How fast is price moving, and is that speed increasing or decreasing?"

### Unique Information Content
- Speed of price changes
- Overbought/oversold conditions
- Momentum divergences (price vs. momentum disagreement)
- Mean-reversion potential

---

### B1. RSI (Relative Strength Index)

| Property | Value |
|----------|-------|
| **Unique Info** | Ratio of up-moves to down-moves; bounded oscillator for OB/OS |
| **Why Not Redundant** | Normalizes momentum to 0-100 scale; comparable across all assets |
| **Parameters** | `period: 14, overbought: 70, oversold: 30` |
| **Output** | Value 0-100 |
| **Signal** | < 30 = oversold (potential bounce); > 70 = overbought (potential pullback) |

```python
# RSI is the canonical momentum oscillator:
# - Bounded (0-100): Universal interpretation across assets
# - Divergences: Price makes new high, RSI doesn't = warning
# Not redundant with Stochastic (different calculation, different sensitivity)
```

---

### B2. Stochastic Oscillator (Slow)

| Property | Value |
|----------|-------|
| **Unique Info** | Close position within recent high-low range |
| **Why Not Redundant** | Uses range (H/L), not just close; captures "where in the range are we?" |
| **Parameters** | `k_period: 14, d_period: 3, slowing: 3` |
| **Output** | `{k: 0-100, d: 0-100}` |
| **Signal** | %K/%D crossovers; < 20 oversold, > 80 overbought |

```python
# Stochastic vs RSI:
# - RSI: Based on average gains vs losses (magnitude)
# - Stochastic: Based on close position in H-L range (location)
# Both provide OB/OS, but from different perspectives.
```

---

### B3. ROC (Rate of Change)

| Property | Value |
|----------|-------|
| **Unique Info** | Simple percentage change over N periods; raw momentum |
| **Why Not Redundant** | Unbounded, raw return; not smoothed or normalized |
| **Parameters** | `periods: [5, 10, 21, 63, 126, 252]` |
| **Output** | Percentage (e.g., 0.05 = 5%) |
| **Signal** | Positive = upward momentum; magnitude = strength |

```python
# ROC at multiple horizons:
# - ROC(5): 1-week momentum
# - ROC(21): 1-month momentum
# - ROC(63): Quarterly momentum
# - ROC(252): Annual momentum (used in factor investing)
# Raw returns are fundamental; not redundant with oscillators.
```

---

### B4. Williams %R

| Property | Value |
|----------|-------|
| **Unique Info** | Inverted stochastic with different scaling; more sensitive |
| **Why Not Redundant** | Faster than Stochastic; better for short-term timing |
| **Parameters** | `period: 14` |
| **Output** | Value -100 to 0 |
| **Signal** | > -20 = overbought; < -80 = oversold |

```python
# Williams %R vs Stochastic:
# - Same concept, but %R is unsmoothed
# - More sensitive to price changes
# - Better for short-term mean reversion signals
# Include only if short-term timing is priority; otherwise, Stochastic suffices.
```

---

### B5. CCI (Commodity Channel Index)

| Property | Value |
|----------|-------|
| **Unique Info** | Price deviation from statistical mean; measures "unusualness" |
| **Why Not Redundant** | Unbounded; uses typical price (HLC/3); statistical basis |
| **Parameters** | `period: 20` |
| **Output** | Unbounded value (typically -300 to +300) |
| **Signal** | > 100 = unusually high; < -100 = unusually low |

```python
# CCI uniquely measures:
# "How far is price from its 'normal' level?"
# Uses standard deviation for context, unlike RSI/Stochastic.
# Unbounded nature allows it to identify extreme moves.
```

---

### B6. TSI (True Strength Index)

| Property | Value |
|----------|-------|
| **Unique Info** | Double-smoothed momentum; low noise, high signal |
| **Why Not Redundant** | Superior noise filtering vs RSI; better for divergences |
| **Parameters** | `long_period: 25, short_period: 13, signal_period: 13` |
| **Output** | `{tsi: -100 to 100, signal: float}` |
| **Signal** | Zero-line crossovers; divergences with price |

```python
# TSI vs RSI:
# - TSI is double-smoothed (less whipsaw)
# - TSI oscillates around zero (more intuitive trend bias)
# - Better for longer-term momentum and divergence detection
```

---

## Group C: Volatility / Risk Indicators

### Purpose
Quantify **price variability and risk**. Volatility indicators answer: "How much is price moving, and what are the risk parameters?"

### Unique Information Content
- Price dispersion and range
- Risk levels for position sizing
- Volatility regimes (high/low vol environments)
- Breakout potential

---

### C1. ATR (Average True Range)

| Property | Value |
|----------|-------|
| **Unique Info** | Average daily price range including gaps |
| **Why Not Redundant** | Absolute volatility measure (in price units); foundation for risk management |
| **Parameters** | `period: 14` |
| **Output** | Price units (e.g., $2.50) |
| **Signal** | Higher ATR = higher volatility; use for stop-loss sizing |

```python
# ATR is the fundamental volatility building block:
# - Used for position sizing (risk per trade)
# - Basis for volatility-adjusted indicators (Keltner, Supertrend)
# - Absolute (not %) so useful for stop placement
```

---

### C2. ATR Percentage (ATRP)

| Property | Value |
|----------|-------|
| **Unique Info** | ATR as percentage of price; normalized volatility |
| **Why Not Redundant** | Comparable across assets (unlike raw ATR) |
| **Parameters** | `period: 14` |
| **Output** | Percentage (e.g., 0.02 = 2%) |
| **Signal** | High ATRP = higher relative risk; useful for cross-asset comparison |

```python
# ATRP = ATR / Close * 100
# Enables comparison: "Is AAPL more volatile than MSFT?"
# Essential for universe-wide volatility rankings.
```

---

### C3. Bollinger Bands

| Property | Value |
|----------|-------|
| **Unique Info** | Statistical price envelope; dynamic support/resistance |
| **Why Not Redundant** | Uses standard deviation (probabilistic); provides actionable levels |
| **Parameters** | `period: 20, std_dev: 2.0` |
| **Output** | `{upper: float, middle: float, lower: float, width: float, %b: float}` |
| **Signal** | Price at upper band = extended; %B for mean reversion |

```python
# Bollinger Bands provide:
# 1. Dynamic S/R levels (upper/lower bands)
# 2. Volatility state (bandwidth)
# 3. Relative position (%B)
# Statistical basis (std dev) is unique among envelope indicators.
```

---

### C4. Keltner Channels

| Property | Value |
|----------|-------|
| **Unique Info** | ATR-based envelope; smoother than Bollinger |
| **Why Not Redundant** | Uses ATR (not std dev); less reactive to outliers |
| **Parameters** | `period: 20, atr_period: 10, multiplier: 1.5` |
| **Output** | `{upper: float, middle: float, lower: float}` |
| **Signal** | Price outside channels = strong momentum; squeeze with BB = breakout setup |

```python
# Keltner vs Bollinger:
# - Keltner uses ATR (more stable)
# - Bollinger uses std dev (more reactive)
# Together: "Bollinger Squeeze" signals compression before breakout
```

---

### C5. Historical Volatility (HV)

| Property | Value |
|----------|-------|
| **Unique Info** | Annualized standard deviation of returns |
| **Why Not Redundant** | Standard financial risk metric; comparable to IV |
| **Parameters** | `periods: [10, 21, 63]` (10-day, 1-month, quarterly) |
| **Output** | Annualized percentage (e.g., 0.25 = 25%) |
| **Signal** | Rising HV = increasing risk; compare to IV for vol trading |

```python
# HV at multiple windows:
# - HV(10): Recent/short-term volatility
# - HV(21): Monthly volatility (most common)
# - HV(63): Quarterly volatility (smoother)
# Annualized for comparability with implied volatility.
```

---

### C6. Volatility Ratio (VR)

| Property | Value |
|----------|-------|
| **Unique Info** | Short-term vs long-term volatility; vol regime changes |
| **Why Not Redundant** | Relative measure; detects vol expansion/contraction |
| **Parameters** | `short_period: 10, long_period: 50` |
| **Output** | Ratio (1.0 = normal; >1.5 = elevated short-term vol) |
| **Signal** | VR > 1.5 = volatility spike; potential regime change |

```python
# Volatility Ratio = HV(short) / HV(long)
# Uniquely captures: "Is current volatility unusual vs. history?"
# Useful for vol timing and risk management.
```

---

### C7. Ulcer Index

| Property | Value |
|----------|-------|
| **Unique Info** | Downside volatility; measures drawdown risk |
| **Why Not Redundant** | Focuses only on negative volatility; risk-adjusted perspective |
| **Parameters** | `period: 14` |
| **Output** | Percentage (lower = less drawdown risk) |
| **Signal** | Low UI = smooth uptrend; high UI = choppy/declining |

```python
# Ulcer Index vs HV:
# - HV measures all volatility (up and down)
# - UI measures only drawdown volatility
# More relevant for risk-averse investors.
```

---

## Group D: Volume Indicators

### Purpose
Analyze **trading activity and conviction** behind price moves. Volume indicators answer: "Is the price move supported by participation?"

### Unique Information Content
- Participation in price moves
- Accumulation vs. distribution
- Smart money activity
- Volume-price confirmation

---

### D1. OBV (On-Balance Volume)

| Property | Value |
|----------|-------|
| **Unique Info** | Cumulative volume direction; tracks buying/selling pressure |
| **Why Not Redundant** | Simplest volume-trend indicator; leading indicator for price |
| **Parameters** | None (cumulative) |
| **Output** | Cumulative volume (unsigned) |
| **Signal** | OBV divergence from price = potential reversal |

```python
# OBV logic:
# - Price up: Add volume to running total
# - Price down: Subtract volume
# Divergences between OBV and price are powerful signals.
```

---

### D2. VWAP (Volume Weighted Average Price)

| Property | Value |
|----------|-------|
| **Unique Info** | Average price weighted by volume; institutional benchmark |
| **Why Not Redundant** | Unique anchor: where did volume actually trade? |
| **Parameters** | `anchor: 'session' or 'rolling_N'` |
| **Output** | Price level |
| **Signal** | Price > VWAP = bullish intraday; institutional support level |

```python
# VWAP is unique:
# - Institutional benchmark for execution quality
# - Different from MA (volume-weighted, not time-weighted)
# - Acts as intraday support/resistance
```

---

### D3. Money Flow Index (MFI)

| Property | Value |
|----------|-------|
| **Unique Info** | Volume-weighted RSI; OB/OS with volume confirmation |
| **Why Not Redundant** | Combines momentum (RSI logic) with volume |
| **Parameters** | `period: 14` |
| **Output** | Value 0-100 |
| **Signal** | < 20 = oversold with volume confirmation; > 80 = overbought |

```python
# MFI vs RSI:
# - RSI: Price momentum only
# - MFI: Price momentum × volume
# MFI oversold/overbought signals have volume confirmation.
```

---

### D4. Accumulation/Distribution Line (ADL)

| Property | Value |
|----------|-------|
| **Unique Info** | Close location value × volume; intrabar accumulation |
| **Why Not Redundant** | Uses close position within bar (not just direction like OBV) |
| **Parameters** | None (cumulative) |
| **Output** | Cumulative value |
| **Signal** | ADL rising while price flat = accumulation (bullish) |

```python
# ADL vs OBV:
# - OBV: Binary (up day adds all volume, down subtracts all)
# - ADL: Proportional (close near high adds more than close near low)
# ADL is more nuanced measure of intrabar buying pressure.
```

---

### D5. Chaikin Money Flow (CMF)

| Property | Value |
|----------|-------|
| **Unique Info** | ADL normalized over period; bounded oscillator |
| **Why Not Redundant** | Bounded version of ADL; easier to interpret |
| **Parameters** | `period: 20` |
| **Output** | Value -1 to +1 |
| **Signal** | > 0.25 = strong buying; < -0.25 = strong selling |

```python
# CMF = Sum(ADL) / Sum(Volume) over period
# Normalized: Comparable across assets and time
# ADL is cumulative; CMF is current state.
```

---

### D6. Volume Rate of Change (VROC)

| Property | Value |
|----------|-------|
| **Unique Info** | Volume momentum; unusual volume detection |
| **Why Not Redundant** | Raw volume change (not price-related) |
| **Parameters** | `period: 14` |
| **Output** | Percentage change in volume |
| **Signal** | High VROC = unusual activity; potential breakout |

```python
# VROC uniquely measures:
# "Is today's volume unusual compared to N periods ago?"
# Volume spikes often precede or confirm moves.
```

---

### D7. Volume Relative to Average

| Property | Value |
|----------|-------|
| **Unique Info** | Current volume as multiple of average; participation metric |
| **Why Not Redundant** | Simple, interpretable activity measure |
| **Parameters** | `period: 20` |
| **Output** | Ratio (1.0 = average; 2.0 = 2x normal) |
| **Signal** | > 1.5 = elevated interest; moves with high volume are more significant |

```python
# Volume Ratio = Current Volume / SMA(Volume, 20)
# Simple but essential for filtering:
# "Is this price move happening on meaningful volume?"
```

---

## Group E: Market Microstructure Indicators

### Purpose
Capture **price action patterns and intrabar dynamics**. Microstructure indicators answer: "What does the shape of price action tell us about buyer/seller control?"

### Unique Information Content
- Intrabar buyer/seller dominance
- Price rejection patterns
- Gap analysis
- Range dynamics

---

### E1. Candlestick Pattern Recognition

| Property | Value |
|----------|-------|
| **Unique Info** | OHLC relationship patterns; reversal/continuation signals |
| **Why Not Redundant** | Only indicator using full OHLC relationship |
| **Parameters** | Pattern-specific |
| **Output** | `{pattern: str, direction: 1/-1, strength: float}` |
| **Signal** | Pattern-dependent (e.g., hammer = bullish reversal) |

```python
# Key patterns to detect:
PATTERNS = [
    # Single candle
    'doji', 'hammer', 'shooting_star', 'marubozu',
    # Two candle
    'engulfing', 'harami', 'piercing', 'dark_cloud',
    # Three candle
    'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows'
]
# Each pattern encodes specific buyer/seller dynamics.
```

---

### E2. True Range (TR)

| Property | Value |
|----------|-------|
| **Unique Info** | Single-bar volatility including gaps |
| **Why Not Redundant** | Foundation metric; captures full bar range |
| **Parameters** | None |
| **Output** | Price units |
| **Signal** | Expanding TR = increased volatility; use with ATR |

```python
# TR = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
# Captures gaps that High-Low misses.
# Building block for ATR.
```

---

### E3. Body-to-Range Ratio

| Property | Value |
|----------|-------|
| **Unique Info** | Candle body as proportion of total range; conviction measure |
| **Why Not Redundant** | Unique measure of intrabar conviction |
| **Parameters** | None |
| **Output** | Ratio 0-1 |
| **Signal** | High ratio (>0.8) = strong conviction; low ratio = indecision |

```python
# Body/Range = abs(Close - Open) / (High - Low)
# High: Strong directional move
# Low: Doji-like, indecision
# Simple but powerful filter for move quality.
```

---

### E4. Gap Analysis

| Property | Value |
|----------|-------|
| **Unique Info** | Overnight price jumps; overnight sentiment shifts |
| **Why Not Redundant** | Only indicator capturing inter-session dynamics |
| **Parameters** | `min_gap_pct: 0.5` |
| **Output** | `{gap_pct: float, gap_type: 'up'/'down'/'none', filled: bool}` |
| **Signal** | Unfilled gaps = strong trend; gap fills = mean reversion |

```python
# Gap types:
# - Common: Usually fills
# - Breakaway: Start of new trend
# - Runaway: Continuation
# - Exhaustion: End of trend
# Gaps encode overnight institutional activity.
```

---

### E5. Range Position

| Property | Value |
|----------|-------|
| **Unique Info** | Close position within period's high-low range |
| **Why Not Redundant** | Multi-bar version of stochastic concept |
| **Parameters** | `period: 20` |
| **Output** | Percentage 0-100 |
| **Signal** | Near 100 = near period high; near 0 = near period low |

```python
# Range Position = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
# Simpler than stochastic, clearer interpretation.
```

---

## Group F: Regime Detection Indicators

### Purpose
Identify **market environment states**. Regime indicators answer: "What type of market are we in, and how should strategy adapt?"

### Unique Information Content
- Trending vs. ranging classification
- Volatility regime (high/low/normal)
- Market state transitions
- Optimal strategy selection

---

### F1. ADX Regime Classification

| Property | Value |
|----------|-------|
| **Unique Info** | Trend strength bucketed into regimes |
| **Why Not Redundant** | Transforms continuous ADX into actionable regime |
| **Parameters** | `thresholds: [15, 25, 40]` |
| **Output** | `{regime: 'ABSENT'/'WEAK'/'MODERATE'/'STRONG', adx: float}` |
| **Signal** | STRONG = trend-following works; ABSENT = mean-reversion works |

```python
# ADX Regime mapping:
# < 15: No trend (range-bound)
# 15-25: Weak trend (mixed signals)
# 25-40: Moderate trend (trend-following favorable)
# > 40: Strong trend (strong trend-following)
```

---

### F2. Volatility Regime

| Property | Value |
|----------|-------|
| **Unique Info** | Volatility percentile vs. history |
| **Why Not Redundant** | Historical context for current volatility |
| **Parameters** | `lookback: 252, thresholds: [25, 75]` |
| **Output** | `{regime: 'LOW'/'NORMAL'/'HIGH', percentile: float, hv: float}` |
| **Signal** | LOW vol = potential breakout; HIGH vol = potential mean reversion |

```python
# Volatility regimes affect:
# - Position sizing (smaller in high vol)
# - Strategy selection (vol-selling in high vol)
# - Risk expectations
```

---

### F3. Trend-Range Classification (Choppiness Index)

| Property | Value |
|----------|-------|
| **Unique Info** | Measures market choppiness vs. trend |
| **Why Not Redundant** | Direct trending/ranging classifier |
| **Parameters** | `period: 14` |
| **Output** | Value 0-100 |
| **Signal** | > 61.8 = choppy (avoid trend strategies); < 38.2 = trending |

```python
# Choppiness Index = 100 * LOG10(SUM(ATR) / (Max-Min)) / LOG10(period)
# High: Choppy, ranging market
# Low: Directional, trending market
# Uses Fibonacci levels (61.8, 38.2) as thresholds.
```

---

### F4. Composite Regime Indicator

| Property | Value |
|----------|-------|
| **Unique Info** | Multi-factor regime classification |
| **Why Not Redundant** | Combines trend + volatility + volume into single regime |
| **Parameters** | Derived from ADX, HV, Volume |
| **Output** | `{regime: 'BULL_QUIET'/'BULL_VOLATILE'/'BEAR_QUIET'/'BEAR_VOLATILE'/'RANGE', confidence: float}` |
| **Signal** | Regime-appropriate strategy selection |

```python
# Composite regime considers:
# 1. Trend: ADX + price vs MA
# 2. Volatility: HV percentile
# 3. Volume: Participation level
# Output: Clear market characterization
```

---

## Group G: Relative Strength Indicators

### Purpose
Compare **performance against benchmarks or peers**. RS indicators answer: "Is this asset outperforming or underperforming its reference?"

### Unique Information Content
- Relative performance vs. benchmark
- Sector/peer ranking
- Leadership vs. laggard status
- Rotation signals

---

### G1. Relative Strength vs. Benchmark (RSB)

| Property | Value |
|----------|-------|
| **Unique Info** | Asset return minus benchmark return |
| **Why Not Redundant** | Only indicator comparing to external reference |
| **Parameters** | `benchmark: 'SPY', periods: [21, 63, 252]` |
| **Output** | Excess return (percentage) |
| **Signal** | Positive = outperforming; negative = underperforming |

```python
# RS = Asset Return - Benchmark Return
# Multi-period for short/medium/long-term comparison
# Essential for relative value and rotation strategies.
```

---

### G2. Relative Strength Ratio Line

| Property | Value |
|----------|-------|
| **Unique Info** | Price ratio trend; relative momentum |
| **Why Not Redundant** | Ratio line (not return difference); shows trend of relative performance |
| **Parameters** | `benchmark: 'SPY'` |
| **Output** | Ratio (asset price / benchmark price) |
| **Signal** | Rising ratio = outperformance; falling = underperformance |

```python
# RS Ratio = Asset Price / Benchmark Price
# Trend of ratio line matters:
# - Rising: Gaining relative strength
# - Falling: Losing relative strength
# Apply trend indicators (MA) to ratio for signals.
```

---

### G3. Sector Relative Strength

| Property | Value |
|----------|-------|
| **Unique Info** | Performance vs. sector ETF |
| **Why Not Redundant** | Sector-specific benchmark (more relevant than broad market) |
| **Parameters** | `sector_etf: auto-detected` |
| **Output** | `{vs_sector: float, sector: str}` |
| **Signal** | Best-in-sector = leadership potential |

```python
# Compare AAPL to XLK (Tech sector ETF)
# Stock can outperform sector while sector underperforms market
# Distinguishes stock-specific alpha from sector beta.
```

---

### G4. Percentile Rank (Universe)

| Property | Value |
|----------|-------|
| **Unique Info** | Cross-sectional ranking within universe |
| **Why Not Redundant** | Only indicator providing peer context |
| **Parameters** | `universe: 'SP500', metric: 'return_21d'` |
| **Output** | Percentile 0-100 |
| **Signal** | Top decile = leader; bottom decile = laggard |

```python
# Percentile rank across universe:
# "Where does this stock rank among peers?"
# Applicable to any metric (return, volatility, volume, etc.)
```

---

### G5. Mansfield Relative Strength

| Property | Value |
|----------|-------|
| **Unique Info** | Long-term RS with zero-line normalization |
| **Why Not Redundant** | Normalized RS for consistent interpretation |
| **Parameters** | `benchmark: 'SPY', ma_period: 52` (weekly) |
| **Output** | Normalized value (0 = average RS) |
| **Signal** | > 0 = long-term outperformance; < 0 = underperformance |

```python
# Mansfield RS = ((Price/Benchmark) / MA(Price/Benchmark)) - 1) * 100
# Zero-centered: Easy to see above/below average RS
# Popular in Weinstein-style analysis.
```

---

## Group H: Structure / Pattern Indicators

### Purpose
Identify **price structure, support/resistance, and pattern formations**. Structure indicators answer: "What are the key price levels and formations?"

### Unique Information Content
- Support and resistance levels
- Chart pattern detection
- Fibonacci levels
- Market structure (higher highs/lower lows)

---

### H1. Pivot Points (Standard + Fibonacci)

| Property | Value |
|----------|-------|
| **Unique Info** | Calculated S/R levels from prior period |
| **Why Not Redundant** | Only calculated (not derived from price action) S/R |
| **Parameters** | `type: 'standard'/'fibonacci'/'woodie'` |
| **Output** | `{pivot: float, s1-s3: float, r1-r3: float}` |
| **Signal** | Key decision levels; watch for reactions |

```python
# Pivot calculations (Standard):
# P = (H + L + C) / 3
# R1 = 2P - L, S1 = 2P - H
# R2 = P + (H - L), S2 = P - (H - L)
# Self-fulfilling: Widely watched levels.
```

---

### H2. Fibonacci Retracement Levels

| Property | Value |
|----------|-------|
| **Unique Info** | Key retracement levels from swing high to low |
| **Why Not Redundant** | Based on swing detection + Fib ratios |
| **Parameters** | `swing_period: 20, levels: [0.236, 0.382, 0.5, 0.618, 0.786]` |
| **Output** | `{levels: Dict[float, float], swing_high: float, swing_low: float}` |
| **Signal** | Common reversal zones; watch for confluence |

```python
# Fibonacci levels calculated from significant swings
# Key levels: 38.2%, 50%, 61.8%
# Self-fulfilling: Heavily watched by traders.
```

---

### H3. Swing High/Low Detection

| Property | Value |
|----------|-------|
| **Unique Info** | Identifies significant turning points |
| **Why Not Redundant** | Foundation for structure analysis |
| **Parameters** | `lookback: 5, threshold: 0` |
| **Output** | `{swing_highs: List[tuple], swing_lows: List[tuple]}` |
| **Signal** | Higher highs/lows = uptrend; lower highs/lows = downtrend |

```python
# Swing High: High > N bars before AND N bars after
# Swing Low: Low < N bars before AND N bars after
# Foundation for pattern recognition and structure analysis.
```

---

### H4. Market Structure (Higher Highs/Lows)

| Property | Value |
|----------|-------|
| **Unique Info** | Classical trend structure analysis |
| **Why Not Redundant** | Discrete structure states (not continuous) |
| **Parameters** | `swing_lookback: 10` |
| **Output** | `{structure: 'UPTREND'/'DOWNTREND'/'CONSOLIDATION', last_hh: float, last_ll: float}` |
| **Signal** | Structure break = potential trend change |

```python
# Uptrend: Higher Highs + Higher Lows
# Downtrend: Lower Highs + Lower Lows
# Break of structure = early reversal warning
```

---

### H5. Price Channels (Donchian)

| Property | Value |
|----------|-------|
| **Unique Info** | Simple high-low channel; breakout levels |
| **Why Not Redundant** | Range-based (not moving average based) |
| **Parameters** | `period: 20` |
| **Output** | `{upper: float, lower: float, middle: float, width: float}` |
| **Signal** | Break above upper = bullish breakout; below lower = bearish |

```python
# Donchian Channels:
# Upper = Highest High over period
# Lower = Lowest Low over period
# Classic breakout indicator (Turtle Trading)
```

---

### H6. Support/Resistance Clusters

| Property | Value |
|----------|-------|
| **Unique Info** | Historical price concentration zones |
| **Why Not Redundant** | Volume-weighted historical S/R |
| **Parameters** | `lookback: 252, num_levels: 5` |
| **Output** | `{levels: List[{price: float, strength: float}]}` |
| **Signal** | Strong levels = likely reaction zones |

```python
# Identify levels where:
# 1. Price reversed multiple times
# 2. High volume traded
# Algorithm: Cluster high-volume price points
```

---

## Group I: Fundamental Indicators (Optional)

### Purpose
Incorporate **company financial data** into analysis. Fundamental indicators answer: "How does valuation and financial health look?"

### Unique Information Content
- Valuation metrics
- Quality metrics
- Growth metrics
- Value vs. momentum integration

---

### I1. Valuation Score

| Property | Value |
|----------|-------|
| **Unique Info** | Multi-factor valuation assessment |
| **Why Not Redundant** | Only valuation-based indicator |
| **Parameters** | `metrics: ['pe', 'pb', 'ps', 'ev_ebitda']` |
| **Output** | Score 0-100 (lower = cheaper) |
| **Signal** | Low score = potentially undervalued |

```python
# Composite of:
# - P/E percentile vs. history and peers
# - P/B percentile
# - P/S percentile
# - EV/EBITDA percentile
# Lower score = cheaper on multiple metrics
```

---

### I2. Quality Score

| Property | Value |
|----------|-------|
| **Unique Info** | Financial quality and stability |
| **Why Not Redundant** | Balance sheet / profitability focus |
| **Parameters** | `metrics: ['roe', 'debt_equity', 'current_ratio', 'margin']` |
| **Output** | Score 0-100 (higher = better quality) |
| **Signal** | High quality + low valuation = potential opportunity |

```python
# Composite of:
# - ROE percentile
# - Debt/Equity (lower better)
# - Current Ratio (higher better)
# - Profit Margin percentile
```

---

### I3. Earnings Surprise Momentum

| Property | Value |
|----------|-------|
| **Unique Info** | Recent earnings beat/miss pattern |
| **Why Not Redundant** | Event-driven fundamental signal |
| **Parameters** | `quarters: 4` |
| **Output** | `{avg_surprise: float, beat_rate: float, trend: float}` |
| **Signal** | Consistent beats = positive fundamental momentum |

```python
# Track last N quarters:
# - Average surprise %
# - Beat rate (% of quarters beating estimates)
# - Surprise trend (improving or deteriorating)
# Post-earnings drift is a documented anomaly.
```

---

### I4. Piotroski F-Score

| Property | Value |
|----------|-------|
| **Unique Info** | 9-point financial health score |
| **Why Not Redundant** | Academic-validated composite |
| **Parameters** | None (uses latest financials) |
| **Output** | Score 0-9 |
| **Signal** | 8-9 = high quality; 0-2 = potential distress |

```python
# 9 binary signals:
# Profitability (4): ROA, CFO, ΔROA, Accruals
# Leverage (3): ΔLeverage, ΔLiquidity, Equity Issuance
# Efficiency (2): ΔMargin, ΔTurnover
# Validated factor in academic research.
```

---

### I5. PEG Ratio (Forward)

| Property | Value |
|----------|-------|
| **Unique Info** | Valuation adjusted for growth |
| **Why Not Redundant** | Only growth-adjusted valuation metric |
| **Parameters** | None |
| **Output** | Ratio (lower = cheaper relative to growth) |
| **Signal** | PEG < 1 = potentially undervalued vs. growth |

```python
# PEG = Forward P/E / Expected EPS Growth
# Normalizes valuation for growth differences
# Low PEG + high quality = GARP opportunity
```

---

## Group J: Sentiment Indicators (Optional)

### Purpose
Capture **market psychology and positioning**. Sentiment indicators answer: "How are other participants positioned?"

### Unique Information Content
- Short interest and positioning
- Analyst sentiment
- Institutional activity
- Contrarian signals

---

### J1. Short Interest Ratio

| Property | Value |
|----------|-------|
| **Unique Info** | Days to cover short positions |
| **Why Not Redundant** | Only indicator of short positioning |
| **Parameters** | None (uses latest data) |
| **Output** | Days to cover |
| **Signal** | High short interest = potential squeeze; also potential distress |

```python
# Short Ratio = Short Interest / Average Daily Volume
# Interpretation depends on context:
# - High short + positive catalyst = squeeze potential
# - High short + negative trend = shorts may be right
```

---

### J2. Analyst Rating Composite

| Property | Value |
|----------|-------|
| **Unique Info** | Wall Street sentiment summary |
| **Why Not Redundant** | Expert opinion aggregation |
| **Parameters** | None |
| **Output** | `{rating: float, changes_30d: int, target_upside: float}` |
| **Signal** | Upgrades = positive; downgrades = negative; contrarian use for extremes |

```python
# Aggregate analyst data:
# - Average rating (1-5 scale)
# - Recent rating changes
# - Price target vs. current price
# Note: Analysts are often lagging indicators.
```

---

### J3. Institutional Ownership Change

| Property | Value |
|----------|-------|
| **Unique Info** | Smart money positioning changes |
| **Why Not Redundant** | Only institutional flow indicator |
| **Parameters** | `quarters: 2` |
| **Output** | `{ownership_pct: float, change: float, new_positions: int}` |
| **Signal** | Rising institutional ownership = accumulation |

```python
# 13F filing data (quarterly, 45-day lag)
# Track:
# - Total institutional ownership %
# - Quarter-over-quarter change
# - New positions vs. closed positions
```

---

### J4. Insider Transaction Score

| Property | Value |
|----------|-------|
| **Unique Info** | Company insider buying/selling |
| **Why Not Redundant** | Most informed participants |
| **Parameters** | `lookback_days: 90` |
| **Output** | `{net_shares: int, buy_sell_ratio: float, notable_buys: List}` |
| **Signal** | Cluster buying = bullish; CEO selling (large) = concerning |

```python
# Insider transactions:
# - Buys are generally more informative (discretionary)
# - Sells can be routine (diversification, liquidity)
# Cluster buying by multiple insiders is strong signal.
```

---

### J5. Options Put/Call Ratio

| Property | Value |
|----------|-------|
| **Unique Info** | Options market sentiment |
| **Why Not Redundant** | Derivatives market positioning |
| **Parameters** | `type: 'volume' or 'open_interest'` |
| **Output** | Ratio (>1 = more puts; <1 = more calls) |
| **Signal** | Extreme high P/C = fear (contrarian bullish); extreme low = complacency |

```python
# Put/Call Ratio:
# - Stock-specific: Individual sentiment
# - Index (VIX): Market-wide sentiment
# Extreme readings often precede reversals (contrarian).
```

---

## Summary: Non-Redundancy Matrix

| Group | Indicators | Unique Information |
|-------|------------|-------------------|
| **A: Trend** | EMA Suite, MACD, ADX, Supertrend, Aroon | Direction, strength, freshness, volatility-adjusted levels |
| **B: Momentum** | RSI, Stochastic, ROC, Williams %R, CCI, TSI | Speed, OB/OS, raw returns, statistical deviation |
| **C: Volatility** | ATR, ATRP, Bollinger, Keltner, HV, VR, Ulcer | Absolute/relative vol, bands, regimes, downside risk |
| **D: Volume** | OBV, VWAP, MFI, ADL, CMF, VROC, Vol Ratio | Cumulative flow, institutional levels, participation |
| **E: Microstructure** | Candles, TR, Body Ratio, Gaps, Range Position | Intrabar dynamics, overnight gaps, conviction |
| **F: Regime** | ADX Regime, Vol Regime, Choppiness, Composite | Market state classification |
| **G: Relative Strength** | RSB, RS Ratio, Sector RS, Percentile, Mansfield | Benchmark comparison, peer ranking |
| **H: Structure** | Pivots, Fibonacci, Swings, Structure, Channels, S/R Clusters | Price levels, patterns, breakout zones |
| **I: Fundamentals** | Valuation, Quality, Earnings, F-Score, PEG | Financial assessment |
| **J: Sentiment** | Short Interest, Analyst, Institutional, Insider, P/C Ratio | Positioning and psychology |

---

## Parameter Summary

| Indicator | Key Parameters | Default | Valid Range |
|-----------|---------------|---------|-------------|
| EMA | periods | [8, 21, 50, 200] | 2-500 |
| MACD | fast, slow, signal | 12, 26, 9 | 2-50 each |
| ADX | period | 14 | 7-30 |
| RSI | period | 14 | 2-50 |
| Stochastic | k, d, slowing | 14, 3, 3 | 2-50 each |
| ROC | periods | [5, 10, 21, 63] | 1-252 |
| ATR | period | 14 | 5-50 |
| Bollinger | period, std_dev | 20, 2.0 | 5-50, 1.0-3.0 |
| HV | periods | [10, 21, 63] | 5-252 |
| OBV | none | - | - |
| VWAP | anchor | 'session' | session/rolling |
| MFI | period | 14 | 5-50 |
| Choppiness | period | 14 | 7-50 |
| Pivots | type | 'standard' | standard/fib/woodie |
| Donchian | period | 20 | 5-100 |

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
