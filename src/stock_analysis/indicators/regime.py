"""
Regime Detection indicators (Group F).

Indicators:
- ADX Regime Classification
- Volatility Regime
- Trend-Range Classification (Choppiness Index)
- Composite Regime Indicator
"""

import numpy as np
import pandas as pd

from stock_analysis.indicators.registry import indicator


@indicator(
    name="adx_regime",
    group="regime",
    description="ADX-based Trend Regime Classification",
    parameters={"period": 14},
    required_periods=30,
)
def compute_adx_regime(
    prices: pd.DataFrame,
    period: int = 14,
) -> dict[str, float]:
    """Classify market regime based on ADX trend strength."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < period * 2:
        return {}

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Directional Movement
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1  # Make positive for comparison

    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
    minus_dm = low.shift(1) - low
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Smooth with Wilder's method
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    current_adx = adx.iloc[-1]

    # Regime classification
    if current_adx < 15:
        regime = "absent"
        regime_score = 0
    elif current_adx < 25:
        regime = "weak"
        regime_score = 1
    elif current_adx < 40:
        regime = "moderate"
        regime_score = 2
    else:
        regime = "strong"
        regime_score = 3

    # Trend direction from DI
    trend_direction = 1 if plus_di.iloc[-1] > minus_di.iloc[-1] else -1

    return {
        "adx": current_adx,
        "plus_di": plus_di.iloc[-1],
        "minus_di": minus_di.iloc[-1],
        "trend_regime": regime,
        "regime_score": regime_score,
        "trend_direction": trend_direction,
        "adx_rising": 1 if adx.iloc[-1] > adx.iloc[-5] else 0,
    }


@indicator(
    name="volatility_regime",
    group="regime",
    description="Volatility Regime Classification",
    parameters={"lookback": 252, "hv_period": 21},
    required_periods=252,
)
def compute_volatility_regime(
    prices: pd.DataFrame,
    lookback: int = 252,
    hv_period: int = 21,
) -> dict[str, float]:
    """Classify volatility regime based on historical percentile."""
    close = prices["adj_close"]

    if len(close) < lookback:
        return {}

    # Calculate historical volatility
    returns = close.pct_change()
    hv = returns.rolling(window=hv_period).std() * np.sqrt(252)

    current_hv = hv.iloc[-1]

    # Calculate percentile rank
    hv_history = hv.iloc[-lookback:]
    percentile = (hv_history < current_hv).sum() / len(hv_history) * 100

    # Regime classification
    if percentile < 25:
        regime = "low"
        regime_score = 0
    elif percentile < 75:
        regime = "normal"
        regime_score = 1
    else:
        regime = "high"
        regime_score = 2

    # Volatility trend
    hv_5d_ago = hv.iloc[-6] if len(hv) > 5 else hv.iloc[0]
    vol_trend = 1 if current_hv > hv_5d_ago else -1

    return {
        "historical_volatility": current_hv,
        "vol_percentile": percentile,
        "vol_regime": regime,
        "vol_regime_score": regime_score,
        "vol_trend": vol_trend,
        "vol_expansion": 1 if percentile > 75 else 0,
        "vol_compression": 1 if percentile < 25 else 0,
    }


@indicator(
    name="choppiness_index",
    group="regime",
    description="Choppiness Index (Trend vs Range)",
    parameters={"period": 14},
    required_periods=30,
)
def compute_choppiness_index(
    prices: pd.DataFrame,
    period: int = 14,
) -> dict[str, float]:
    """Compute Choppiness Index to detect trending vs ranging markets."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < period + 1:
        return {}

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Sum of ATR
    atr_sum = tr.rolling(window=period).sum()

    # Highest high - Lowest low
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    hl_range = highest - lowest

    # Choppiness Index
    chop = 100 * np.log10(atr_sum / hl_range) / np.log10(period)

    current_chop = chop.iloc[-1]

    # Classification using Fibonacci levels
    if current_chop > 61.8:
        market_type = "choppy"
        is_trending = 0
    elif current_chop < 38.2:
        market_type = "trending"
        is_trending = 1
    else:
        market_type = "transitional"
        is_trending = 0.5

    return {
        "choppiness_index": current_chop,
        "market_type": market_type,
        "is_trending": is_trending,
        "chop_rising": 1 if chop.iloc[-1] > chop.iloc[-5] else 0,
    }


@indicator(
    name="composite_regime",
    group="regime",
    description="Multi-factor Composite Regime",
    parameters={},
    required_periods=252,
)
def compute_composite_regime(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Compute composite regime from trend, volatility, and volume."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]
    volume = prices["volume"]

    if len(close) < 252:
        return {}

    # Trend component (price vs 50-day EMA)
    ema_50 = close.ewm(span=50, adjust=False).mean()
    trend_bullish = close.iloc[-1] > ema_50.iloc[-1]

    # Calculate ADX for trend strength
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = low.shift(1) - low

    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean().abs() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean().abs() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    is_trending = adx.iloc[-1] > 25

    # Volatility component
    returns = close.pct_change()
    hv = returns.rolling(window=21).std() * np.sqrt(252)
    hv_percentile = (hv.iloc[-252:] < hv.iloc[-1]).sum() / 252 * 100
    is_volatile = hv_percentile > 75

    # Volume component
    vol_sma = volume.rolling(window=20).mean()
    high_volume = volume.iloc[-1] > vol_sma.iloc[-1] * 1.5

    # Composite regime classification
    if trend_bullish and is_trending and not is_volatile:
        regime = "bull_quiet"
        regime_score = 4
    elif trend_bullish and is_trending and is_volatile:
        regime = "bull_volatile"
        regime_score = 3
    elif not trend_bullish and is_trending and not is_volatile:
        regime = "bear_quiet"
        regime_score = 1
    elif not trend_bullish and is_trending and is_volatile:
        regime = "bear_volatile"
        regime_score = 0
    else:
        regime = "range"
        regime_score = 2

    # Confidence based on signal agreement
    signals = [
        1 if trend_bullish else 0,
        1 if is_trending else 0,
        1 if not is_volatile else 0,
    ]
    confidence = abs(sum(signals) - 1.5) / 1.5  # Higher when signals agree

    return {
        "composite_regime": regime,
        "regime_score": regime_score,
        "trend_bullish": 1 if trend_bullish else 0,
        "is_trending": 1 if is_trending else 0,
        "is_volatile": 1 if is_volatile else 0,
        "high_volume": 1 if high_volume else 0,
        "regime_confidence": confidence,
    }


@indicator(
    name="market_breadth",
    group="regime",
    description="Simple Market Breadth Proxy",
    parameters={"period": 20},
    required_periods=30,
)
def compute_market_breadth(
    prices: pd.DataFrame,
    period: int = 20,
) -> dict[str, float]:
    """Compute simple breadth metrics from single stock (proxy)."""
    close = prices["adj_close"]
    high = prices["high"]
    low = prices["low"]

    if len(close) < period:
        return {}

    # Percentage of days up in period
    returns = close.pct_change()
    up_days = (returns > 0).rolling(window=period).sum()
    up_ratio = up_days / period

    # New highs/lows in period
    period_high = high.rolling(window=period).max()
    period_low = low.rolling(window=period).min()

    at_period_high = 1 if high.iloc[-1] == period_high.iloc[-1] else 0
    at_period_low = 1 if low.iloc[-1] == period_low.iloc[-1] else 0

    # Advance-decline proxy (positive returns ratio)
    adv_dec_ratio = up_ratio.iloc[-1]

    return {
        "up_day_ratio": adv_dec_ratio,
        "at_period_high": at_period_high,
        "at_period_low": at_period_low,
        "breadth_bullish": 1 if adv_dec_ratio > 0.6 else 0,
        "breadth_bearish": 1 if adv_dec_ratio < 0.4 else 0,
    }
