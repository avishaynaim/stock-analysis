"""
Candlestick Pattern Recognition indicators (Group E - Microstructure).

Indicators:
- Single Candle Patterns (Doji, Hammer, Shooting Star, Marubozu)
- Two Candle Patterns (Engulfing, Harami, Piercing, Dark Cloud)
- Three Candle Patterns (Morning Star, Evening Star, Three Soldiers/Crows)
- Bar Analysis (Body Ratio, Range Position)
"""

import numpy as np
import pandas as pd

from stock_analysis.indicators.registry import indicator


def _body_size(open_price: float, close: float) -> float:
    """Calculate candle body size."""
    return abs(close - open_price)


def _range_size(high: float, low: float) -> float:
    """Calculate candle range size."""
    return high - low


def _is_bullish(open_price: float, close: float) -> bool:
    """Check if candle is bullish."""
    return close > open_price


def _upper_shadow(open_price: float, high: float, close: float) -> float:
    """Calculate upper shadow size."""
    return high - max(open_price, close)


def _lower_shadow(open_price: float, low: float, close: float) -> float:
    """Calculate lower shadow size."""
    return min(open_price, close) - low


@indicator(
    name="candle_metrics",
    group="patterns",
    description="Basic Candle Metrics",
    parameters={},
    required_periods=5,
)
def compute_candle_metrics(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Compute basic candlestick metrics."""
    open_price = prices["open"]
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < 1:
        return {}

    o, h, l, c = open_price.iloc[-1], high.iloc[-1], low.iloc[-1], close.iloc[-1]

    body = _body_size(o, c)
    range_size = _range_size(h, l)
    upper_shadow = _upper_shadow(o, h, c)
    lower_shadow = _lower_shadow(o, l, c)

    body_ratio = body / range_size if range_size > 0 else 0
    upper_shadow_ratio = upper_shadow / range_size if range_size > 0 else 0
    lower_shadow_ratio = lower_shadow / range_size if range_size > 0 else 0

    is_bullish = 1 if c > o else 0

    return {
        "body_size": body,
        "range_size": range_size,
        "body_ratio": body_ratio,
        "upper_shadow_ratio": upper_shadow_ratio,
        "lower_shadow_ratio": lower_shadow_ratio,
        "is_bullish_candle": is_bullish,
        "candle_strength": body_ratio * (1 if is_bullish else -1),
    }


@indicator(
    name="single_candle_patterns",
    group="patterns",
    description="Single Candle Patterns",
    parameters={"doji_threshold": 0.1, "shadow_ratio": 2.0},
    required_periods=5,
)
def compute_single_candle_patterns(
    prices: pd.DataFrame,
    doji_threshold: float = 0.1,
    shadow_ratio: float = 2.0,
) -> dict[str, float]:
    """Detect single candlestick patterns."""
    open_price = prices["open"]
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < 1:
        return {}

    o, h, l, c = open_price.iloc[-1], high.iloc[-1], low.iloc[-1], close.iloc[-1]

    body = _body_size(o, c)
    range_size = _range_size(h, l)
    upper_shadow = _upper_shadow(o, h, c)
    lower_shadow = _lower_shadow(o, l, c)

    body_ratio = body / range_size if range_size > 0 else 0

    # Pattern detection
    patterns = {
        "is_doji": 0,
        "is_hammer": 0,
        "is_inverted_hammer": 0,
        "is_shooting_star": 0,
        "is_hanging_man": 0,
        "is_marubozu": 0,
        "is_spinning_top": 0,
        "pattern_signal": 0,  # -1 bearish, 0 neutral, 1 bullish
    }

    # Doji: Very small body
    if body_ratio < doji_threshold:
        patterns["is_doji"] = 1
        patterns["pattern_signal"] = 0  # Neutral, indecision

    # Hammer: Small body at top, long lower shadow (bullish reversal)
    if (
        body_ratio < 0.3
        and lower_shadow > body * shadow_ratio
        and upper_shadow < body * 0.5
    ):
        patterns["is_hammer"] = 1
        patterns["pattern_signal"] = 1  # Bullish

    # Inverted Hammer: Small body at bottom, long upper shadow (bullish reversal)
    if (
        body_ratio < 0.3
        and upper_shadow > body * shadow_ratio
        and lower_shadow < body * 0.5
    ):
        patterns["is_inverted_hammer"] = 1
        patterns["pattern_signal"] = 1  # Bullish (needs confirmation)

    # Shooting Star: Like inverted hammer but at top of uptrend (bearish)
    if (
        body_ratio < 0.3
        and upper_shadow > body * shadow_ratio
        and lower_shadow < body * 0.5
        and c < o  # Bearish close helps
    ):
        patterns["is_shooting_star"] = 1
        patterns["pattern_signal"] = -1  # Bearish

    # Hanging Man: Like hammer but at top of uptrend (bearish)
    if (
        body_ratio < 0.3
        and lower_shadow > body * shadow_ratio
        and upper_shadow < body * 0.5
        and c < o
    ):
        patterns["is_hanging_man"] = 1
        patterns["pattern_signal"] = -1  # Bearish (needs confirmation)

    # Marubozu: Almost no shadows (strong conviction)
    if body_ratio > 0.9:
        patterns["is_marubozu"] = 1
        patterns["pattern_signal"] = 1 if c > o else -1

    # Spinning Top: Small body with shadows on both sides
    if (
        body_ratio < 0.3
        and upper_shadow > body * 0.5
        and lower_shadow > body * 0.5
    ):
        patterns["is_spinning_top"] = 1
        patterns["pattern_signal"] = 0  # Neutral

    return patterns


@indicator(
    name="two_candle_patterns",
    group="patterns",
    description="Two Candle Patterns",
    parameters={},
    required_periods=5,
)
def compute_two_candle_patterns(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Detect two-candlestick patterns."""
    open_price = prices["open"]
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < 2:
        return {}

    # Previous candle
    o1, h1, l1, c1 = open_price.iloc[-2], high.iloc[-2], low.iloc[-2], close.iloc[-2]
    # Current candle
    o2, h2, l2, c2 = open_price.iloc[-1], high.iloc[-1], low.iloc[-1], close.iloc[-1]

    body1 = _body_size(o1, c1)
    body2 = _body_size(o2, c2)

    prev_bullish = c1 > o1
    curr_bullish = c2 > o2

    patterns = {
        "is_bullish_engulfing": 0,
        "is_bearish_engulfing": 0,
        "is_bullish_harami": 0,
        "is_bearish_harami": 0,
        "is_piercing_line": 0,
        "is_dark_cloud_cover": 0,
        "is_tweezer_top": 0,
        "is_tweezer_bottom": 0,
        "two_candle_signal": 0,
    }

    # Bullish Engulfing: Bearish candle followed by larger bullish candle
    if (
        not prev_bullish
        and curr_bullish
        and o2 < c1
        and c2 > o1
        and body2 > body1
    ):
        patterns["is_bullish_engulfing"] = 1
        patterns["two_candle_signal"] = 1

    # Bearish Engulfing: Bullish candle followed by larger bearish candle
    if (
        prev_bullish
        and not curr_bullish
        and o2 > c1
        and c2 < o1
        and body2 > body1
    ):
        patterns["is_bearish_engulfing"] = 1
        patterns["two_candle_signal"] = -1

    # Bullish Harami: Large bearish candle followed by small bullish candle inside
    if (
        not prev_bullish
        and curr_bullish
        and o2 > c1
        and c2 < o1
        and body2 < body1 * 0.5
    ):
        patterns["is_bullish_harami"] = 1
        patterns["two_candle_signal"] = 1

    # Bearish Harami: Large bullish candle followed by small bearish candle inside
    if (
        prev_bullish
        and not curr_bullish
        and o2 < c1
        and c2 > o1
        and body2 < body1 * 0.5
    ):
        patterns["is_bearish_harami"] = 1
        patterns["two_candle_signal"] = -1

    # Piercing Line: Bearish candle, then bullish opens below and closes above midpoint
    midpoint1 = (o1 + c1) / 2
    if (
        not prev_bullish
        and curr_bullish
        and o2 < l1
        and c2 > midpoint1
        and c2 < o1
    ):
        patterns["is_piercing_line"] = 1
        patterns["two_candle_signal"] = 1

    # Dark Cloud Cover: Bullish candle, then bearish opens above and closes below midpoint
    if (
        prev_bullish
        and not curr_bullish
        and o2 > h1
        and c2 < midpoint1
        and c2 > o1
    ):
        patterns["is_dark_cloud_cover"] = 1
        patterns["two_candle_signal"] = -1

    # Tweezer Top: Two candles with similar highs at resistance
    if abs(h1 - h2) / h1 < 0.001:
        patterns["is_tweezer_top"] = 1
        if not curr_bullish:
            patterns["two_candle_signal"] = -1

    # Tweezer Bottom: Two candles with similar lows at support
    if abs(l1 - l2) / l1 < 0.001:
        patterns["is_tweezer_bottom"] = 1
        if curr_bullish:
            patterns["two_candle_signal"] = 1

    return patterns


@indicator(
    name="three_candle_patterns",
    group="patterns",
    description="Three Candle Patterns",
    parameters={},
    required_periods=5,
)
def compute_three_candle_patterns(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Detect three-candlestick patterns."""
    open_price = prices["open"]
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < 3:
        return {}

    # Three candles
    o1, h1, l1, c1 = open_price.iloc[-3], high.iloc[-3], low.iloc[-3], close.iloc[-3]
    o2, h2, l2, c2 = open_price.iloc[-2], high.iloc[-2], low.iloc[-2], close.iloc[-2]
    o3, h3, l3, c3 = open_price.iloc[-1], high.iloc[-1], low.iloc[-1], close.iloc[-1]

    body1 = _body_size(o1, c1)
    body2 = _body_size(o2, c2)
    body3 = _body_size(o3, c3)

    range1 = _range_size(h1, l1)

    bull1 = c1 > o1
    bull2 = c2 > o2
    bull3 = c3 > o3

    patterns = {
        "is_morning_star": 0,
        "is_evening_star": 0,
        "is_three_white_soldiers": 0,
        "is_three_black_crows": 0,
        "is_three_inside_up": 0,
        "is_three_inside_down": 0,
        "three_candle_signal": 0,
    }

    # Morning Star: Bearish, small body, bullish (reversal)
    small_body2 = body2 < body1 * 0.3 if body1 > 0 else body2 < range1 * 0.2
    if (
        not bull1
        and bull3
        and small_body2
        and c3 > (o1 + c1) / 2
    ):
        patterns["is_morning_star"] = 1
        patterns["three_candle_signal"] = 1

    # Evening Star: Bullish, small body, bearish (reversal)
    if (
        bull1
        and not bull3
        and small_body2
        and c3 < (o1 + c1) / 2
    ):
        patterns["is_evening_star"] = 1
        patterns["three_candle_signal"] = -1

    # Three White Soldiers: Three consecutive bullish candles
    if (
        bull1
        and bull2
        and bull3
        and c2 > c1
        and c3 > c2
        and body1 > range1 * 0.5
        and body2 > body1 * 0.5
        and body3 > body2 * 0.5
    ):
        patterns["is_three_white_soldiers"] = 1
        patterns["three_candle_signal"] = 1

    # Three Black Crows: Three consecutive bearish candles
    if (
        not bull1
        and not bull2
        and not bull3
        and c2 < c1
        and c3 < c2
        and body1 > range1 * 0.5
        and body2 > body1 * 0.5
        and body3 > body2 * 0.5
    ):
        patterns["is_three_black_crows"] = 1
        patterns["three_candle_signal"] = -1

    # Three Inside Up: Bearish, bullish harami, bullish confirmation
    if (
        not bull1
        and bull2
        and bull3
        and o2 > c1
        and c2 < o1
        and body2 < body1 * 0.5
        and c3 > o1
    ):
        patterns["is_three_inside_up"] = 1
        patterns["three_candle_signal"] = 1

    # Three Inside Down: Bullish, bearish harami, bearish confirmation
    if (
        bull1
        and not bull2
        and not bull3
        and o2 < c1
        and c2 > o1
        and body2 < body1 * 0.5
        and c3 < o1
    ):
        patterns["is_three_inside_down"] = 1
        patterns["three_candle_signal"] = -1

    return patterns


@indicator(
    name="pattern_composite",
    group="patterns",
    description="Composite Pattern Signal",
    parameters={},
    required_periods=10,
)
def compute_pattern_composite(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Compute composite pattern signal from all patterns."""
    open_price = prices["open"]
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < 3:
        return {}

    # Compute individual pattern signals
    single = compute_single_candle_patterns(prices)
    two = compute_two_candle_patterns(prices)
    three = compute_three_candle_patterns(prices)

    # Aggregate signals (weighted by reliability)
    signals = []

    # Three candle patterns are most reliable (weight 3)
    if three.get("three_candle_signal", 0) != 0:
        signals.extend([three["three_candle_signal"]] * 3)

    # Two candle patterns (weight 2)
    if two.get("two_candle_signal", 0) != 0:
        signals.extend([two["two_candle_signal"]] * 2)

    # Single candle patterns (weight 1)
    if single.get("pattern_signal", 0) != 0:
        signals.append(single["pattern_signal"])

    # Calculate composite
    if signals:
        composite_signal = sum(signals) / len(signals)
        pattern_count = len(signals)
    else:
        composite_signal = 0
        pattern_count = 0

    # Determine overall bias
    if composite_signal > 0.3:
        pattern_bias = "bullish"
    elif composite_signal < -0.3:
        pattern_bias = "bearish"
    else:
        pattern_bias = "neutral"

    return {
        "pattern_composite_signal": composite_signal,
        "pattern_count": pattern_count,
        "pattern_bias": pattern_bias,
        "has_bullish_pattern": 1 if composite_signal > 0.3 else 0,
        "has_bearish_pattern": 1 if composite_signal < -0.3 else 0,
        "single_pattern_signal": single.get("pattern_signal", 0),
        "two_pattern_signal": two.get("two_candle_signal", 0),
        "three_pattern_signal": three.get("three_candle_signal", 0),
    }
