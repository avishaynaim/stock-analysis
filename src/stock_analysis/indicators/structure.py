"""
Structure and Pattern indicators (Group H).

Indicators:
- Support/Resistance Levels
- Pivot Points
- Fibonacci Retracements
- Higher Highs/Lower Lows
- Trend Line Detection
- Price Patterns
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from stock_analysis.indicators.registry import indicator


@indicator(
    name="pivot_points",
    group="structure",
    description="Pivot Points (Standard)",
    parameters={},
    required_periods=5,
)
def compute_pivot_points(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Compute Standard Pivot Points."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < 2:
        return {}

    # Use previous day's data
    prev_high = high.iloc[-2]
    prev_low = low.iloc[-2]
    prev_close = close.iloc[-2]

    # Pivot Point
    pp = (prev_high + prev_low + prev_close) / 3

    # Support and Resistance levels
    r1 = 2 * pp - prev_low
    r2 = pp + (prev_high - prev_low)
    r3 = prev_high + 2 * (pp - prev_low)

    s1 = 2 * pp - prev_high
    s2 = pp - (prev_high - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)

    current_price = close.iloc[-1]

    # Determine position relative to pivot
    if current_price > r1:
        position = "above_r1"
    elif current_price > pp:
        position = "above_pivot"
    elif current_price > s1:
        position = "below_pivot"
    else:
        position = "below_s1"

    return {
        "pivot_point": pp,
        "resistance_1": r1,
        "resistance_2": r2,
        "resistance_3": r3,
        "support_1": s1,
        "support_2": s2,
        "support_3": s3,
        "pivot_position": position,
        "distance_to_pivot": (current_price - pp) / pp,
    }


@indicator(
    name="fibonacci_levels",
    group="structure",
    description="Fibonacci Retracement Levels",
    parameters={"lookback": 63},
    required_periods=63,
)
def compute_fibonacci_levels(
    prices: pd.DataFrame,
    lookback: int = 63,
) -> dict[str, float]:
    """Compute Fibonacci retracement levels."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < lookback:
        return {}

    # Find swing high and low in lookback period
    period_high = high.iloc[-lookback:].max()
    period_low = low.iloc[-lookback:].min()

    price_range = period_high - period_low

    # Fibonacci levels
    fib_levels = {
        "0.0": period_low,
        "0.236": period_low + 0.236 * price_range,
        "0.382": period_low + 0.382 * price_range,
        "0.5": period_low + 0.5 * price_range,
        "0.618": period_low + 0.618 * price_range,
        "0.786": period_low + 0.786 * price_range,
        "1.0": period_high,
    }

    current_price = close.iloc[-1]

    # Find nearest levels
    levels = sorted(fib_levels.values())
    nearest_support = max([l for l in levels if l < current_price], default=levels[0])
    nearest_resistance = min([l for l in levels if l > current_price], default=levels[-1])

    # Retracement ratio
    retracement = (current_price - period_low) / price_range if price_range > 0 else 0.5

    return {
        "fib_swing_high": period_high,
        "fib_swing_low": period_low,
        "fib_236": fib_levels["0.236"],
        "fib_382": fib_levels["0.382"],
        "fib_500": fib_levels["0.5"],
        "fib_618": fib_levels["0.618"],
        "fib_786": fib_levels["0.786"],
        "fib_nearest_support": nearest_support,
        "fib_nearest_resistance": nearest_resistance,
        "fib_retracement": retracement,
    }


@indicator(
    name="swing_points",
    group="structure",
    description="Swing Highs and Lows",
    parameters={"order": 5},
    required_periods=30,
)
def compute_swing_points(
    prices: pd.DataFrame,
    order: int = 5,
) -> dict[str, float]:
    """Identify swing highs and lows."""
    high = prices["high"].values
    low = prices["low"].values
    close = prices["adj_close"]

    if len(close) < order * 3:
        return {}

    # Find local maxima (swing highs)
    swing_high_idx = argrelextrema(high, np.greater, order=order)[0]

    # Find local minima (swing lows)
    swing_low_idx = argrelextrema(low, np.less, order=order)[0]

    result = {}

    # Recent swing high
    if len(swing_high_idx) > 0:
        last_swing_high_idx = swing_high_idx[-1]
        result["last_swing_high"] = high[last_swing_high_idx]
        result["bars_since_swing_high"] = len(high) - 1 - last_swing_high_idx

        if len(swing_high_idx) > 1:
            prev_swing_high = high[swing_high_idx[-2]]
            result["higher_high"] = 1 if high[last_swing_high_idx] > prev_swing_high else 0

    # Recent swing low
    if len(swing_low_idx) > 0:
        last_swing_low_idx = swing_low_idx[-1]
        result["last_swing_low"] = low[last_swing_low_idx]
        result["bars_since_swing_low"] = len(low) - 1 - last_swing_low_idx

        if len(swing_low_idx) > 1:
            prev_swing_low = low[swing_low_idx[-2]]
            result["higher_low"] = 1 if low[last_swing_low_idx] > prev_swing_low else 0

    # Trend structure
    if "higher_high" in result and "higher_low" in result:
        if result.get("higher_high") == 1 and result.get("higher_low") == 1:
            result["trend_structure"] = "uptrend"
        elif result.get("higher_high") == 0 and result.get("higher_low") == 0:
            result["trend_structure"] = "downtrend"
        else:
            result["trend_structure"] = "mixed"

    return result


@indicator(
    name="price_channels",
    group="structure",
    description="Price Channel Analysis",
    parameters={"period": 20},
    required_periods=30,
)
def compute_price_channels(
    prices: pd.DataFrame,
    period: int = 20,
) -> dict[str, float]:
    """Compute price channel characteristics."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < period:
        return {}

    # Calculate channel bounds using linear regression
    x = np.arange(period)

    high_recent = high.iloc[-period:].values
    low_recent = low.iloc[-period:].values

    # Fit lines to highs and lows
    high_slope, high_intercept = np.polyfit(x, high_recent, 1)
    low_slope, low_intercept = np.polyfit(x, low_recent, 1)

    # Channel width
    channel_width = (high_recent.mean() - low_recent.mean()) / close.iloc[-1]

    # Channel direction
    avg_slope = (high_slope + low_slope) / 2
    channel_direction = "up" if avg_slope > 0 else "down" if avg_slope < 0 else "flat"

    # Price position in channel
    current_price = close.iloc[-1]
    upper_bound = high_intercept + high_slope * (period - 1)
    lower_bound = low_intercept + low_slope * (period - 1)

    channel_position = (current_price - lower_bound) / (upper_bound - lower_bound)

    return {
        "channel_upper": upper_bound,
        "channel_lower": lower_bound,
        "channel_width": channel_width,
        "channel_slope": avg_slope / close.iloc[-1],  # Normalized
        "channel_direction": channel_direction,
        "channel_position": channel_position,
        "near_channel_top": 1 if channel_position > 0.8 else 0,
        "near_channel_bottom": 1 if channel_position < 0.2 else 0,
    }


@indicator(
    name="range_analysis",
    group="structure",
    description="Trading Range Analysis",
    parameters={"period": 20},
    required_periods=30,
)
def compute_range_analysis(
    prices: pd.DataFrame,
    period: int = 20,
) -> dict[str, float]:
    """Analyze trading range characteristics."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < period:
        return {}

    # Average True Range as percentage
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    atr_pct = atr / close

    # Range metrics
    period_high = high.iloc[-period:].max()
    period_low = low.iloc[-period:].min()
    period_range = period_high - period_low
    range_pct = period_range / close.iloc[-1]

    # Inside/Outside bars
    is_inside_bar = (high.iloc[-1] < high.iloc[-2]) and (low.iloc[-1] > low.iloc[-2])
    is_outside_bar = (high.iloc[-1] > high.iloc[-2]) and (low.iloc[-1] < low.iloc[-2])

    # Range expansion/contraction
    recent_range = (high.iloc[-5:] - low.iloc[-5:]).mean()
    prior_range = (high.iloc[-10:-5] - low.iloc[-10:-5]).mean()
    range_change = recent_range / prior_range - 1 if prior_range > 0 else 0

    return {
        "period_high": period_high,
        "period_low": period_low,
        "period_range": period_range,
        "range_pct": range_pct,
        "atr_pct": atr_pct.iloc[-1],
        "is_inside_bar": 1 if is_inside_bar else 0,
        "is_outside_bar": 1 if is_outside_bar else 0,
        "range_expanding": 1 if range_change > 0.1 else 0,
        "range_contracting": 1 if range_change < -0.1 else 0,
    }


@indicator(
    name="gap_analysis",
    group="structure",
    description="Gap Analysis",
    parameters={},
    required_periods=30,
)
def compute_gap_analysis(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Analyze price gaps."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]
    open_price = prices["open"]

    if len(close) < 20:
        return {}

    # Today's gap
    gap = open_price.iloc[-1] - close.iloc[-2]
    gap_pct = gap / close.iloc[-2]

    # Gap type
    if gap_pct > 0.02:
        gap_type = "gap_up_large"
    elif gap_pct > 0.005:
        gap_type = "gap_up_small"
    elif gap_pct < -0.02:
        gap_type = "gap_down_large"
    elif gap_pct < -0.005:
        gap_type = "gap_down_small"
    else:
        gap_type = "no_gap"

    # Gap fill check
    if gap > 0:
        gap_filled = 1 if low.iloc[-1] <= close.iloc[-2] else 0
    elif gap < 0:
        gap_filled = 1 if high.iloc[-1] >= close.iloc[-2] else 0
    else:
        gap_filled = 0

    # Historical gap metrics
    gaps = open_price - close.shift(1)
    gap_pcts = gaps / close.shift(1)

    avg_gap = gap_pcts.iloc[-20:].abs().mean()
    max_gap = gap_pcts.iloc[-20:].abs().max()

    return {
        "gap_today": gap,
        "gap_pct": gap_pct,
        "gap_type": gap_type,
        "gap_filled": gap_filled,
        "avg_gap_20d": avg_gap,
        "max_gap_20d": max_gap,
    }
