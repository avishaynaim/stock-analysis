"""
Trend indicators (Group A).

Indicators:
- SMA (multiple periods)
- EMA (multiple periods)
- MACD
- ADX
- Parabolic SAR
"""

import numpy as np
import pandas as pd

from stock_analysis.indicators.registry import indicator


@indicator(
    name="sma",
    group="trend",
    description="Simple Moving Averages",
    parameters={"periods": [20, 50, 200]},
    required_periods=200,
)
def compute_sma(
    prices: pd.DataFrame,
    periods: list[int] = [20, 50, 200],
) -> dict[str, float]:
    """Compute Simple Moving Averages."""
    close = prices["adj_close"]
    result = {}

    for period in periods:
        if len(close) >= period:
            sma = close.rolling(window=period).mean()
            result[f"sma_{period}"] = sma.iloc[-1]
            # Price vs SMA
            result[f"price_vs_sma_{period}"] = close.iloc[-1] / sma.iloc[-1] - 1

    # SMA alignment (trend confirmation)
    if all(f"sma_{p}" in result for p in [20, 50, 200]):
        # 1 if SMA20 > SMA50 > SMA200, -1 if reverse, 0 otherwise
        sma20, sma50, sma200 = result["sma_20"], result["sma_50"], result["sma_200"]
        if sma20 > sma50 > sma200:
            result["sma_alignment"] = 1.0
        elif sma20 < sma50 < sma200:
            result["sma_alignment"] = -1.0
        else:
            result["sma_alignment"] = 0.0

    return result


@indicator(
    name="ema",
    group="trend",
    description="Exponential Moving Averages",
    parameters={"periods": [12, 26]},
    required_periods=50,
)
def compute_ema(
    prices: pd.DataFrame,
    periods: list[int] = [12, 26],
) -> dict[str, float]:
    """Compute Exponential Moving Averages."""
    close = prices["adj_close"]
    result = {}

    for period in periods:
        if len(close) >= period:
            ema = close.ewm(span=period, adjust=False).mean()
            result[f"ema_{period}"] = ema.iloc[-1]
            result[f"price_vs_ema_{period}"] = close.iloc[-1] / ema.iloc[-1] - 1

    return result


@indicator(
    name="macd",
    group="trend",
    description="Moving Average Convergence Divergence",
    parameters={"fast": 12, "slow": 26, "signal": 9},
    required_periods=50,
)
def compute_macd(
    prices: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float]:
    """Compute MACD indicator."""
    close = prices["adj_close"]

    if len(close) < slow + signal:
        return {}

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        "macd": macd_line.iloc[-1],
        "macd_signal": signal_line.iloc[-1],
        "macd_histogram": histogram.iloc[-1],
        "macd_histogram_slope": histogram.diff().iloc[-1],
        # Normalized MACD (as % of price)
        "macd_normalized": macd_line.iloc[-1] / close.iloc[-1] * 100,
    }


@indicator(
    name="adx",
    group="trend",
    description="Average Directional Index",
    parameters={"period": 14},
    required_periods=50,
)
def compute_adx(
    prices: pd.DataFrame,
    period: int = 14,
) -> dict[str, float]:
    """Compute ADX (trend strength indicator)."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < period * 2:
        return {}

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Smoothed averages
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()

    return {
        "adx": adx.iloc[-1],
        "plus_di": plus_di.iloc[-1],
        "minus_di": minus_di.iloc[-1],
        "di_spread": plus_di.iloc[-1] - minus_di.iloc[-1],
        # ADX interpretation
        "trend_strength": (
            "strong" if adx.iloc[-1] > 25 else "weak" if adx.iloc[-1] < 20 else "moderate"
        ),
    }


@indicator(
    name="parabolic_sar",
    group="trend",
    description="Parabolic Stop and Reverse",
    parameters={"af_start": 0.02, "af_max": 0.2},
    required_periods=50,
)
def compute_parabolic_sar(
    prices: pd.DataFrame,
    af_start: float = 0.02,
    af_max: float = 0.2,
) -> dict[str, float]:
    """Compute Parabolic SAR."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < 10:
        return {}

    # Initialize
    sar = [low.iloc[0]]
    ep = high.iloc[0]  # Extreme point
    af = af_start
    trend = 1  # 1 = up, -1 = down

    for i in range(1, len(close)):
        if trend == 1:
            sar_new = sar[-1] + af * (ep - sar[-1])
            sar_new = min(sar_new, low.iloc[i - 1], low.iloc[max(0, i - 2)])

            if low.iloc[i] < sar_new:
                trend = -1
                sar_new = ep
                ep = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_start, af_max)
        else:
            sar_new = sar[-1] + af * (ep - sar[-1])
            sar_new = max(sar_new, high.iloc[i - 1], high.iloc[max(0, i - 2)])

            if high.iloc[i] > sar_new:
                trend = 1
                sar_new = ep
                ep = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_start, af_max)

        sar.append(sar_new)

    current_sar = sar[-1]
    current_price = close.iloc[-1]

    return {
        "parabolic_sar": current_sar,
        "sar_trend": 1 if current_price > current_sar else -1,
        "sar_distance": (current_price - current_sar) / current_price,
    }


@indicator(
    name="supertrend",
    group="trend",
    description="Supertrend indicator",
    parameters={"period": 10, "multiplier": 3.0},
    required_periods=50,
)
def compute_supertrend(
    prices: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> dict[str, float]:
    """Compute Supertrend indicator."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < period:
        return {}

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Supertrend calculation
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    supertrend.iloc[period - 1] = upper_band.iloc[period - 1]
    direction.iloc[period - 1] = 1

    for i in range(period, len(close)):
        if close.iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = -1

    return {
        "supertrend": supertrend.iloc[-1],
        "supertrend_direction": direction.iloc[-1],
        "supertrend_distance": (close.iloc[-1] - supertrend.iloc[-1]) / close.iloc[-1],
    }
