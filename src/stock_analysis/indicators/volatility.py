"""
Volatility indicators (Group C).

Indicators:
- ATR
- Bollinger Bands
- Keltner Channels
- Historical Volatility
- Normalized ATR
- Standard Deviation
- Chaikin Volatility
"""

import numpy as np
import pandas as pd

from stock_analysis.indicators.registry import indicator


@indicator(
    name="atr",
    group="volatility",
    description="Average True Range",
    parameters={"period": 14},
    required_periods=30,
)
def compute_atr(
    prices: pd.DataFrame,
    period: int = 14,
) -> dict[str, float]:
    """Compute Average True Range."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < period + 1:
        return {}

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    current_atr = atr.iloc[-1]
    current_price = close.iloc[-1]

    # Normalized ATR (as percentage of price)
    natr = (current_atr / current_price) * 100

    # ATR percentile (vs historical)
    atr_percentile = (atr < current_atr).mean()

    return {
        "atr": current_atr,
        "atr_percent": natr,
        "atr_percentile": atr_percentile,
        "atr_expanding": 1 if atr.iloc[-1] > atr.iloc[-5] else 0,
    }


@indicator(
    name="bollinger_bands",
    group="volatility",
    description="Bollinger Bands",
    parameters={"period": 20, "std_dev": 2.0},
    required_periods=30,
)
def compute_bollinger_bands(
    prices: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, float]:
    """Compute Bollinger Bands."""
    close = prices["adj_close"]

    if len(close) < period:
        return {}

    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std

    current_price = close.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    current_sma = sma.iloc[-1]

    # Bandwidth (volatility measure)
    bandwidth = (current_upper - current_lower) / current_sma

    # %B (position within bands)
    pct_b = (current_price - current_lower) / (current_upper - current_lower)

    return {
        "bollinger_upper": current_upper,
        "bollinger_middle": current_sma,
        "bollinger_lower": current_lower,
        "bollinger_bandwidth": bandwidth,
        "bollinger_pct_b": pct_b,
        "bollinger_squeeze": 1 if bandwidth < 0.1 else 0,
        "price_above_upper": 1 if current_price > current_upper else 0,
        "price_below_lower": 1 if current_price < current_lower else 0,
    }


@indicator(
    name="keltner_channels",
    group="volatility",
    description="Keltner Channels",
    parameters={"ema_period": 20, "atr_period": 10, "multiplier": 2.0},
    required_periods=30,
)
def compute_keltner_channels(
    prices: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> dict[str, float]:
    """Compute Keltner Channels."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < max(ema_period, atr_period):
        return {}

    # EMA middle line
    ema = close.ewm(span=ema_period, adjust=False).mean()

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    upper_channel = ema + multiplier * atr
    lower_channel = ema - multiplier * atr

    current_price = close.iloc[-1]

    return {
        "keltner_upper": upper_channel.iloc[-1],
        "keltner_middle": ema.iloc[-1],
        "keltner_lower": lower_channel.iloc[-1],
        "keltner_position": (current_price - lower_channel.iloc[-1]) / (
            upper_channel.iloc[-1] - lower_channel.iloc[-1]
        ),
    }


@indicator(
    name="historical_volatility",
    group="volatility",
    description="Historical Volatility (annualized)",
    parameters={"periods": [21, 63, 252]},
    required_periods=63,
)
def compute_historical_volatility(
    prices: pd.DataFrame,
    periods: list[int] = [21, 63, 252],
) -> dict[str, float]:
    """Compute Historical Volatility."""
    close = prices["adj_close"]
    log_returns = np.log(close / close.shift(1))

    result = {}
    annualization_factor = np.sqrt(252)

    for period in periods:
        if len(close) > period:
            vol = log_returns.rolling(window=period).std() * annualization_factor
            result[f"volatility_{period}d"] = vol.iloc[-1]

    # Volatility ratio (short-term vs long-term)
    if "volatility_21d" in result and "volatility_63d" in result:
        result["volatility_ratio"] = result["volatility_21d"] / result["volatility_63d"]

    # Volatility percentile
    if len(close) > 252:
        vol_21 = log_returns.rolling(window=21).std() * annualization_factor
        result["volatility_percentile"] = (vol_21 < vol_21.iloc[-1]).mean()

    return result


@indicator(
    name="chaikin_volatility",
    group="volatility",
    description="Chaikin Volatility",
    parameters={"ema_period": 10, "roc_period": 10},
    required_periods=30,
)
def compute_chaikin_volatility(
    prices: pd.DataFrame,
    ema_period: int = 10,
    roc_period: int = 10,
) -> dict[str, float]:
    """Compute Chaikin Volatility."""
    high = prices["high"]
    low = prices["low"]

    if len(high) < ema_period + roc_period:
        return {}

    # High-Low range
    hl_range = high - low

    # EMA of range
    ema_range = hl_range.ewm(span=ema_period, adjust=False).mean()

    # Rate of change of EMA
    chaikin_vol = ((ema_range - ema_range.shift(roc_period)) / ema_range.shift(roc_period)) * 100

    return {
        "chaikin_volatility": chaikin_vol.iloc[-1],
        "chaikin_vol_expanding": 1 if chaikin_vol.iloc[-1] > 0 else 0,
    }


@indicator(
    name="ulcer_index",
    group="volatility",
    description="Ulcer Index (downside volatility)",
    parameters={"period": 14},
    required_periods=30,
)
def compute_ulcer_index(
    prices: pd.DataFrame,
    period: int = 14,
) -> dict[str, float]:
    """Compute Ulcer Index (measures downside risk)."""
    close = prices["adj_close"]

    if len(close) < period:
        return {}

    # Rolling maximum
    rolling_max = close.rolling(window=period).max()

    # Percentage drawdown
    pct_drawdown = ((close - rolling_max) / rolling_max) * 100

    # Ulcer Index is the quadratic mean of drawdowns
    ulcer = np.sqrt((pct_drawdown**2).rolling(window=period).mean())

    return {
        "ulcer_index": ulcer.iloc[-1],
        "current_drawdown": pct_drawdown.iloc[-1],
    }


@indicator(
    name="donchian_channels",
    group="volatility",
    description="Donchian Channels",
    parameters={"period": 20},
    required_periods=30,
)
def compute_donchian_channels(
    prices: pd.DataFrame,
    period: int = 20,
) -> dict[str, float]:
    """Compute Donchian Channels."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < period:
        return {}

    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2

    current_price = close.iloc[-1]
    current_upper = upper.iloc[-1]
    current_lower = lower.iloc[-1]

    # Channel width as percentage
    channel_width = (current_upper - current_lower) / current_price

    return {
        "donchian_upper": current_upper,
        "donchian_middle": middle.iloc[-1],
        "donchian_lower": current_lower,
        "donchian_width": channel_width,
        "donchian_position": (current_price - current_lower) / (current_upper - current_lower),
        "at_donchian_high": 1 if current_price >= current_upper else 0,
        "at_donchian_low": 1 if current_price <= current_lower else 0,
    }
