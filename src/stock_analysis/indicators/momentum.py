"""
Momentum indicators (Group B).

Indicators:
- RSI
- Stochastic Oscillator
- Williams %R
- CCI
- ROC
- Momentum
"""

import numpy as np
import pandas as pd

from stock_analysis.indicators.registry import indicator


@indicator(
    name="rsi",
    group="momentum",
    description="Relative Strength Index",
    parameters={"period": 14},
    required_periods=30,
)
def compute_rsi(
    prices: pd.DataFrame,
    period: int = 14,
) -> dict[str, float]:
    """Compute RSI indicator."""
    close = prices["adj_close"]

    if len(close) < period + 1:
        return {}

    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    current_rsi = rsi.iloc[-1]

    return {
        "rsi": current_rsi,
        "rsi_oversold": 1 if current_rsi < 30 else 0,
        "rsi_overbought": 1 if current_rsi > 70 else 0,
        "rsi_neutral": 1 if 40 <= current_rsi <= 60 else 0,
        "rsi_slope": rsi.diff(5).iloc[-1] / 5,  # 5-day slope
    }


@indicator(
    name="stochastic",
    group="momentum",
    description="Stochastic Oscillator",
    parameters={"k_period": 14, "d_period": 3},
    required_periods=30,
)
def compute_stochastic(
    prices: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, float]:
    """Compute Stochastic Oscillator."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < k_period + d_period:
        return {}

    # %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # %D (signal line)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    k_val = stoch_k.iloc[-1]
    d_val = stoch_d.iloc[-1]

    return {
        "stochastic_k": k_val,
        "stochastic_d": d_val,
        "stochastic_crossover": 1 if k_val > d_val and stoch_k.iloc[-2] <= stoch_d.iloc[-2] else 0,
        "stochastic_crossunder": 1 if k_val < d_val and stoch_k.iloc[-2] >= stoch_d.iloc[-2] else 0,
        "stochastic_oversold": 1 if k_val < 20 else 0,
        "stochastic_overbought": 1 if k_val > 80 else 0,
    }


@indicator(
    name="williams_r",
    group="momentum",
    description="Williams %R",
    parameters={"period": 14},
    required_periods=30,
)
def compute_williams_r(
    prices: pd.DataFrame,
    period: int = 14,
) -> dict[str, float]:
    """Compute Williams %R."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < period:
        return {}

    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)

    current = williams_r.iloc[-1]

    return {
        "williams_r": current,
        "williams_r_oversold": 1 if current < -80 else 0,
        "williams_r_overbought": 1 if current > -20 else 0,
    }


@indicator(
    name="cci",
    group="momentum",
    description="Commodity Channel Index",
    parameters={"period": 20},
    required_periods=30,
)
def compute_cci(
    prices: pd.DataFrame,
    period: int = 20,
) -> dict[str, float]:
    """Compute Commodity Channel Index."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]

    if len(close) < period:
        return {}

    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )

    cci = (typical_price - sma) / (0.015 * mad)

    current = cci.iloc[-1]

    return {
        "cci": current,
        "cci_oversold": 1 if current < -100 else 0,
        "cci_overbought": 1 if current > 100 else 0,
        "cci_trend": 1 if current > 0 else -1,
    }


@indicator(
    name="roc",
    group="momentum",
    description="Rate of Change",
    parameters={"periods": [5, 10, 21]},
    required_periods=30,
)
def compute_roc(
    prices: pd.DataFrame,
    periods: list[int] = [5, 10, 21],
) -> dict[str, float]:
    """Compute Rate of Change for multiple periods."""
    close = prices["adj_close"]
    result = {}

    for period in periods:
        if len(close) > period:
            roc = (close / close.shift(period) - 1) * 100
            result[f"roc_{period}"] = roc.iloc[-1]

    # ROC momentum (acceleration)
    if "roc_5" in result and len(close) > 10:
        roc_5_series = (close / close.shift(5) - 1) * 100
        result["roc_momentum"] = roc_5_series.diff(5).iloc[-1]

    return result


@indicator(
    name="momentum",
    group="momentum",
    description="Price Momentum",
    parameters={"period": 10},
    required_periods=20,
)
def compute_momentum(
    prices: pd.DataFrame,
    period: int = 10,
) -> dict[str, float]:
    """Compute simple momentum indicator."""
    close = prices["adj_close"]

    if len(close) <= period:
        return {}

    momentum = close - close.shift(period)
    momentum_pct = (close / close.shift(period) - 1) * 100

    return {
        "momentum": momentum.iloc[-1],
        "momentum_pct": momentum_pct.iloc[-1],
        "momentum_positive": 1 if momentum.iloc[-1] > 0 else 0,
        "momentum_slope": momentum.diff(5).iloc[-1] / 5,
    }


@indicator(
    name="tsi",
    group="momentum",
    description="True Strength Index",
    parameters={"long_period": 25, "short_period": 13},
    required_periods=50,
)
def compute_tsi(
    prices: pd.DataFrame,
    long_period: int = 25,
    short_period: int = 13,
) -> dict[str, float]:
    """Compute True Strength Index."""
    close = prices["adj_close"]

    if len(close) < long_period + short_period:
        return {}

    delta = close.diff()

    # Double smoothed price change
    pc_first = delta.ewm(span=long_period, adjust=False).mean()
    pc_second = pc_first.ewm(span=short_period, adjust=False).mean()

    # Double smoothed absolute price change
    apc_first = delta.abs().ewm(span=long_period, adjust=False).mean()
    apc_second = apc_first.ewm(span=short_period, adjust=False).mean()

    tsi = 100 * pc_second / apc_second

    current = tsi.iloc[-1]

    return {
        "tsi": current,
        "tsi_signal": tsi.ewm(span=7, adjust=False).mean().iloc[-1],
        "tsi_positive": 1 if current > 0 else 0,
    }


@indicator(
    name="awesome_oscillator",
    group="momentum",
    description="Awesome Oscillator",
    parameters={"fast": 5, "slow": 34},
    required_periods=50,
)
def compute_awesome_oscillator(
    prices: pd.DataFrame,
    fast: int = 5,
    slow: int = 34,
) -> dict[str, float]:
    """Compute Awesome Oscillator."""
    high = prices["high"]
    low = prices["low"]

    if len(high) < slow:
        return {}

    median_price = (high + low) / 2

    ao = median_price.rolling(window=fast).mean() - median_price.rolling(window=slow).mean()

    current = ao.iloc[-1]
    prev = ao.iloc[-2]

    return {
        "awesome_oscillator": current,
        "ao_positive": 1 if current > 0 else 0,
        "ao_increasing": 1 if current > prev else 0,
        "ao_color": "green" if current > prev else "red",
    }
