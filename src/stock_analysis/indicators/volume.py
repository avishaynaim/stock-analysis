"""
Volume indicators (Group D).

Indicators:
- Volume SMA Ratio
- OBV (On-Balance Volume)
- Accumulation/Distribution
- MFI (Money Flow Index)
- VWAP
- Volume Profile
- CMF (Chaikin Money Flow)
"""

import numpy as np
import pandas as pd

from stock_analysis.indicators.registry import indicator


@indicator(
    name="volume_sma",
    group="volume",
    description="Volume vs Moving Average",
    parameters={"period": 20},
    required_periods=30,
)
def compute_volume_sma(
    prices: pd.DataFrame,
    period: int = 20,
) -> dict[str, float]:
    """Compute volume relative to moving average."""
    volume = prices["volume"]

    if len(volume) < period:
        return {}

    vol_sma = volume.rolling(window=period).mean()
    vol_ratio = volume / vol_sma

    return {
        "volume_sma": vol_sma.iloc[-1],
        "volume_sma_ratio": vol_ratio.iloc[-1],
        "volume_above_avg": 1 if vol_ratio.iloc[-1] > 1.0 else 0,
        "volume_spike": 1 if vol_ratio.iloc[-1] > 2.0 else 0,
        "avg_volume_20d": vol_sma.iloc[-1],
    }


@indicator(
    name="obv",
    group="volume",
    description="On-Balance Volume",
    parameters={},
    required_periods=30,
)
def compute_obv(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Compute On-Balance Volume."""
    close = prices["adj_close"]
    volume = prices["volume"]

    if len(close) < 20:
        return {}

    # Calculate OBV
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]

    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    # OBV trend (using EMA)
    obv_ema = obv.ewm(span=20, adjust=False).mean()

    # OBV slope (normalized)
    obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / abs(obv.iloc[-5]) if obv.iloc[-5] != 0 else 0

    return {
        "obv": obv.iloc[-1],
        "obv_ema": obv_ema.iloc[-1],
        "obv_trend": 1 if obv.iloc[-1] > obv_ema.iloc[-1] else -1,
        "obv_slope": obv_slope,
        "obv_divergence": _check_divergence(close, obv),
    }


def _check_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 14) -> int:
    """Check for price/indicator divergence."""
    if len(price) < lookback:
        return 0

    price_change = price.iloc[-1] - price.iloc[-lookback]
    ind_change = indicator.iloc[-1] - indicator.iloc[-lookback]

    # Bullish divergence: price down, indicator up
    if price_change < 0 and ind_change > 0:
        return 1
    # Bearish divergence: price up, indicator down
    elif price_change > 0 and ind_change < 0:
        return -1
    return 0


@indicator(
    name="accumulation_distribution",
    group="volume",
    description="Accumulation/Distribution Line",
    parameters={},
    required_periods=30,
)
def compute_accumulation_distribution(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Compute Accumulation/Distribution Line."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]
    volume = prices["volume"]

    if len(close) < 20:
        return {}

    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)

    # Money Flow Volume
    mfv = mfm * volume

    # A/D Line
    ad_line = mfv.cumsum()

    # A/D trend
    ad_ema = ad_line.ewm(span=20, adjust=False).mean()

    return {
        "ad_line": ad_line.iloc[-1],
        "ad_ema": ad_ema.iloc[-1],
        "ad_trend": 1 if ad_line.iloc[-1] > ad_ema.iloc[-1] else -1,
        "ad_slope": (ad_line.iloc[-1] - ad_line.iloc[-5]) / abs(ad_line.iloc[-5]) if ad_line.iloc[-5] != 0 else 0,
    }


@indicator(
    name="mfi",
    group="volume",
    description="Money Flow Index",
    parameters={"period": 14},
    required_periods=30,
)
def compute_mfi(
    prices: pd.DataFrame,
    period: int = 14,
) -> dict[str, float]:
    """Compute Money Flow Index (volume-weighted RSI)."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]
    volume = prices["volume"]

    if len(close) < period + 1:
        return {}

    # Typical Price
    typical_price = (high + low + close) / 3

    # Raw Money Flow
    raw_money_flow = typical_price * volume

    # Positive and Negative Money Flow
    tp_change = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_change > 0, 0)
    negative_flow = raw_money_flow.where(tp_change < 0, 0)

    # Money Flow Ratio
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    money_ratio = positive_mf / negative_mf

    # MFI
    mfi = 100 - (100 / (1 + money_ratio))

    current_mfi = mfi.iloc[-1]

    return {
        "mfi": current_mfi,
        "mfi_oversold": 1 if current_mfi < 20 else 0,
        "mfi_overbought": 1 if current_mfi > 80 else 0,
        "mfi_trend": 1 if current_mfi > 50 else -1,
    }


@indicator(
    name="vwap",
    group="volume",
    description="Volume Weighted Average Price",
    parameters={},
    required_periods=1,
)
def compute_vwap(
    prices: pd.DataFrame,
) -> dict[str, float]:
    """Compute VWAP (for daily data, uses rolling calculation)."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]
    volume = prices["volume"]

    if len(close) < 5:
        return {}

    # Typical price
    typical_price = (high + low + close) / 3

    # Cumulative VWAP (rolling for daily data)
    cumulative_tp_vol = (typical_price * volume).rolling(window=20).sum()
    cumulative_vol = volume.rolling(window=20).sum()

    vwap = cumulative_tp_vol / cumulative_vol

    current_price = close.iloc[-1]
    current_vwap = vwap.iloc[-1]

    return {
        "vwap": current_vwap,
        "price_vs_vwap": (current_price - current_vwap) / current_vwap,
        "above_vwap": 1 if current_price > current_vwap else 0,
    }


@indicator(
    name="cmf",
    group="volume",
    description="Chaikin Money Flow",
    parameters={"period": 20},
    required_periods=30,
)
def compute_cmf(
    prices: pd.DataFrame,
    period: int = 20,
) -> dict[str, float]:
    """Compute Chaikin Money Flow."""
    high = prices["high"]
    low = prices["low"]
    close = prices["adj_close"]
    volume = prices["volume"]

    if len(close) < period:
        return {}

    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)

    # Money Flow Volume
    mfv = mfm * volume

    # CMF
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()

    current_cmf = cmf.iloc[-1]

    return {
        "cmf": current_cmf,
        "cmf_positive": 1 if current_cmf > 0 else 0,
        "cmf_strong": 1 if abs(current_cmf) > 0.25 else 0,
    }


@indicator(
    name="force_index",
    group="volume",
    description="Force Index",
    parameters={"period": 13},
    required_periods=30,
)
def compute_force_index(
    prices: pd.DataFrame,
    period: int = 13,
) -> dict[str, float]:
    """Compute Force Index."""
    close = prices["adj_close"]
    volume = prices["volume"]

    if len(close) < period + 1:
        return {}

    # Force Index = Price Change * Volume
    force = close.diff() * volume

    # Smoothed Force Index
    force_ema = force.ewm(span=period, adjust=False).mean()

    current = force_ema.iloc[-1]

    return {
        "force_index": current,
        "force_positive": 1 if current > 0 else 0,
        "force_trend": 1 if current > force_ema.iloc[-5] else -1,
    }


@indicator(
    name="ease_of_movement",
    group="volume",
    description="Ease of Movement",
    parameters={"period": 14},
    required_periods=30,
)
def compute_ease_of_movement(
    prices: pd.DataFrame,
    period: int = 14,
) -> dict[str, float]:
    """Compute Ease of Movement indicator."""
    high = prices["high"]
    low = prices["low"]
    volume = prices["volume"]

    if len(high) < period:
        return {}

    # Distance Moved
    dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)

    # Box Ratio
    box_ratio = (volume / 100000000) / (high - low)

    # EMV
    emv = dm / box_ratio

    # Smoothed EMV
    emv_sma = emv.rolling(window=period).mean()

    current = emv_sma.iloc[-1]

    return {
        "ease_of_movement": current,
        "eom_positive": 1 if current > 0 else 0,
    }
