"""
Indicator computation engine.

Responsible for computing all indicators for a given stock.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from stock_analysis.indicators.registry import IndicatorRegistry

if TYPE_CHECKING:
    from stock_analysis.data.models import StockData

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """Engine for computing technical indicators."""

    def __init__(
        self,
        registry: IndicatorRegistry | None = None,
        enabled_groups: list[str] | None = None,
        disabled_indicators: list[str] | None = None,
    ):
        """Initialize indicator engine.

        Args:
            registry: Indicator registry to use. If None, uses global registry.
            enabled_groups: List of indicator groups to enable. If None, enables all.
            disabled_indicators: List of specific indicators to disable.
        """
        self.registry = registry or IndicatorRegistry.get_instance()
        self.enabled_groups = enabled_groups
        self.disabled_indicators = disabled_indicators or []

        # Import all indicator modules to register them
        self._load_indicator_modules()

    def _load_indicator_modules(self) -> None:
        """Import all indicator modules to trigger registration."""
        try:
            from stock_analysis.indicators import (
                momentum,
                patterns,
                regime,
                relative,
                structure,
                trend,
                volatility,
                volume,
            )
        except ImportError as e:
            logger.warning(f"Failed to import some indicator modules: {e}")

    def compute_all(
        self,
        prices: pd.DataFrame,
        benchmark_prices: pd.DataFrame | None = None,
        return_dataframe: bool = False,
    ) -> dict[str, Any] | pd.DataFrame:
        """Compute all enabled indicators for the given price data.

        Args:
            prices: DataFrame with OHLCV data (columns: open, high, low, adj_close, volume)
            benchmark_prices: Optional benchmark price data for relative indicators
            return_dataframe: If True, returns a DataFrame with indicator time series

        Returns:
            Dictionary of indicator names to computed values, or DataFrame with time series
        """
        results: dict[str, Any] = {}
        series_results: dict[str, pd.Series] = {}

        for name, info in self.registry.get_all().items():
            # Skip disabled indicators
            if name in self.disabled_indicators:
                continue

            # Skip indicators from disabled groups
            if self.enabled_groups and info.group not in self.enabled_groups:
                continue

            # Check if we have enough data
            if len(prices) < info.required_periods:
                logger.debug(
                    f"Skipping {name}: insufficient data "
                    f"({len(prices)} < {info.required_periods})"
                )
                continue

            try:
                # Prepare arguments
                kwargs = dict(info.parameters)

                # Add benchmark if required
                if info.requires_benchmark and benchmark_prices is not None:
                    kwargs["benchmark_prices"] = benchmark_prices

                # Compute indicator
                indicator_result = info.func(prices, **kwargs)

                # Flatten nested results if necessary
                if isinstance(indicator_result, dict):
                    results.update(indicator_result)
                    # For DataFrame mode, also collect series
                    if return_dataframe:
                        for k, v in indicator_result.items():
                            if isinstance(v, pd.Series):
                                series_results[k] = v
                else:
                    results[name] = indicator_result
                    if return_dataframe and isinstance(indicator_result, pd.Series):
                        series_results[name] = indicator_result

            except Exception as e:
                logger.warning(f"Failed to compute indicator {name}: {e}")
                continue

        if return_dataframe:
            # Build DataFrame from series results and compute time series for scalar results
            return self._build_indicator_dataframe(prices, results, series_results)

        return results

    def _build_indicator_dataframe(
        self,
        prices: pd.DataFrame,
        results: dict[str, Any],
        series_results: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Build a DataFrame with indicator time series."""
        import numpy as np

        # Start with price-derived basic indicators computed over time
        close = prices["adj_close"] if "adj_close" in prices.columns else prices["close"]
        high = prices["high"]
        low = prices["low"]
        volume = prices["volume"]

        df = pd.DataFrame(index=prices.index)

        # Add any Series results directly
        for name, series in series_results.items():
            if len(series) == len(prices):
                df[name] = series.values

        # Compute core indicators as time series
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd_line"] = ema12 - ema26
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

        # EMAs
        for period in [8, 21, 50, 200]:
            df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_percent"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / sma20

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / close * 100

        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df["adx"] = dx.rolling(14).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        # Stochastic
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        df["stoch_k"] = 100 * (close - low14) / (high14 - low14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # Volume indicators
        df["volume_sma"] = volume.rolling(20).mean()
        df["volume_sma_ratio"] = volume / df["volume_sma"]

        # OBV
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv

        # MFI
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        mf_positive = raw_money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        mf_negative = raw_money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mf_ratio = mf_positive / mf_negative
        df["mfi"] = 100 - (100 / (1 + mf_ratio))

        # Momentum
        df["roc_10"] = close.pct_change(10) * 100
        df["momentum_10"] = close - close.shift(10)

        # Volatility
        df["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

        # Price position
        df["price_vs_ema50"] = (close / df["ema_50"] - 1) * 100
        df["price_vs_ema200"] = (close / df["ema_200"] - 1) * 100

        # Replace infinities and fill NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def compute_single(
        self,
        name: str,
        prices: pd.DataFrame,
        benchmark_prices: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute a single indicator.

        Args:
            name: Name of the indicator to compute
            prices: DataFrame with OHLCV data
            benchmark_prices: Optional benchmark data
            **kwargs: Override default parameters

        Returns:
            Dictionary of computed values

        Raises:
            KeyError: If indicator not found
            ValueError: If insufficient data
        """
        info = self.registry.get(name)
        if info is None:
            raise KeyError(f"Indicator '{name}' not found in registry")

        if len(prices) < info.required_periods:
            raise ValueError(
                f"Insufficient data for {name}: "
                f"{len(prices)} < {info.required_periods}"
            )

        # Merge default parameters with overrides
        params = dict(info.parameters)
        params.update(kwargs)

        if info.requires_benchmark and benchmark_prices is not None:
            params["benchmark_prices"] = benchmark_prices

        return info.func(prices, **params)

    def compute_group(
        self,
        group: str,
        prices: pd.DataFrame,
        benchmark_prices: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Compute all indicators in a specific group.

        Args:
            group: Name of the indicator group
            prices: DataFrame with OHLCV data
            benchmark_prices: Optional benchmark data

        Returns:
            Dictionary of computed values for the group
        """
        results: dict[str, Any] = {}

        for name, info in self.registry.get_by_group(group).items():
            if name in self.disabled_indicators:
                continue

            if len(prices) < info.required_periods:
                continue

            try:
                kwargs = dict(info.parameters)
                if info.requires_benchmark and benchmark_prices is not None:
                    kwargs["benchmark_prices"] = benchmark_prices

                result = info.func(prices, **kwargs)
                if isinstance(result, dict):
                    results.update(result)
                else:
                    results[name] = result

            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")

        return results

    def get_available_indicators(self) -> list[str]:
        """Get list of all available indicator names."""
        return list(self.registry.get_all().keys())

    def get_indicator_info(self, name: str) -> dict[str, Any] | None:
        """Get information about a specific indicator."""
        info = self.registry.get(name)
        if info is None:
            return None

        return {
            "name": info.name,
            "group": info.group,
            "description": info.description,
            "parameters": info.parameters,
            "required_periods": info.required_periods,
            "requires_benchmark": info.requires_benchmark,
        }

    def get_groups(self) -> list[str]:
        """Get list of all indicator groups."""
        groups = set()
        for info in self.registry.get_all().values():
            groups.add(info.group)
        return sorted(groups)
