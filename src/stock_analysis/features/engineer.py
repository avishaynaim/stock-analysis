"""
Feature engineering pipeline.

Transforms raw indicators into ML-ready features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from stock_analysis.indicators.engine import IndicatorEngine

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline that transforms indicator outputs
    into ML-ready features.
    """

    def __init__(
        self,
        indicator_engine: IndicatorEngine | None = None,
        include_lagged: bool = True,
        lag_periods: list[int] | None = None,
        include_rolling: bool = True,
        rolling_windows: list[int] | None = None,
        include_interactions: bool = False,
    ):
        """Initialize feature engineer.

        Args:
            indicator_engine: Engine for computing indicators
            include_lagged: Whether to include lagged features
            lag_periods: Periods for lagged features
            include_rolling: Whether to include rolling statistics
            rolling_windows: Windows for rolling calculations
            include_interactions: Whether to include feature interactions
        """
        self.indicator_engine = indicator_engine or IndicatorEngine()
        self.include_lagged = include_lagged
        self.lag_periods = lag_periods or [1, 5, 10]
        self.include_rolling = include_rolling
        self.rolling_windows = rolling_windows or [5, 10, 20]
        self.include_interactions = include_interactions

        self._feature_names: list[str] = []

    def compute_features(
        self,
        prices: pd.DataFrame,
        benchmark_prices: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Compute all features for the given price data.

        Args:
            prices: DataFrame with OHLCV data
            benchmark_prices: Optional benchmark data

        Returns:
            Dictionary of feature names to values
        """
        features: dict[str, Any] = {}

        # Get base indicators
        indicators = self.indicator_engine.compute_all(prices, benchmark_prices)
        features.update(indicators)

        # Add derived features
        features.update(self._compute_price_features(prices))
        features.update(self._compute_return_features(prices))
        features.update(self._compute_volatility_features(prices))

        # Filter out non-numeric and invalid values
        features = self._clean_features(features)

        self._feature_names = list(features.keys())

        return features

    def compute_features_timeseries(
        self,
        prices: pd.DataFrame,
        benchmark_prices: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute features for entire time series (for training).

        Args:
            prices: Full historical price data
            benchmark_prices: Optional benchmark data

        Returns:
            DataFrame with features for each date
        """
        all_features = []
        min_periods = 252  # Minimum data needed

        for i in range(min_periods, len(prices)):
            # Get data up to current date
            current_prices = prices.iloc[: i + 1]
            current_benchmark = None
            if benchmark_prices is not None:
                current_benchmark = benchmark_prices.iloc[: i + 1]

            try:
                features = self.compute_features(current_prices, current_benchmark)
                features["date"] = prices.index[i]
                all_features.append(features)
            except Exception as e:
                logger.debug(f"Failed to compute features for index {i}: {e}")
                continue

        if not all_features:
            return pd.DataFrame()

        df = pd.DataFrame(all_features)
        df.set_index("date", inplace=True)

        return df

    def _compute_price_features(self, prices: pd.DataFrame) -> dict[str, float]:
        """Compute price-based features."""
        close = prices["adj_close"]
        high = prices["high"]
        low = prices["low"]

        features = {}

        if len(close) < 20:
            return features

        # Price position in various ranges
        range_20 = high.iloc[-20:].max() - low.iloc[-20:].min()
        if range_20 > 0:
            features["price_position_20d"] = (
                close.iloc[-1] - low.iloc[-20:].min()
            ) / range_20

        range_50 = high.iloc[-50:].max() - low.iloc[-50:].min() if len(close) >= 50 else 0
        if range_50 > 0:
            features["price_position_50d"] = (
                close.iloc[-1] - low.iloc[-50:].min()
            ) / range_50

        # Distance from moving averages
        for period in [10, 20, 50, 200]:
            if len(close) >= period:
                ma = close.iloc[-period:].mean()
                features[f"dist_from_ma{period}"] = (close.iloc[-1] - ma) / ma

        # New high/low detection
        features["new_20d_high"] = 1 if close.iloc[-1] >= high.iloc[-20:].max() else 0
        features["new_20d_low"] = 1 if close.iloc[-1] <= low.iloc[-20:].min() else 0

        if len(close) >= 52:
            features["new_52w_high"] = 1 if close.iloc[-1] >= high.iloc[-252:].max() else 0
            features["new_52w_low"] = 1 if close.iloc[-1] <= low.iloc[-252:].min() else 0

        return features

    def _compute_return_features(self, prices: pd.DataFrame) -> dict[str, float]:
        """Compute return-based features."""
        close = prices["adj_close"]
        features = {}

        if len(close) < 5:
            return features

        returns = close.pct_change()

        # Multi-period returns
        for period in [1, 5, 10, 21, 63]:
            if len(close) > period:
                features[f"return_{period}d"] = (
                    close.iloc[-1] / close.iloc[-period - 1] - 1
                )

        # Return statistics
        if len(returns) >= 20:
            recent_returns = returns.iloc[-20:]
            features["return_mean_20d"] = recent_returns.mean()
            features["return_std_20d"] = recent_returns.std()
            features["return_skew_20d"] = recent_returns.skew()
            features["return_kurt_20d"] = recent_returns.kurtosis()

            # Positive return ratio
            features["positive_return_ratio_20d"] = (recent_returns > 0).mean()

        # Consecutive up/down days
        if len(returns) >= 10:
            recent = returns.iloc[-10:]
            up_streak = 0
            down_streak = 0
            for r in recent[::-1]:
                if r > 0:
                    up_streak += 1
                else:
                    break
            for r in recent[::-1]:
                if r < 0:
                    down_streak += 1
                else:
                    break
            features["up_streak"] = up_streak
            features["down_streak"] = down_streak

        return features

    def _compute_volatility_features(self, prices: pd.DataFrame) -> dict[str, float]:
        """Compute volatility-based features."""
        close = prices["adj_close"]
        high = prices["high"]
        low = prices["low"]
        features = {}

        if len(close) < 20:
            return features

        returns = close.pct_change()

        # Realized volatility at different windows
        for period in [5, 10, 20]:
            if len(returns) >= period:
                vol = returns.iloc[-period:].std() * np.sqrt(252)
                features[f"realized_vol_{period}d"] = vol

        # Parkinson volatility (using high-low range)
        if len(high) >= 20:
            log_hl = np.log(high / low)
            parkinson = np.sqrt((1 / (4 * np.log(2))) * (log_hl**2).rolling(20).mean())
            features["parkinson_vol_20d"] = parkinson.iloc[-1] * np.sqrt(252)

        # Volatility ratio (short vs long)
        if len(returns) >= 50:
            vol_short = returns.iloc[-10:].std()
            vol_long = returns.iloc[-50:].std()
            features["vol_ratio_10_50"] = vol_short / vol_long if vol_long > 0 else 1

        # Average true range percentage
        if len(close) >= 14:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            features["atr_pct"] = atr.iloc[-1] / close.iloc[-1]

        return features

    def _clean_features(self, features: dict[str, Any]) -> dict[str, float]:
        """Clean and filter features."""
        cleaned = {}

        for name, value in features.items():
            # Skip non-numeric values (like string regime names)
            if isinstance(value, str):
                continue

            # Convert to float if possible
            try:
                val = float(value)

                # Skip invalid values
                if np.isnan(val) or np.isinf(val):
                    continue

                cleaned[name] = val

            except (TypeError, ValueError):
                continue

        return cleaned

    def get_feature_names(self) -> list[str]:
        """Get list of computed feature names."""
        return self._feature_names.copy()

    def create_target_variable(
        self,
        prices: pd.DataFrame,
        horizon: int = 21,
        target_type: str = "return",
        threshold: float = 0.0,
    ) -> pd.Series:
        """Create target variable for ML training.

        Args:
            prices: Price data
            horizon: Forward-looking horizon in days
            target_type: Type of target ('return', 'binary', 'class')
            threshold: Threshold for binary/class targets

        Returns:
            Series with target values
        """
        close = prices["adj_close"]

        # Forward returns
        forward_return = close.shift(-horizon) / close - 1

        if target_type == "return":
            return forward_return

        elif target_type == "binary":
            return (forward_return > threshold).astype(int)

        elif target_type == "class":
            # Three classes: down, neutral, up
            target = pd.Series(index=close.index, dtype=int)
            target[forward_return < -threshold] = 0  # Down
            target[
                (forward_return >= -threshold) & (forward_return <= threshold)
            ] = 1  # Neutral
            target[forward_return > threshold] = 2  # Up
            return target

        else:
            raise ValueError(f"Unknown target type: {target_type}")
