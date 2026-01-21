"""
ML Data Builder - Creates training datasets from historical indicator patterns and returns.

Enhanced with:
- Advanced pattern features (momentum breakouts, mean reversion)
- Market regime features
- Volatility-adjusted features
- Time-based features (day of week, month)
- Lagged feature interactions
"""
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class MLDataset:
    """Container for ML training data."""
    features: pd.DataFrame
    target: pd.Series
    feature_names: list[str]
    target_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def X(self) -> np.ndarray:
        return self.features.values

    @property
    def y(self) -> np.ndarray:
        return self.target.values


class MLDataBuilder:
    """Builds ML training data from indicator values and forward returns."""

    def __init__(
        self,
        forward_days: int = 20,
        gain_threshold: float = 0.10,
        big_gain_threshold: float = 0.20,
    ):
        self.forward_days = forward_days
        self.gain_threshold = gain_threshold
        self.big_gain_threshold = big_gain_threshold

    def build_dataset(
        self,
        prices: pd.DataFrame,
        indicators: pd.DataFrame,
        target_type: str = "big_gain",
    ) -> MLDataset:
        forward_returns = self._calculate_forward_returns(prices)
        target = self._create_target(forward_returns, target_type)
        features = self._prepare_features(indicators)
        aligned_features, aligned_target = self._align_data(features, target)

        mask = ~(aligned_features.isna().any(axis=1) | aligned_target.isna())
        clean_features = aligned_features[mask]
        clean_target = aligned_target[mask]

        return MLDataset(
            features=clean_features,
            target=clean_target,
            feature_names=list(clean_features.columns),
            target_name=target_type,
            metadata={
                "forward_days": self.forward_days,
                "gain_threshold": self.gain_threshold,
                "big_gain_threshold": self.big_gain_threshold,
                "n_samples": len(clean_features),
                "n_features": len(clean_features.columns),
                "target_distribution": clean_target.value_counts().to_dict() if target_type != "return" else {},
            }
        )

    def _calculate_forward_returns(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"] if "close" in prices.columns else prices["adj_close"]
        forward_returns = close.shift(-self.forward_days) / close - 1
        return forward_returns

    def _create_target(self, forward_returns: pd.Series, target_type: str) -> pd.Series:
        if target_type == "return":
            return forward_returns
        elif target_type == "direction":
            return (forward_returns > 0).astype(int)
        elif target_type == "gain":
            return (forward_returns >= self.gain_threshold).astype(int)
        elif target_type == "big_gain":
            return (forward_returns >= self.big_gain_threshold).astype(int)
        elif target_type == "multi_class":
            conditions = [
                forward_returns < 0,
                (forward_returns >= 0) & (forward_returns < self.gain_threshold),
                (forward_returns >= self.gain_threshold) & (forward_returns < self.big_gain_threshold),
                forward_returns >= self.big_gain_threshold,
            ]
            choices = [0, 1, 2, 3]
            return pd.Series(np.select(conditions, choices), index=forward_returns.index)
        else:
            raise ValueError(f"Unknown target type: {target_type}")

    def _prepare_features(self, indicators: pd.DataFrame) -> pd.DataFrame:
        features = indicators.copy()
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_cols]
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()
        return features

    def _align_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        common_index = features.index.intersection(target.index)
        return features.loc[common_index], target.loc[common_index]

    def create_pattern_features(
        self,
        indicators: pd.DataFrame,
        lookback: int = 5,
    ) -> pd.DataFrame:
        pattern_features = pd.DataFrame(index=indicators.index)
        numeric_cols = indicators.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = indicators[col]
            pattern_features[f"{col}_roc_{lookback}d"] = series.pct_change(lookback)
            pattern_features[f"{col}_vs_ma"] = series / series.rolling(lookback).mean() - 1
            rolling_mean = series.rolling(lookback * 4).mean()
            rolling_std = series.rolling(lookback * 4).std()
            pattern_features[f"{col}_zscore"] = (series - rolling_mean) / (rolling_std + 1e-10)
            pattern_features[f"{col}_trend"] = series.rolling(lookback).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == lookback else np.nan,
                raw=False
            )

        return pattern_features.replace([np.inf, -np.inf], np.nan)

    def create_cross_features(self, indicators: pd.DataFrame) -> pd.DataFrame:
        cross_features = pd.DataFrame(index=indicators.index)
        pairs = [
            ("rsi", "macd"),
            ("rsi", "adx"),
            ("macd", "volume_sma_ratio"),
            ("bb_percent", "rsi"),
            ("momentum", "volatility"),
            ("stoch_k", "rsi"),  # Momentum confirmation
            ("adx", "volume"),   # Trend + volume
        ]

        for col1, col2 in pairs:
            match1 = [c for c in indicators.columns if col1.lower() in c.lower()]
            match2 = [c for c in indicators.columns if col2.lower() in c.lower()]

            if match1 and match2:
                c1, c2 = match1[0], match2[0]
                cross_features[f"{col1}_{col2}_product"] = indicators[c1] * indicators[c2]
                cross_features[f"{col1}_{col2}_ratio"] = indicators[c1] / (indicators[c2] + 1e-10)
                # Add difference feature
                cross_features[f"{col1}_{col2}_diff"] = indicators[c1] - indicators[c2]

        return cross_features.replace([np.inf, -np.inf], np.nan)

    def create_regime_features(self, prices: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
        """Create market regime features."""
        regime_features = pd.DataFrame(index=indicators.index)
        close = prices["adj_close"] if "adj_close" in prices.columns else prices["close"]

        # Volatility regime (using rolling realized vol)
        returns = close.pct_change()
        short_vol = returns.rolling(10).std() * np.sqrt(252)
        long_vol = returns.rolling(60).std() * np.sqrt(252)
        regime_features["vol_regime"] = short_vol / (long_vol + 1e-10)
        regime_features["vol_expanding"] = (short_vol > long_vol).astype(int)

        # Trend regime
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()

        regime_features["trend_score"] = (
            (close > ema_20).astype(int) +
            (close > ema_50).astype(int) +
            (ema_20 > ema_50).astype(int) +
            (ema_50 > ema_200).astype(int)
        ) / 4

        # Momentum regime
        momentum_10 = close.pct_change(10)
        momentum_20 = close.pct_change(20)
        regime_features["momentum_regime"] = (momentum_10 + momentum_20) / 2

        # Mean reversion setup
        bb_col = [c for c in indicators.columns if "bb_percent" in c.lower()]
        if bb_col:
            bb_pct = indicators[bb_col[0]]
            regime_features["mean_reversion_setup"] = (
                ((bb_pct < 0.2) & (momentum_10 < 0)).astype(int) -
                ((bb_pct > 0.8) & (momentum_10 > 0)).astype(int)
            )

        # Breakout detection
        high_20 = prices["high"].rolling(20).max()
        low_20 = prices["low"].rolling(20).min()
        regime_features["near_high_breakout"] = (close >= high_20 * 0.98).astype(int)
        regime_features["near_low_breakdown"] = (close <= low_20 * 1.02).astype(int)

        return regime_features.replace([np.inf, -np.inf], np.nan)

    def create_time_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create time-based cyclical features."""
        time_features = pd.DataFrame(index=prices.index)

        if hasattr(prices.index, 'dayofweek'):
            # Day of week (0=Monday, 4=Friday)
            time_features["day_of_week"] = prices.index.dayofweek
            time_features["is_monday"] = (prices.index.dayofweek == 0).astype(int)
            time_features["is_friday"] = (prices.index.dayofweek == 4).astype(int)

            # Month
            time_features["month"] = prices.index.month
            time_features["is_jan"] = (prices.index.month == 1).astype(int)  # January effect
            time_features["is_dec"] = (prices.index.month == 12).astype(int)  # Tax loss selling

            # Quarter end
            time_features["is_quarter_end"] = prices.index.is_quarter_end.astype(int)

        return time_features

    def build_enhanced_dataset(
        self,
        prices: pd.DataFrame,
        indicators: pd.DataFrame,
        target_type: str = "big_gain",
        include_patterns: bool = True,
        include_cross: bool = True,
        include_regime: bool = True,
        include_time: bool = True,
    ) -> MLDataset:
        all_features = indicators.copy()

        if include_patterns:
            pattern_features = self.create_pattern_features(indicators)
            all_features = pd.concat([all_features, pattern_features], axis=1)

        if include_cross:
            cross_features = self.create_cross_features(indicators)
            all_features = pd.concat([all_features, cross_features], axis=1)

        if include_regime:
            regime_features = self.create_regime_features(prices, indicators)
            all_features = pd.concat([all_features, regime_features], axis=1)

        if include_time:
            time_features = self.create_time_features(prices)
            all_features = pd.concat([all_features, time_features], axis=1)

        return self.build_dataset(prices, all_features, target_type)
