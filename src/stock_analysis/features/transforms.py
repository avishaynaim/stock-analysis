"""
Feature transformations and preprocessing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """
    Transforms features for ML models.

    Handles scaling, encoding, and feature selection.
    """

    def __init__(
        self,
        scaling_method: str = "robust",
        handle_outliers: bool = True,
        outlier_threshold: float = 3.0,
    ):
        """Initialize transformer.

        Args:
            scaling_method: 'standard', 'minmax', or 'robust'
            handle_outliers: Whether to clip outliers
            outlier_threshold: Z-score threshold for outliers
        """
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold

        self._scaler = None
        self._feature_stats: dict[str, dict[str, float]] = {}
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> "FeatureTransformer":
        """Fit transformer on training data.

        Args:
            features: DataFrame of features

        Returns:
            self
        """
        # Store feature statistics
        for col in features.columns:
            self._feature_stats[col] = {
                "mean": features[col].mean(),
                "std": features[col].std(),
                "median": features[col].median(),
                "min": features[col].min(),
                "max": features[col].max(),
                "q25": features[col].quantile(0.25),
                "q75": features[col].quantile(0.75),
            }

        # Initialize scaler
        if self.scaling_method == "standard":
            self._scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self._scaler = MinMaxScaler()
        elif self.scaling_method == "robust":
            self._scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

        # Handle outliers before fitting
        clean_features = self._clip_outliers(features) if self.handle_outliers else features

        # Fit scaler
        self._scaler.fit(clean_features)

        self._fitted = True
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features.

        Args:
            features: DataFrame of features

        Returns:
            Transformed features
        """
        if not self._fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        # Handle outliers
        clean_features = self._clip_outliers(features) if self.handle_outliers else features

        # Apply scaling
        scaled = self._scaler.transform(clean_features)

        return pd.DataFrame(
            scaled,
            columns=features.columns,
            index=features.index,
        )

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)

    def _clip_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers based on z-score or IQR."""
        result = features.copy()

        for col in result.columns:
            if col in self._feature_stats:
                stats = self._feature_stats[col]
                iqr = stats["q75"] - stats["q25"]
                lower = stats["q25"] - self.outlier_threshold * iqr
                upper = stats["q75"] + self.outlier_threshold * iqr
            else:
                # Use current data stats if not fitted
                q25 = result[col].quantile(0.25)
                q75 = result[col].quantile(0.75)
                iqr = q75 - q25
                lower = q25 - self.outlier_threshold * iqr
                upper = q75 + self.outlier_threshold * iqr

            result[col] = result[col].clip(lower, upper)

        return result

    def get_feature_importance_filter(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        threshold: float = 0.01,
    ) -> list[str]:
        """Get features with correlation above threshold.

        Args:
            features: Feature DataFrame
            target: Target series
            threshold: Minimum absolute correlation

        Returns:
            List of important feature names
        """
        correlations = features.corrwith(target).abs()
        important = correlations[correlations >= threshold].index.tolist()
        return important

    def remove_highly_correlated(
        self,
        features: pd.DataFrame,
        threshold: float = 0.95,
    ) -> pd.DataFrame:
        """Remove highly correlated features.

        Args:
            features: Feature DataFrame
            threshold: Correlation threshold for removal

        Returns:
            Features with redundant columns removed
        """
        corr_matrix = features.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find columns to drop
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        logger.info(f"Removing {len(to_drop)} highly correlated features")

        return features.drop(columns=to_drop)

    def get_feature_stats(self) -> dict[str, dict[str, float]]:
        """Get stored feature statistics."""
        return self._feature_stats.copy()


class FeatureSelector:
    """
    Feature selection utilities.
    """

    @staticmethod
    def select_by_importance(
        features: pd.DataFrame,
        target: pd.Series,
        n_features: int = 50,
        method: str = "correlation",
    ) -> list[str]:
        """Select top N features by importance.

        Args:
            features: Feature DataFrame
            target: Target series
            n_features: Number of features to select
            method: Selection method ('correlation', 'mutual_info')

        Returns:
            List of selected feature names
        """
        if method == "correlation":
            importance = features.corrwith(target).abs().sort_values(ascending=False)
            return importance.head(n_features).index.tolist()

        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_regression

            mi = mutual_info_regression(features.fillna(0), target.fillna(0))
            importance = pd.Series(mi, index=features.columns).sort_values(ascending=False)
            return importance.head(n_features).index.tolist()

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def variance_threshold(
        features: pd.DataFrame,
        threshold: float = 0.01,
    ) -> list[str]:
        """Select features with variance above threshold.

        Args:
            features: Feature DataFrame
            threshold: Minimum variance

        Returns:
            List of selected feature names
        """
        variances = features.var()
        return variances[variances >= threshold].index.tolist()

    @staticmethod
    def get_feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
        """Group features by indicator type.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary of group name to feature list
        """
        groups = {
            "trend": [],
            "momentum": [],
            "volatility": [],
            "volume": [],
            "structure": [],
            "regime": [],
            "relative": [],
            "pattern": [],
            "price": [],
            "return": [],
            "other": [],
        }

        group_keywords = {
            "trend": ["ema", "macd", "adx", "supertrend", "aroon", "sma"],
            "momentum": ["rsi", "stoch", "roc", "williams", "cci", "tsi"],
            "volatility": ["atr", "bollinger", "keltner", "vol", "ulcer"],
            "volume": ["obv", "vwap", "mfi", "cmf", "volume"],
            "structure": ["pivot", "fib", "swing", "channel", "gap"],
            "regime": ["regime", "chop", "breadth"],
            "relative": ["rs_", "mansfield", "drawdown", "sharpe", "sortino"],
            "pattern": ["candle", "pattern", "engulf", "doji", "hammer"],
            "price": ["price", "dist_from", "new_"],
            "return": ["return_"],
        }

        for feature in feature_names:
            feature_lower = feature.lower()
            assigned = False

            for group, keywords in group_keywords.items():
                if any(kw in feature_lower for kw in keywords):
                    groups[group].append(feature)
                    assigned = True
                    break

            if not assigned:
                groups["other"].append(feature)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
