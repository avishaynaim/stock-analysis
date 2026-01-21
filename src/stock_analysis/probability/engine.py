"""
Probability estimation engine.

Main interface for probability calculations.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from stock_analysis.probability.estimators import (
    BaseProbabilityEstimator,
    EnsembleEstimator,
    HistoricalEstimator,
    BayesianEstimator,
)

logger = logging.getLogger(__name__)


class ProbabilityEngine:
    """
    Main probability estimation engine.

    Orchestrates multiple estimators and provides unified interface.
    """

    def __init__(
        self,
        estimator: BaseProbabilityEstimator | None = None,
        horizons: list[int] | None = None,
    ):
        """Initialize probability engine.

        Args:
            estimator: Probability estimator to use
            horizons: Forecast horizons in days
        """
        self.estimator = estimator or EnsembleEstimator()
        self.horizons = horizons or [5, 10, 21, 63]

        self._calibrated = False

    def calibrate(
        self,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        target_horizon: int = 21,
    ) -> None:
        """Calibrate estimator on historical data.

        Args:
            features: Historical feature DataFrame
            prices: Historical price DataFrame
            target_horizon: Default horizon for outcome calculation
        """
        # Create binary outcome (up/down)
        close = prices["adj_close"]
        forward_return = close.shift(-target_horizon) / close - 1
        outcome = (forward_return > 0).astype(int)

        # Align features and outcome
        common_idx = features.index.intersection(outcome.dropna().index)
        aligned_features = features.loc[common_idx]
        aligned_outcome = outcome.loc[common_idx]

        # Calibrate estimator
        self.estimator.calibrate(aligned_features, aligned_outcome)

        self._calibrated = True
        logger.info(f"Probability engine calibrated on {len(aligned_features)} samples")

    def estimate_probabilities(
        self,
        features: dict[str, float],
        historical_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Estimate probabilities for current state.

        Args:
            features: Current feature values
            historical_data: Optional historical data for context

        Returns:
            Dictionary with probability estimates
        """
        result = self.estimator.estimate(features, historical_data)

        # Add interpretation
        prob_up = result.get("prob_up", 0.5)

        if prob_up >= 0.7:
            result["signal"] = "strong_bullish"
            result["signal_strength"] = 3
        elif prob_up >= 0.6:
            result["signal"] = "bullish"
            result["signal_strength"] = 2
        elif prob_up >= 0.55:
            result["signal"] = "slightly_bullish"
            result["signal_strength"] = 1
        elif prob_up >= 0.45:
            result["signal"] = "neutral"
            result["signal_strength"] = 0
        elif prob_up >= 0.4:
            result["signal"] = "slightly_bearish"
            result["signal_strength"] = -1
        elif prob_up >= 0.3:
            result["signal"] = "bearish"
            result["signal_strength"] = -2
        else:
            result["signal"] = "strong_bearish"
            result["signal_strength"] = -3

        return result

    def estimate_scenario_probabilities(
        self,
        features: dict[str, float],
        scenarios: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Estimate probabilities for multiple scenarios.

        Args:
            features: Current feature values
            scenarios: Return thresholds for scenarios

        Returns:
            Dictionary of scenario probabilities
        """
        if scenarios is None:
            scenarios = {
                "strong_up": 0.10,     # >10% return
                "moderate_up": 0.05,   # >5% return
                "slight_up": 0.02,     # >2% return
                "slight_down": -0.02,  # <-2% return
                "moderate_down": -0.05,  # <-5% return
                "strong_down": -0.10,  # <-10% return
            }

        # Get base probability
        base = self.estimator.estimate(features)
        prob_up = base.get("prob_up", 0.5)

        # Estimate scenario probabilities using normal approximation
        # These are rough estimates based on the base probability
        results = {}

        # Use the base probability to estimate the mean of returns
        # Map prob_up to expected return (rough approximation)
        expected_return = (prob_up - 0.5) * 0.2  # Scale to ±10%

        # Assume some volatility for distribution
        vol = 0.15 / (12**0.5)  # Monthly vol roughly

        import scipy.stats as stats

        for scenario_name, threshold in scenarios.items():
            if "up" in scenario_name:
                # Probability of return > threshold
                z = (threshold - expected_return) / vol
                prob = 1 - stats.norm.cdf(z)
            else:
                # Probability of return < threshold
                z = (threshold - expected_return) / vol
                prob = stats.norm.cdf(z)

            results[scenario_name] = {
                "threshold": threshold,
                "probability": prob,
            }

        return results

    def get_confidence_intervals(
        self,
        features: dict[str, float],
        confidence_levels: list[float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Get confidence intervals for probability estimate.

        Args:
            features: Current feature values
            confidence_levels: Confidence levels to compute

        Returns:
            Dictionary of confidence intervals
        """
        if confidence_levels is None:
            confidence_levels = [0.5, 0.8, 0.95]

        base = self.estimator.estimate(features)
        prob_up = base.get("prob_up", 0.5)
        confidence = base.get("confidence", 0.5)

        # Estimate uncertainty based on confidence
        # Higher confidence = narrower intervals
        uncertainty = (1 - confidence) * 0.2

        import scipy.stats as stats

        intervals = {}
        for level in confidence_levels:
            z = stats.norm.ppf((1 + level) / 2)
            margin = z * uncertainty

            intervals[f"ci_{int(level*100)}"] = {
                "lower": max(0, prob_up - margin),
                "upper": min(1, prob_up + margin),
                "level": level,
            }

        return intervals

    def compare_estimators(
        self,
        features: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """Compare estimates from different methods.

        Args:
            features: Current feature values

        Returns:
            Dictionary of estimates from each method
        """
        estimators = {
            "historical": HistoricalEstimator(),
            "bayesian": BayesianEstimator(),
            "ensemble": self.estimator,
        }

        results = {}
        for name, estimator in estimators.items():
            result = estimator.estimate(features)
            results[name] = {
                "prob_up": result.get("prob_up", 0.5),
                "confidence": result.get("confidence", 0.5),
            }

        # Add agreement measure
        probs = [r["prob_up"] for r in results.values()]
        results["agreement"] = {
            "mean": sum(probs) / len(probs),
            "std": (sum((p - sum(probs)/len(probs))**2 for p in probs) / len(probs))**0.5,
        }

        return results
