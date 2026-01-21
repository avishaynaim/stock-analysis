"""
Probability estimators for stock analysis.

Three approaches:
1. Historical/Empirical - Based on historical conditional frequencies
2. Bayesian - Incorporates prior beliefs and updates with evidence
3. Ensemble - Combines multiple estimators with calibration
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class BaseProbabilityEstimator(ABC):
    """Base class for probability estimators."""

    @abstractmethod
    def estimate(
        self,
        features: dict[str, float],
        historical_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Estimate probabilities.

        Args:
            features: Current feature values
            historical_data: Historical feature/outcome data for calibration

        Returns:
            Dictionary with probability estimates
        """
        pass

    @abstractmethod
    def calibrate(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
    ) -> None:
        """Calibrate estimator on historical data."""
        pass


class HistoricalEstimator(BaseProbabilityEstimator):
    """
    Historical/Empirical probability estimator.

    Estimates probabilities based on historical conditional frequencies.
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_samples: int = 30,
    ):
        """Initialize historical estimator.

        Args:
            n_bins: Number of bins for discretization
            min_samples: Minimum samples needed for reliable estimate
        """
        self.n_bins = n_bins
        self.min_samples = min_samples

        # Calibration data
        self._feature_bins: dict[str, np.ndarray] = {}
        self._conditional_probs: dict[str, np.ndarray] = {}
        self._base_rate: float = 0.5
        self._calibrated = False

    def calibrate(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
    ) -> None:
        """Calibrate on historical data.

        Computes conditional probabilities for each feature bin.
        """
        # Store base rate
        self._base_rate = outcomes.mean()

        for col in features.columns:
            try:
                # Create bins
                bins = np.percentile(
                    features[col].dropna(),
                    np.linspace(0, 100, self.n_bins + 1),
                )
                bins = np.unique(bins)  # Remove duplicates

                if len(bins) < 3:
                    continue

                self._feature_bins[col] = bins

                # Compute conditional probabilities
                bin_indices = np.digitize(features[col], bins[1:-1])
                probs = []

                for i in range(len(bins) - 1):
                    mask = bin_indices == i
                    if mask.sum() >= self.min_samples:
                        prob = outcomes[mask].mean()
                    else:
                        prob = self._base_rate  # Fall back to base rate

                    probs.append(prob)

                self._conditional_probs[col] = np.array(probs)

            except Exception as e:
                logger.debug(f"Failed to calibrate feature {col}: {e}")
                continue

        self._calibrated = True
        logger.info(
            f"Historical estimator calibrated on {len(self._feature_bins)} features"
        )

    def estimate(
        self,
        features: dict[str, float],
        historical_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Estimate probability based on historical frequencies."""
        if not self._calibrated:
            return {
                "prob_up": 0.5,
                "prob_down": 0.5,
                "confidence": 0.0,
                "method": "historical_uncalibrated",
            }

        feature_probs = []
        feature_count = 0

        for col, value in features.items():
            if col not in self._feature_bins or np.isnan(value):
                continue

            bins = self._feature_bins[col]
            probs = self._conditional_probs[col]

            # Find which bin the value falls into
            bin_idx = np.digitize(value, bins[1:-1])
            bin_idx = min(bin_idx, len(probs) - 1)  # Handle edge case

            feature_probs.append(probs[bin_idx])
            feature_count += 1

        if not feature_probs:
            return {
                "prob_up": self._base_rate,
                "prob_down": 1 - self._base_rate,
                "confidence": 0.1,
                "method": "historical_base_rate",
            }

        # Average probability across features
        prob_up = np.mean(feature_probs)

        # Confidence based on number of features and deviation from 0.5
        confidence = min(
            feature_count / 20,  # More features = more confidence
            abs(prob_up - 0.5) * 2,  # Stronger signal = more confidence
            1.0,
        )

        return {
            "prob_up": prob_up,
            "prob_down": 1 - prob_up,
            "confidence": confidence,
            "n_features": feature_count,
            "method": "historical",
        }


class BayesianEstimator(BaseProbabilityEstimator):
    """
    Bayesian probability estimator.

    Uses prior beliefs and updates with observed evidence.
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        learning_rate: float = 0.1,
    ):
        """Initialize Bayesian estimator.

        Args:
            prior_alpha: Alpha parameter for Beta prior (successes + 1)
            prior_beta: Beta parameter for Beta prior (failures + 1)
            learning_rate: How much to weight recent observations
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.learning_rate = learning_rate

        # Posterior parameters
        self._alpha = prior_alpha
        self._beta = prior_beta

        # Feature likelihoods
        self._feature_likelihoods: dict[str, dict] = {}
        self._calibrated = False

    def calibrate(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
    ) -> None:
        """Calibrate on historical data.

        Estimates likelihood of features given outcomes.
        """
        # Update posterior with historical success rate
        successes = outcomes.sum()
        failures = len(outcomes) - successes

        self._alpha = self.prior_alpha + successes * self.learning_rate
        self._beta = self.prior_beta + failures * self.learning_rate

        # Compute feature likelihoods P(feature | outcome)
        for col in features.columns:
            try:
                # Compute mean and std for each outcome class
                up_mask = outcomes == 1
                down_mask = outcomes == 0

                up_mean = features.loc[up_mask, col].mean()
                up_std = features.loc[up_mask, col].std()
                down_mean = features.loc[down_mask, col].mean()
                down_std = features.loc[down_mask, col].std()

                # Only store if distributions are meaningfully different
                if not (np.isnan(up_mean) or np.isnan(down_mean)):
                    self._feature_likelihoods[col] = {
                        "up_mean": up_mean,
                        "up_std": max(up_std, 0.01),
                        "down_mean": down_mean,
                        "down_std": max(down_std, 0.01),
                    }

            except Exception as e:
                logger.debug(f"Failed to compute likelihood for {col}: {e}")

        self._calibrated = True
        logger.info(
            f"Bayesian estimator calibrated with {len(self._feature_likelihoods)} features"
        )

    def estimate(
        self,
        features: dict[str, float],
        historical_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Estimate probability using Bayesian updating."""
        # Prior probability from Beta distribution
        prior_up = self._alpha / (self._alpha + self._beta)

        if not self._calibrated or not self._feature_likelihoods:
            return {
                "prob_up": prior_up,
                "prob_down": 1 - prior_up,
                "confidence": 0.1,
                "method": "bayesian_prior",
            }

        # Compute likelihood ratio for each feature
        log_likelihood_ratio = 0.0
        n_features = 0

        for col, value in features.items():
            if col not in self._feature_likelihoods or np.isnan(value):
                continue

            lik = self._feature_likelihoods[col]

            # P(feature | up)
            p_feature_up = stats.norm.pdf(value, lik["up_mean"], lik["up_std"])
            # P(feature | down)
            p_feature_down = stats.norm.pdf(value, lik["down_mean"], lik["down_std"])

            # Avoid division by zero and log of zero
            if p_feature_down > 1e-10 and p_feature_up > 1e-10:
                log_likelihood_ratio += np.log(p_feature_up / p_feature_down)
                n_features += 1

        # Apply Bayes rule: P(up|features) ∝ P(features|up) * P(up)
        # Using log-odds for numerical stability
        log_prior_odds = np.log(prior_up / (1 - prior_up + 1e-10))
        log_posterior_odds = log_prior_odds + log_likelihood_ratio

        # Convert back to probability
        posterior_up = 1 / (1 + np.exp(-log_posterior_odds))

        # Clip to reasonable range
        posterior_up = np.clip(posterior_up, 0.01, 0.99)

        # Confidence based on evidence strength
        confidence = min(
            n_features / 15,  # More features = more confidence
            abs(log_likelihood_ratio) / 5,  # Stronger evidence = more confidence
            1.0,
        )

        return {
            "prob_up": posterior_up,
            "prob_down": 1 - posterior_up,
            "confidence": confidence,
            "log_likelihood_ratio": log_likelihood_ratio,
            "n_features": n_features,
            "method": "bayesian",
        }


class EnsembleEstimator(BaseProbabilityEstimator):
    """
    Ensemble probability estimator.

    Combines multiple estimators with calibration.
    """

    def __init__(
        self,
        estimators: list[BaseProbabilityEstimator] | None = None,
        weights: list[float] | None = None,
        use_isotonic_calibration: bool = True,
    ):
        """Initialize ensemble estimator.

        Args:
            estimators: List of estimators to combine
            weights: Weights for each estimator
            use_isotonic_calibration: Whether to apply isotonic calibration
        """
        if estimators is None:
            estimators = [
                HistoricalEstimator(),
                BayesianEstimator(),
            ]

        self.estimators = estimators
        self.weights = weights or [1.0 / len(estimators)] * len(estimators)
        self.use_isotonic_calibration = use_isotonic_calibration

        # Calibration map
        self._calibration_bins: np.ndarray | None = None
        self._calibration_probs: np.ndarray | None = None
        self._calibrated = False

    def calibrate(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
    ) -> None:
        """Calibrate all estimators and the ensemble."""
        # Calibrate individual estimators
        for estimator in self.estimators:
            estimator.calibrate(features, outcomes)

        # If using isotonic calibration, compute mapping
        if self.use_isotonic_calibration:
            self._calibrate_isotonic(features, outcomes)

        self._calibrated = True
        logger.info("Ensemble estimator calibrated")

    def _calibrate_isotonic(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
    ) -> None:
        """Apply isotonic calibration to ensemble predictions."""
        # Get ensemble predictions for training data
        predictions = []

        for idx in range(len(features)):
            row_features = features.iloc[idx].to_dict()
            prob = self._get_raw_probability(row_features)
            predictions.append(prob)

        predictions = np.array(predictions)

        # Create calibration bins
        n_bins = 10
        bin_boundaries = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
        bin_boundaries = np.unique(bin_boundaries)

        self._calibration_bins = bin_boundaries
        self._calibration_probs = np.zeros(len(bin_boundaries) - 1)

        # Compute actual probability in each bin
        bin_indices = np.digitize(predictions, bin_boundaries[1:-1])

        for i in range(len(bin_boundaries) - 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                self._calibration_probs[i] = outcomes[mask].mean()
            else:
                self._calibration_probs[i] = 0.5

    def _get_raw_probability(self, features: dict[str, float]) -> float:
        """Get uncalibrated ensemble probability."""
        probs = []

        for estimator, weight in zip(self.estimators, self.weights):
            result = estimator.estimate(features)
            prob = result.get("prob_up", 0.5)
            probs.append(prob * weight)

        return sum(probs)

    def _apply_calibration(self, raw_prob: float) -> float:
        """Apply isotonic calibration."""
        if self._calibration_bins is None:
            return raw_prob

        bin_idx = np.digitize(raw_prob, self._calibration_bins[1:-1])
        bin_idx = min(bin_idx, len(self._calibration_probs) - 1)

        return self._calibration_probs[bin_idx]

    def estimate(
        self,
        features: dict[str, float],
        historical_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Estimate probability using ensemble."""
        # Get individual estimates
        estimates = []
        confidences = []

        for estimator, weight in zip(self.estimators, self.weights):
            result = estimator.estimate(features, historical_data)
            estimates.append(result)
            confidences.append(result.get("confidence", 0.5) * weight)

        # Weighted average
        raw_prob = sum(
            est.get("prob_up", 0.5) * w
            for est, w in zip(estimates, self.weights)
        )

        # Apply calibration if available
        if self.use_isotonic_calibration and self._calibration_bins is not None:
            prob_up = self._apply_calibration(raw_prob)
        else:
            prob_up = raw_prob

        # Combined confidence
        confidence = sum(confidences) / sum(self.weights)

        return {
            "prob_up": prob_up,
            "prob_down": 1 - prob_up,
            "raw_prob": raw_prob,
            "confidence": confidence,
            "individual_estimates": [
                {"method": est.get("method"), "prob": est.get("prob_up")}
                for est in estimates
            ],
            "method": "ensemble",
        }


class MLProbabilityEstimator(BaseProbabilityEstimator):
    """
    Machine learning based probability estimator.

    Uses trained ML models for probability estimation.
    """

    def __init__(self, model: Any = None):
        """Initialize ML estimator.

        Args:
            model: Trained sklearn model with predict_proba method
        """
        self.model = model
        self._feature_names: list[str] = []
        self._calibrated = False

    def calibrate(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
    ) -> None:
        """Train model on historical data."""
        from sklearn.ensemble import GradientBoostingClassifier

        if self.model is None:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )

        # Store feature names
        self._feature_names = list(features.columns)

        # Train model
        X = features.fillna(0)
        y = outcomes.astype(int)

        self.model.fit(X, y)
        self._calibrated = True

        logger.info(f"ML estimator trained on {len(features)} samples")

    def estimate(
        self,
        features: dict[str, float],
        historical_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Estimate probability using ML model."""
        if not self._calibrated or self.model is None:
            return {
                "prob_up": 0.5,
                "prob_down": 0.5,
                "confidence": 0.0,
                "method": "ml_uncalibrated",
            }

        # Create feature vector
        X = np.array([
            features.get(name, 0) for name in self._feature_names
        ]).reshape(1, -1)

        # Replace NaN with 0
        X = np.nan_to_num(X, nan=0)

        # Get probability
        proba = self.model.predict_proba(X)[0]

        # Assuming class 1 is "up"
        prob_up = proba[1] if len(proba) > 1 else proba[0]

        return {
            "prob_up": prob_up,
            "prob_down": 1 - prob_up,
            "confidence": abs(prob_up - 0.5) * 2,
            "method": "ml",
        }
