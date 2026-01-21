"""
Main stock scorer that orchestrates all scoring components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from stock_analysis.features.engineer import FeatureEngineer
from stock_analysis.indicators.engine import IndicatorEngine
from stock_analysis.probability.engine import ProbabilityEngine
from stock_analysis.scoring.components import (
    CompositeScore,
    MomentumScore,
    RiskScore,
    TechnicalScore,
)

logger = logging.getLogger(__name__)


@dataclass
class StockAnalysis:
    """Complete analysis result for a stock."""

    symbol: str
    date: str
    price: float

    # Scores
    composite_score: CompositeScore
    technical_score: TechnicalScore
    momentum_score: MomentumScore
    risk_score: RiskScore

    # Probability estimates
    probability: dict[str, Any]

    # Raw data
    indicators: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "date": self.date,
            "price": self.price,
            "composite_score": self.composite_score.value,
            "composite_rating": self.composite_score.components.get("rating", "N/A"),
            "composite_interpretation": self.composite_score.interpretation,
            "technical_score": self.technical_score.value,
            "technical_interpretation": self.technical_score.interpretation,
            "momentum_score": self.momentum_score.value,
            "momentum_interpretation": self.momentum_score.interpretation,
            "risk_score": self.risk_score.value,
            "risk_interpretation": self.risk_score.interpretation,
            "prob_up": self.probability.get("prob_up", 0.5),
            "prob_confidence": self.probability.get("confidence", 0),
            "signal": self.probability.get("signal", "neutral"),
        }

    def summary(self) -> str:
        """Get text summary of analysis."""
        lines = [
            f"=== {self.symbol} Analysis ({self.date}) ===",
            f"Price: ${self.price:.2f}",
            "",
            f"COMPOSITE SCORE: {self.composite_score.value:.1f}/100",
            f"Rating: {self.composite_score.components.get('rating', 'N/A')}",
            f"Signal: {self.composite_score.interpretation}",
            "",
            "Component Scores:",
            f"  Technical: {self.technical_score.value:.1f} - {self.technical_score.interpretation}",
            f"  Momentum:  {self.momentum_score.value:.1f} - {self.momentum_score.interpretation}",
            f"  Risk:      {self.risk_score.value:.1f} - {self.risk_score.interpretation}",
            "",
            "Probability Estimate:",
            f"  P(Up):       {self.probability.get('prob_up', 0.5)*100:.1f}%",
            f"  Confidence:  {self.probability.get('confidence', 0)*100:.1f}%",
            f"  Signal:      {self.probability.get('signal', 'neutral')}",
        ]
        return "\n".join(lines)


class StockScorer:
    """
    Main stock scoring engine.

    Orchestrates indicators, features, probability estimation, and scoring.
    """

    def __init__(
        self,
        indicator_engine: IndicatorEngine | None = None,
        feature_engineer: FeatureEngineer | None = None,
        probability_engine: ProbabilityEngine | None = None,
        score_weights: dict[str, float] | None = None,
    ):
        """Initialize stock scorer.

        Args:
            indicator_engine: Engine for computing indicators
            feature_engineer: Feature engineering pipeline
            probability_engine: Probability estimation engine
            score_weights: Weights for composite score components
        """
        self.indicator_engine = indicator_engine or IndicatorEngine()
        self.feature_engineer = feature_engineer or FeatureEngineer(
            indicator_engine=self.indicator_engine
        )
        self.probability_engine = probability_engine or ProbabilityEngine()

        self.score_weights = score_weights or {
            "technical": 0.30,
            "momentum": 0.30,
            "risk": 0.20,
            "probability": 0.20,
        }

    def analyze(
        self,
        symbol: str,
        prices: pd.DataFrame,
        benchmark_prices: pd.DataFrame | None = None,
    ) -> StockAnalysis:
        """Perform complete analysis on a stock.

        Args:
            symbol: Stock symbol
            prices: OHLCV price data
            benchmark_prices: Optional benchmark data

        Returns:
            StockAnalysis object with all scores and probabilities
        """
        # Compute features (includes indicators)
        features = self.feature_engineer.compute_features(prices, benchmark_prices)

        # Auto-calibrate probability engine if not already calibrated
        if not self.probability_engine._calibrated and len(prices) > 100:
            self._auto_calibrate(prices, benchmark_prices)

        # Get probability estimate
        probability = self.probability_engine.estimate_probabilities(features)

        # Compute scores
        technical_score = TechnicalScore.compute(
            features, self.score_weights.get("technical", 0.3)
        )
        momentum_score = MomentumScore.compute(
            features, self.score_weights.get("momentum", 0.3)
        )
        risk_score = RiskScore.compute(
            features, self.score_weights.get("risk", 0.2)
        )
        composite_score = CompositeScore.compute(
            features, probability, self.score_weights
        )

        # Get current price and date
        current_price = prices["adj_close"].iloc[-1]
        current_date = str(prices.index[-1].date() if hasattr(prices.index[-1], "date") else prices.index[-1])

        return StockAnalysis(
            symbol=symbol,
            date=current_date,
            price=current_price,
            composite_score=composite_score,
            technical_score=technical_score,
            momentum_score=momentum_score,
            risk_score=risk_score,
            probability=probability,
            indicators=features,
            features=features,
        )

    def _auto_calibrate(
        self,
        prices: pd.DataFrame,
        benchmark_prices: pd.DataFrame | None = None,
        target_horizon: int = 21,
    ) -> None:
        """Auto-calibrate probability engine on provided data.

        Uses indicators computed as time series for efficient calibration.

        Args:
            prices: Historical price data
            benchmark_prices: Optional benchmark data
            target_horizon: Days ahead for outcome calculation
        """
        import numpy as np

        # Get indicator time series (much faster than point-by-point)
        indicator_df = self.indicator_engine.compute_all(prices, benchmark_prices, return_dataframe=True)

        # Calculate forward returns and binary outcome
        close = prices["adj_close"]
        forward_return = close.shift(-target_horizon) / close - 1
        outcome = (forward_return > 0).astype(int)

        # Align data - use only rows where we have both indicators and valid outcome
        common_idx = indicator_df.index.intersection(outcome.dropna().index)
        if len(common_idx) < 50:
            logger.warning("Insufficient aligned data for calibration")
            return

        aligned_features = indicator_df.loc[common_idx]
        aligned_outcome = outcome.loc[common_idx]

        # Remove rows with NaN in features
        valid_mask = ~aligned_features.isna().any(axis=1)
        aligned_features = aligned_features[valid_mask]
        aligned_outcome = aligned_outcome[valid_mask]

        if len(aligned_features) < 50:
            logger.warning("Insufficient valid data for calibration")
            return

        # Remove non-numeric columns
        numeric_features = aligned_features.select_dtypes(include=[np.number])

        # Calibrate the probability estimator directly (bypass engine's calibrate which expects prices)
        self.probability_engine.estimator.calibrate(numeric_features, aligned_outcome)
        self.probability_engine._calibrated = True
        logger.info(f"Auto-calibrated probability engine on {len(aligned_features)} samples")

    def analyze_universe(
        self,
        symbols: list[str],
        price_data: dict[str, pd.DataFrame],
        benchmark_prices: pd.DataFrame | None = None,
    ) -> list[StockAnalysis]:
        """Analyze multiple stocks.

        Args:
            symbols: List of stock symbols
            price_data: Dictionary of symbol to price DataFrame
            benchmark_prices: Optional benchmark data

        Returns:
            List of StockAnalysis objects
        """
        results = []

        for symbol in symbols:
            if symbol not in price_data:
                logger.warning(f"No price data for {symbol}, skipping")
                continue

            try:
                analysis = self.analyze(symbol, price_data[symbol], benchmark_prices)
                results.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                continue

        return results

    def rank_universe(
        self,
        analyses: list[StockAnalysis],
        sort_by: str = "composite_score",
        ascending: bool = False,
    ) -> list[StockAnalysis]:
        """Rank analyzed stocks by specified criteria.

        Args:
            analyses: List of stock analyses
            sort_by: Field to sort by
            ascending: Sort order

        Returns:
            Sorted list of analyses
        """
        def get_sort_key(a: StockAnalysis) -> float:
            if sort_by == "composite_score":
                return a.composite_score.value
            elif sort_by == "technical_score":
                return a.technical_score.value
            elif sort_by == "momentum_score":
                return a.momentum_score.value
            elif sort_by == "risk_score":
                return a.risk_score.value
            elif sort_by == "probability":
                return a.probability.get("prob_up", 0.5)
            else:
                return a.composite_score.value

        return sorted(analyses, key=get_sort_key, reverse=not ascending)

    def screen(
        self,
        analyses: list[StockAnalysis],
        min_composite: float | None = None,
        min_technical: float | None = None,
        min_momentum: float | None = None,
        min_risk: float | None = None,
        min_probability: float | None = None,
        max_results: int | None = None,
    ) -> list[StockAnalysis]:
        """Screen stocks based on criteria.

        Args:
            analyses: List of stock analyses
            min_composite: Minimum composite score
            min_technical: Minimum technical score
            min_momentum: Minimum momentum score
            min_risk: Minimum risk score
            min_probability: Minimum probability of up
            max_results: Maximum number of results

        Returns:
            Filtered and sorted list of analyses
        """
        filtered = analyses.copy()

        if min_composite is not None:
            filtered = [a for a in filtered if a.composite_score.value >= min_composite]

        if min_technical is not None:
            filtered = [a for a in filtered if a.technical_score.value >= min_technical]

        if min_momentum is not None:
            filtered = [a for a in filtered if a.momentum_score.value >= min_momentum]

        if min_risk is not None:
            filtered = [a for a in filtered if a.risk_score.value >= min_risk]

        if min_probability is not None:
            filtered = [
                a for a in filtered
                if a.probability.get("prob_up", 0) >= min_probability
            ]

        # Sort by composite score
        filtered = self.rank_universe(filtered, "composite_score")

        # Limit results
        if max_results is not None:
            filtered = filtered[:max_results]

        return filtered

    def calibrate(
        self,
        training_data: dict[str, pd.DataFrame],
        benchmark_prices: pd.DataFrame | None = None,
        target_horizon: int = 21,
    ) -> None:
        """Calibrate the scorer on historical data.

        Args:
            training_data: Dictionary of symbol to historical price data
            benchmark_prices: Optional benchmark data
            target_horizon: Forward-looking horizon for outcome
        """
        # Collect all features and outcomes
        all_features = []
        all_prices = []

        for symbol, prices in training_data.items():
            try:
                # Compute features for each date
                features_df = self.feature_engineer.compute_features_timeseries(
                    prices, benchmark_prices
                )
                all_features.append(features_df)
                all_prices.append(prices)
            except Exception as e:
                logger.warning(f"Failed to process {symbol} for calibration: {e}")

        if not all_features:
            logger.warning("No data available for calibration")
            return

        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_prices = pd.concat(all_prices, ignore_index=True)

        # Calibrate probability engine
        self.probability_engine.calibrate(
            combined_features, combined_prices, target_horizon
        )

        logger.info(f"Scorer calibrated on {len(combined_features)} samples")

    def to_dataframe(self, analyses: list[StockAnalysis]) -> pd.DataFrame:
        """Convert analyses to DataFrame.

        Args:
            analyses: List of stock analyses

        Returns:
            DataFrame with analysis results
        """
        records = [a.to_dict() for a in analyses]
        return pd.DataFrame(records)
