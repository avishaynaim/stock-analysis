"""
Scoring components for different aspects of stock analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ScoreComponent:
    """Base class for score components."""

    name: str = ""
    value: float = 50.0  # 0-100 scale
    weight: float = 1.0
    components: dict[str, float] = field(default_factory=dict)
    interpretation: str = ""

    def weighted_value(self) -> float:
        """Get weighted score value."""
        return self.value * self.weight


@dataclass
class TechnicalScore(ScoreComponent):
    """Technical analysis score component."""

    name: str = "technical"

    @classmethod
    def compute(cls, indicators: dict[str, Any], weight: float = 1.0) -> "TechnicalScore":
        """Compute technical score from indicators."""
        components = {}
        scores = []

        # Trend score (0-100)
        trend_signals = []

        # EMA alignment
        if all(k in indicators for k in ["ema_8", "ema_21", "ema_50"]):
            if indicators["ema_8"] > indicators["ema_21"] > indicators["ema_50"]:
                trend_signals.append(100)  # Perfect bullish alignment
            elif indicators["ema_8"] < indicators["ema_21"] < indicators["ema_50"]:
                trend_signals.append(0)  # Perfect bearish alignment
            else:
                trend_signals.append(50)  # Mixed

        # MACD
        if "macd_histogram" in indicators:
            hist = indicators["macd_histogram"]
            if hist > 0:
                trend_signals.append(min(100, 50 + hist * 100))
            else:
                trend_signals.append(max(0, 50 + hist * 100))

        # ADX trend strength
        if "adx" in indicators:
            adx = indicators["adx"]
            plus_di = indicators.get("plus_di", 0)
            minus_di = indicators.get("minus_di", 0)

            if adx > 25 and plus_di > minus_di:
                trend_signals.append(min(100, 50 + adx))
            elif adx > 25 and minus_di > plus_di:
                trend_signals.append(max(0, 50 - adx))
            else:
                trend_signals.append(50)

        if trend_signals:
            components["trend"] = np.mean(trend_signals)
            scores.append(components["trend"])

        # Momentum score
        momentum_signals = []

        if "rsi" in indicators:
            rsi = indicators["rsi"]
            if rsi < 30:
                momentum_signals.append(30)  # Oversold
            elif rsi > 70:
                momentum_signals.append(70)  # Overbought
            else:
                momentum_signals.append(rsi)

        if "stoch_k" in indicators:
            stoch = indicators["stoch_k"]
            momentum_signals.append(stoch)

        if momentum_signals:
            components["momentum"] = np.mean(momentum_signals)
            scores.append(components["momentum"])

        # Volatility score (inverse - lower vol = higher score)
        if "atr_pct" in indicators:
            atr_pct = indicators["atr_pct"]
            # Map ATR% to score (lower ATR = higher score)
            vol_score = max(0, min(100, 100 - atr_pct * 1000))
            components["volatility"] = vol_score
            scores.append(vol_score)

        # Volume confirmation
        if "volume_sma_ratio" in indicators:
            vol_ratio = indicators["volume_sma_ratio"]
            # Higher volume on up days is good
            if indicators.get("is_bullish_candle", 0) and vol_ratio > 1:
                components["volume"] = min(100, 50 + vol_ratio * 25)
            elif not indicators.get("is_bullish_candle", 0) and vol_ratio > 1:
                components["volume"] = max(0, 50 - vol_ratio * 25)
            else:
                components["volume"] = 50
            scores.append(components["volume"])

        # Calculate final score
        final_score = np.mean(scores) if scores else 50

        # Interpretation
        if final_score >= 70:
            interpretation = "Strong technical setup"
        elif final_score >= 55:
            interpretation = "Moderately bullish technicals"
        elif final_score >= 45:
            interpretation = "Neutral technical picture"
        elif final_score >= 30:
            interpretation = "Moderately bearish technicals"
        else:
            interpretation = "Weak technical setup"

        return cls(
            name="technical",
            value=final_score,
            weight=weight,
            components=components,
            interpretation=interpretation,
        )


@dataclass
class MomentumScore(ScoreComponent):
    """Momentum and relative strength score."""

    name: str = "momentum"

    @classmethod
    def compute(cls, indicators: dict[str, Any], weight: float = 1.0) -> "MomentumScore":
        """Compute momentum score."""
        components = {}
        scores = []

        # Price momentum (returns)
        momentum_periods = [
            ("return_5d", 0.15),
            ("return_21d", 0.25),
            ("return_63d", 0.30),
            ("return_252d", 0.30),
        ]

        for key, period_weight in momentum_periods:
            if key in indicators:
                ret = indicators[key]
                # Convert return to 0-100 score
                # Roughly: -20% = 0, 0% = 50, +20% = 100
                score = max(0, min(100, 50 + ret * 250))
                components[key] = score
                scores.append(score * period_weight)

        # Relative strength
        if "rs_composite" in indicators:
            rs = indicators["rs_composite"]
            # Convert RS to score
            rs_score = max(0, min(100, 50 + rs * 5))
            components["relative_strength"] = rs_score
            scores.append(rs_score * 0.2)

        # Momentum consistency
        if "momentum_consistency" in indicators:
            consistency = indicators["momentum_consistency"]
            components["consistency"] = consistency * 100
            scores.append(consistency * 100 * 0.1)

        # Calculate final score
        total_weight = sum(w for _, w in momentum_periods) + 0.3
        final_score = sum(scores) / total_weight if scores else 50

        # Interpretation
        if final_score >= 70:
            interpretation = "Strong positive momentum"
        elif final_score >= 55:
            interpretation = "Building momentum"
        elif final_score >= 45:
            interpretation = "Neutral momentum"
        elif final_score >= 30:
            interpretation = "Weakening momentum"
        else:
            interpretation = "Strong negative momentum"

        return cls(
            name="momentum",
            value=final_score,
            weight=weight,
            components=components,
            interpretation=interpretation,
        )


@dataclass
class RiskScore(ScoreComponent):
    """Risk assessment score (higher = lower risk)."""

    name: str = "risk"

    @classmethod
    def compute(cls, indicators: dict[str, Any], weight: float = 1.0) -> "RiskScore":
        """Compute risk score (higher score = lower risk)."""
        components = {}
        scores = []

        # Volatility (lower is better)
        if "annualized_volatility" in indicators:
            vol = indicators["annualized_volatility"]
            # Map vol to score (lower vol = higher score)
            vol_score = max(0, min(100, 100 - vol * 2))
            components["volatility"] = vol_score
            scores.append(vol_score)

        # Drawdown (smaller is better)
        if "current_drawdown" in indicators:
            dd = abs(indicators["current_drawdown"])
            # Map drawdown to score
            dd_score = max(0, min(100, 100 - dd * 3))
            components["drawdown"] = dd_score
            scores.append(dd_score)

        # Risk-adjusted returns
        if "sharpe_ratio" in indicators:
            sharpe = indicators["sharpe_ratio"]
            # Map Sharpe to score
            sharpe_score = max(0, min(100, 50 + sharpe * 25))
            components["sharpe"] = sharpe_score
            scores.append(sharpe_score)

        if "sortino_ratio" in indicators:
            sortino = indicators["sortino_ratio"]
            sortino_score = max(0, min(100, 50 + sortino * 20))
            components["sortino"] = sortino_score
            scores.append(sortino_score)

        # Volatility regime
        if "vol_regime_score" in indicators:
            regime = indicators["vol_regime_score"]
            # Lower regime = lower vol = higher score
            regime_score = 100 - regime * 25
            components["vol_regime"] = regime_score
            scores.append(regime_score)

        # Calculate final score
        final_score = np.mean(scores) if scores else 50

        # Interpretation
        if final_score >= 70:
            interpretation = "Low risk profile"
        elif final_score >= 55:
            interpretation = "Moderate risk"
        elif final_score >= 45:
            interpretation = "Average risk"
        elif final_score >= 30:
            interpretation = "Elevated risk"
        else:
            interpretation = "High risk profile"

        return cls(
            name="risk",
            value=final_score,
            weight=weight,
            components=components,
            interpretation=interpretation,
        )


@dataclass
class CompositeScore(ScoreComponent):
    """Combined composite score from all components."""

    name: str = "composite"
    sub_scores: list[ScoreComponent] = field(default_factory=list)

    @classmethod
    def compute(
        cls,
        indicators: dict[str, Any],
        probability: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ) -> "CompositeScore":
        """Compute composite score from all components."""
        if weights is None:
            weights = {
                "technical": 0.30,
                "momentum": 0.30,
                "risk": 0.20,
                "probability": 0.20,
            }

        sub_scores = []
        components = {}

        # Technical score
        tech_score = TechnicalScore.compute(indicators, weights.get("technical", 0.3))
        sub_scores.append(tech_score)
        components["technical"] = tech_score.value

        # Momentum score
        mom_score = MomentumScore.compute(indicators, weights.get("momentum", 0.3))
        sub_scores.append(mom_score)
        components["momentum"] = mom_score.value

        # Risk score
        risk_score = RiskScore.compute(indicators, weights.get("risk", 0.2))
        sub_scores.append(risk_score)
        components["risk"] = risk_score.value

        # Probability score (if available)
        if probability:
            prob_up = probability.get("prob_up", 0.5)
            prob_score = prob_up * 100
            components["probability"] = prob_score

            prob_component = ScoreComponent(
                name="probability",
                value=prob_score,
                weight=weights.get("probability", 0.2),
                interpretation=probability.get("signal", "neutral"),
            )
            sub_scores.append(prob_component)

        # Calculate weighted composite
        total_weight = sum(s.weight for s in sub_scores)
        final_score = (
            sum(s.weighted_value() for s in sub_scores) / total_weight
            if total_weight > 0
            else 50
        )

        # Overall interpretation
        if final_score >= 75:
            interpretation = "Strong Buy"
            rating = "A"
        elif final_score >= 65:
            interpretation = "Buy"
            rating = "B+"
        elif final_score >= 55:
            interpretation = "Moderate Buy"
            rating = "B"
        elif final_score >= 45:
            interpretation = "Hold"
            rating = "C"
        elif final_score >= 35:
            interpretation = "Moderate Sell"
            rating = "D"
        elif final_score >= 25:
            interpretation = "Sell"
            rating = "D-"
        else:
            interpretation = "Strong Sell"
            rating = "F"

        components["rating"] = rating

        return cls(
            name="composite",
            value=final_score,
            weight=1.0,
            components=components,
            interpretation=interpretation,
            sub_scores=sub_scores,
        )
