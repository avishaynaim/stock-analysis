# Machine Learning & Statistical Models

## 1. Overview

This document defines all machine learning and statistical models used in the stock analysis system. The models complement the rule-based scoring system by providing data-driven probability estimates, pattern detection, and anomaly identification.

### Model Philosophy

1. **Interpretability First:** Prefer explainable models over black boxes
2. **Robustness Over Accuracy:** Small, stable improvements over overfitted perfection
3. **Ensemble Diversity:** Combine multiple approaches for robustness
4. **Temporal Integrity:** Strict prevention of lookahead bias
5. **Uncertainty Quantification:** All predictions include confidence measures

### Model Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL HIERARCHY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIER 1: RULE-BASED (Baseline)                                             │
│  ─────────────────────────────                                             │
│  • Deterministic, interpretable                                            │
│  • Always available, no training required                                  │
│  • Provides baseline for ML comparison                                     │
│                                                                             │
│  TIER 2: SUPERVISED LEARNING                                               │
│  ───────────────────────────                                               │
│  • Classification: Binary (high gain / not)                                │
│  • Regression: Expected return prediction                                  │
│  • Calibrated probability outputs                                          │
│                                                                             │
│  TIER 3: UNSUPERVISED LEARNING                                             │
│  ─────────────────────────────                                             │
│  • Clustering: Market regime detection                                     │
│  • Dimensionality reduction: Feature patterns                              │
│  • Anomaly detection: Unusual states                                       │
│                                                                             │
│  TIER 4: ENSEMBLE                                                          │
│  ───────────────                                                           │
│  • Combine all tiers for final prediction                                  │
│  • Weighted by validation performance                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Rule-Based Model

### 2.1 Overview

The rule-based model provides interpretable, deterministic predictions based on expert-defined rules. It serves as:
- Baseline for ML model comparison
- Fallback when ML models have insufficient data
- Explainable component of ensemble

### 2.2 Rule Engine Architecture

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

class RuleOutcome(Enum):
    STRONG_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    STRONG_BEARISH = -2


@dataclass
class RuleResult:
    """Result from a single rule evaluation."""
    rule_name: str
    outcome: RuleOutcome
    confidence: float           # 0-1
    explanation: str
    indicators_used: List[str]
    triggered: bool


@dataclass
class RuleSetResult:
    """Aggregated result from all rules."""
    bullish_score: float        # 0-10
    bearish_score: float        # 0-10
    net_score: float            # -10 to 10
    probability_estimate: float # 0-1
    confidence: float           # 0-1
    rules_triggered: List[RuleResult]
    explanation: str


class Rule(ABC):
    """Abstract base class for trading rules."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(self, state: Dict) -> RuleResult:
        """Evaluate rule against current state."""
        pass

    @property
    @abstractmethod
    def required_indicators(self) -> List[str]:
        """List of required indicators."""
        pass
```

### 2.3 Rule Definitions

```python
class RSIOversoldRule(Rule):
    """
    RSI Oversold Bounce Rule

    Logic: Deeply oversold RSI in uptrending market suggests bounce potential.

    Conditions:
    - RSI < 30 (oversold)
    - Price above 200 EMA (long-term uptrend)
    - Volume elevated (capitulation)
    """

    def __init__(self):
        super().__init__("RSI_Oversold_Bounce", weight=1.5)

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'ema_200_pct', 'volume_ratio']

    def evaluate(self, state: Dict) -> RuleResult:
        rsi = state.get('rsi', 50)
        ema_200_pct = state.get('ema_200_pct', 0)
        volume_ratio = state.get('volume_ratio', 1.0)

        # Check conditions
        oversold = rsi < 30
        uptrend = ema_200_pct > 0
        elevated_volume = volume_ratio > 1.5

        if oversold and uptrend and elevated_volume:
            outcome = RuleOutcome.STRONG_BULLISH
            confidence = min(1.0, (30 - rsi) / 15 * 0.5 + volume_ratio / 4 * 0.5)
            explanation = f"RSI={rsi:.0f} oversold, above 200 EMA, volume {volume_ratio:.1f}x"
            triggered = True
        elif oversold and uptrend:
            outcome = RuleOutcome.BULLISH
            confidence = 0.6
            explanation = f"RSI={rsi:.0f} oversold, above 200 EMA"
            triggered = True
        elif oversold:
            outcome = RuleOutcome.NEUTRAL
            confidence = 0.4
            explanation = f"RSI={rsi:.0f} oversold but trend unclear"
            triggered = True
        else:
            outcome = RuleOutcome.NEUTRAL
            confidence = 0.0
            explanation = "RSI not oversold"
            triggered = False

        return RuleResult(
            rule_name=self.name,
            outcome=outcome,
            confidence=confidence,
            explanation=explanation,
            indicators_used=self.required_indicators,
            triggered=triggered
        )


class MACDCrossoverRule(Rule):
    """
    MACD Bullish Crossover Rule

    Logic: MACD crossing above signal line indicates momentum shift.

    Conditions:
    - MACD crosses above signal line
    - Histogram turning positive
    - ADX > 20 (trending market)
    """

    def __init__(self):
        super().__init__("MACD_Bullish_Crossover", weight=1.2)

    @property
    def required_indicators(self) -> List[str]:
        return ['macd', 'macd_signal', 'macd_histogram', 'macd_histogram_prev', 'adx']

    def evaluate(self, state: Dict) -> RuleResult:
        macd = state.get('macd', 0)
        signal = state.get('macd_signal', 0)
        histogram = state.get('macd_histogram', 0)
        histogram_prev = state.get('macd_histogram_prev', 0)
        adx = state.get('adx', 20)

        # Crossover detection
        bullish_cross = macd > signal and histogram > 0 and histogram_prev <= 0
        trending = adx > 20

        if bullish_cross and trending:
            outcome = RuleOutcome.BULLISH
            confidence = min(1.0, 0.6 + adx / 100 * 0.4)
            explanation = f"MACD bullish crossover, ADX={adx:.0f}"
            triggered = True
        elif bullish_cross:
            outcome = RuleOutcome.BULLISH
            confidence = 0.5
            explanation = "MACD bullish crossover, weak trend"
            triggered = True
        elif macd > signal:
            outcome = RuleOutcome.NEUTRAL
            confidence = 0.3
            explanation = "MACD above signal but no fresh crossover"
            triggered = False
        else:
            outcome = RuleOutcome.NEUTRAL
            confidence = 0.0
            explanation = "No MACD bullish signal"
            triggered = False

        return RuleResult(
            rule_name=self.name,
            outcome=outcome,
            confidence=confidence,
            explanation=explanation,
            indicators_used=self.required_indicators,
            triggered=triggered
        )


class TrendAlignmentRule(Rule):
    """
    Multi-Timeframe Trend Alignment Rule

    Logic: When all timeframe EMAs align bullishly, trend is strong.

    Conditions:
    - Price > EMA 8 > EMA 21 > EMA 50 > EMA 200
    - ADX > 25 (strong trend)
    """

    def __init__(self):
        super().__init__("Trend_Alignment", weight=1.8)

    @property
    def required_indicators(self) -> List[str]:
        return ['ema_8_pct', 'ema_21_pct', 'ema_50_pct', 'ema_200_pct', 'adx']

    def evaluate(self, state: Dict) -> RuleResult:
        emas = [
            state.get('ema_8_pct', 0),
            state.get('ema_21_pct', 0),
            state.get('ema_50_pct', 0),
            state.get('ema_200_pct', 0)
        ]
        adx = state.get('adx', 20)

        # Check alignment
        all_above = all(e > 0 for e in emas)
        properly_stacked = all(emas[i] >= emas[i+1] for i in range(len(emas)-1))
        strong_trend = adx > 25

        if all_above and properly_stacked and strong_trend:
            outcome = RuleOutcome.STRONG_BULLISH
            confidence = min(1.0, 0.7 + adx / 100 * 0.3)
            explanation = f"Perfect bullish EMA alignment, ADX={adx:.0f}"
            triggered = True
        elif all_above and properly_stacked:
            outcome = RuleOutcome.BULLISH
            confidence = 0.65
            explanation = "Bullish EMA alignment, moderate trend"
            triggered = True
        elif all_above:
            outcome = RuleOutcome.BULLISH
            confidence = 0.5
            explanation = "Price above all EMAs, not perfectly stacked"
            triggered = True
        elif all(e < 0 for e in emas):
            outcome = RuleOutcome.STRONG_BEARISH
            confidence = 0.7
            explanation = "Price below all EMAs - bearish"
            triggered = True
        else:
            outcome = RuleOutcome.NEUTRAL
            confidence = 0.3
            explanation = "Mixed EMA signals"
            triggered = False

        return RuleResult(
            rule_name=self.name,
            outcome=outcome,
            confidence=confidence,
            explanation=explanation,
            indicators_used=self.required_indicators,
            triggered=triggered
        )


class VolumeBreakoutRule(Rule):
    """
    Volume Breakout Rule

    Logic: Price breakout with high volume confirms move.

    Conditions:
    - Price at 20-day high
    - Volume > 2x average
    - CMF positive (accumulation)
    """

    def __init__(self):
        super().__init__("Volume_Breakout", weight=1.4)

    @property
    def required_indicators(self) -> List[str]:
        return ['range_position_20d', 'volume_ratio', 'cmf']

    def evaluate(self, state: Dict) -> RuleResult:
        range_pos = state.get('range_position_20d', 50)
        volume_ratio = state.get('volume_ratio', 1.0)
        cmf = state.get('cmf', 0)

        at_high = range_pos > 95
        high_volume = volume_ratio > 2.0
        accumulation = cmf > 0.1

        if at_high and high_volume and accumulation:
            outcome = RuleOutcome.STRONG_BULLISH
            confidence = min(1.0, volume_ratio / 4)
            explanation = f"Breakout with {volume_ratio:.1f}x volume, CMF={cmf:.2f}"
            triggered = True
        elif at_high and high_volume:
            outcome = RuleOutcome.BULLISH
            confidence = 0.6
            explanation = f"Breakout with {volume_ratio:.1f}x volume"
            triggered = True
        elif at_high and accumulation:
            outcome = RuleOutcome.BULLISH
            confidence = 0.5
            explanation = "At highs with accumulation"
            triggered = True
        else:
            outcome = RuleOutcome.NEUTRAL
            confidence = 0.0
            explanation = "No volume breakout"
            triggered = False

        return RuleResult(
            rule_name=self.name,
            outcome=outcome,
            confidence=confidence,
            explanation=explanation,
            indicators_used=self.required_indicators,
            triggered=triggered
        )


class MeanReversionRule(Rule):
    """
    Mean Reversion Setup Rule

    Logic: Extreme deviation from mean with signs of reversal.

    Conditions:
    - Bollinger %B < 0 or > 1 (outside bands)
    - RSI divergence or extreme
    - Reversal candle pattern
    """

    def __init__(self):
        super().__init__("Mean_Reversion", weight=1.3)

    @property
    def required_indicators(self) -> List[str]:
        return ['bb_pct_b', 'rsi', 'body_ratio', 'candle_pattern']

    def evaluate(self, state: Dict) -> RuleResult:
        bb_pct_b = state.get('bb_pct_b', 0.5)
        rsi = state.get('rsi', 50)
        body_ratio = state.get('body_ratio', 0.5)
        candle = state.get('candle_pattern', 'none')

        below_lower = bb_pct_b < 0
        above_upper = bb_pct_b > 1
        oversold = rsi < 30
        overbought = rsi > 70
        reversal_candle = candle in ['hammer', 'engulfing_bullish', 'morning_star']

        # Bullish mean reversion
        if below_lower and oversold and reversal_candle:
            outcome = RuleOutcome.STRONG_BULLISH
            confidence = 0.75
            explanation = f"Below BB, RSI={rsi:.0f}, {candle} pattern"
            triggered = True
        elif below_lower and oversold:
            outcome = RuleOutcome.BULLISH
            confidence = 0.6
            explanation = f"Below BB with RSI={rsi:.0f}"
            triggered = True
        elif below_lower:
            outcome = RuleOutcome.BULLISH
            confidence = 0.4
            explanation = "Below lower Bollinger Band"
            triggered = True
        # Bearish mean reversion
        elif above_upper and overbought:
            outcome = RuleOutcome.BEARISH
            confidence = 0.6
            explanation = f"Above BB with RSI={rsi:.0f}"
            triggered = True
        else:
            outcome = RuleOutcome.NEUTRAL
            confidence = 0.0
            explanation = "No mean reversion setup"
            triggered = False

        return RuleResult(
            rule_name=self.name,
            outcome=outcome,
            confidence=confidence,
            explanation=explanation,
            indicators_used=self.required_indicators,
            triggered=triggered
        )
```

### 2.4 Rule Aggregator

```python
class RuleBasedModel:
    """
    Aggregate multiple rules into a unified prediction.
    """

    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules = rules or self._default_rules()

    def _default_rules(self) -> List[Rule]:
        return [
            RSIOversoldRule(),
            MACDCrossoverRule(),
            TrendAlignmentRule(),
            VolumeBreakoutRule(),
            MeanReversionRule(),
            # Add more rules as needed
        ]

    def predict(self, state: Dict) -> RuleSetResult:
        """
        Evaluate all rules and aggregate results.
        """
        results = []
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0

        for rule in self.rules:
            # Check required indicators are present
            missing = [ind for ind in rule.required_indicators if ind not in state]
            if missing:
                continue

            result = rule.evaluate(state)
            results.append(result)

            if result.triggered:
                weight = rule.weight * result.confidence

                if result.outcome in [RuleOutcome.STRONG_BULLISH, RuleOutcome.BULLISH]:
                    bullish_score += weight * (2 if result.outcome == RuleOutcome.STRONG_BULLISH else 1)
                elif result.outcome in [RuleOutcome.STRONG_BEARISH, RuleOutcome.BEARISH]:
                    bearish_score += weight * (2 if result.outcome == RuleOutcome.STRONG_BEARISH else 1)

                total_weight += rule.weight

        # Normalize scores to 0-10
        if total_weight > 0:
            max_possible = total_weight * 2  # Maximum if all strong signals
            bullish_score = min(10, bullish_score / max_possible * 10)
            bearish_score = min(10, bearish_score / max_possible * 10)
        else:
            bullish_score = 5.0
            bearish_score = 5.0

        net_score = bullish_score - bearish_score

        # Convert to probability estimate
        # Map net_score from [-10, 10] to [0.05, 0.50]
        probability = 0.05 + (net_score + 10) / 20 * 0.45

        # Confidence based on rule agreement
        triggered_rules = [r for r in results if r.triggered]
        if len(triggered_rules) >= 3:
            confidence = 0.8
        elif len(triggered_rules) >= 2:
            confidence = 0.6
        elif len(triggered_rules) >= 1:
            confidence = 0.4
        else:
            confidence = 0.2

        # Generate explanation
        top_rules = sorted(
            [r for r in results if r.triggered],
            key=lambda x: x.confidence,
            reverse=True
        )[:3]

        explanation = "; ".join(r.explanation for r in top_rules) if top_rules else "No rules triggered"

        return RuleSetResult(
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            net_score=net_score,
            probability_estimate=probability,
            confidence=confidence,
            rules_triggered=results,
            explanation=explanation
        )
```

---

## 3. Supervised Models

### 3.1 Classification Models

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


@dataclass
class ClassificationConfig:
    """Configuration for classification models."""

    # Target definition
    target_horizon: str = '21d'
    target_threshold: float = 0.10  # 10% gain

    # Model parameters
    model_type: str = 'gradient_boosting'
    calibration_method: str = 'isotonic'  # 'isotonic', 'sigmoid', or None

    # Training parameters
    class_weight: str = 'balanced'
    n_estimators: int = 100
    max_depth: int = 5
    min_samples_leaf: int = 50
    learning_rate: float = 0.05

    # Feature selection
    max_features: int = 30
    feature_selection_method: str = 'importance'


class HighGainClassifier:
    """
    Binary classifier for high-gain prediction.

    Predicts: P(return >= threshold | features)
    """

    def __init__(self, config: ClassificationConfig = None):
        self.config = config or ClassificationConfig()
        self.model = None
        self.calibrated_model = None
        self.feature_names = None
        self.feature_importance = None
        self.training_metrics = {}

    def _build_base_model(self) -> BaseEstimator:
        """Build base classifier."""
        if self.config.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                learning_rate=self.config.learning_rate,
                subsample=0.8,
                random_state=42
            )
        elif self.config.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight=self.config.class_weight,
                random_state=42,
                n_jobs=-1
            )
        elif self.config.model_type == 'logistic':
            return LogisticRegression(
                C=1.0,
                class_weight=self.config.class_weight,
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        sample_weight: Optional[np.ndarray] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict:
        """
        Train the classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary target (1 = high gain, 0 = not)
            feature_names: Names of features
            sample_weight: Optional sample weights
            validation_data: Optional (X_val, y_val) for evaluation

        Returns:
            Training metrics dictionary
        """
        self.feature_names = feature_names
        self.model = self._build_base_model()

        # Fit base model
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

        # Calibrate if requested
        if self.config.calibration_method:
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method=self.config.calibration_method,
                cv=5
            )
            self.calibrated_model.fit(X, y)
        else:
            self.calibrated_model = self.model

        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                feature_names,
                self.model.feature_importances_
            ))
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = dict(zip(
                feature_names,
                np.abs(self.model.coef_[0])
            ))

        # Compute training metrics
        self.training_metrics = self._compute_metrics(X, y, 'train')

        if validation_data:
            X_val, y_val = validation_data
            self.training_metrics.update(
                self._compute_metrics(X_val, y_val, 'val')
            )

        return self.training_metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of high gain.

        Returns:
            Array of probabilities
        """
        if self.calibrated_model is None:
            raise ValueError("Model not trained")

        return self.calibrated_model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def _compute_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prefix: str
    ) -> Dict:
        """Compute classification metrics."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            brier_score_loss, log_loss
        )

        proba = self.predict_proba(X)

        return {
            f'{prefix}_roc_auc': roc_auc_score(y, proba),
            f'{prefix}_avg_precision': average_precision_score(y, proba),
            f'{prefix}_brier_score': brier_score_loss(y, proba),
            f'{prefix}_log_loss': log_loss(y, proba),
            f'{prefix}_n_samples': len(y),
            f'{prefix}_positive_rate': y.mean()
        }

    def get_feature_importance(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if self.feature_importance is None:
            return []

        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: -x[1]
        )
        return sorted_features[:top_n]


class MultiHorizonClassifier:
    """
    Train separate classifiers for each time horizon.
    """

    def __init__(
        self,
        horizons: List[str] = None,
        base_config: ClassificationConfig = None
    ):
        self.horizons = horizons or ['5d', '21d', '63d']
        self.base_config = base_config or ClassificationConfig()
        self.classifiers: Dict[str, HighGainClassifier] = {}

    def fit(
        self,
        X: np.ndarray,
        forward_returns: pd.DataFrame,
        feature_names: List[str],
        thresholds: Dict[str, float] = None
    ) -> Dict[str, Dict]:
        """
        Train classifiers for all horizons.
        """
        thresholds = thresholds or {
            '5d': 0.05, '21d': 0.10, '63d': 0.15
        }

        all_metrics = {}

        for horizon in self.horizons:
            # Create target
            y = (forward_returns[f'fwd_return_{horizon}'] >= thresholds[horizon]).astype(int).values

            # Remove NaN
            valid_mask = ~np.isnan(forward_returns[f'fwd_return_{horizon}'].values)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            # Configure and train
            config = ClassificationConfig(
                target_horizon=horizon,
                target_threshold=thresholds[horizon],
                **{k: v for k, v in vars(self.base_config).items()
                   if k not in ['target_horizon', 'target_threshold']}
            )

            classifier = HighGainClassifier(config)
            metrics = classifier.fit(X_valid, y_valid, feature_names)

            self.classifiers[horizon] = classifier
            all_metrics[horizon] = metrics

        return all_metrics

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict probabilities for all horizons.
        """
        return {
            horizon: clf.predict_proba(X)
            for horizon, clf in self.classifiers.items()
        }
```

### 3.2 Regression Models

```python
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, HuberRegressor


@dataclass
class RegressionConfig:
    """Configuration for regression models."""

    target_horizon: str = '21d'
    model_type: str = 'gradient_boosting'

    # Model parameters
    n_estimators: int = 100
    max_depth: int = 4
    min_samples_leaf: int = 50
    learning_rate: float = 0.05

    # Target transformation
    target_transform: str = 'none'  # 'none', 'log', 'winsorize'
    winsorize_percentile: float = 0.01


class ReturnRegressor:
    """
    Regression model for expected return prediction.

    Predicts: E[return | features]
    """

    def __init__(self, config: RegressionConfig = None):
        self.config = config or RegressionConfig()
        self.model = None
        self.feature_names = None
        self.target_stats = {}

    def _build_model(self) -> BaseEstimator:
        """Build regression model."""
        if self.config.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                learning_rate=self.config.learning_rate,
                loss='huber',  # Robust to outliers
                random_state=42
            )
        elif self.config.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
        elif self.config.model_type == 'huber':
            return HuberRegressor(epsilon=1.35)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train the regressor.
        """
        self.feature_names = feature_names
        self.model = self._build_model()

        # Store target statistics
        self.target_stats = {
            'mean': np.mean(y),
            'std': np.std(y),
            'median': np.median(y),
            'min': np.min(y),
            'max': np.max(y)
        }

        # Transform target if needed
        y_transformed = self._transform_target(y)

        # Fit
        if sample_weight is not None and hasattr(self.model, 'fit'):
            try:
                self.model.fit(X, y_transformed, sample_weight=sample_weight)
            except TypeError:
                self.model.fit(X, y_transformed)
        else:
            self.model.fit(X, y_transformed)

        # Compute metrics
        y_pred = self.predict(X)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'ic': np.corrcoef(y, y_pred)[0, 1],  # Information coefficient
            'n_samples': len(y)
        }

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict expected returns.
        """
        if self.model is None:
            raise ValueError("Model not trained")

        predictions = self.model.predict(X)

        # Inverse transform if needed
        return self._inverse_transform(predictions)

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        """Apply target transformation."""
        if self.config.target_transform == 'winsorize':
            lower = np.percentile(y, self.config.winsorize_percentile * 100)
            upper = np.percentile(y, (1 - self.config.winsorize_percentile) * 100)
            return np.clip(y, lower, upper)
        elif self.config.target_transform == 'log':
            # Log transform for positive returns only
            return np.sign(y) * np.log1p(np.abs(y))
        return y

    def _inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse target transformation."""
        if self.config.target_transform == 'log':
            return np.sign(y) * (np.exp(np.abs(y)) - 1)
        return y

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_bootstrap: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with bootstrap uncertainty.

        Returns:
            (predictions, lower_bound, upper_bound)
        """
        if self.config.model_type not in ['random_forest', 'gradient_boosting']:
            # For models without built-in uncertainty
            pred = self.predict(X)
            std = self.target_stats['std'] * 0.5  # Rough estimate
            return pred, pred - 1.96 * std, pred + 1.96 * std

        if self.config.model_type == 'random_forest':
            # Use tree predictions
            all_preds = np.array([
                tree.predict(X)
                for tree in self.model.estimators_
            ])
            return (
                np.mean(all_preds, axis=0),
                np.percentile(all_preds, 5, axis=0),
                np.percentile(all_preds, 95, axis=0)
            )

        # Default: single prediction
        pred = self.predict(X)
        return pred, pred, pred
```

### 3.3 Ensemble Classifier

```python
class EnsembleClassifier:
    """
    Ensemble of multiple classification approaches.
    """

    def __init__(
        self,
        include_rule_based: bool = True,
        include_gradient_boosting: bool = True,
        include_random_forest: bool = True,
        include_logistic: bool = True
    ):
        self.models = {}
        self.weights = {}
        self.include_rule_based = include_rule_based

        if include_gradient_boosting:
            self.models['gradient_boosting'] = HighGainClassifier(
                ClassificationConfig(model_type='gradient_boosting')
            )
        if include_random_forest:
            self.models['random_forest'] = HighGainClassifier(
                ClassificationConfig(model_type='random_forest')
            )
        if include_logistic:
            self.models['logistic'] = HighGainClassifier(
                ClassificationConfig(model_type='logistic')
            )

        if include_rule_based:
            self.rule_model = RuleBasedModel()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train all models and compute ensemble weights.
        """
        all_metrics = {}

        for name, model in self.models.items():
            metrics = model.fit(
                X, y, feature_names,
                validation_data=(X_val, y_val) if X_val is not None else None
            )
            all_metrics[name] = metrics

        # Compute weights based on validation performance
        if X_val is not None and y_val is not None:
            self._compute_weights(X_val, y_val)
        else:
            # Equal weights
            n_models = len(self.models) + (1 if self.include_rule_based else 0)
            self.weights = {name: 1.0 / n_models for name in self.models}
            if self.include_rule_based:
                self.weights['rule_based'] = 1.0 / n_models

        all_metrics['weights'] = self.weights
        return all_metrics

    def _compute_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Compute weights based on validation AUC.
        """
        from sklearn.metrics import roc_auc_score

        aucs = {}
        for name, model in self.models.items():
            proba = model.predict_proba(X_val)
            aucs[name] = roc_auc_score(y_val, proba)

        # Rule-based gets fixed weight
        if self.include_rule_based:
            aucs['rule_based'] = 0.55  # Slight above random

        # Softmax weighting based on AUC
        auc_values = np.array(list(aucs.values()))
        # Exaggerate differences
        exp_aucs = np.exp((auc_values - 0.5) * 10)
        normalized = exp_aucs / exp_aucs.sum()

        self.weights = dict(zip(aucs.keys(), normalized))

    def predict_proba(
        self,
        X: np.ndarray,
        state_dicts: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """
        Ensemble probability prediction.

        Args:
            X: Feature matrix
            state_dicts: Optional state dictionaries for rule-based model
        """
        weighted_proba = np.zeros(len(X))

        for name, model in self.models.items():
            proba = model.predict_proba(X)
            weighted_proba += self.weights[name] * proba

        if self.include_rule_based and state_dicts:
            rule_proba = np.array([
                self.rule_model.predict(state).probability_estimate
                for state in state_dicts
            ])
            weighted_proba += self.weights.get('rule_based', 0) * rule_proba

        return weighted_proba
```

---

## 4. Unsupervised Clustering

### 4.1 Market Regime Clustering

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringConfig:
    """Configuration for clustering models."""

    n_clusters: int = 5
    method: str = 'gmm'  # 'kmeans', 'gmm', 'dbscan'

    # Feature selection
    features: List[str] = field(default_factory=lambda: [
        'hv_21_pctl',      # Volatility
        'adx_norm',        # Trend strength
        'rsi_norm',        # Momentum
        'volume_ratio',    # Activity
        'rs_vs_spy_21',    # Relative strength
    ])


class MarketRegimeClusterer:
    """
    Unsupervised clustering for market regime detection.

    Identifies distinct market states that may require
    different trading approaches.
    """

    REGIME_LABELS = {
        0: 'LOW_VOL_UPTREND',
        1: 'HIGH_VOL_UPTREND',
        2: 'LOW_VOL_RANGE',
        3: 'HIGH_VOL_DOWNTREND',
        4: 'CRISIS'
    }

    def __init__(self, config: ClusteringConfig = None):
        self.config = config or ClusteringConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.cluster_centers = None
        self.cluster_profiles = {}

    def _build_model(self):
        """Build clustering model."""
        if self.config.method == 'kmeans':
            return KMeans(
                n_clusters=self.config.n_clusters,
                random_state=42,
                n_init=10
            )
        elif self.config.method == 'gmm':
            return GaussianMixture(
                n_components=self.config.n_clusters,
                covariance_type='full',
                random_state=42,
                n_init=5
            )
        elif self.config.method == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=50)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str],
        forward_returns: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Fit clustering model.

        Args:
            X: Feature matrix
            feature_names: Names of features
            forward_returns: Optional forward returns for cluster profiling
        """
        # Select relevant features
        feature_idx = [feature_names.index(f) for f in self.config.features
                       if f in feature_names]
        X_selected = X[:, feature_idx]

        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)

        # Fit model
        self.model = self._build_model()
        labels = self.model.fit_predict(X_scaled)

        # Store cluster centers
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers = self.scaler.inverse_transform(
                self.model.cluster_centers_
            )
        elif hasattr(self.model, 'means_'):
            self.cluster_centers = self.scaler.inverse_transform(
                self.model.means_
            )

        # Profile clusters
        self._profile_clusters(X_selected, labels, forward_returns)

        return {
            'n_clusters': len(np.unique(labels)),
            'cluster_sizes': {
                int(c): int((labels == c).sum())
                for c in np.unique(labels)
            },
            'cluster_profiles': self.cluster_profiles
        }

    def _profile_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        forward_returns: Optional[np.ndarray]
    ):
        """
        Create profiles for each cluster.
        """
        for cluster in np.unique(labels):
            if cluster == -1:  # DBSCAN noise
                continue

            mask = labels == cluster

            profile = {
                'size': int(mask.sum()),
                'feature_means': dict(zip(
                    self.config.features,
                    X[mask].mean(axis=0).tolist()
                )),
                'feature_stds': dict(zip(
                    self.config.features,
                    X[mask].std(axis=0).tolist()
                ))
            }

            if forward_returns is not None:
                valid_returns = forward_returns[mask]
                valid_returns = valid_returns[~np.isnan(valid_returns)]

                if len(valid_returns) > 0:
                    profile['return_mean'] = float(np.mean(valid_returns))
                    profile['return_std'] = float(np.std(valid_returns))
                    profile['win_rate'] = float((valid_returns > 0).mean())

            self.cluster_profiles[int(cluster)] = profile

        # Auto-label clusters based on characteristics
        self._auto_label_clusters()

    def _auto_label_clusters(self):
        """
        Automatically assign semantic labels to clusters.
        """
        for cluster, profile in self.cluster_profiles.items():
            means = profile['feature_means']

            vol = means.get('hv_21_pctl', 50)
            trend = means.get('adx_norm', 0.5)
            momentum = means.get('rsi_norm', 0.5)

            if vol < 30 and trend > 0.5 and momentum > 0.5:
                label = 'LOW_VOL_UPTREND'
            elif vol > 70 and trend > 0.5 and momentum > 0.5:
                label = 'HIGH_VOL_UPTREND'
            elif vol < 40 and trend < 0.4:
                label = 'LOW_VOL_RANGE'
            elif vol > 70 and momentum < 0.4:
                label = 'HIGH_VOL_DOWNTREND'
            elif vol > 90:
                label = 'CRISIS'
            else:
                label = 'TRANSITION'

            self.cluster_profiles[cluster]['label'] = label

    def predict(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Predict cluster for new samples.
        """
        feature_idx = [feature_names.index(f) for f in self.config.features
                       if f in feature_names]
        X_selected = X[:, feature_idx]
        X_scaled = self.scaler.transform(X_selected)

        if hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled)
        else:
            return self.model.fit_predict(X_scaled)

    def get_regime_label(self, cluster: int) -> str:
        """Get semantic label for cluster."""
        if cluster in self.cluster_profiles:
            return self.cluster_profiles[cluster].get('label', 'UNKNOWN')
        return 'UNKNOWN'
```

### 4.2 Stock Clustering

```python
class StockClusterer:
    """
    Cluster stocks based on behavior patterns.

    Use cases:
    - Find similar stocks for pair trading
    - Diversification analysis
    - Peer comparison
    """

    def __init__(
        self,
        n_clusters: int = 20,
        features: List[str] = None
    ):
        self.n_clusters = n_clusters
        self.features = features or [
            'roc_21', 'roc_63', 'hv_21', 'volume_ratio',
            'rs_vs_spy_21', 'beta', 'sector_encoded'
        ]
        self.model = None
        self.scaler = StandardScaler()

    def fit(
        self,
        stock_features: pd.DataFrame
    ) -> Dict:
        """
        Cluster stocks based on features.

        Args:
            stock_features: DataFrame with ticker as index, features as columns
        """
        # Select and prepare features
        X = stock_features[self.features].values
        tickers = stock_features.index.tolist()

        # Handle missing values
        X = np.nan_to_num(X, nan=0)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Cluster
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = self.model.fit_predict(X_scaled)

        # Build cluster membership
        self.cluster_members = {
            int(c): [t for t, l in zip(tickers, labels) if l == c]
            for c in range(self.n_clusters)
        }

        return {
            'cluster_sizes': {c: len(m) for c, m in self.cluster_members.items()},
            'cluster_members': self.cluster_members
        }

    def find_similar(
        self,
        ticker: str,
        stock_features: pd.DataFrame,
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find most similar stocks to given ticker.
        """
        if ticker not in stock_features.index:
            return []

        X = stock_features[self.features].values
        X_scaled = self.scaler.transform(X)

        ticker_idx = stock_features.index.get_loc(ticker)
        ticker_vector = X_scaled[ticker_idx]

        # Compute distances
        distances = np.linalg.norm(X_scaled - ticker_vector, axis=1)

        # Sort and return (excluding self)
        sorted_idx = np.argsort(distances)

        similar = []
        for idx in sorted_idx:
            if idx != ticker_idx and len(similar) < n:
                similar.append((
                    stock_features.index[idx],
                    float(distances[idx])
                ))

        return similar
```

---

## 5. Anomaly Detection

### 5.1 State Anomaly Detector

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


class StateAnomalyDetector:
    """
    Detect unusual/anomalous indicator states.

    Use cases:
    - Identify extreme conditions
    - Flag potential data errors
    - Find unique opportunities
    """

    def __init__(
        self,
        contamination: float = 0.05,
        method: str = 'isolation_forest'
    ):
        """
        Args:
            contamination: Expected proportion of anomalies
            method: 'isolation_forest', 'lof', or 'elliptic'
        """
        self.contamination = contamination
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None

    def _build_model(self):
        """Build anomaly detection model."""
        if self.method == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
        elif self.method == 'lof':
            return LocalOutlierFactor(
                n_neighbors=50,
                contamination=self.contamination,
                novelty=True
            )
        elif self.method == 'elliptic':
            return EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Fit anomaly detector on historical states.
        """
        self.feature_names = feature_names

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model = self._build_model()
        self.model.fit(X_scaled)

        # Compute anomaly scores for calibration
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X_scaled)
        else:
            scores = -self.model.score_samples(X_scaled)

        self.threshold = np.percentile(scores, self.contamination * 100)

        # Find anomaly rate
        predictions = self.model.predict(X_scaled)
        anomaly_rate = (predictions == -1).mean()

        return {
            'anomaly_rate': float(anomaly_rate),
            'threshold': float(self.threshold),
            'n_samples': len(X)
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies.

        Returns:
            (is_anomaly, anomaly_scores)
        """
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        is_anomaly = predictions == -1

        if hasattr(self.model, 'decision_function'):
            scores = -self.model.decision_function(X_scaled)  # Higher = more anomalous
        else:
            scores = -self.model.score_samples(X_scaled)

        return is_anomaly, scores

    def explain_anomaly(
        self,
        X: np.ndarray,
        baseline_mean: np.ndarray,
        baseline_std: np.ndarray
    ) -> List[Dict]:
        """
        Explain why samples are anomalous.

        Returns feature-level explanations.
        """
        explanations = []

        for i, sample in enumerate(X):
            # Compute z-scores for each feature
            z_scores = (sample - baseline_mean) / (baseline_std + 1e-8)

            # Find extreme features
            extreme_features = []
            for j, (name, z) in enumerate(zip(self.feature_names, z_scores)):
                if abs(z) > 2:
                    extreme_features.append({
                        'feature': name,
                        'value': float(sample[j]),
                        'z_score': float(z),
                        'direction': 'high' if z > 0 else 'low'
                    })

            # Sort by absolute z-score
            extreme_features.sort(key=lambda x: -abs(x['z_score']))

            explanations.append({
                'sample_index': i,
                'extreme_features': extreme_features[:5],
                'summary': self._summarize_anomaly(extreme_features[:3])
            })

        return explanations

    def _summarize_anomaly(self, features: List[Dict]) -> str:
        """Generate human-readable anomaly summary."""
        if not features:
            return "No clear anomaly pattern"

        parts = []
        for f in features:
            direction = "extremely high" if f['direction'] == 'high' else "extremely low"
            parts.append(f"{f['feature']} is {direction} ({f['z_score']:.1f}σ)")

        return "; ".join(parts)


class ReturnAnomalyDetector:
    """
    Detect anomalous returns that may indicate data errors or events.
    """

    def __init__(self, window: int = 252, threshold_std: float = 4.0):
        self.window = window
        self.threshold_std = threshold_std

    def detect(
        self,
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        Detect return anomalies.

        Returns:
            DataFrame with anomaly flags and details
        """
        rolling_mean = returns.rolling(window=self.window, min_periods=20).mean()
        rolling_std = returns.rolling(window=self.window, min_periods=20).std()

        z_scores = (returns - rolling_mean) / (rolling_std + 1e-8)

        anomalies = pd.DataFrame({
            'return': returns,
            'z_score': z_scores,
            'is_anomaly': abs(z_scores) > self.threshold_std,
            'anomaly_type': 'none'
        })

        # Classify anomaly types
        anomalies.loc[z_scores > self.threshold_std, 'anomaly_type'] = 'extreme_positive'
        anomalies.loc[z_scores < -self.threshold_std, 'anomaly_type'] = 'extreme_negative'

        return anomalies[anomalies['is_anomaly']]
```

---

## 6. Walk-Forward Validation Strategy

### 6.1 Time-Series Cross-Validation

```python
from typing import Generator


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    # Window sizes
    train_window_days: int = 756         # 3 years training
    validation_window_days: int = 63     # 3 months validation
    test_window_days: int = 21           # 1 month test

    # Embargo to prevent leakage
    embargo_days: int = 21               # Gap between train and test

    # Expanding vs. rolling
    expanding_window: bool = True        # Use expanding training window

    # Retraining frequency
    retrain_frequency_days: int = 21     # Retrain monthly

    # Minimum requirements
    min_train_samples: int = 1000
    min_positive_samples: int = 100


class WalkForwardValidator:
    """
    Walk-forward validation for time-series models.

    Ensures no future information leakage and realistic
    evaluation of model performance over time.
    """

    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()

    def generate_splits(
        self,
        dates: pd.DatetimeIndex,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Generator[Dict, None, None]:
        """
        Generate train/validation/test splits.

        Yields:
            Dict with 'train_idx', 'val_idx', 'test_idx', 'split_info'
        """
        if start_date is None:
            start_date = dates.min() + pd.Timedelta(days=self.config.train_window_days)
        if end_date is None:
            end_date = dates.max()

        current_date = start_date
        split_num = 0

        while current_date + pd.Timedelta(days=self.config.test_window_days) <= end_date:
            # Define split boundaries
            test_end = current_date + pd.Timedelta(days=self.config.test_window_days)
            test_start = current_date

            val_end = test_start - pd.Timedelta(days=self.config.embargo_days)
            val_start = val_end - pd.Timedelta(days=self.config.validation_window_days)

            if self.config.expanding_window:
                train_start = dates.min()
            else:
                train_start = val_start - pd.Timedelta(days=self.config.train_window_days)

            train_end = val_start - pd.Timedelta(days=self.config.embargo_days)

            # Get indices
            train_mask = (dates >= train_start) & (dates < train_end)
            val_mask = (dates >= val_start) & (dates < val_end)
            test_mask = (dates >= test_start) & (dates < test_end)

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            test_idx = np.where(test_mask)[0]

            # Check minimum requirements
            if len(train_idx) >= self.config.min_train_samples:
                yield {
                    'train_idx': train_idx,
                    'val_idx': val_idx,
                    'test_idx': test_idx,
                    'split_info': {
                        'split_num': split_num,
                        'train_start': train_start,
                        'train_end': train_end,
                        'val_start': val_start,
                        'val_end': val_end,
                        'test_start': test_start,
                        'test_end': test_end,
                        'train_size': len(train_idx),
                        'val_size': len(val_idx),
                        'test_size': len(test_idx)
                    }
                }
                split_num += 1

            # Move to next period
            current_date += pd.Timedelta(days=self.config.retrain_frequency_days)

    def validate_model(
        self,
        model_class,
        model_config: Dict,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.DatetimeIndex,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Run complete walk-forward validation.

        Returns:
            DataFrame with metrics for each split
        """
        results = []

        for split in self.generate_splits(dates):
            train_idx = split['train_idx']
            val_idx = split['val_idx']
            test_idx = split['test_idx']
            info = split['split_info']

            # Train model
            model = model_class(**model_config)
            model.fit(
                X[train_idx], y[train_idx],
                feature_names,
                validation_data=(X[val_idx], y[val_idx])
            )

            # Evaluate on test
            test_proba = model.predict_proba(X[test_idx])
            test_y = y[test_idx]

            from sklearn.metrics import roc_auc_score, brier_score_loss

            metrics = {
                'split': info['split_num'],
                'test_start': info['test_start'],
                'test_end': info['test_end'],
                'train_size': info['train_size'],
                'test_size': info['test_size'],
                'test_positive_rate': test_y.mean(),
                'roc_auc': roc_auc_score(test_y, test_proba) if test_y.sum() > 0 else np.nan,
                'brier_score': brier_score_loss(test_y, test_proba)
            }

            results.append(metrics)

        return pd.DataFrame(results)


class PurgedKFold:
    """
    K-Fold cross-validation with purging and embargo.

    Addresses overlapping prediction horizons that cause leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01
    ):
        """
        Args:
            n_splits: Number of folds
            embargo_pct: Fraction of samples to embargo after test
            purge_pct: Fraction of samples to purge before test
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate purged train/test splits.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples

            test_indices = indices[test_start:test_end]

            # Purge: Remove samples before test that overlap
            purge_start = max(0, test_start - purge_size)

            # Embargo: Remove samples after test
            embargo_end = min(n_samples, test_end + embargo_size)

            train_indices = np.concatenate([
                indices[:purge_start],
                indices[embargo_end:]
            ])

            yield train_indices, test_indices
```

### 6.2 Performance Tracking

```python
class ModelPerformanceTracker:
    """
    Track model performance over time for monitoring and alerting.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.performance_history = []
        self.baseline_metrics = None

    def record_performance(
        self,
        date: pd.Timestamp,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ):
        """Record performance for a time period."""
        from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

        metrics = {
            'date': date,
            'n_predictions': len(predictions),
            'positive_rate': actuals.mean(),
            'predicted_positive_rate': predictions.mean()
        }

        if probabilities is not None and len(np.unique(actuals)) > 1:
            metrics['roc_auc'] = roc_auc_score(actuals, probabilities)
            metrics['brier_score'] = brier_score_loss(actuals, probabilities)

        metrics['accuracy'] = accuracy_score(actuals, predictions)

        self.performance_history.append(metrics)

    def detect_drift(
        self,
        window: int = 10,
        threshold_std: float = 2.0
    ) -> Dict:
        """
        Detect performance drift.
        """
        if len(self.performance_history) < window * 2:
            return {'drift_detected': False, 'reason': 'Insufficient history'}

        df = pd.DataFrame(self.performance_history)

        recent = df.tail(window)
        historical = df.iloc[:-window]

        drift_signals = {}

        for metric in ['roc_auc', 'brier_score', 'accuracy']:
            if metric not in df.columns:
                continue

            hist_mean = historical[metric].mean()
            hist_std = historical[metric].std()
            recent_mean = recent[metric].mean()

            z_score = (recent_mean - hist_mean) / (hist_std + 1e-8)

            if abs(z_score) > threshold_std:
                drift_signals[metric] = {
                    'z_score': float(z_score),
                    'historical_mean': float(hist_mean),
                    'recent_mean': float(recent_mean),
                    'direction': 'degradation' if (
                        (metric == 'roc_auc' and z_score < 0) or
                        (metric == 'brier_score' and z_score > 0)
                    ) else 'improvement'
                }

        return {
            'drift_detected': len(drift_signals) > 0,
            'signals': drift_signals
        }
```

---

## 7. Leakage Prevention Rules

### 7.1 Leakage Types and Prevention

```python
class LeakagePreventionFramework:
    """
    Framework for preventing data leakage in ML pipeline.
    """

    # Known leakage patterns
    LEAKAGE_PATTERNS = {
        'future_data': {
            'description': 'Using data that was not available at prediction time',
            'examples': [
                'Using adjusted prices before adjustment date',
                'Including future earnings in features',
                'Forward-looking fundamental data'
            ],
            'prevention': 'Point-in-time data retrieval'
        },

        'target_leakage': {
            'description': 'Features derived from or correlated with target',
            'examples': [
                'Forward returns in features',
                'Next-day price in feature set',
                'Features computed on overlapping period with target'
            ],
            'prevention': 'Strict feature/target separation'
        },

        'overlap_leakage': {
            'description': 'Train/test overlap due to sequential dependencies',
            'examples': [
                'Rolling features computed across train/test boundary',
                'Same company appearing in train and test',
                'Overlapping prediction horizons'
            ],
            'prevention': 'Purging and embargo periods'
        },

        'preprocessing_leakage': {
            'description': 'Using test data statistics in preprocessing',
            'examples': [
                'Scaling with full dataset mean/std',
                'Feature selection using all data',
                'Imputation using future values'
            ],
            'prevention': 'Fit preprocessing only on training data'
        }
    }


class LeakageDetector:
    """
    Detect potential data leakage in features and pipeline.
    """

    def check_feature_target_correlation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.95
    ) -> List[Dict]:
        """
        Detect suspiciously high feature-target correlations.
        """
        warnings = []

        for i, name in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]

            if abs(corr) > threshold:
                warnings.append({
                    'type': 'high_correlation',
                    'feature': name,
                    'correlation': float(corr),
                    'message': f"Feature '{name}' has {corr:.2f} correlation with target - possible leakage"
                })

        return warnings

    def check_perfect_prediction(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        threshold: float = 0.99
    ) -> Dict:
        """
        Detect suspiciously perfect predictions.
        """
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(actuals, predictions)

        if auc > threshold:
            return {
                'warning': True,
                'auc': float(auc),
                'message': f"AUC of {auc:.4f} is suspiciously high - check for leakage"
            }

        return {'warning': False, 'auc': float(auc)}

    def check_temporal_consistency(
        self,
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
        embargo_days: int = 21
    ) -> Dict:
        """
        Check for temporal overlap between train and test.
        """
        train_max = train_dates.max()
        test_min = test_dates.min()

        gap_days = (test_min - train_max).days

        if gap_days < embargo_days:
            return {
                'valid': False,
                'gap_days': int(gap_days),
                'required_gap': embargo_days,
                'message': f"Insufficient gap: {gap_days} days (need {embargo_days})"
            }

        return {
            'valid': True,
            'gap_days': int(gap_days),
            'message': "Temporal split is valid"
        }


class PointInTimeDataManager:
    """
    Ensure data is accessed in point-in-time manner.
    """

    def __init__(self, data_store):
        self.data_store = data_store

    def get_features_as_of(
        self,
        ticker: str,
        as_of_date: pd.Timestamp,
        feature_list: List[str]
    ) -> Dict[str, float]:
        """
        Get features as they would have been known at as_of_date.

        This prevents using future information.
        """
        features = {}

        for feature in feature_list:
            # Get latest value available before as_of_date
            value = self.data_store.get_latest_before(
                ticker=ticker,
                feature=feature,
                date=as_of_date
            )
            features[feature] = value

        return features

    def get_fundamentals_as_of(
        self,
        ticker: str,
        as_of_date: pd.Timestamp
    ) -> Dict:
        """
        Get fundamental data with proper release lag.

        Fundamentals have significant reporting lag:
        - Quarterly reports: ~45 days after quarter end
        - Annual reports: ~60-90 days after year end
        """
        # Apply standard 45-day lag
        effective_date = as_of_date - pd.Timedelta(days=45)

        return self.data_store.get_fundamentals(
            ticker=ticker,
            effective_date=effective_date
        )

    def validate_no_future_data(
        self,
        features: pd.DataFrame,
        feature_dates: pd.DatetimeIndex,
        prediction_date: pd.Timestamp
    ) -> bool:
        """
        Validate that no features use data after prediction date.
        """
        for date in feature_dates:
            if date > prediction_date:
                raise ValueError(
                    f"Feature uses data from {date}, after prediction date {prediction_date}"
                )
        return True
```

### 7.2 Safe Feature Engineering

```python
class SafeFeatureEngineer:
    """
    Feature engineering with built-in leakage prevention.
    """

    def __init__(self):
        self.fitted_params = {}

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: List[str],
        is_train: bool = True
    ) -> np.ndarray:
        """
        Fit parameters on training data, transform any data.
        """
        if is_train:
            # Fit scaling parameters
            self.fitted_params['mean'] = np.nanmean(X, axis=0)
            self.fitted_params['std'] = np.nanstd(X, axis=0)
            self.fitted_params['feature_names'] = feature_names

        # Transform using fitted parameters
        X_scaled = (X - self.fitted_params['mean']) / (self.fitted_params['std'] + 1e-8)

        return X_scaled

    def create_lagged_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Create lagged features (safe - only uses past data).
        """
        result = df.copy()

        for col in feature_cols:
            for lag in lags:
                if lag > 0:  # Only past lags
                    result[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return result

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        feature_col: str,
        windows: List[int],
        min_periods: int = 1
    ) -> pd.DataFrame:
        """
        Create rolling features (safe - only uses past/current data).
        """
        result = df.copy()

        for window in windows:
            result[f'{feature_col}_rolling_mean_{window}'] = (
                df[feature_col]
                .rolling(window=window, min_periods=min_periods)
                .mean()
            )
            result[f'{feature_col}_rolling_std_{window}'] = (
                df[feature_col]
                .rolling(window=window, min_periods=min_periods)
                .std()
            )

        return result

    def validate_features(
        self,
        feature_names: List[str]
    ) -> List[str]:
        """
        Check feature names for potential leakage indicators.
        """
        warnings = []

        dangerous_patterns = [
            'forward', 'future', 'next', 'lead',
            'target', 'return_fwd', 'price_next'
        ]

        for name in feature_names:
            for pattern in dangerous_patterns:
                if pattern in name.lower():
                    warnings.append(
                        f"Feature '{name}' contains dangerous pattern '{pattern}'"
                    )

        return warnings
```

---

## 8. Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ML & STATISTICAL MODELS SUMMARY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIER 1: RULE-BASED                                                        │
│  ─────────────────                                                          │
│  • 5+ Trading rules (RSI oversold, MACD cross, trend alignment, etc.)      │
│  • Deterministic, interpretable                                            │
│  • Aggregated into probability estimate                                    │
│                                                                             │
│  TIER 2: SUPERVISED                                                        │
│  ─────────────────                                                          │
│  • Classification: Gradient Boosting, Random Forest, Logistic              │
│  • Regression: Return prediction with uncertainty                          │
│  • Calibrated probability outputs                                          │
│  • Multi-horizon models (5d, 21d, 63d)                                     │
│                                                                             │
│  TIER 3: UNSUPERVISED                                                      │
│  ───────────────────                                                        │
│  • Market regime clustering (GMM/KMeans)                                   │
│  • Stock similarity clustering                                             │
│  • Auto-labeled regimes (uptrend, crisis, etc.)                            │
│                                                                             │
│  TIER 4: ANOMALY DETECTION                                                 │
│  ─────────────────────────                                                  │
│  • Isolation Forest for state anomalies                                    │
│  • Return anomaly detection                                                │
│  • Explainable anomaly summaries                                           │
│                                                                             │
│  VALIDATION                                                                │
│  ──────────                                                                 │
│  • Walk-forward with 21-day embargo                                        │
│  • Purged K-Fold for cross-validation                                      │
│  • Performance drift detection                                             │
│                                                                             │
│  LEAKAGE PREVENTION                                                        │
│  ──────────────────                                                         │
│  • Point-in-time data access                                               │
│  • Feature-target correlation checks                                       │
│  • Temporal consistency validation                                         │
│  • Safe feature engineering                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Assumptions

1. **Stationarity:** Market relationships are sufficiently stable for learning
2. **Representative History:** Training data captures relevant market regimes
3. **Feature Informativeness:** Selected features contain predictive signal
4. **Model Generalization:** Models trained on past data apply to future
5. **Calibration Stability:** Probability calibration remains valid
6. **Regime Continuity:** Detected regimes persist long enough to be actionable
7. **Data Quality:** Input data is clean and accurate

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
