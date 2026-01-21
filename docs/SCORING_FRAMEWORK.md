# Scoring Framework (0-10)

## 1. Overview

The Scoring Framework transforms raw indicator states, probability estimates, and risk metrics into a single, interpretable 0-10 recommendation score. The system is designed to be **explainable**, **deterministic**, and **actionable**.

### Score Interpretation

| Score | Label | Interpretation | Action Guidance |
|-------|-------|----------------|-----------------|
| 9-10 | **Exceptional** | Rare, high-conviction opportunity | Strong consideration for position |
| 7-8 | **Strong** | Multiple factors align favorably | Worthy of detailed analysis |
| 5-6 | **Moderate** | Mixed signals, some positive | Monitor for improvement |
| 3-4 | **Weak** | More negatives than positives | Caution warranted |
| 1-2 | **Poor** | Significant concerns | Avoid or consider short |
| 0 | **Critical** | Major red flags | Strong avoid |

### Scoring Philosophy

1. **Base Score from Probability:** Historical edge probability is the foundation
2. **Subscore Adjustments:** Category-specific factors modify the base
3. **Risk Penalties:** Excessive risk reduces the final score
4. **Ensemble Weighting:** Multiple timeframes combined intelligently
5. **Explainability:** Every score has a clear breakdown

---

## 2. Scoring Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCORING ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS                                                                     │
│  ──────                                                                     │
│  ├── Probability Estimates (per horizon)                                   │
│  ├── Indicator State Vector                                                │
│  ├── Risk Metrics                                                          │
│  └── Market Context                                                        │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: COMPUTE SUBSCORES (0-10 each)                               │   │
│  │                                                                     │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ TREND   │ │MOMENTUM │ │ VOLUME  │ │RELATIVE │ │ FUNDA-  │       │   │
│  │  │ Score   │ │ Score   │ │ Score   │ │STRENGTH │ │ MENTAL  │       │   │
│  │  │  (T)    │ │  (M)    │ │  (V)    │ │  (RS)   │ │  (F)    │       │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: COMPUTE EDGE SCORES (per horizon)                           │   │
│  │                                                                     │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │   │
│  │  │  5-Day Edge │ │ 21-Day Edge │ │ 63-Day Edge │                   │   │
│  │  │   Score     │ │   Score     │ │   Score     │                   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: APPLY RISK PENALTIES                                        │   │
│  │                                                                     │   │
│  │  Raw Score - Volatility Penalty - Drawdown Penalty - Liquidity Pen. │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: ENSEMBLE & FINAL MAPPING                                    │   │
│  │                                                                     │   │
│  │  Weighted combination → Clamp to [0, 10] → Round to 1 decimal       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  FINAL SCORE: 7.3 / 10                                                     │
│  Breakdown: T=7.5, M=8.0, V=6.5, RS=7.0, F=6.0 | Edge=7.8 | Risk=-0.5     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Subscores per Category

### 3.1 Subscore Design Principles

Each subscore:
- Ranges from 0 to 10
- Combines multiple related indicators
- Has clear interpretation at each level
- Contributes independently to final score

### 3.2 Trend Subscore (T)

```python
class TrendSubscore:
    """
    Measures trend direction, strength, and alignment.

    Components:
    - Trend direction (price vs. MAs)
    - Trend strength (ADX)
    - Multi-MA alignment
    - Trend freshness (Aroon)
    """

    WEIGHTS = {
        'direction': 0.35,      # Price position relative to MAs
        'strength': 0.25,       # ADX level
        'alignment': 0.25,      # EMA stack alignment
        'freshness': 0.15       # Aroon-based trend age
    }

    def compute(self, state: IndicatorStateVector) -> SubscoreResult:
        components = {}

        # 1. Direction Score (0-10)
        # Based on price position vs. key EMAs
        ema_positions = [
            state.continuous.get('ema_8_pct', 0),
            state.continuous.get('ema_21_pct', 0),
            state.continuous.get('ema_50_pct', 0),
            state.continuous.get('ema_200_pct', 0)
        ]

        # Count how many MAs price is above
        above_count = sum(1 for p in ema_positions if p > 0)

        # Weight by MA importance (longer = more important)
        weighted_position = (
            0.15 * (1 if ema_positions[0] > 0 else 0) +  # EMA 8
            0.20 * (1 if ema_positions[1] > 0 else 0) +  # EMA 21
            0.30 * (1 if ema_positions[2] > 0 else 0) +  # EMA 50
            0.35 * (1 if ema_positions[3] > 0 else 0)    # EMA 200
        )

        direction_score = weighted_position * 10
        components['direction'] = direction_score

        # 2. Strength Score (0-10)
        # ADX-based
        adx = state.continuous.get('adx_norm', 0.5) * 100  # Denormalize

        if adx >= 50:
            strength_score = 10.0
        elif adx >= 40:
            strength_score = 8.0 + (adx - 40) / 10 * 2
        elif adx >= 25:
            strength_score = 5.0 + (adx - 25) / 15 * 3
        elif adx >= 15:
            strength_score = 2.0 + (adx - 15) / 10 * 3
        else:
            strength_score = adx / 15 * 2

        components['strength'] = strength_score

        # 3. Alignment Score (0-10)
        # Perfect alignment: EMA8 > EMA21 > EMA50 > EMA200
        alignment = state.meta_features.get('ema_alignment', 0)

        # alignment is -1 to 1, convert to 0-10
        alignment_score = (alignment + 1) / 2 * 10
        components['alignment'] = alignment_score

        # 4. Freshness Score (0-10)
        # Aroon-based: Recent new highs = fresh uptrend
        aroon_up = state.continuous.get('aroon_up', 50)
        aroon_down = state.continuous.get('aroon_down', 50)

        aroon_diff = aroon_up - aroon_down
        # aroon_diff ranges -100 to 100, convert to 0-10
        freshness_score = (aroon_diff + 100) / 200 * 10
        components['freshness'] = freshness_score

        # Weighted combination
        final_score = sum(
            self.WEIGHTS[k] * components[k]
            for k in self.WEIGHTS
        )

        return SubscoreResult(
            name='trend',
            score=final_score,
            components=components,
            weights=self.WEIGHTS,
            interpretation=self._interpret(final_score)
        )

    def _interpret(self, score: float) -> str:
        if score >= 8:
            return "Strong uptrend with good alignment"
        elif score >= 6:
            return "Moderate uptrend"
        elif score >= 4:
            return "Neutral/weak trend"
        elif score >= 2:
            return "Moderate downtrend"
        else:
            return "Strong downtrend"


@dataclass
class SubscoreResult:
    """Result from subscore computation."""
    name: str
    score: float                    # 0-10
    components: Dict[str, float]    # Individual component scores
    weights: Dict[str, float]       # Component weights
    interpretation: str             # Human-readable interpretation
```

### 3.3 Momentum Subscore (M)

```python
class MomentumSubscore:
    """
    Measures price momentum and mean-reversion potential.

    Components:
    - RSI position (oversold = opportunity)
    - Rate of change (momentum direction)
    - MACD signal
    - Stochastic position
    """

    WEIGHTS = {
        'rsi': 0.30,
        'roc': 0.25,
        'macd': 0.25,
        'stochastic': 0.20
    }

    def compute(self, state: IndicatorStateVector) -> SubscoreResult:
        components = {}

        # 1. RSI Score (0-10)
        # Scoring depends on strategy:
        # For momentum: higher RSI = more bullish (but overbought risk)
        # For mean-reversion: oversold = opportunity
        # We use a balanced approach
        rsi = state.continuous.get('rsi_norm', 0.5) * 100

        if rsi <= 20:
            # Deeply oversold - high opportunity
            rsi_score = 9.0
        elif rsi <= 30:
            # Oversold - good opportunity
            rsi_score = 8.0
        elif rsi <= 40:
            # Approaching oversold
            rsi_score = 6.5
        elif rsi <= 60:
            # Neutral zone
            rsi_score = 5.0
        elif rsi <= 70:
            # Approaching overbought
            rsi_score = 4.0
        elif rsi <= 80:
            # Overbought - caution
            rsi_score = 2.5
        else:
            # Extremely overbought
            rsi_score = 1.0

        components['rsi'] = rsi_score

        # 2. Rate of Change Score (0-10)
        # Positive momentum = higher score
        roc_21 = state.continuous.get('roc_21', 0)

        # Map ROC to score: -20% to +20% → 0 to 10
        roc_score = np.clip((roc_21 + 0.20) / 0.40 * 10, 0, 10)
        components['roc'] = roc_score

        # 3. MACD Score (0-10)
        macd_signal = state.signals.get('macd_signal', 0)
        macd_z = state.continuous.get('macd_z', 0)

        # Combine signal and magnitude
        if macd_signal == 1:  # Bullish
            macd_score = 6.0 + min(4.0, abs(macd_z))
        elif macd_signal == -1:  # Bearish
            macd_score = 4.0 - min(4.0, abs(macd_z))
        else:
            macd_score = 5.0

        macd_score = np.clip(macd_score, 0, 10)
        components['macd'] = macd_score

        # 4. Stochastic Score (0-10)
        stoch_k = state.continuous.get('stoch_k_norm', 0.5) * 100

        # Similar logic to RSI
        if stoch_k <= 20:
            stoch_score = 9.0
        elif stoch_k <= 30:
            stoch_score = 7.5
        elif stoch_k <= 70:
            stoch_score = 5.0
        elif stoch_k <= 80:
            stoch_score = 3.0
        else:
            stoch_score = 1.5

        components['stochastic'] = stoch_score

        # Weighted combination
        final_score = sum(
            self.WEIGHTS[k] * components[k]
            for k in self.WEIGHTS
        )

        return SubscoreResult(
            name='momentum',
            score=final_score,
            components=components,
            weights=self.WEIGHTS,
            interpretation=self._interpret(final_score)
        )

    def _interpret(self, score: float) -> str:
        if score >= 8:
            return "Strong bullish momentum / oversold bounce potential"
        elif score >= 6:
            return "Positive momentum"
        elif score >= 4:
            return "Neutral momentum"
        elif score >= 2:
            return "Negative momentum"
        else:
            return "Strong bearish momentum / overbought"
```

### 3.4 Volume Subscore (V)

```python
class VolumeSubscore:
    """
    Measures volume confirmation and accumulation/distribution.

    Components:
    - Volume level (vs. average)
    - Money flow (CMF)
    - On-balance volume trend
    - Volume-price confirmation
    """

    WEIGHTS = {
        'volume_level': 0.25,
        'money_flow': 0.30,
        'obv_trend': 0.25,
        'confirmation': 0.20
    }

    def compute(
        self,
        state: IndicatorStateVector,
        price_direction: int  # 1 = up, -1 = down
    ) -> SubscoreResult:
        components = {}

        # 1. Volume Level Score (0-10)
        # Higher volume = more conviction (good for any direction)
        vol_ratio = state.continuous.get('volume_ratio', 1.0)

        if vol_ratio >= 3.0:
            vol_score = 10.0
        elif vol_ratio >= 2.0:
            vol_score = 8.0
        elif vol_ratio >= 1.5:
            vol_score = 7.0
        elif vol_ratio >= 1.0:
            vol_score = 5.0
        elif vol_ratio >= 0.7:
            vol_score = 3.0
        else:
            vol_score = 1.0

        components['volume_level'] = vol_score

        # 2. Money Flow Score (0-10)
        # CMF ranges -1 to 1
        cmf = state.continuous.get('cmf', 0)

        # Map to 0-10
        money_flow_score = (cmf + 1) / 2 * 10
        components['money_flow'] = money_flow_score

        # 3. OBV Trend Score (0-10)
        # Based on OBV rate of change z-score
        obv_z = state.continuous.get('obv_roc_21_z', 0)

        # Map z-score to 0-10: -3 to +3 → 0 to 10
        obv_score = np.clip((obv_z + 3) / 6 * 10, 0, 10)
        components['obv_trend'] = obv_score

        # 4. Volume-Price Confirmation Score (0-10)
        # High volume + price up = bullish confirmation
        # High volume + price down = bearish confirmation
        # We score based on bullish confirmation

        if price_direction > 0 and vol_ratio > 1.2:
            confirm_score = 8.0 + min(2.0, (vol_ratio - 1.2) * 2)
        elif price_direction < 0 and vol_ratio > 1.2:
            confirm_score = 2.0 - min(2.0, (vol_ratio - 1.2))
        elif price_direction > 0:
            confirm_score = 6.0
        elif price_direction < 0:
            confirm_score = 4.0
        else:
            confirm_score = 5.0

        confirm_score = np.clip(confirm_score, 0, 10)
        components['confirmation'] = confirm_score

        # Weighted combination
        final_score = sum(
            self.WEIGHTS[k] * components[k]
            for k in self.WEIGHTS
        )

        return SubscoreResult(
            name='volume',
            score=final_score,
            components=components,
            weights=self.WEIGHTS,
            interpretation=self._interpret(final_score)
        )

    def _interpret(self, score: float) -> str:
        if score >= 8:
            return "Strong accumulation with volume confirmation"
        elif score >= 6:
            return "Positive money flow"
        elif score >= 4:
            return "Neutral volume activity"
        elif score >= 2:
            return "Mild distribution"
        else:
            return "Strong distribution / selling pressure"
```

### 3.5 Relative Strength Subscore (RS)

```python
class RelativeStrengthSubscore:
    """
    Measures performance vs. benchmarks and peers.

    Components:
    - Short-term RS (21-day)
    - Medium-term RS (63-day)
    - Sector rank
    - RS trend (improving vs. deteriorating)
    """

    WEIGHTS = {
        'rs_21d': 0.30,
        'rs_63d': 0.25,
        'sector_rank': 0.25,
        'rs_trend': 0.20
    }

    def compute(self, state: IndicatorStateVector) -> SubscoreResult:
        components = {}

        # 1. 21-Day RS Score (0-10)
        rs_21 = state.continuous.get('rs_vs_spy_21', 0)

        # Map excess return to score
        # -10% to +10% → 0 to 10
        rs_21_score = np.clip((rs_21 + 0.10) / 0.20 * 10, 0, 10)
        components['rs_21d'] = rs_21_score

        # 2. 63-Day RS Score (0-10)
        rs_63 = state.continuous.get('rs_vs_spy_63', 0)

        rs_63_score = np.clip((rs_63 + 0.15) / 0.30 * 10, 0, 10)
        components['rs_63d'] = rs_63_score

        # 3. Sector Rank Score (0-10)
        # Percentile within sector (0-100 → 0-10)
        sector_pctl = state.continuous.get('sector_rank_pctl', 50)

        sector_score = sector_pctl / 10
        components['sector_rank'] = sector_score

        # 4. RS Trend Score (0-10)
        # Is RS improving or deteriorating?
        rs_21 = state.continuous.get('rs_vs_spy_21', 0)
        rs_63 = state.continuous.get('rs_vs_spy_63', 0)

        # Short-term vs long-term RS
        rs_momentum = rs_21 - rs_63 / 3  # Normalize for time

        # Map to score
        rs_trend_score = np.clip((rs_momentum + 0.05) / 0.10 * 10, 0, 10)
        components['rs_trend'] = rs_trend_score

        # Weighted combination
        final_score = sum(
            self.WEIGHTS[k] * components[k]
            for k in self.WEIGHTS
        )

        return SubscoreResult(
            name='relative_strength',
            score=final_score,
            components=components,
            weights=self.WEIGHTS,
            interpretation=self._interpret(final_score)
        )

    def _interpret(self, score: float) -> str:
        if score >= 8:
            return "Market leader, strong outperformance"
        elif score >= 6:
            return "Outperforming market and peers"
        elif score >= 4:
            return "In-line with market"
        elif score >= 2:
            return "Underperforming"
        else:
            return "Significant laggard"
```

### 3.6 Fundamental Subscore (F) - Optional

```python
class FundamentalSubscore:
    """
    Measures fundamental quality and valuation.

    Components:
    - Valuation (P/E, P/B relative)
    - Quality (ROE, margins)
    - Earnings momentum
    - Financial health
    """

    WEIGHTS = {
        'valuation': 0.30,
        'quality': 0.30,
        'earnings': 0.25,
        'health': 0.15
    }

    def compute(
        self,
        state: IndicatorStateVector,
        fundamentals: Optional[Dict] = None
    ) -> SubscoreResult:
        if fundamentals is None:
            # Return neutral score if no fundamental data
            return SubscoreResult(
                name='fundamental',
                score=5.0,
                components={k: 5.0 for k in self.WEIGHTS},
                weights=self.WEIGHTS,
                interpretation="Fundamental data unavailable"
            )

        components = {}

        # 1. Valuation Score (0-10)
        # Lower valuation percentile = higher score
        pe_pctl = fundamentals.get('pe_percentile', 50)
        pb_pctl = fundamentals.get('pb_percentile', 50)

        # Invert: low percentile = cheap = high score
        valuation_score = 10 - (pe_pctl * 0.6 + pb_pctl * 0.4) / 10
        components['valuation'] = valuation_score

        # 2. Quality Score (0-10)
        roe_pctl = fundamentals.get('roe_percentile', 50)
        margin_pctl = fundamentals.get('profit_margin_percentile', 50)

        quality_score = (roe_pctl * 0.5 + margin_pctl * 0.5) / 10
        components['quality'] = quality_score

        # 3. Earnings Score (0-10)
        # Based on earnings surprise and revision
        surprise_avg = fundamentals.get('earnings_surprise_avg', 0)

        # Map surprise to score: -10% to +10% → 0 to 10
        earnings_score = np.clip((surprise_avg + 0.10) / 0.20 * 10, 0, 10)
        components['earnings'] = earnings_score

        # 4. Financial Health Score (0-10)
        # F-Score or similar composite
        f_score = fundamentals.get('piotroski_f_score', 5)

        health_score = f_score / 9 * 10
        components['health'] = health_score

        # Weighted combination
        final_score = sum(
            self.WEIGHTS[k] * components[k]
            for k in self.WEIGHTS
        )

        return SubscoreResult(
            name='fundamental',
            score=final_score,
            components=components,
            weights=self.WEIGHTS,
            interpretation=self._interpret(final_score)
        )

    def _interpret(self, score: float) -> str:
        if score >= 8:
            return "Excellent fundamentals, attractively valued"
        elif score >= 6:
            return "Good fundamentals"
        elif score >= 4:
            return "Average fundamentals"
        elif score >= 2:
            return "Weak fundamentals or expensive"
        else:
            return "Poor fundamentals, concerns present"
```

### 3.7 Subscore Summary

```python
# Complete subscore configuration
SUBSCORE_CONFIG = {
    'trend': {
        'class': TrendSubscore,
        'weight': 0.20,
        'required': True
    },
    'momentum': {
        'class': MomentumSubscore,
        'weight': 0.20,
        'required': True
    },
    'volume': {
        'class': VolumeSubscore,
        'weight': 0.15,
        'required': True
    },
    'relative_strength': {
        'class': RelativeStrengthSubscore,
        'weight': 0.15,
        'required': True
    },
    'fundamental': {
        'class': FundamentalSubscore,
        'weight': 0.10,
        'required': False  # Optional
    },
    # Edge score gets remaining weight (0.20)
}
```

---

## 4. Edge Score per Timeframe

### 4.1 Edge Score Definition

The Edge Score transforms probability estimates from the Probability Engine into a 0-10 score for each timeframe.

```python
class EdgeScore:
    """
    Transform probability estimates into 0-10 edge scores.

    Edge Score = f(probability, confidence, reliability)
    """

    # Probability to base score mapping
    # Designed so that:
    # - Base rate (~10-15% for very high gain) → score ~4
    # - 25% probability → score ~6
    # - 40% probability → score ~8
    # - 50%+ probability → score 9-10

    PROBABILITY_BREAKPOINTS = [
        (0.05, 1.0),   # 5% → 1.0
        (0.10, 3.0),   # 10% → 3.0 (near base rate)
        (0.15, 4.5),   # 15% → 4.5
        (0.20, 5.5),   # 20% → 5.5
        (0.25, 6.5),   # 25% → 6.5
        (0.30, 7.2),   # 30% → 7.2
        (0.35, 7.8),   # 35% → 7.8
        (0.40, 8.3),   # 40% → 8.3
        (0.45, 8.7),   # 45% → 8.7
        (0.50, 9.0),   # 50% → 9.0
        (0.60, 9.5),   # 60% → 9.5
        (0.70, 9.8),   # 70% → 9.8
        (1.00, 10.0),  # 100% → 10.0
    ]

    RELIABILITY_MULTIPLIERS = {
        'very_low': 0.6,
        'low': 0.75,
        'moderate': 0.9,
        'high': 1.0,
        'very_high': 1.0,
        'model_based': 0.85
    }

    def compute(
        self,
        probability_estimate: ProbabilityEstimate,
        horizon: str
    ) -> EdgeScoreResult:
        """
        Compute edge score from probability estimate.
        """
        prob = probability_estimate.probability

        if np.isnan(prob):
            return EdgeScoreResult(
                horizon=horizon,
                raw_score=np.nan,
                adjusted_score=np.nan,
                probability=np.nan,
                confidence_interval=(np.nan, np.nan),
                reliability='insufficient_data',
                interpretation="Insufficient historical data"
            )

        # 1. Map probability to base score using interpolation
        raw_score = self._probability_to_score(prob)

        # 2. Adjust for reliability
        reliability = probability_estimate.reliability
        reliability_mult = self.RELIABILITY_MULTIPLIERS.get(reliability, 0.8)

        # 3. Adjust for confidence width
        # Narrow confidence = more certain = bonus
        # Wide confidence = less certain = penalty
        conf_width = probability_estimate.confidence_width

        if conf_width < 0.10:
            conf_adjustment = 0.3
        elif conf_width < 0.20:
            conf_adjustment = 0.1
        elif conf_width < 0.30:
            conf_adjustment = 0.0
        elif conf_width < 0.40:
            conf_adjustment = -0.2
        else:
            conf_adjustment = -0.5

        # 4. Compute adjusted score
        adjusted_score = raw_score * reliability_mult + conf_adjustment
        adjusted_score = np.clip(adjusted_score, 0, 10)

        return EdgeScoreResult(
            horizon=horizon,
            raw_score=raw_score,
            adjusted_score=adjusted_score,
            probability=prob,
            confidence_interval=(
                probability_estimate.confidence_low,
                probability_estimate.confidence_high
            ),
            reliability=reliability,
            n_samples=probability_estimate.n_samples,
            interpretation=self._interpret(adjusted_score, prob)
        )

    def _probability_to_score(self, prob: float) -> float:
        """Interpolate probability to score using breakpoints."""
        for i, (p, s) in enumerate(self.PROBABILITY_BREAKPOINTS):
            if prob <= p:
                if i == 0:
                    return s * prob / p
                prev_p, prev_s = self.PROBABILITY_BREAKPOINTS[i - 1]
                # Linear interpolation
                return prev_s + (s - prev_s) * (prob - prev_p) / (p - prev_p)
        return 10.0

    def _interpret(self, score: float, prob: float) -> str:
        if score >= 8:
            return f"Strong historical edge ({prob:.1%} probability)"
        elif score >= 6:
            return f"Moderate edge ({prob:.1%} probability)"
        elif score >= 4:
            return f"Near base rate ({prob:.1%} probability)"
        elif score >= 2:
            return f"Below average odds ({prob:.1%} probability)"
        else:
            return f"Poor historical performance ({prob:.1%} probability)"


@dataclass
class EdgeScoreResult:
    """Result from edge score computation."""
    horizon: str
    raw_score: float
    adjusted_score: float
    probability: float
    confidence_interval: Tuple[float, float]
    reliability: str
    n_samples: int = 0
    interpretation: str = ""
```

### 4.2 Multi-Horizon Edge Aggregation

```python
class MultiHorizonEdgeAggregator:
    """
    Aggregate edge scores across multiple time horizons.
    """

    # Horizon weights (can be customized based on investment style)
    HORIZON_WEIGHTS = {
        '5d': 0.20,    # Short-term trading
        '21d': 0.40,   # Swing trading (primary)
        '63d': 0.30,   # Position trading
        '126d': 0.10,  # Long-term
    }

    def aggregate(
        self,
        edge_scores: Dict[str, EdgeScoreResult],
        horizon_weights: Optional[Dict[str, float]] = None
    ) -> AggregatedEdgeScore:
        """
        Aggregate multiple horizon edge scores.
        """
        weights = horizon_weights or self.HORIZON_WEIGHTS

        # Filter to available horizons
        available = {
            h: s for h, s in edge_scores.items()
            if not np.isnan(s.adjusted_score)
        }

        if not available:
            return AggregatedEdgeScore(
                score=np.nan,
                by_horizon={},
                primary_horizon=None,
                alignment='unknown',
                interpretation="No valid edge scores"
            )

        # Normalize weights for available horizons
        total_weight = sum(weights.get(h, 0.1) for h in available)
        norm_weights = {
            h: weights.get(h, 0.1) / total_weight
            for h in available
        }

        # Weighted average
        agg_score = sum(
            norm_weights[h] * available[h].adjusted_score
            for h in available
        )

        # Find primary (highest weight) horizon
        primary = max(available.keys(), key=lambda h: norm_weights[h])

        # Check horizon alignment
        scores = [s.adjusted_score for s in available.values()]

        if max(scores) - min(scores) < 1.5:
            alignment = 'aligned'
        elif all(s > 5 for s in scores):
            alignment = 'bullish_aligned'
        elif all(s < 5 for s in scores):
            alignment = 'bearish_aligned'
        else:
            alignment = 'mixed'

        return AggregatedEdgeScore(
            score=agg_score,
            by_horizon={h: s.adjusted_score for h, s in available.items()},
            primary_horizon=primary,
            primary_score=available[primary].adjusted_score,
            alignment=alignment,
            weights_used=norm_weights,
            interpretation=self._interpret(agg_score, alignment)
        )

    def _interpret(self, score: float, alignment: str) -> str:
        base = ""
        if score >= 7:
            base = "Strong edge across timeframes"
        elif score >= 5:
            base = "Moderate edge"
        else:
            base = "Weak or no edge"

        if alignment == 'mixed':
            base += " (conflicting signals across horizons)"

        return base


@dataclass
class AggregatedEdgeScore:
    """Aggregated edge score across horizons."""
    score: float
    by_horizon: Dict[str, float]
    primary_horizon: Optional[str]
    primary_score: float = 0.0
    alignment: str = ''
    weights_used: Dict[str, float] = field(default_factory=dict)
    interpretation: str = ""
```

---

## 5. Risk Penalty Logic

### 5.1 Risk Penalty Framework

```python
class RiskPenaltyCalculator:
    """
    Calculate risk-based penalties to reduce final score.

    Philosophy:
    - High risk should reduce score even if other factors are positive
    - Multiple risk factors compound
    - Severe risks can cap maximum score
    """

    def compute(
        self,
        state: IndicatorStateVector,
        market_context: Dict,
        base_score: float
    ) -> RiskPenaltyResult:
        """
        Compute total risk penalty.

        Returns:
            RiskPenaltyResult with individual penalties and total
        """
        penalties = {}
        caps = []

        # 1. Volatility Penalty
        vol_penalty, vol_cap = self._volatility_penalty(state)
        penalties['volatility'] = vol_penalty
        if vol_cap:
            caps.append(vol_cap)

        # 2. Drawdown Penalty
        dd_penalty, dd_cap = self._drawdown_penalty(state)
        penalties['drawdown'] = dd_penalty
        if dd_cap:
            caps.append(dd_cap)

        # 3. Liquidity Penalty
        liq_penalty, liq_cap = self._liquidity_penalty(state, market_context)
        penalties['liquidity'] = liq_penalty
        if liq_cap:
            caps.append(liq_cap)

        # 4. Concentration/Gap Risk
        gap_penalty = self._gap_risk_penalty(state)
        penalties['gap_risk'] = gap_penalty

        # 5. Regime Risk (high vol environment)
        regime_penalty = self._regime_risk_penalty(state, market_context)
        penalties['regime'] = regime_penalty

        # Total penalty (additive with diminishing returns)
        raw_total = sum(penalties.values())

        # Apply diminishing returns: penalty = sqrt(sum of squares)
        total_penalty = np.sqrt(sum(p**2 for p in penalties.values()))

        # Cap penalty at 4 points (can't reduce more than 4)
        total_penalty = min(total_penalty, 4.0)

        # Apply score cap if any severe risks
        score_cap = min(caps) if caps else 10.0

        return RiskPenaltyResult(
            penalties=penalties,
            total_penalty=total_penalty,
            score_cap=score_cap,
            interpretation=self._interpret(penalties, total_penalty)
        )

    def _volatility_penalty(
        self,
        state: IndicatorStateVector
    ) -> Tuple[float, Optional[float]]:
        """
        Penalty for high volatility.
        """
        hv_pctl = state.continuous.get('hv_21_pctl', 50)
        atrp = state.continuous.get('atrp', 0.02)

        # Percentile-based penalty
        if hv_pctl >= 95:
            penalty = 1.5
            cap = 7.0  # Cap at 7 in extreme vol
        elif hv_pctl >= 85:
            penalty = 1.0
            cap = 8.0
        elif hv_pctl >= 75:
            penalty = 0.5
            cap = None
        else:
            penalty = 0.0
            cap = None

        # Additional penalty for very high ATR%
        if atrp > 0.05:  # >5% daily range
            penalty += 0.5

        return penalty, cap

    def _drawdown_penalty(
        self,
        state: IndicatorStateVector
    ) -> Tuple[float, Optional[float]]:
        """
        Penalty for being in significant drawdown.
        """
        # Distance from 52-week high
        pct_from_high = state.continuous.get('pct_from_52w_high', 0)

        if pct_from_high < -0.50:  # Down 50%+
            return 2.0, 5.0  # Heavy penalty, cap at 5
        elif pct_from_high < -0.30:  # Down 30-50%
            return 1.0, 7.0
        elif pct_from_high < -0.20:  # Down 20-30%
            return 0.5, None
        else:
            return 0.0, None

    def _liquidity_penalty(
        self,
        state: IndicatorStateVector,
        market_context: Dict
    ) -> Tuple[float, Optional[float]]:
        """
        Penalty for low liquidity.
        """
        avg_volume = market_context.get('avg_daily_volume', 1_000_000)
        avg_dollar_volume = market_context.get('avg_dollar_volume', 10_000_000)

        # Very low volume
        if avg_dollar_volume < 500_000:
            return 2.0, 5.0  # Serious liquidity concern
        elif avg_dollar_volume < 1_000_000:
            return 1.0, 7.0
        elif avg_dollar_volume < 5_000_000:
            return 0.5, None
        else:
            return 0.0, None

    def _gap_risk_penalty(
        self,
        state: IndicatorStateVector
    ) -> float:
        """
        Penalty for gap risk (earnings, events).
        """
        # Check if earnings imminent
        days_to_earnings = state.continuous.get('days_to_earnings', 999)

        if days_to_earnings <= 3:
            return 1.0  # High gap risk
        elif days_to_earnings <= 7:
            return 0.5
        else:
            return 0.0

    def _regime_risk_penalty(
        self,
        state: IndicatorStateVector,
        market_context: Dict
    ) -> float:
        """
        Penalty based on market regime.
        """
        market_regime = market_context.get('market_regime', 'NORMAL')

        if market_regime == 'CRISIS':
            return 1.5
        elif market_regime == 'HIGH_VOL':
            return 0.5
        elif market_regime == 'BEAR':
            return 0.3
        else:
            return 0.0

    def _interpret(
        self,
        penalties: Dict[str, float],
        total: float
    ) -> str:
        if total < 0.5:
            return "Minimal risk concerns"

        # Find top risk factors
        top_risks = sorted(penalties.items(), key=lambda x: -x[1])[:2]
        risk_names = [r[0] for r in top_risks if r[1] > 0.3]

        if total >= 2.0:
            return f"Significant risks: {', '.join(risk_names)}"
        elif total >= 1.0:
            return f"Moderate risks: {', '.join(risk_names)}"
        else:
            return f"Minor risks: {', '.join(risk_names)}"


@dataclass
class RiskPenaltyResult:
    """Result from risk penalty calculation."""
    penalties: Dict[str, float]
    total_penalty: float
    score_cap: float
    interpretation: str
```

---

## 6. Ensemble Weighting Rules

### 6.1 Component Weight Configuration

```python
class ScoringWeights:
    """
    Configurable weights for score components.
    """

    # Default weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        'trend': 0.18,
        'momentum': 0.18,
        'volume': 0.12,
        'relative_strength': 0.12,
        'fundamental': 0.08,
        'edge': 0.22,         # Historical probability edge
        'risk_adjustment': 0.10  # Risk-adjusted component
    }

    # Style-based weight presets
    WEIGHT_PRESETS = {
        'momentum': {
            'trend': 0.20,
            'momentum': 0.25,
            'volume': 0.10,
            'relative_strength': 0.15,
            'fundamental': 0.05,
            'edge': 0.20,
            'risk_adjustment': 0.05
        },
        'value': {
            'trend': 0.10,
            'momentum': 0.10,
            'volume': 0.10,
            'relative_strength': 0.10,
            'fundamental': 0.30,
            'edge': 0.20,
            'risk_adjustment': 0.10
        },
        'technical': {
            'trend': 0.25,
            'momentum': 0.25,
            'volume': 0.15,
            'relative_strength': 0.10,
            'fundamental': 0.00,
            'edge': 0.20,
            'risk_adjustment': 0.05
        },
        'balanced': DEFAULT_WEIGHTS.copy()
    }

    @classmethod
    def get_weights(
        cls,
        style: str = 'balanced',
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Get scoring weights for a given style.
        """
        if custom_weights:
            return cls._normalize_weights(custom_weights)

        return cls.WEIGHT_PRESETS.get(style, cls.DEFAULT_WEIGHTS).copy()

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
```

### 6.2 Dynamic Weight Adjustment

```python
class DynamicWeightAdjuster:
    """
    Adjust weights based on data availability and confidence.
    """

    def adjust_weights(
        self,
        base_weights: Dict[str, float],
        subscores: Dict[str, SubscoreResult],
        edge_score: AggregatedEdgeScore,
        data_availability: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Dynamically adjust weights based on:
        1. Data availability (redistribute if data missing)
        2. Confidence (higher weight if more confident)
        """
        adjusted = base_weights.copy()

        # Handle missing data
        missing_components = []
        for component, available in data_availability.items():
            if not available and component in adjusted:
                missing_components.append(component)
                adjusted[component] = 0.0

        # Redistribute missing weight
        if missing_components:
            missing_weight = sum(base_weights[c] for c in missing_components)
            available_components = [c for c in adjusted if c not in missing_components]

            for c in available_components:
                adjusted[c] += missing_weight * adjusted[c] / sum(
                    adjusted[c2] for c2 in available_components
                )

        # Confidence-based adjustment for edge score
        if edge_score.alignment == 'aligned':
            # Boost edge weight if horizons agree
            adjusted['edge'] *= 1.1
        elif edge_score.alignment == 'mixed':
            # Reduce edge weight if horizons disagree
            adjusted['edge'] *= 0.85

        # Renormalize
        return self._normalize_weights(adjusted)

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}
```

---

## 7. Final Score Mapping (0-10)

### 7.1 Score Combiner

```python
class FinalScoreCombiner:
    """
    Combine all components into final 0-10 score.
    """

    def compute(
        self,
        subscores: Dict[str, SubscoreResult],
        edge_score: AggregatedEdgeScore,
        risk_penalty: RiskPenaltyResult,
        weights: Dict[str, float]
    ) -> FinalScoreResult:
        """
        Compute final score.

        Formula:
        1. Weighted average of subscores and edge score
        2. Subtract risk penalty
        3. Apply score cap if present
        4. Clamp to [0, 10]
        5. Round to 1 decimal
        """

        # 1. Collect component scores
        component_scores = {}

        for name, result in subscores.items():
            if name in weights:
                component_scores[name] = result.score

        if not np.isnan(edge_score.score):
            component_scores['edge'] = edge_score.score

        # 2. Compute weighted average
        total_weight = sum(weights.get(k, 0) for k in component_scores)

        if total_weight == 0:
            raw_score = 5.0  # Default to neutral
        else:
            raw_score = sum(
                component_scores[k] * weights.get(k, 0)
                for k in component_scores
            ) / total_weight

        # 3. Apply risk penalty
        penalized_score = raw_score - risk_penalty.total_penalty

        # 4. Apply score cap
        capped_score = min(penalized_score, risk_penalty.score_cap)

        # 5. Clamp and round
        final_score = round(np.clip(capped_score, 0, 10), 1)

        # Determine label
        label = self._score_to_label(final_score)

        return FinalScoreResult(
            score=final_score,
            label=label,
            raw_score=round(raw_score, 2),
            risk_adjusted_score=round(penalized_score, 2),
            component_scores=component_scores,
            weights_used=weights,
            risk_penalty=risk_penalty.total_penalty,
            score_cap_applied=risk_penalty.score_cap < 10,
            breakdown=self._create_breakdown(
                subscores, edge_score, risk_penalty, weights
            )
        )

    def _score_to_label(self, score: float) -> str:
        """Map score to categorical label."""
        if score >= 9.0:
            return 'EXCEPTIONAL'
        elif score >= 7.0:
            return 'STRONG'
        elif score >= 5.0:
            return 'MODERATE'
        elif score >= 3.0:
            return 'WEAK'
        elif score >= 1.0:
            return 'POOR'
        else:
            return 'CRITICAL'

    def _create_breakdown(
        self,
        subscores: Dict[str, SubscoreResult],
        edge_score: AggregatedEdgeScore,
        risk_penalty: RiskPenaltyResult,
        weights: Dict[str, float]
    ) -> Dict:
        """Create detailed score breakdown for explanation."""
        breakdown = {
            'subscores': {},
            'edge': {},
            'risk': {},
            'weights': weights
        }

        for name, result in subscores.items():
            breakdown['subscores'][name] = {
                'score': result.score,
                'weight': weights.get(name, 0),
                'contribution': result.score * weights.get(name, 0),
                'interpretation': result.interpretation,
                'components': result.components
            }

        breakdown['edge'] = {
            'score': edge_score.score,
            'weight': weights.get('edge', 0),
            'by_horizon': edge_score.by_horizon,
            'alignment': edge_score.alignment,
            'interpretation': edge_score.interpretation
        }

        breakdown['risk'] = {
            'total_penalty': risk_penalty.total_penalty,
            'penalties': risk_penalty.penalties,
            'score_cap': risk_penalty.score_cap,
            'interpretation': risk_penalty.interpretation
        }

        return breakdown


@dataclass
class FinalScoreResult:
    """Complete final score with breakdown."""
    score: float                    # 0-10, 1 decimal
    label: str                      # EXCEPTIONAL, STRONG, etc.
    raw_score: float                # Before risk adjustment
    risk_adjusted_score: float      # After penalty, before cap
    component_scores: Dict[str, float]
    weights_used: Dict[str, float]
    risk_penalty: float
    score_cap_applied: bool
    breakdown: Dict

    def to_summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Score: {self.score}/10 ({self.label})",
            "",
            "Component Breakdown:",
        ]

        for name, data in self.breakdown['subscores'].items():
            lines.append(f"  {name}: {data['score']:.1f} × {data['weight']:.0%} = {data['contribution']:.2f}")

        edge = self.breakdown['edge']
        lines.append(f"  edge: {edge['score']:.1f} × {self.weights_used.get('edge', 0):.0%}")

        if self.risk_penalty > 0:
            lines.append(f"\nRisk Penalty: -{self.risk_penalty:.1f}")

        if self.score_cap_applied:
            lines.append(f"Score Cap Applied: {self.breakdown['risk']['score_cap']}")

        return "\n".join(lines)
```

### 7.2 Score Normalization Rules

```python
class ScoreNormalizer:
    """
    Ensure final scores follow expected distribution.
    """

    # Target distribution (approximate)
    TARGET_DISTRIBUTION = {
        'EXCEPTIONAL (9-10)': 0.02,   # 2% of universe
        'STRONG (7-8)': 0.13,         # 13%
        'MODERATE (5-6)': 0.40,       # 40%
        'WEAK (3-4)': 0.30,           # 30%
        'POOR (1-2)': 0.13,           # 13%
        'CRITICAL (0)': 0.02          # 2%
    }

    def calibrate_to_universe(
        self,
        scores: List[FinalScoreResult],
        method: str = 'percentile'
    ) -> List[FinalScoreResult]:
        """
        Optional: Calibrate scores relative to universe.

        This ensures score distribution matches expectations.
        Use sparingly - raw scores are often more informative.
        """
        if method == 'percentile':
            return self._percentile_calibration(scores)
        elif method == 'z_score':
            return self._zscore_calibration(scores)
        else:
            return scores

    def _percentile_calibration(
        self,
        scores: List[FinalScoreResult]
    ) -> List[FinalScoreResult]:
        """
        Map scores to percentiles, then to 0-10.
        """
        raw_scores = [s.score for s in scores]
        percentiles = [
            (np.sum(np.array(raw_scores) < s) / len(raw_scores)) * 100
            for s in raw_scores
        ]

        # Map percentile to 0-10
        calibrated = []
        for score_result, pctl in zip(scores, percentiles):
            calibrated_score = pctl / 10

            # Create new result with calibrated score
            new_result = FinalScoreResult(
                score=round(calibrated_score, 1),
                label=self._score_to_label(calibrated_score),
                raw_score=score_result.raw_score,
                risk_adjusted_score=score_result.risk_adjusted_score,
                component_scores=score_result.component_scores,
                weights_used=score_result.weights_used,
                risk_penalty=score_result.risk_penalty,
                score_cap_applied=score_result.score_cap_applied,
                breakdown={
                    **score_result.breakdown,
                    'calibration': {
                        'method': 'percentile',
                        'percentile': pctl,
                        'original_score': score_result.score
                    }
                }
            )
            calibrated.append(new_result)

        return calibrated
```

---

## 8. Complete Scoring Pipeline

```python
class ScoringPipeline:
    """
    Complete end-to-end scoring pipeline.
    """

    def __init__(
        self,
        style: str = 'balanced',
        horizons: List[str] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ):
        self.weights = ScoringWeights.get_weights(style, custom_weights)
        self.horizons = horizons or ['5d', '21d', '63d']

        # Initialize components
        self.trend_scorer = TrendSubscore()
        self.momentum_scorer = MomentumSubscore()
        self.volume_scorer = VolumeSubscore()
        self.rs_scorer = RelativeStrengthSubscore()
        self.fundamental_scorer = FundamentalSubscore()

        self.edge_scorer = EdgeScore()
        self.edge_aggregator = MultiHorizonEdgeAggregator()

        self.risk_calculator = RiskPenaltyCalculator()
        self.weight_adjuster = DynamicWeightAdjuster()
        self.score_combiner = FinalScoreCombiner()

    def score(
        self,
        state: IndicatorStateVector,
        probability_estimates: Dict[str, ProbabilityEstimate],
        market_context: Dict,
        fundamentals: Optional[Dict] = None
    ) -> FinalScoreResult:
        """
        Compute complete score for a ticker.

        Args:
            state: Current indicator state vector
            probability_estimates: Probability estimates per horizon
            market_context: Market-level context (regime, etc.)
            fundamentals: Optional fundamental data

        Returns:
            Complete FinalScoreResult
        """
        # 1. Compute subscores
        price_direction = 1 if state.continuous.get('roc_5', 0) > 0 else -1

        subscores = {
            'trend': self.trend_scorer.compute(state),
            'momentum': self.momentum_scorer.compute(state),
            'volume': self.volume_scorer.compute(state, price_direction),
            'relative_strength': self.rs_scorer.compute(state),
            'fundamental': self.fundamental_scorer.compute(state, fundamentals)
        }

        # 2. Compute edge scores
        edge_scores = {}
        for horizon in self.horizons:
            if horizon in probability_estimates:
                edge_scores[horizon] = self.edge_scorer.compute(
                    probability_estimates[horizon],
                    horizon
                )

        aggregated_edge = self.edge_aggregator.aggregate(edge_scores)

        # 3. Compute risk penalties
        risk_penalty = self.risk_calculator.compute(
            state, market_context,
            base_score=aggregated_edge.score
        )

        # 4. Adjust weights dynamically
        data_availability = {
            'fundamental': fundamentals is not None,
            'edge': len(edge_scores) > 0
        }

        adjusted_weights = self.weight_adjuster.adjust_weights(
            self.weights,
            subscores,
            aggregated_edge,
            data_availability
        )

        # 5. Combine into final score
        final_result = self.score_combiner.compute(
            subscores,
            aggregated_edge,
            risk_penalty,
            adjusted_weights
        )

        return final_result

    def score_universe(
        self,
        universe_data: List[Dict]
    ) -> List[Tuple[str, FinalScoreResult]]:
        """
        Score entire universe.

        Args:
            universe_data: List of dicts with ticker data

        Returns:
            List of (ticker, FinalScoreResult) sorted by score desc
        """
        results = []

        for data in universe_data:
            ticker = data['ticker']
            state = data['state']
            probs = data['probability_estimates']
            context = data.get('market_context', {})
            fundamentals = data.get('fundamentals')

            result = self.score(state, probs, context, fundamentals)
            results.append((ticker, result))

        # Sort by score descending
        results.sort(key=lambda x: -x[1].score)

        return results
```

---

## 9. Score Interpretation Guide

```python
SCORE_INTERPRETATION = {
    10.0: {
        'label': 'EXCEPTIONAL',
        'description': 'Extremely rare, all factors strongly aligned',
        'frequency': '<0.5% of observations',
        'action': 'High conviction opportunity'
    },
    9.0: {
        'label': 'EXCEPTIONAL',
        'description': 'Outstanding setup across most dimensions',
        'frequency': '~1-2% of observations',
        'action': 'Strong consideration for position'
    },
    8.0: {
        'label': 'STRONG',
        'description': 'Multiple strong factors, limited concerns',
        'frequency': '~5% of observations',
        'action': 'Worthy of detailed analysis'
    },
    7.0: {
        'label': 'STRONG',
        'description': 'Good overall profile with some strengths',
        'frequency': '~10% of observations',
        'action': 'Add to watchlist, look for entry'
    },
    6.0: {
        'label': 'MODERATE',
        'description': 'Above average, mixed signals',
        'frequency': '~20% of observations',
        'action': 'Monitor for improvement'
    },
    5.0: {
        'label': 'MODERATE',
        'description': 'Neutral, no strong edge either way',
        'frequency': '~25% of observations',
        'action': 'No action recommended'
    },
    4.0: {
        'label': 'WEAK',
        'description': 'Below average, some concerns',
        'frequency': '~20% of observations',
        'action': 'Avoid or exit existing positions'
    },
    3.0: {
        'label': 'WEAK',
        'description': 'Multiple negative factors',
        'frequency': '~10% of observations',
        'action': 'Avoid'
    },
    2.0: {
        'label': 'POOR',
        'description': 'Significant issues across dimensions',
        'frequency': '~5% of observations',
        'action': 'Strong avoid, potential short'
    },
    1.0: {
        'label': 'POOR',
        'description': 'Major red flags',
        'frequency': '~2% of observations',
        'action': 'Avoid entirely'
    },
    0.0: {
        'label': 'CRITICAL',
        'description': 'Extreme concerns, data issues, or halted',
        'frequency': '<0.5% of observations',
        'action': 'Do not trade'
    }
}
```

---

## 10. Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCORING FRAMEWORK SUMMARY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SUBSCORES (each 0-10)                                                      │
│  ─────────────────────                                                      │
│  • Trend (18%): Direction, strength, alignment, freshness                  │
│  • Momentum (18%): RSI, ROC, MACD, Stochastic                              │
│  • Volume (12%): Level, money flow, OBV, confirmation                      │
│  • Relative Strength (12%): vs SPY, sector rank, RS trend                  │
│  • Fundamental (8%): Valuation, quality, earnings, health                  │
│                                                                             │
│  EDGE SCORE (22%)                                                           │
│  ───────────────                                                            │
│  • Probability → Score mapping (10% prob → 3, 25% → 6.5, 50% → 9)          │
│  • Adjusted for reliability and confidence width                           │
│  • Multi-horizon aggregation (5d, 21d, 63d)                                │
│                                                                             │
│  RISK PENALTIES                                                             │
│  ──────────────                                                             │
│  • Volatility: Up to 1.5 points                                            │
│  • Drawdown: Up to 2.0 points                                              │
│  • Liquidity: Up to 2.0 points                                             │
│  • Gap risk: Up to 1.0 points                                              │
│  • Regime: Up to 1.5 points                                                │
│  • Score caps for severe risks                                             │
│                                                                             │
│  FINAL SCORE                                                                │
│  ───────────                                                                │
│  • Weighted combination of all components                                   │
│  • Risk penalty subtracted                                                  │
│  • Score cap applied if needed                                             │
│  • Clamped to [0, 10], rounded to 1 decimal                                │
│                                                                             │
│  LABELS                                                                     │
│  ──────                                                                     │
│  9-10: EXCEPTIONAL | 7-8: STRONG | 5-6: MODERATE                           │
│  3-4: WEAK | 1-2: POOR | 0: CRITICAL                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Assumptions

1. **Determinism:** Same inputs always produce same score (no randomness)
2. **Linearity:** Subscores combine linearly (weighted average)
3. **Independence:** Subscores are approximately independent
4. **Risk Asymmetry:** Risk penalties are additive, not multiplicative
5. **Score Stability:** Small input changes produce small score changes
6. **Interpretability Priority:** Prefer explainability over marginal accuracy
7. **Equal Scaling:** All subscores use 0-10 scale for comparability

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
