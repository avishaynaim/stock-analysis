# Historical High-Gain Probability Engine

## 1. Overview

The Historical High-Gain Probability Engine is the core analytical component that estimates the probability of achieving "very high gains" based on current indicator states. It leverages historical data to compute conditional probabilities using multiple estimation approaches.

### Core Question

> "Given the current state of indicators for this ticker, what is the probability of achieving a very high gain over the forward horizon, based on historical evidence?"

### Design Principles

1. **Evidence-Based:** All probabilities derived from historical data
2. **Uncertainty-Aware:** Explicit confidence intervals and sample size considerations
3. **Regime-Adaptive:** Accounts for changing market conditions
4. **Multi-Estimator:** Ensemble of approaches for robustness
5. **Transparent:** Clear audit trail for all probability estimates

---

## 2. Definition of "Very High Gain"

### 2.1 Gain Threshold Philosophy

"Very high gain" is defined relative to:
1. **Timeframe-specific expectations** (5-day vs. 252-day returns differ)
2. **Asset-specific volatility** (normalize by typical volatility)
3. **Market regime** (different thresholds for bull/bear markets)

### 2.2 Absolute Gain Thresholds by Timeframe

```python
@dataclass
class GainThresholds:
    """
    Definition of gain thresholds per forward horizon.

    Based on historical S&P 500 return distributions:
    - Daily: σ ≈ 1%, mean ≈ 0.04%
    - Weekly: σ ≈ 2.3%, mean ≈ 0.2%
    - Monthly: σ ≈ 4.5%, mean ≈ 0.8%
    - Quarterly: σ ≈ 8%, mean ≈ 2.5%
    - Annual: σ ≈ 16%, mean ≈ 10%
    """

    # Forward horizon definitions
    horizons: Dict[str, int] = field(default_factory=lambda: {
        '1d': 1,      # 1 trading day
        '5d': 5,      # 1 week
        '10d': 10,    # 2 weeks
        '21d': 21,    # 1 month
        '63d': 63,    # 1 quarter
        '126d': 126,  # 6 months
        '252d': 252,  # 1 year
    })

    # "Very high gain" thresholds (absolute %)
    # Approximately top 10-15% of historical returns
    very_high_gain: Dict[str, float] = field(default_factory=lambda: {
        '1d': 0.025,    # 2.5% in 1 day (~2.5σ move)
        '5d': 0.05,     # 5% in 1 week (~2.2σ)
        '10d': 0.07,    # 7% in 2 weeks (~2.0σ)
        '21d': 0.10,    # 10% in 1 month (~2.2σ)
        '63d': 0.15,    # 15% in 1 quarter (~1.9σ)
        '126d': 0.20,   # 20% in 6 months (~1.5σ)
        '252d': 0.30,   # 30% in 1 year (~1.25σ)
    })

    # "High gain" thresholds (moderate threshold)
    # Approximately top 25-30% of historical returns
    high_gain: Dict[str, float] = field(default_factory=lambda: {
        '1d': 0.015,    # 1.5%
        '5d': 0.03,     # 3%
        '10d': 0.045,   # 4.5%
        '21d': 0.06,    # 6%
        '63d': 0.10,    # 10%
        '126d': 0.14,   # 14%
        '252d': 0.20,   # 20%
    })

    # Positive gain (any positive return)
    positive_gain: Dict[str, float] = field(default_factory=lambda: {
        horizon: 0.0 for horizon in horizons
    })


# Default thresholds instance
DEFAULT_THRESHOLDS = GainThresholds()
```

### 2.3 Volatility-Adjusted Gain Thresholds

```python
class VolatilityAdjustedThresholds:
    """
    Adjust gain thresholds based on asset's own volatility.

    Rationale: A 10% move in a low-vol utility stock is more significant
    than a 10% move in a high-vol biotech.
    """

    def __init__(
        self,
        base_thresholds: GainThresholds,
        vol_multiplier: float = 2.0
    ):
        """
        Args:
            base_thresholds: Absolute thresholds as baseline
            vol_multiplier: Number of standard deviations for "very high"
        """
        self.base = base_thresholds
        self.vol_multiplier = vol_multiplier

    def get_threshold(
        self,
        horizon: str,
        asset_volatility: float,
        threshold_type: str = 'very_high'
    ) -> float:
        """
        Get volatility-adjusted threshold.

        Args:
            horizon: Forward horizon ('5d', '21d', etc.)
            asset_volatility: Asset's annualized volatility (e.g., 0.25 = 25%)
            threshold_type: 'very_high', 'high', or 'positive'

        Returns:
            Adjusted gain threshold

        Example:
            Asset with 40% annual vol (high):
            - 21d threshold = 2.0 * (0.40 / sqrt(252/21)) ≈ 23%

            Asset with 15% annual vol (low):
            - 21d threshold = 2.0 * (0.15 / sqrt(252/21)) ≈ 8.7%
        """
        # Get base threshold
        if threshold_type == 'very_high':
            base = self.base.very_high_gain[horizon]
        elif threshold_type == 'high':
            base = self.base.high_gain[horizon]
        else:
            return 0.0

        # Days in horizon
        days = self.base.horizons[horizon]

        # Scale volatility to horizon
        horizon_vol = asset_volatility * np.sqrt(days / 252)

        # Volatility-adjusted threshold
        vol_adjusted = self.vol_multiplier * horizon_vol

        # Use maximum of base and vol-adjusted
        # (don't lower threshold for low-vol stocks too much)
        return max(base, vol_adjusted * 0.7, base * 0.5)

    def classify_return(
        self,
        forward_return: float,
        horizon: str,
        asset_volatility: float
    ) -> str:
        """
        Classify a return into gain categories.

        Returns:
            'VERY_HIGH', 'HIGH', 'POSITIVE', 'NEGATIVE', 'VERY_NEGATIVE'
        """
        very_high = self.get_threshold(horizon, asset_volatility, 'very_high')
        high = self.get_threshold(horizon, asset_volatility, 'high')

        if forward_return >= very_high:
            return 'VERY_HIGH'
        elif forward_return >= high:
            return 'HIGH'
        elif forward_return >= 0:
            return 'POSITIVE'
        elif forward_return >= -high:
            return 'NEGATIVE'
        else:
            return 'VERY_NEGATIVE'
```

### 2.4 Regime-Specific Thresholds

```python
class RegimeAwareThresholds:
    """
    Adjust thresholds based on market regime.

    In high-volatility regimes, larger moves are more common,
    so thresholds should be higher.
    """

    REGIME_MULTIPLIERS = {
        # Volatility regime adjustments
        'LOW_VOL': 0.7,       # Lower thresholds in calm markets
        'NORMAL_VOL': 1.0,    # Base thresholds
        'HIGH_VOL': 1.3,      # Higher thresholds in volatile markets
        'EXTREME_VOL': 1.6,   # Much higher in extreme conditions

        # Trend regime adjustments (for directional thresholds)
        'STRONG_UPTREND': 1.1,   # Slightly higher in strong trends
        'UPTREND': 1.0,
        'RANGE': 0.9,            # Lower in ranging markets
        'DOWNTREND': 1.0,
        'STRONG_DOWNTREND': 1.2, # Higher (bounces are bigger)
    }

    def adjust_threshold(
        self,
        base_threshold: float,
        vol_regime: str,
        trend_regime: Optional[str] = None
    ) -> float:
        """
        Adjust threshold for current regime.
        """
        multiplier = self.REGIME_MULTIPLIERS.get(vol_regime, 1.0)

        if trend_regime:
            trend_mult = self.REGIME_MULTIPLIERS.get(trend_regime, 1.0)
            multiplier = (multiplier + trend_mult) / 2

        return base_threshold * multiplier
```

---

## 3. State Matching Logic

### 3.1 State Matching Overview

State matching finds historical instances where indicator states were similar to the current state. This forms the basis for empirical probability estimation.

```
Current State → Find Historical Matches → Compute Forward Returns → Estimate Probability
```

### 3.2 State Representation for Matching

```python
@dataclass
class MatchableState:
    """
    State representation optimized for historical matching.
    """

    # Discrete state components (exact match)
    discrete_state: Dict[str, str]

    # Continuous state components (similarity match)
    continuous_state: Dict[str, float]

    # Context (for stratification)
    regime: str
    sector: str
    market_cap_bucket: str

    def get_discrete_key(self) -> str:
        """
        Create hashable key from discrete states.
        Used for exact matching.
        """
        items = sorted(self.discrete_state.items())
        return '|'.join(f"{k}:{v}" for k, v in items)

    def get_continuous_vector(self) -> np.ndarray:
        """
        Create numpy array from continuous states.
        Used for similarity matching.
        """
        return np.array(list(self.continuous_state.values()))


class StateKeyBuilder:
    """
    Build matching keys from indicator states.
    """

    # Default discrete features for matching
    DEFAULT_DISCRETE_FEATURES = [
        'rsi_zone',           # OVERSOLD, NEUTRAL, OVERBOUGHT, etc.
        'trend_regime',       # UPTREND, RANGE, DOWNTREND
        'adx_zone',           # NO_TREND, WEAK, MODERATE, STRONG
        'volume_zone',        # LOW, NORMAL, HIGH
        'volatility_regime',  # LOW_VOL, NORMAL_VOL, HIGH_VOL
        'macd_zone',          # BEARISH, NEUTRAL, BULLISH
    ]

    # Default continuous features for similarity
    DEFAULT_CONTINUOUS_FEATURES = [
        'rsi_norm',
        'macd_z',
        'adx_norm',
        'ema_50_pct',
        'atrp',
        'volume_ratio',
        'roc_21',
        'rs_vs_spy_21',
    ]

    def __init__(
        self,
        discrete_features: Optional[List[str]] = None,
        continuous_features: Optional[List[str]] = None
    ):
        self.discrete_features = discrete_features or self.DEFAULT_DISCRETE_FEATURES
        self.continuous_features = continuous_features or self.DEFAULT_CONTINUOUS_FEATURES

    def build_matchable_state(
        self,
        state_vector: IndicatorStateVector
    ) -> MatchableState:
        """
        Convert full state vector to matchable state.
        """
        discrete = {
            f: state_vector.categorical.get(f, 'UNKNOWN')
            for f in self.discrete_features
        }

        continuous = {
            f: state_vector.continuous.get(f, np.nan)
            for f in self.continuous_features
        }

        return MatchableState(
            discrete_state=discrete,
            continuous_state=continuous,
            regime=state_vector.regimes.get('composite', 'UNKNOWN'),
            sector=state_vector.categorical.get('sector', 'UNKNOWN'),
            market_cap_bucket=state_vector.categorical.get('market_cap_bucket', 'UNKNOWN')
        )
```

### 3.3 Exact State Matching

```python
class ExactStateMatcher:
    """
    Find historical instances with exact discrete state matches.
    """

    def __init__(self, min_matches: int = 30):
        self.min_matches = min_matches
        self.state_index: Dict[str, List[int]] = {}

    def build_index(
        self,
        historical_states: pd.DataFrame,
        discrete_cols: List[str]
    ) -> None:
        """
        Build index for fast exact matching.

        Args:
            historical_states: DataFrame with columns for each discrete feature
            discrete_cols: Columns to use for matching
        """
        self.state_index = {}

        for idx, row in historical_states.iterrows():
            key = '|'.join(str(row[col]) for col in discrete_cols)
            if key not in self.state_index:
                self.state_index[key] = []
            self.state_index[key].append(idx)

    def find_exact_matches(
        self,
        query_state: MatchableState
    ) -> List[int]:
        """
        Find all historical instances with exact state match.

        Returns:
            List of row indices in historical data
        """
        key = query_state.get_discrete_key()
        return self.state_index.get(key, [])

    def find_relaxed_matches(
        self,
        query_state: MatchableState,
        relax_features: List[str]
    ) -> List[int]:
        """
        Find matches with some features relaxed (ignored).

        Used when exact matches are insufficient.
        """
        # Build partial key excluding relaxed features
        partial_key = '|'.join(
            f"{k}:{v}" for k, v in sorted(query_state.discrete_state.items())
            if k not in relax_features
        )

        matches = []
        for key, indices in self.state_index.items():
            # Check if partial key matches
            if self._partial_match(key, partial_key, relax_features):
                matches.extend(indices)

        return matches

    def hierarchical_match(
        self,
        query_state: MatchableState,
        feature_priority: List[str]
    ) -> Tuple[List[int], int]:
        """
        Hierarchical matching: Start exact, relax features until min_matches met.

        Args:
            query_state: State to match
            feature_priority: Features in order of importance (least important first for relaxation)

        Returns:
            (matched_indices, num_features_used)
        """
        # Start with all features
        matches = self.find_exact_matches(query_state)

        if len(matches) >= self.min_matches:
            return matches, len(feature_priority)

        # Progressively relax features
        relax_features = []
        for feature in feature_priority:
            relax_features.append(feature)
            matches = self.find_relaxed_matches(query_state, relax_features)

            if len(matches) >= self.min_matches:
                return matches, len(feature_priority) - len(relax_features)

        # Return whatever we have
        return matches, 0
```

### 3.4 Similarity-Based Matching

```python
class SimilarityMatcher:
    """
    Find historical instances with similar continuous states.
    Uses distance metrics for flexible matching.
    """

    def __init__(
        self,
        n_neighbors: int = 100,
        distance_metric: str = 'euclidean'
    ):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.index = None
        self.scaler = StandardScaler()

    def build_index(
        self,
        historical_vectors: np.ndarray
    ) -> None:
        """
        Build spatial index for fast nearest neighbor search.

        Uses different indexing strategies based on dimensionality:
        - Low dim (<20): KD-Tree
        - High dim (>=20): Ball Tree or LSH
        """
        # Normalize features
        normalized = self.scaler.fit_transform(historical_vectors)

        # Build appropriate index
        if normalized.shape[1] < 20:
            from sklearn.neighbors import KDTree
            self.index = KDTree(normalized, metric=self.distance_metric)
        else:
            from sklearn.neighbors import BallTree
            self.index = BallTree(normalized, metric=self.distance_metric)

    def find_similar(
        self,
        query_vector: np.ndarray,
        n_neighbors: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest neighbors to query state.

        Returns:
            (distances, indices) arrays
        """
        n = n_neighbors or self.n_neighbors

        # Normalize query
        query_norm = self.scaler.transform(query_vector.reshape(1, -1))

        # Query index
        distances, indices = self.index.query(query_norm, k=n)

        return distances[0], indices[0]

    def find_similar_in_regime(
        self,
        query_vector: np.ndarray,
        regime: str,
        regime_labels: np.ndarray,
        n_neighbors: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar states within the same regime.
        """
        # Filter to regime
        regime_mask = regime_labels == regime
        regime_indices = np.where(regime_mask)[0]

        if len(regime_indices) == 0:
            return np.array([]), np.array([])

        # Query within regime
        distances, indices = self.find_similar(query_vector, len(regime_indices))

        # Map back to original indices
        return distances, regime_indices[indices]


class HybridStateMatcher:
    """
    Combines exact and similarity matching.

    Strategy:
    1. Find exact discrete matches
    2. Rank by continuous similarity
    3. Return top-K most similar within exact matches
    """

    def __init__(
        self,
        exact_matcher: ExactStateMatcher,
        similarity_matcher: SimilarityMatcher,
        min_matches: int = 30,
        max_matches: int = 500
    ):
        self.exact_matcher = exact_matcher
        self.similarity_matcher = similarity_matcher
        self.min_matches = min_matches
        self.max_matches = max_matches

    def find_matches(
        self,
        query_state: MatchableState,
        historical_vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Find best historical matches.

        Returns:
            (indices, similarities, metadata)
        """
        metadata = {
            'match_type': None,
            'num_exact_matches': 0,
            'features_used': 0,
            'similarity_threshold': None
        }

        # Step 1: Exact matching
        exact_indices, features_used = self.exact_matcher.hierarchical_match(
            query_state,
            feature_priority=self.exact_matcher.DEFAULT_DISCRETE_FEATURES[::-1]
        )

        metadata['num_exact_matches'] = len(exact_indices)
        metadata['features_used'] = features_used

        if len(exact_indices) >= self.min_matches:
            metadata['match_type'] = 'exact' if features_used == len(
                self.exact_matcher.DEFAULT_DISCRETE_FEATURES
            ) else 'relaxed'

            # Rank by similarity within exact matches
            query_vector = query_state.get_continuous_vector()
            exact_vectors = historical_vectors[exact_indices]

            # Compute similarities
            similarities = self._compute_similarities(query_vector, exact_vectors)

            # Sort and take top matches
            sorted_idx = np.argsort(similarities)[::-1][:self.max_matches]

            return (
                np.array(exact_indices)[sorted_idx],
                similarities[sorted_idx],
                metadata
            )

        # Step 2: Fall back to pure similarity matching
        metadata['match_type'] = 'similarity'

        query_vector = query_state.get_continuous_vector()
        distances, indices = self.similarity_matcher.find_similar(
            query_vector,
            n_neighbors=self.max_matches
        )

        # Convert distances to similarities
        similarities = 1 / (1 + distances)
        metadata['similarity_threshold'] = float(np.min(similarities))

        return indices, similarities, metadata

    def _compute_similarities(
        self,
        query: np.ndarray,
        candidates: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and candidates.
        """
        # Normalize
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        cand_norms = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)

        return cand_norms @ query_norm
```

---

## 4. Estimator A: Empirical Conditional Probability

### 4.1 Core Algorithm

```python
class EmpiricalProbabilityEstimator:
    """
    Estimator A: Compute probability from historical frequency.

    P(high_gain | state) = count(high_gain AND state) / count(state)

    This is the most transparent and interpretable estimator.
    """

    def __init__(
        self,
        thresholds: GainThresholds,
        min_samples: int = 30,
        use_bayesian_prior: bool = True,
        prior_alpha: float = 1.0,
        prior_beta: float = 10.0
    ):
        """
        Args:
            thresholds: Gain threshold definitions
            min_samples: Minimum matches for reliable estimate
            use_bayesian_prior: Use Beta prior for small samples
            prior_alpha, prior_beta: Beta prior parameters
                Default: Beta(1, 10) = prior belief of ~9% probability
        """
        self.thresholds = thresholds
        self.min_samples = min_samples
        self.use_bayesian_prior = use_bayesian_prior
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def estimate(
        self,
        matched_indices: np.ndarray,
        forward_returns: pd.DataFrame,
        horizon: str,
        threshold_type: str = 'very_high',
        weights: Optional[np.ndarray] = None
    ) -> ProbabilityEstimate:
        """
        Estimate probability of exceeding threshold.

        Args:
            matched_indices: Indices of matched historical states
            forward_returns: DataFrame with forward returns by horizon
            horizon: Forward horizon ('5d', '21d', etc.)
            threshold_type: 'very_high', 'high', or 'positive'
            weights: Optional similarity weights for matches

        Returns:
            ProbabilityEstimate with point estimate and confidence interval
        """
        # Get forward returns for matched states
        returns = forward_returns.loc[matched_indices, f'fwd_return_{horizon}'].values

        # Remove NaN (happens at end of data)
        valid_mask = ~np.isnan(returns)
        returns = returns[valid_mask]

        if weights is not None:
            weights = weights[valid_mask]

        n_samples = len(returns)

        if n_samples == 0:
            return ProbabilityEstimate(
                probability=np.nan,
                confidence_low=np.nan,
                confidence_high=np.nan,
                n_samples=0,
                method='empirical',
                reliability='insufficient_data'
            )

        # Get threshold
        if threshold_type == 'very_high':
            threshold = self.thresholds.very_high_gain[horizon]
        elif threshold_type == 'high':
            threshold = self.thresholds.high_gain[horizon]
        else:
            threshold = 0.0

        # Count successes
        successes = (returns >= threshold)

        if weights is not None:
            # Weighted count
            weights = weights / weights.sum()
            n_success = (successes * weights).sum() * n_samples
        else:
            n_success = successes.sum()

        # Compute probability
        if self.use_bayesian_prior:
            # Bayesian estimate with Beta prior
            # Posterior: Beta(alpha + successes, beta + failures)
            posterior_alpha = self.prior_alpha + n_success
            posterior_beta = self.prior_beta + (n_samples - n_success)

            # Point estimate (posterior mean)
            probability = posterior_alpha / (posterior_alpha + posterior_beta)

            # 90% credible interval
            from scipy.stats import beta
            conf_low = beta.ppf(0.05, posterior_alpha, posterior_beta)
            conf_high = beta.ppf(0.95, posterior_alpha, posterior_beta)
        else:
            # Frequentist estimate
            probability = n_success / n_samples

            # Wilson score interval for proportions
            conf_low, conf_high = self._wilson_interval(n_success, n_samples, 0.90)

        # Determine reliability
        if n_samples < self.min_samples // 3:
            reliability = 'very_low'
        elif n_samples < self.min_samples:
            reliability = 'low'
        elif n_samples < self.min_samples * 3:
            reliability = 'moderate'
        else:
            reliability = 'high'

        return ProbabilityEstimate(
            probability=probability,
            confidence_low=conf_low,
            confidence_high=conf_high,
            n_samples=n_samples,
            n_successes=int(n_success),
            threshold=threshold,
            method='empirical_bayesian' if self.use_bayesian_prior else 'empirical_frequentist',
            reliability=reliability,
            metadata={
                'horizon': horizon,
                'threshold_type': threshold_type,
                'mean_return': float(np.mean(returns)),
                'median_return': float(np.median(returns)),
                'return_std': float(np.std(returns))
            }
        )

    def _wilson_interval(
        self,
        successes: float,
        n: int,
        confidence: float
    ) -> Tuple[float, float]:
        """
        Wilson score interval for binomial proportion.
        More accurate than normal approximation for small samples.
        """
        from scipy.stats import norm

        z = norm.ppf(1 - (1 - confidence) / 2)
        p_hat = successes / n

        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

        return max(0, center - margin), min(1, center + margin)

    def estimate_multi_horizon(
        self,
        matched_indices: np.ndarray,
        forward_returns: pd.DataFrame,
        horizons: List[str] = None,
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, ProbabilityEstimate]:
        """
        Estimate probabilities for multiple horizons.
        """
        horizons = horizons or list(self.thresholds.horizons.keys())

        return {
            horizon: self.estimate(
                matched_indices,
                forward_returns,
                horizon,
                weights=weights
            )
            for horizon in horizons
        }


@dataclass
class ProbabilityEstimate:
    """
    Complete probability estimate with uncertainty quantification.
    """

    probability: float              # Point estimate
    confidence_low: float           # Lower bound (e.g., 5th percentile)
    confidence_high: float          # Upper bound (e.g., 95th percentile)
    n_samples: int                  # Number of historical matches
    n_successes: int = 0            # Number exceeding threshold
    threshold: float = 0.0          # Gain threshold used
    method: str = ''                # Estimation method
    reliability: str = ''           # 'very_low', 'low', 'moderate', 'high'
    metadata: Dict = field(default_factory=dict)

    @property
    def confidence_width(self) -> float:
        """Width of confidence interval."""
        return self.confidence_high - self.confidence_low

    @property
    def is_reliable(self) -> bool:
        """Is estimate reliable for decision-making?"""
        return self.reliability in ['moderate', 'high']

    def to_dict(self) -> Dict:
        return asdict(self)
```

### 4.2 Stratified Empirical Estimation

```python
class StratifiedEmpiricalEstimator:
    """
    Compute empirical probabilities stratified by regime, sector, etc.

    Addresses: "Does this signal work differently in different contexts?"
    """

    def __init__(
        self,
        base_estimator: EmpiricalProbabilityEstimator,
        stratification_cols: List[str] = None
    ):
        self.base_estimator = base_estimator
        self.stratification_cols = stratification_cols or [
            'volatility_regime',
            'trend_regime',
            'sector'
        ]

    def estimate_stratified(
        self,
        matched_indices: np.ndarray,
        forward_returns: pd.DataFrame,
        stratification_data: pd.DataFrame,
        current_strata: Dict[str, str],
        horizon: str
    ) -> Dict[str, ProbabilityEstimate]:
        """
        Compute probabilities for each stratum and overall.

        Returns:
            Dict with 'overall' and per-stratum estimates
        """
        results = {}

        # Overall estimate
        results['overall'] = self.base_estimator.estimate(
            matched_indices, forward_returns, horizon
        )

        # Per-stratum estimates
        for col in self.stratification_cols:
            if col not in stratification_data.columns:
                continue

            current_value = current_strata.get(col)
            if current_value is None:
                continue

            # Filter to current stratum
            stratum_mask = stratification_data.loc[matched_indices, col] == current_value
            stratum_indices = matched_indices[stratum_mask]

            if len(stratum_indices) >= 10:  # Minimum for stratum estimate
                results[f'{col}:{current_value}'] = self.base_estimator.estimate(
                    stratum_indices, forward_returns, horizon
                )

        return results
```

---

## 5. Estimator B: Supervised Probability Model

### 5.1 Core Algorithm

```python
class SupervisedProbabilityEstimator:
    """
    Estimator B: Train a classifier to predict high-gain probability.

    Advantages over empirical:
    - Handles high-dimensional continuous features
    - Generalizes to unseen state combinations
    - Can capture non-linear interactions

    Model choices:
    - Logistic Regression: Interpretable, fast
    - Gradient Boosting: Better accuracy, feature importance
    - Calibrated models: Better probability estimates
    """

    def __init__(
        self,
        model_type: str = 'gradient_boosting',
        calibration: str = 'isotonic',
        cv_folds: int = 5
    ):
        """
        Args:
            model_type: 'logistic', 'gradient_boosting', 'random_forest'
            calibration: 'isotonic', 'sigmoid', or None
            cv_folds: Cross-validation folds for calibration
        """
        self.model_type = model_type
        self.calibration = calibration
        self.cv_folds = cv_folds
        self.model = None
        self.feature_names = None
        self.thresholds = GainThresholds()

    def train(
        self,
        X: np.ndarray,
        forward_returns: np.ndarray,
        horizon: str,
        feature_names: List[str],
        threshold_type: str = 'very_high',
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train probability model.

        Args:
            X: Feature matrix (n_samples, n_features)
            forward_returns: Forward returns array
            horizon: Horizon for which model is trained
            feature_names: Names of features
            threshold_type: Target threshold type
            sample_weights: Optional time-decay or importance weights

        Returns:
            Training metrics
        """
        self.feature_names = feature_names

        # Create binary target
        threshold = self.thresholds.very_high_gain[horizon]
        y = (forward_returns >= threshold).astype(int)

        # Remove NaN
        valid_mask = ~np.isnan(forward_returns)
        X = X[valid_mask]
        y = y[valid_mask]
        if sample_weights is not None:
            sample_weights = sample_weights[valid_mask]

        # Build base model
        base_model = self._build_base_model()

        # Train with calibration
        if self.calibration:
            from sklearn.calibration import CalibratedClassifierCV
            self.model = CalibratedClassifierCV(
                base_model,
                method=self.calibration,
                cv=self.cv_folds
            )
        else:
            self.model = base_model

        # Fit
        if sample_weights is not None:
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            self.model.fit(X, y)

        # Compute training metrics
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

        metrics = {
            'n_samples': len(y),
            'n_positive': y.sum(),
            'base_rate': y.mean(),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'brier_score': brier_score_loss(y, y_pred_proba),
            'log_loss': log_loss(y, y_pred_proba),
            'horizon': horizon,
            'threshold': threshold
        }

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            metrics['feature_importance'] = dict(zip(
                feature_names,
                self.model.feature_importances_
            ))
        elif hasattr(self.model, 'coef_'):
            metrics['feature_importance'] = dict(zip(
                feature_names,
                np.abs(self.model.coef_[0])
            ))

        return metrics

    def _build_base_model(self):
        """Build base classifier based on model_type."""
        if self.model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )

        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42
            )

        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=42
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict_probability(
        self,
        X: np.ndarray
    ) -> ProbabilityEstimate:
        """
        Predict probability for new state(s).

        Args:
            X: Feature vector(s) (1, n_features) or (n_samples, n_features)

        Returns:
            ProbabilityEstimate
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get probability
        proba = self.model.predict_proba(X)[:, 1]

        # Estimate uncertainty using prediction variance
        # (for ensemble models) or bootstrap
        conf_low, conf_high = self._estimate_confidence(X)

        return ProbabilityEstimate(
            probability=float(proba.mean()),
            confidence_low=float(conf_low),
            confidence_high=float(conf_high),
            n_samples=-1,  # Not applicable for model-based
            method=f'supervised_{self.model_type}',
            reliability='model_based',
            metadata={
                'calibration': self.calibration,
                'model_type': self.model_type
            }
        )

    def _estimate_confidence(
        self,
        X: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate confidence interval for predictions.
        """
        # For gradient boosting/random forest, use staged predictions or tree variance
        if hasattr(self.model, 'estimators_'):
            # Ensemble model - use prediction variance across trees
            if hasattr(self.model, 'staged_predict_proba'):
                # Gradient boosting
                staged = list(self.model.staged_predict_proba(X))
                probas = [s[:, 1] for s in staged[-10:]]  # Last 10 stages
            else:
                # Random forest
                probas = [tree.predict_proba(X)[:, 1]
                         for tree in self.model.estimators_]

            probas = np.array(probas)
            conf_low = np.percentile(probas, 5, axis=0).mean()
            conf_high = np.percentile(probas, 95, axis=0).mean()

        else:
            # Logistic regression - use approximate variance
            proba = self.model.predict_proba(X)[:, 1].mean()
            # Rough approximation
            conf_low = max(0, proba - 0.1)
            conf_high = min(1, proba + 0.1)

        return conf_low, conf_high

    def get_feature_contributions(
        self,
        X: np.ndarray
    ) -> Dict[str, float]:
        """
        Get feature contributions to prediction (SHAP-like).
        """
        if self.model_type == 'logistic' and hasattr(self.model, 'coef_'):
            # Linear model: contribution = coefficient * feature value
            contributions = self.model.coef_[0] * X.flatten()
            return dict(zip(self.feature_names, contributions))

        # For tree models, would use SHAP or similar
        # Placeholder: return feature importances
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))

        return {}
```

### 5.2 Time-Aware Training

```python
class TimeAwareModelTrainer:
    """
    Train models with proper temporal considerations.

    Addresses:
    - Time-series cross-validation (no future leakage)
    - Recency weighting (recent data more relevant)
    - Regime-specific models
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 21,
        recency_halflife_days: int = 252
    ):
        """
        Args:
            n_splits: Number of time-series CV splits
            embargo_days: Gap between train and test to prevent leakage
            recency_halflife_days: Half-life for exponential recency weighting
        """
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.recency_halflife = recency_halflife_days

    def create_cv_splits(
        self,
        dates: pd.DatetimeIndex,
        min_train_size: int = 252
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-series cross-validation splits.

        Uses expanding window with embargo period.
        """
        from sklearn.model_selection import TimeSeriesSplit

        n_samples = len(dates)
        splits = []

        # Use TimeSeriesSplit with gap
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.embargo_days
        )

        for train_idx, test_idx in tscv.split(np.arange(n_samples)):
            if len(train_idx) >= min_train_size:
                splits.append((train_idx, test_idx))

        return splits

    def compute_recency_weights(
        self,
        dates: pd.DatetimeIndex,
        reference_date: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Compute exponential recency weights.

        More recent observations get higher weight.
        """
        if reference_date is None:
            reference_date = dates.max()

        # Days ago
        days_ago = (reference_date - dates).days

        # Exponential decay
        decay_rate = np.log(2) / self.recency_halflife
        weights = np.exp(-decay_rate * days_ago)

        # Normalize
        return weights / weights.sum() * len(weights)

    def train_with_cv(
        self,
        estimator: SupervisedProbabilityEstimator,
        X: np.ndarray,
        forward_returns: np.ndarray,
        dates: pd.DatetimeIndex,
        horizon: str,
        feature_names: List[str]
    ) -> Dict:
        """
        Train with time-series CV and return metrics.
        """
        splits = self.create_cv_splits(dates)
        cv_metrics = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            # Train on this fold
            recency_weights = self.compute_recency_weights(
                dates[train_idx],
                dates[train_idx].max()
            )

            fold_estimator = SupervisedProbabilityEstimator(
                model_type=estimator.model_type,
                calibration=estimator.calibration
            )

            fold_estimator.train(
                X[train_idx],
                forward_returns[train_idx],
                horizon,
                feature_names,
                sample_weights=recency_weights
            )

            # Evaluate on test
            y_test = (forward_returns[test_idx] >=
                     estimator.thresholds.very_high_gain[horizon])
            y_pred = fold_estimator.model.predict_proba(X[test_idx])[:, 1]

            from sklearn.metrics import roc_auc_score, brier_score_loss

            cv_metrics.append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'roc_auc': roc_auc_score(y_test, y_pred),
                'brier_score': brier_score_loss(y_test, y_pred)
            })

        # Train final model on all data
        recency_weights = self.compute_recency_weights(dates)
        final_metrics = estimator.train(
            X, forward_returns, horizon, feature_names,
            sample_weights=recency_weights
        )

        final_metrics['cv_metrics'] = cv_metrics
        final_metrics['cv_roc_auc_mean'] = np.mean([m['roc_auc'] for m in cv_metrics])
        final_metrics['cv_roc_auc_std'] = np.std([m['roc_auc'] for m in cv_metrics])

        return final_metrics
```

---

## 6. Estimator C: Similarity Search (Optional)

### 6.1 Core Algorithm

```python
class SimilaritySearchEstimator:
    """
    Estimator C: Non-parametric probability via similarity-weighted returns.

    Key insight: Weight historical returns by how similar they are to current state.

    P(high_gain | state) ≈ Σ(similarity_i * I(high_gain_i)) / Σ(similarity_i)

    Advantages:
    - No discretization needed
    - Naturally handles continuous features
    - Local adaptation to feature space
    """

    def __init__(
        self,
        n_neighbors: int = 200,
        kernel: str = 'gaussian',
        bandwidth: float = 1.0,
        adaptive_bandwidth: bool = True
    ):
        """
        Args:
            n_neighbors: Number of nearest neighbors to consider
            kernel: 'gaussian', 'epanechnikov', or 'uniform'
            bandwidth: Kernel bandwidth (if not adaptive)
            adaptive_bandwidth: Use k-nearest neighbor adaptive bandwidth
        """
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.adaptive_bandwidth = adaptive_bandwidth
        self.similarity_matcher = SimilarityMatcher(n_neighbors=n_neighbors)

    def fit(
        self,
        X: np.ndarray,
        forward_returns: pd.DataFrame
    ) -> None:
        """
        Build similarity index.
        """
        self.X = X
        self.forward_returns = forward_returns
        self.similarity_matcher.build_index(X)

    def estimate(
        self,
        query: np.ndarray,
        horizon: str,
        threshold: float
    ) -> ProbabilityEstimate:
        """
        Estimate probability using similarity-weighted average.
        """
        # Find neighbors
        distances, indices = self.similarity_matcher.find_similar(
            query, self.n_neighbors
        )

        # Compute kernel weights
        if self.adaptive_bandwidth:
            # Use k-th neighbor distance as bandwidth
            h = distances[min(50, len(distances) - 1)]
        else:
            h = self.bandwidth

        weights = self._kernel_weights(distances, h)

        # Get forward returns
        returns = self.forward_returns.loc[indices, f'fwd_return_{horizon}'].values
        valid_mask = ~np.isnan(returns)
        returns = returns[valid_mask]
        weights = weights[valid_mask]

        if len(returns) == 0:
            return ProbabilityEstimate(
                probability=np.nan,
                confidence_low=np.nan,
                confidence_high=np.nan,
                n_samples=0,
                method='similarity_search',
                reliability='insufficient_data'
            )

        # Compute weighted probability
        successes = (returns >= threshold).astype(float)
        weights_norm = weights / weights.sum()

        probability = (successes * weights_norm).sum()

        # Effective sample size for confidence interval
        ess = (weights.sum() ** 2) / (weights ** 2).sum()

        # Bootstrap confidence interval
        conf_low, conf_high = self._bootstrap_confidence(
            successes, weights_norm, n_bootstrap=500
        )

        # Determine reliability based on effective sample size
        if ess < 10:
            reliability = 'very_low'
        elif ess < 30:
            reliability = 'low'
        elif ess < 100:
            reliability = 'moderate'
        else:
            reliability = 'high'

        return ProbabilityEstimate(
            probability=probability,
            confidence_low=conf_low,
            confidence_high=conf_high,
            n_samples=len(returns),
            method='similarity_search',
            reliability=reliability,
            metadata={
                'effective_sample_size': ess,
                'bandwidth': h,
                'kernel': self.kernel,
                'mean_similarity': float(weights.mean()),
                'horizon': horizon,
                'threshold': threshold
            }
        )

    def _kernel_weights(
        self,
        distances: np.ndarray,
        bandwidth: float
    ) -> np.ndarray:
        """
        Compute kernel weights from distances.
        """
        u = distances / (bandwidth + 1e-8)

        if self.kernel == 'gaussian':
            return np.exp(-0.5 * u ** 2)

        elif self.kernel == 'epanechnikov':
            weights = np.maximum(0, 1 - u ** 2) * 0.75
            return weights

        elif self.kernel == 'uniform':
            return (u <= 1).astype(float)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _bootstrap_confidence(
        self,
        successes: np.ndarray,
        weights: np.ndarray,
        n_bootstrap: int = 500,
        confidence: float = 0.90
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for weighted mean.
        """
        n = len(successes)
        bootstrap_probs = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(n, size=n, replace=True)
            resampled_success = successes[idx]
            resampled_weights = weights[idx]
            resampled_weights = resampled_weights / resampled_weights.sum()

            prob = (resampled_success * resampled_weights).sum()
            bootstrap_probs.append(prob)

        alpha = (1 - confidence) / 2
        conf_low = np.percentile(bootstrap_probs, alpha * 100)
        conf_high = np.percentile(bootstrap_probs, (1 - alpha) * 100)

        return conf_low, conf_high
```

---

## 7. Uncertainty Handling

### 7.1 Sample Size Uncertainty

```python
class SampleSizeUncertainty:
    """
    Handle uncertainty due to limited sample sizes.
    """

    # Minimum samples for different reliability levels
    SAMPLE_THRESHOLDS = {
        'very_low': 10,
        'low': 30,
        'moderate': 100,
        'high': 300,
        'very_high': 1000
    }

    @staticmethod
    def get_reliability_level(n_samples: int) -> str:
        """Get reliability level based on sample count."""
        if n_samples < SampleSizeUncertainty.SAMPLE_THRESHOLDS['very_low']:
            return 'insufficient'
        elif n_samples < SampleSizeUncertainty.SAMPLE_THRESHOLDS['low']:
            return 'very_low'
        elif n_samples < SampleSizeUncertainty.SAMPLE_THRESHOLDS['moderate']:
            return 'low'
        elif n_samples < SampleSizeUncertainty.SAMPLE_THRESHOLDS['high']:
            return 'moderate'
        elif n_samples < SampleSizeUncertainty.SAMPLE_THRESHOLDS['very_high']:
            return 'high'
        else:
            return 'very_high'

    @staticmethod
    def compute_shrinkage_weight(
        n_samples: int,
        min_samples: int = 30
    ) -> float:
        """
        Compute shrinkage weight toward prior.

        With few samples, weight toward base rate (prior).
        With many samples, weight toward observed rate.

        Formula: weight = n / (n + min_samples)
        """
        return n_samples / (n_samples + min_samples)

    @staticmethod
    def shrink_estimate(
        observed_prob: float,
        prior_prob: float,
        n_samples: int,
        min_samples: int = 30
    ) -> float:
        """
        Shrink observed probability toward prior.
        """
        weight = SampleSizeUncertainty.compute_shrinkage_weight(n_samples, min_samples)
        return weight * observed_prob + (1 - weight) * prior_prob
```

### 7.2 Regime Drift Handling

```python
class RegimeDriftHandler:
    """
    Handle uncertainty due to regime changes.

    Key insight: Historical patterns may not apply in different regimes.
    """

    def __init__(
        self,
        recency_halflife_days: int = 252,
        regime_mismatch_penalty: float = 0.5
    ):
        self.recency_halflife = recency_halflife_days
        self.regime_penalty = regime_mismatch_penalty

    def compute_regime_adjustment(
        self,
        matched_regimes: pd.Series,
        current_regime: str,
        matched_dates: pd.DatetimeIndex,
        current_date: datetime
    ) -> np.ndarray:
        """
        Compute adjustment weights for regime drift.

        Returns:
            Weight adjustments for each matched sample
        """
        n = len(matched_regimes)
        weights = np.ones(n)

        # Penalty for regime mismatch
        regime_match = (matched_regimes == current_regime).values
        weights[~regime_match] *= self.regime_penalty

        # Recency weighting
        if matched_dates is not None and current_date is not None:
            days_ago = (current_date - matched_dates).days
            recency = np.exp(-np.log(2) / self.recency_halflife * days_ago)
            weights *= recency

        return weights

    def detect_regime_shift(
        self,
        recent_returns: np.ndarray,
        historical_returns: np.ndarray,
        window: int = 63
    ) -> Dict[str, float]:
        """
        Detect if current regime differs significantly from history.

        Uses statistical tests to detect distribution shift.
        """
        from scipy.stats import ks_2samp, mannwhitneyu

        # Recent vs. historical distribution
        recent = recent_returns[-window:] if len(recent_returns) > window else recent_returns

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(recent, historical_returns)

        # Mann-Whitney U test
        mw_stat, mw_pvalue = mannwhitneyu(recent, historical_returns, alternative='two-sided')

        # Volatility ratio
        vol_ratio = np.std(recent) / (np.std(historical_returns) + 1e-8)

        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mw_pvalue': mw_pvalue,
            'volatility_ratio': vol_ratio,
            'regime_shift_detected': ks_pvalue < 0.05 or abs(vol_ratio - 1) > 0.5
        }

    def adjust_confidence_for_drift(
        self,
        estimate: ProbabilityEstimate,
        drift_metrics: Dict
    ) -> ProbabilityEstimate:
        """
        Widen confidence interval if regime drift detected.
        """
        if not drift_metrics.get('regime_shift_detected', False):
            return estimate

        # Widen confidence interval
        center = estimate.probability
        current_width = estimate.confidence_high - estimate.confidence_low

        # Increase width by 50% if drift detected
        new_width = current_width * 1.5

        return ProbabilityEstimate(
            probability=estimate.probability,
            confidence_low=max(0, center - new_width / 2),
            confidence_high=min(1, center + new_width / 2),
            n_samples=estimate.n_samples,
            n_successes=estimate.n_successes,
            threshold=estimate.threshold,
            method=estimate.method,
            reliability='regime_drift_adjusted',
            metadata={
                **estimate.metadata,
                'drift_adjustment': True,
                'drift_metrics': drift_metrics
            }
        )
```

### 7.3 Ensemble Uncertainty Aggregation

```python
class EnsembleUncertaintyAggregator:
    """
    Aggregate probability estimates from multiple estimators.

    Combines Estimator A, B, C with uncertainty-aware weighting.
    """

    def __init__(
        self,
        estimator_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            estimator_weights: Base weights for each estimator
                Default: {'empirical': 0.5, 'supervised': 0.3, 'similarity': 0.2}
        """
        self.base_weights = estimator_weights or {
            'empirical': 0.5,
            'supervised': 0.3,
            'similarity': 0.2
        }

    def aggregate(
        self,
        estimates: Dict[str, ProbabilityEstimate]
    ) -> ProbabilityEstimate:
        """
        Aggregate multiple estimates into single estimate.

        Weighting considers:
        1. Base estimator weights
        2. Reliability of each estimate
        3. Confidence interval width (narrower = more weight)
        """
        valid_estimates = {
            k: v for k, v in estimates.items()
            if not np.isnan(v.probability)
        }

        if len(valid_estimates) == 0:
            return ProbabilityEstimate(
                probability=np.nan,
                confidence_low=np.nan,
                confidence_high=np.nan,
                n_samples=0,
                method='ensemble',
                reliability='no_valid_estimates'
            )

        # Compute adjusted weights
        weights = {}
        for name, estimate in valid_estimates.items():
            base_weight = self.base_weights.get(name, 0.1)

            # Reliability adjustment
            reliability_mult = {
                'very_low': 0.3,
                'low': 0.5,
                'moderate': 0.8,
                'high': 1.0,
                'very_high': 1.0,
                'model_based': 0.9
            }.get(estimate.reliability, 0.5)

            # Confidence width adjustment (narrower = more weight)
            if estimate.confidence_width > 0:
                width_mult = 1 / (estimate.confidence_width + 0.1)
            else:
                width_mult = 1.0

            weights[name] = base_weight * reliability_mult * width_mult

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Weighted average of probabilities
        agg_prob = sum(
            weights[k] * valid_estimates[k].probability
            for k in weights
        )

        # Combined confidence interval
        # Use weighted mixture of intervals
        agg_conf_low = sum(
            weights[k] * valid_estimates[k].confidence_low
            for k in weights
        )
        agg_conf_high = sum(
            weights[k] * valid_estimates[k].confidence_high
            for k in weights
        )

        # Total samples (sum across estimators)
        total_samples = sum(
            max(0, e.n_samples) for e in valid_estimates.values()
        )

        # Determine aggregate reliability
        reliability_scores = {
            'very_low': 1, 'low': 2, 'moderate': 3,
            'high': 4, 'very_high': 5, 'model_based': 3
        }
        avg_reliability = sum(
            weights[k] * reliability_scores.get(valid_estimates[k].reliability, 2)
            for k in weights
        )

        if avg_reliability >= 4:
            agg_reliability = 'high'
        elif avg_reliability >= 3:
            agg_reliability = 'moderate'
        elif avg_reliability >= 2:
            agg_reliability = 'low'
        else:
            agg_reliability = 'very_low'

        return ProbabilityEstimate(
            probability=agg_prob,
            confidence_low=agg_conf_low,
            confidence_high=agg_conf_high,
            n_samples=total_samples,
            method='ensemble',
            reliability=agg_reliability,
            metadata={
                'estimator_weights': weights,
                'estimator_probs': {k: v.probability for k, v in valid_estimates.items()},
                'estimator_reliabilities': {k: v.reliability for k, v in valid_estimates.items()}
            }
        )

    def compute_disagreement(
        self,
        estimates: Dict[str, ProbabilityEstimate]
    ) -> float:
        """
        Compute disagreement between estimators.

        High disagreement suggests uncertainty in the estimate.

        Returns:
            Disagreement score (0 = perfect agreement, 1 = maximum disagreement)
        """
        probs = [
            e.probability for e in estimates.values()
            if not np.isnan(e.probability)
        ]

        if len(probs) < 2:
            return 0.0

        # Standard deviation of probabilities
        return float(np.std(probs))
```

---

## 8. Complete Probability Engine

```python
class HighGainProbabilityEngine:
    """
    Complete engine combining all estimators and uncertainty handling.
    """

    def __init__(
        self,
        thresholds: GainThresholds = None,
        use_estimator_a: bool = True,
        use_estimator_b: bool = True,
        use_estimator_c: bool = True,
        min_samples: int = 30
    ):
        self.thresholds = thresholds or GainThresholds()

        # Initialize estimators
        self.estimator_a = EmpiricalProbabilityEstimator(
            thresholds=self.thresholds,
            min_samples=min_samples
        ) if use_estimator_a else None

        self.estimator_b = SupervisedProbabilityEstimator(
            model_type='gradient_boosting'
        ) if use_estimator_b else None

        self.estimator_c = SimilaritySearchEstimator(
            n_neighbors=200
        ) if use_estimator_c else None

        # State matcher
        self.state_matcher = HybridStateMatcher(
            ExactStateMatcher(min_matches=min_samples),
            SimilarityMatcher(n_neighbors=500),
            min_matches=min_samples
        )

        # Uncertainty handlers
        self.regime_drift_handler = RegimeDriftHandler()
        self.ensemble_aggregator = EnsembleUncertaintyAggregator()

        # State
        self.is_fitted = False

    def fit(
        self,
        historical_states: pd.DataFrame,
        historical_vectors: np.ndarray,
        forward_returns: pd.DataFrame,
        feature_names: List[str],
        horizons: List[str] = None
    ) -> Dict:
        """
        Fit all estimators on historical data.
        """
        horizons = horizons or ['5d', '21d', '63d']

        # Build state matcher index
        discrete_cols = StateKeyBuilder.DEFAULT_DISCRETE_FEATURES
        self.state_matcher.exact_matcher.build_index(historical_states, discrete_cols)
        self.state_matcher.similarity_matcher.build_index(historical_vectors)

        # Train supervised models for each horizon
        self.supervised_models = {}
        if self.estimator_b:
            trainer = TimeAwareModelTrainer()
            for horizon in horizons:
                returns = forward_returns[f'fwd_return_{horizon}'].values
                valid_mask = ~np.isnan(returns)

                metrics = trainer.train_with_cv(
                    self.estimator_b,
                    historical_vectors[valid_mask],
                    returns[valid_mask],
                    historical_states.index[valid_mask],
                    horizon,
                    feature_names
                )
                self.supervised_models[horizon] = {
                    'model': self.estimator_b.model,
                    'metrics': metrics
                }

        # Fit similarity search
        if self.estimator_c:
            self.estimator_c.fit(historical_vectors, forward_returns)

        self.historical_states = historical_states
        self.historical_vectors = historical_vectors
        self.forward_returns = forward_returns
        self.feature_names = feature_names
        self.is_fitted = True

        return {'status': 'fitted', 'horizons': horizons}

    def estimate_probability(
        self,
        current_state: MatchableState,
        current_vector: np.ndarray,
        horizon: str,
        current_regime: Optional[str] = None,
        current_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Estimate probability of high gain for current state.

        Returns:
            Complete estimation result with all estimators and uncertainty
        """
        if not self.is_fitted:
            raise ValueError("Engine not fitted. Call fit() first.")

        threshold = self.thresholds.very_high_gain[horizon]
        estimates = {}

        # Find matched historical states
        matched_indices, similarities, match_metadata = self.state_matcher.find_matches(
            current_state, self.historical_vectors
        )

        # Estimator A: Empirical
        if self.estimator_a and len(matched_indices) > 0:
            # Apply regime weighting if available
            if current_regime and current_date:
                weights = self.regime_drift_handler.compute_regime_adjustment(
                    self.historical_states.loc[matched_indices, 'regime'],
                    current_regime,
                    self.historical_states.loc[matched_indices].index,
                    current_date
                )
                weights = weights * similarities  # Combine with similarity
            else:
                weights = similarities

            estimates['empirical'] = self.estimator_a.estimate(
                matched_indices,
                self.forward_returns,
                horizon,
                weights=weights
            )

        # Estimator B: Supervised
        if self.estimator_b and horizon in self.supervised_models:
            model = self.supervised_models[horizon]['model']
            self.estimator_b.model = model
            estimates['supervised'] = self.estimator_b.predict_probability(current_vector)

        # Estimator C: Similarity
        if self.estimator_c:
            estimates['similarity'] = self.estimator_c.estimate(
                current_vector, horizon, threshold
            )

        # Aggregate estimates
        ensemble_estimate = self.ensemble_aggregator.aggregate(estimates)
        disagreement = self.ensemble_aggregator.compute_disagreement(estimates)

        # Check for regime drift
        drift_metrics = {}
        if current_date and len(matched_indices) > 0:
            recent_returns = self.forward_returns[f'fwd_return_{horizon}'].iloc[-63:].dropna().values
            historical_returns = self.forward_returns.loc[
                matched_indices, f'fwd_return_{horizon}'
            ].dropna().values

            if len(recent_returns) > 20 and len(historical_returns) > 20:
                drift_metrics = self.regime_drift_handler.detect_regime_shift(
                    recent_returns, historical_returns
                )

                if drift_metrics.get('regime_shift_detected'):
                    ensemble_estimate = self.regime_drift_handler.adjust_confidence_for_drift(
                        ensemble_estimate, drift_metrics
                    )

        return {
            'ensemble': ensemble_estimate,
            'individual_estimates': estimates,
            'match_metadata': match_metadata,
            'disagreement': disagreement,
            'drift_metrics': drift_metrics,
            'horizon': horizon,
            'threshold': threshold
        }
```

---

## 9. Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROBABILITY ENGINE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CURRENT STATE                                                              │
│  ─────────────                                                              │
│  [RSI=32, ADX=35, MACD=bullish, Trend=uptrend, ...]                        │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STATE MATCHING                                                      │   │
│  │ ├── Exact: Match discrete states (regime, zones)                   │   │
│  │ ├── Similarity: Nearest neighbors in continuous space              │   │
│  │ └── Hybrid: Exact matches ranked by similarity                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌───────────────┬───────────────┬───────────────┐                        │
│  │ ESTIMATOR A   │ ESTIMATOR B   │ ESTIMATOR C   │                        │
│  │ Empirical     │ Supervised    │ Similarity    │                        │
│  │               │               │               │                        │
│  │ P = hits/n    │ P = model(X)  │ P = Σ(w*I)/Σw │                        │
│  │ Bayesian prior│ Calibrated GB │ Kernel weights│                        │
│  │               │               │               │                        │
│  │ + Transparent │ + Generalizes │ + Continuous  │                        │
│  │ - Needs bins  │ + Interactions│ + Local       │                        │
│  └───────┬───────┴───────┬───────┴───────┬───────┘                        │
│          │               │               │                                 │
│          └───────────────┼───────────────┘                                 │
│                          ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ UNCERTAINTY HANDLING                                                │   │
│  │ ├── Sample size → Shrinkage toward prior                           │   │
│  │ ├── Regime drift → Wider confidence intervals                      │   │
│  │ └── Estimator disagreement → Flag uncertainty                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                 │
│                          ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ENSEMBLE AGGREGATION                                                │   │
│  │ ├── Weight by reliability                                          │   │
│  │ ├── Weight by confidence width                                     │   │
│  │ └── Output: probability + confidence interval + reliability        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                 │
│                          ▼                                                 │
│  FINAL OUTPUT                                                              │
│  ────────────                                                              │
│  P(very_high_gain | state) = 0.23 [0.15, 0.31]                            │
│  Reliability: moderate | n_samples: 147 | Horizon: 21d                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Assumptions

1. **Historical Relevance:** Past patterns provide information about future probabilities
2. **State Sufficiency:** Selected indicators capture relevant market dynamics
3. **Stationarity (Partial):** Regime-conditional relationships are relatively stable
4. **Threshold Stability:** Gain thresholds are appropriate for current market
5. **Sample Independence:** Matched samples are approximately independent (addressed via regime weighting)
6. **Model Calibration:** Supervised model probabilities are well-calibrated
7. **Ensemble Benefit:** Multiple estimators improve robustness over single approach

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
