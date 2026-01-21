# Feature Engineering & Indicator State Encoding

## 1. Overview

This document defines the transformation pipeline that converts raw indicator values into model-ready features and state signatures. The goal is to create a standardized, comparable, and information-rich feature set suitable for scoring, screening, and future ML applications.

### Design Principles

1. **Comparability:** Features should be comparable across assets and time
2. **Stationarity:** Transform non-stationary indicators into stationary features
3. **Information Preservation:** Minimize information loss during transformation
4. **Interpretability:** Maintain clear semantic meaning where possible
5. **Efficiency:** Enable fast computation for universe-wide scans

---

## 2. Normalization Strategies

### 2.1 Normalization Strategy Selection Matrix

| Indicator Type | Recommended Strategy | Rationale |
|----------------|---------------------|-----------|
| Bounded oscillators (RSI, Stochastic) | Min-Max to [0,1] | Already bounded; preserve natural scale |
| Unbounded oscillators (CCI, MACD) | Z-Score (rolling) | Need standardization; preserve sign |
| Price-based (EMA, VWAP) | Price-relative | Must be price-independent |
| Volume-based (OBV, ADL) | Percentage change | Cumulative; levels meaningless |
| Volatility (ATR, HV) | Percentile rank | Compare to own history |
| Ratios (P/E, PEG) | Cross-sectional percentile | Compare to peers |
| Binary signals | One-hot encoding | Categorical by nature |

---

### 2.2 Normalization Methods

#### 2.2.1 Min-Max Normalization (Bounded Indicators)

```python
class MinMaxNormalizer:
    """
    Scale bounded indicators to [0, 1] range.
    Use for: RSI, Stochastic, MFI, Williams %R, Aroon
    """

    def __init__(self, feature_min: float, feature_max: float):
        self.min = feature_min
        self.max = feature_max

    def transform(self, value: float) -> float:
        """
        Formula: (x - min) / (max - min)

        Example (RSI):
            min=0, max=100
            RSI=30 -> 0.30
            RSI=70 -> 0.70
        """
        if self.max == self.min:
            return 0.5
        normalized = (value - self.min) / (self.max - self.min)
        return np.clip(normalized, 0.0, 1.0)

# Configuration for bounded indicators
BOUNDED_INDICATORS = {
    'rsi':        {'min': 0, 'max': 100},
    'stochastic_k': {'min': 0, 'max': 100},
    'stochastic_d': {'min': 0, 'max': 100},
    'mfi':        {'min': 0, 'max': 100},
    'williams_r': {'min': -100, 'max': 0},  # Note: inverted scale
    'aroon_up':   {'min': 0, 'max': 100},
    'aroon_down': {'min': 0, 'max': 100},
    'aroon_osc':  {'min': -100, 'max': 100},
    'adx':        {'min': 0, 'max': 100},
    'plus_di':    {'min': 0, 'max': 100},
    'minus_di':   {'min': 0, 'max': 100},
    'choppiness': {'min': 0, 'max': 100},
    'cmf':        {'min': -1, 'max': 1},
}
```

#### 2.2.2 Z-Score Normalization (Unbounded Indicators)

```python
class RollingZScoreNormalizer:
    """
    Standardize unbounded indicators using rolling statistics.
    Use for: MACD, CCI, TSI, OBV change, momentum
    """

    def __init__(self, lookback: int = 252, min_periods: int = 20):
        self.lookback = lookback
        self.min_periods = min_periods

    def transform(self, series: pd.Series) -> pd.Series:
        """
        Formula: (x - rolling_mean) / rolling_std

        Output interpretation:
            z = 0: At mean
            z = 2: 2 standard deviations above mean
            z = -2: 2 standard deviations below mean
        """
        rolling_mean = series.rolling(
            window=self.lookback,
            min_periods=self.min_periods
        ).mean()

        rolling_std = series.rolling(
            window=self.lookback,
            min_periods=self.min_periods
        ).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        z_score = (series - rolling_mean) / rolling_std

        # Winsorize extreme values
        return z_score.clip(-4, 4)

    def transform_single(
        self,
        value: float,
        mean: float,
        std: float
    ) -> float:
        """Single value transformation with pre-computed stats."""
        if std == 0 or np.isnan(std):
            return 0.0
        z = (value - mean) / std
        return np.clip(z, -4, 4)

# Configuration for unbounded indicators
UNBOUNDED_INDICATORS = {
    'macd':       {'lookback': 252},
    'macd_histogram': {'lookback': 252},
    'cci':        {'lookback': 252},
    'tsi':        {'lookback': 252},
    'roc':        {'lookback': 63},   # Shorter for momentum
    'obv_change': {'lookback': 252},
    'vroc':       {'lookback': 63},
}
```

#### 2.2.3 Price-Relative Normalization

```python
class PriceRelativeNormalizer:
    """
    Convert price-based indicators to price-independent ratios.
    Use for: EMA, SMA, VWAP, Bollinger Bands, pivot levels
    """

    def transform_distance(
        self,
        price: float,
        level: float
    ) -> float:
        """
        Price distance as percentage.

        Formula: (price - level) / price * 100

        Example:
            price=100, ema_50=95
            -> (100-95)/100 = 5% above EMA
        """
        if price == 0:
            return 0.0
        return (price - level) / price

    def transform_multiple_levels(
        self,
        price: float,
        levels: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Transform multiple levels relative to price.

        Example output:
        {
            'ema_8_pct': 0.02,    # 2% above EMA(8)
            'ema_21_pct': 0.05,   # 5% above EMA(21)
            'ema_50_pct': 0.08,   # 8% above EMA(50)
            'ema_200_pct': 0.15,  # 15% above EMA(200)
        }
        """
        return {
            f"{name}_pct": self.transform_distance(price, level)
            for name, level in levels.items()
        }

    def transform_band_position(
        self,
        price: float,
        upper: float,
        lower: float
    ) -> float:
        """
        Position within band as 0-1 ratio.

        Formula: (price - lower) / (upper - lower)

        Example (Bollinger):
            price=102, lower=95, upper=105
            -> (102-95)/(105-95) = 0.70 (70th percentile of band)
        """
        if upper == lower:
            return 0.5
        return (price - lower) / (upper - lower)
```

#### 2.2.4 Percentile Rank Normalization

```python
class PercentileRankNormalizer:
    """
    Convert values to percentile ranks.
    Use for: Volatility, volume, cross-sectional comparisons
    """

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def transform_time_series(self, series: pd.Series) -> pd.Series:
        """
        Rolling percentile rank (0-100).

        "Where does today's value rank in the last N observations?"
        """
        def percentile_rank(x):
            if len(x) < 2:
                return 50.0
            current = x.iloc[-1]
            return (x < current).sum() / (len(x) - 1) * 100

        return series.rolling(
            window=self.lookback,
            min_periods=20
        ).apply(percentile_rank, raw=False)

    def transform_cross_sectional(
        self,
        value: float,
        universe_values: np.ndarray
    ) -> float:
        """
        Percentile rank within universe.

        "Where does this stock rank among its peers?"
        """
        if len(universe_values) < 2:
            return 50.0
        return (universe_values < value).sum() / len(universe_values) * 100

# Configuration for percentile-based features
PERCENTILE_INDICATORS = {
    'atr':        {'lookback': 252, 'type': 'time_series'},
    'hv_21':      {'lookback': 252, 'type': 'time_series'},
    'volume':     {'lookback': 63,  'type': 'time_series'},
    'pe_ratio':   {'lookback': None, 'type': 'cross_sectional'},
    'market_cap': {'lookback': None, 'type': 'cross_sectional'},
}
```

#### 2.2.5 Robust Normalization (Outlier-Resistant)

```python
class RobustNormalizer:
    """
    Median-based normalization for outlier-heavy distributions.
    Use for: Fundamental ratios, extreme momentum
    """

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def transform(self, series: pd.Series) -> pd.Series:
        """
        Robust Z-score using median and MAD.

        Formula: (x - median) / MAD
        where MAD = median(|x - median|)
        """
        rolling_median = series.rolling(
            window=self.lookback,
            min_periods=20
        ).median()

        def mad(x):
            return np.median(np.abs(x - np.median(x)))

        rolling_mad = series.rolling(
            window=self.lookback,
            min_periods=20
        ).apply(mad, raw=True)

        # Scale MAD to approximate std (for normal distribution)
        rolling_mad = rolling_mad * 1.4826

        robust_z = (series - rolling_median) / rolling_mad
        return robust_z.clip(-4, 4)
```

---

### 2.3 Normalization Pipeline

```python
class NormalizationPipeline:
    """
    Unified normalization pipeline for all indicators.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.normalizers = self._build_normalizers()

    def _build_normalizers(self) -> Dict[str, Callable]:
        return {
            'min_max': MinMaxNormalizer,
            'z_score': RollingZScoreNormalizer,
            'price_relative': PriceRelativeNormalizer,
            'percentile': PercentileRankNormalizer,
            'robust': RobustNormalizer,
        }

    def transform(
        self,
        indicator_name: str,
        values: pd.Series,
        price: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Apply appropriate normalization based on indicator type.
        """
        config = self.config.get(indicator_name, {})
        method = config.get('method', 'z_score')
        normalizer = self.normalizers[method](**config.get('params', {}))
        return normalizer.transform(values)

# Master normalization configuration
NORMALIZATION_CONFIG = {
    # Bounded oscillators
    'rsi': {'method': 'min_max', 'params': {'feature_min': 0, 'feature_max': 100}},
    'stochastic_k': {'method': 'min_max', 'params': {'feature_min': 0, 'feature_max': 100}},
    'mfi': {'method': 'min_max', 'params': {'feature_min': 0, 'feature_max': 100}},
    'adx': {'method': 'min_max', 'params': {'feature_min': 0, 'feature_max': 100}},

    # Unbounded oscillators
    'macd': {'method': 'z_score', 'params': {'lookback': 252}},
    'cci': {'method': 'z_score', 'params': {'lookback': 252}},
    'tsi': {'method': 'z_score', 'params': {'lookback': 252}},

    # Price-based (handled separately with price input)
    'ema_8': {'method': 'price_relative'},
    'ema_21': {'method': 'price_relative'},
    'ema_50': {'method': 'price_relative'},
    'ema_200': {'method': 'price_relative'},
    'vwap': {'method': 'price_relative'},

    # Volatility
    'atr': {'method': 'percentile', 'params': {'lookback': 252}},
    'hv_21': {'method': 'percentile', 'params': {'lookback': 252}},

    # Fundamentals (outlier-prone)
    'pe_ratio': {'method': 'robust', 'params': {'lookback': 252}},
    'peg_ratio': {'method': 'robust', 'params': {'lookback': 252}},
}
```

---

## 3. Discretization Logic

### 3.1 Discretization Strategy Overview

Discretization converts continuous indicator values into discrete bins or regime labels. This enables:
- Categorical analysis ("What happens when RSI is oversold?")
- Regime-based strategy selection
- Reduced noise and improved interpretability
- Compatibility with decision-tree models

---

### 3.2 Bin-Based Discretization

#### 3.2.1 Fixed-Threshold Binning

```python
class FixedThresholdBinner:
    """
    Discretize using fixed, domain-specific thresholds.
    Use for: Indicators with established interpretation levels.
    """

    def __init__(self, thresholds: List[float], labels: List[str]):
        """
        Args:
            thresholds: Sorted list of cutoff values
            labels: Labels for each bin (len = len(thresholds) + 1)
        """
        self.thresholds = thresholds
        self.labels = labels

    def transform(self, value: float) -> str:
        """Assign value to appropriate bin."""
        for i, threshold in enumerate(self.thresholds):
            if value < threshold:
                return self.labels[i]
        return self.labels[-1]

    def transform_series(self, series: pd.Series) -> pd.Series:
        return pd.cut(
            series,
            bins=[-np.inf] + self.thresholds + [np.inf],
            labels=self.labels
        )

# Domain-specific binning configurations
FIXED_THRESHOLD_BINS = {
    'rsi': {
        'thresholds': [20, 30, 50, 70, 80],
        'labels': ['EXTREME_OVERSOLD', 'OVERSOLD', 'NEUTRAL_BEAR',
                   'NEUTRAL_BULL', 'OVERBOUGHT', 'EXTREME_OVERBOUGHT']
    },

    'stochastic_k': {
        'thresholds': [20, 40, 60, 80],
        'labels': ['OVERSOLD', 'WEAK', 'NEUTRAL', 'STRONG', 'OVERBOUGHT']
    },

    'adx': {
        'thresholds': [15, 25, 40, 55],
        'labels': ['NO_TREND', 'WEAK_TREND', 'MODERATE_TREND',
                   'STRONG_TREND', 'EXTREME_TREND']
    },

    'mfi': {
        'thresholds': [20, 40, 60, 80],
        'labels': ['OVERSOLD', 'WEAK', 'NEUTRAL', 'STRONG', 'OVERBOUGHT']
    },

    'williams_r': {
        'thresholds': [-80, -50, -20],
        'labels': ['OVERSOLD', 'WEAK', 'STRONG', 'OVERBOUGHT']
    },

    'cci': {
        'thresholds': [-200, -100, 0, 100, 200],
        'labels': ['EXTREME_OVERSOLD', 'OVERSOLD', 'BEARISH',
                   'BULLISH', 'OVERBOUGHT', 'EXTREME_OVERBOUGHT']
    },

    'choppiness': {
        'thresholds': [38.2, 50, 61.8],
        'labels': ['TRENDING', 'TRANSITIONING', 'CONSOLIDATING', 'CHOPPY']
    },

    'cmf': {
        'thresholds': [-0.25, -0.05, 0.05, 0.25],
        'labels': ['STRONG_DISTRIBUTION', 'DISTRIBUTION', 'NEUTRAL',
                   'ACCUMULATION', 'STRONG_ACCUMULATION']
    },

    'volume_ratio': {
        'thresholds': [0.5, 0.8, 1.2, 2.0, 3.0],
        'labels': ['VERY_LOW', 'LOW', 'NORMAL', 'ELEVATED', 'HIGH', 'EXTREME']
    },
}
```

#### 3.2.2 Quantile-Based Binning

```python
class QuantileBinner:
    """
    Discretize using rolling quantiles.
    Use for: Indicators without established thresholds.
    """

    def __init__(
        self,
        n_bins: int = 5,
        lookback: int = 252,
        labels: Optional[List[str]] = None
    ):
        self.n_bins = n_bins
        self.lookback = lookback
        self.labels = labels or [f'Q{i+1}' for i in range(n_bins)]

    def transform(self, series: pd.Series) -> pd.Series:
        """
        Assign values to quantile bins based on rolling history.
        """
        def quantile_bin(window):
            current = window.iloc[-1]
            quantiles = np.percentile(
                window[:-1],
                np.linspace(0, 100, self.n_bins + 1)[1:-1]
            )
            bin_idx = np.searchsorted(quantiles, current)
            return self.labels[bin_idx]

        return series.rolling(
            window=self.lookback,
            min_periods=self.lookback // 2
        ).apply(lambda x: self.labels.index(quantile_bin(x)), raw=False)

# Quantile binning configurations
QUANTILE_BINS = {
    'macd': {
        'n_bins': 5,
        'lookback': 252,
        'labels': ['VERY_BEARISH', 'BEARISH', 'NEUTRAL', 'BULLISH', 'VERY_BULLISH']
    },

    'roc_21': {
        'n_bins': 5,
        'lookback': 252,
        'labels': ['STRONG_DOWN', 'DOWN', 'FLAT', 'UP', 'STRONG_UP']
    },

    'hv_21': {
        'n_bins': 5,
        'lookback': 252,
        'labels': ['VERY_LOW_VOL', 'LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL', 'VERY_HIGH_VOL']
    },

    'relative_strength': {
        'n_bins': 5,
        'lookback': 63,
        'labels': ['LAGGARD', 'UNDERPERFORM', 'INLINE', 'OUTPERFORM', 'LEADER']
    },
}
```

#### 3.2.3 Adaptive Binning (Regime-Aware)

```python
class AdaptiveBinner:
    """
    Adjust bin thresholds based on market regime.
    Use for: Indicators whose interpretation varies by regime.
    """

    def __init__(
        self,
        base_thresholds: Dict[str, List[float]],
        regime_adjustments: Dict[str, Dict[str, float]]
    ):
        """
        Args:
            base_thresholds: Default thresholds
            regime_adjustments: Multipliers for each regime
        """
        self.base_thresholds = base_thresholds
        self.regime_adjustments = regime_adjustments

    def get_thresholds(
        self,
        indicator: str,
        regime: str
    ) -> List[float]:
        """Get adjusted thresholds for current regime."""
        base = self.base_thresholds[indicator]
        adjustment = self.regime_adjustments.get(regime, {}).get(indicator, 1.0)
        return [t * adjustment for t in base]

# Example: RSI thresholds adjusted by volatility regime
ADAPTIVE_CONFIG = {
    'base_thresholds': {
        'rsi': [30, 70],  # Standard oversold/overbought
    },
    'regime_adjustments': {
        'HIGH_VOL': {'rsi': 0.85},     # 25.5/59.5 - tighter in high vol
        'NORMAL_VOL': {'rsi': 1.0},    # 30/70 - standard
        'LOW_VOL': {'rsi': 1.15},      # 34.5/80.5 - wider in low vol
    }
}
```

---

### 3.3 Regime Classification

```python
class RegimeClassifier:
    """
    Classify market into discrete regimes.
    """

    def classify_trend_regime(
        self,
        adx: float,
        price_vs_ma: float
    ) -> str:
        """
        Trend regime classification.

        Returns:
            'STRONG_UPTREND', 'UPTREND', 'RANGE', 'DOWNTREND', 'STRONG_DOWNTREND'
        """
        if adx < 20:
            return 'RANGE'

        if price_vs_ma > 0:
            return 'STRONG_UPTREND' if adx > 40 else 'UPTREND'
        else:
            return 'STRONG_DOWNTREND' if adx > 40 else 'DOWNTREND'

    def classify_volatility_regime(
        self,
        hv_percentile: float
    ) -> str:
        """
        Volatility regime classification.

        Returns:
            'LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL', 'EXTREME_VOL'
        """
        if hv_percentile < 25:
            return 'LOW_VOL'
        elif hv_percentile < 75:
            return 'NORMAL_VOL'
        elif hv_percentile < 95:
            return 'HIGH_VOL'
        else:
            return 'EXTREME_VOL'

    def classify_momentum_regime(
        self,
        rsi: float,
        roc_21: float
    ) -> str:
        """
        Momentum regime classification.

        Returns:
            'OVERBOUGHT_ACCELERATING', 'OVERBOUGHT_DECELERATING',
            'OVERSOLD_ACCELERATING', 'OVERSOLD_DECELERATING', 'NEUTRAL'
        """
        if rsi > 70:
            return 'OVERBOUGHT_ACCELERATING' if roc_21 > 0 else 'OVERBOUGHT_DECELERATING'
        elif rsi < 30:
            return 'OVERSOLD_DECELERATING' if roc_21 < 0 else 'OVERSOLD_ACCELERATING'
        else:
            return 'NEUTRAL'

    def classify_composite_regime(
        self,
        indicators: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Full regime classification.
        """
        return {
            'trend_regime': self.classify_trend_regime(
                indicators['adx'],
                indicators['price_vs_ema_50']
            ),
            'volatility_regime': self.classify_volatility_regime(
                indicators['hv_percentile']
            ),
            'momentum_regime': self.classify_momentum_regime(
                indicators['rsi'],
                indicators['roc_21']
            ),
        }

# Regime encoding for features
REGIME_ENCODINGS = {
    'trend_regime': {
        'STRONG_UPTREND': 2,
        'UPTREND': 1,
        'RANGE': 0,
        'DOWNTREND': -1,
        'STRONG_DOWNTREND': -2
    },
    'volatility_regime': {
        'LOW_VOL': -1,
        'NORMAL_VOL': 0,
        'HIGH_VOL': 1,
        'EXTREME_VOL': 2
    },
    'momentum_regime': {
        'OVERBOUGHT_ACCELERATING': 2,
        'OVERBOUGHT_DECELERATING': 1,
        'NEUTRAL': 0,
        'OVERSOLD_ACCELERATING': -1,
        'OVERSOLD_DECELERATING': -2
    }
}
```

---

## 4. Feature Pruning & Collinearity Reduction

### 4.1 Collinearity Detection

```python
class CollinearityAnalyzer:
    """
    Detect and handle highly correlated features.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def compute_correlation_matrix(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute pairwise correlations."""
        return features.corr(method='spearman')

    def find_collinear_pairs(
        self,
        corr_matrix: pd.DataFrame
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs exceeding correlation threshold.

        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        pairs = []
        cols = corr_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr >= self.threshold:
                    pairs.append((cols[i], cols[j], corr))

        return sorted(pairs, key=lambda x: -x[2])

    def select_representative(
        self,
        pair: Tuple[str, str],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Select which feature to keep from a collinear pair.

        Priority:
        1. Higher feature importance (if available)
        2. More interpretable
        3. Simpler computation
        """
        f1, f2 = pair[0], pair[1]

        if feature_importance:
            imp1 = feature_importance.get(f1, 0)
            imp2 = feature_importance.get(f2, 0)
            return f1 if imp1 >= imp2 else f2

        # Default priority rules
        return FEATURE_PRIORITY.get(f1, 0) >= FEATURE_PRIORITY.get(f2, 0)

# Feature priority for tie-breaking (higher = keep)
FEATURE_PRIORITY = {
    # Prefer simple, interpretable features
    'rsi': 10,
    'macd': 9,
    'adx': 9,
    'ema_50_pct': 8,

    # Derived features lower priority
    'rsi_z_score': 5,
    'macd_percentile': 5,

    # Complex features lowest priority
    'rsi_stochastic_avg': 3,
}
```

### 4.2 Feature Selection Methods

```python
class FeatureSelector:
    """
    Select optimal feature subset.
    """

    def __init__(
        self,
        max_features: int = 50,
        correlation_threshold: float = 0.85,
        variance_threshold: float = 0.01
    ):
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold

    def remove_low_variance(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Remove features with near-zero variance.
        """
        variances = features.var()
        keep_cols = variances[variances > self.variance_threshold].index
        return features[keep_cols]

    def remove_collinear(
        self,
        features: pd.DataFrame,
        keep_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Iteratively remove highly correlated features.

        Algorithm:
        1. Compute correlation matrix
        2. Find highest correlated pair
        3. Remove lower-priority feature
        4. Repeat until no pairs exceed threshold
        """
        keep_features = keep_features or []
        df = features.copy()

        while True:
            corr_matrix = df.corr(method='spearman').abs()

            # Zero out diagonal and lower triangle
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find maximum correlation
            max_corr = upper.max().max()
            if max_corr < self.correlation_threshold:
                break

            # Find the pair
            max_pair = upper.stack().idxmax()
            f1, f2 = max_pair

            # Decide which to drop
            if f1 in keep_features:
                drop = f2
            elif f2 in keep_features:
                drop = f1
            else:
                drop = f2 if FEATURE_PRIORITY.get(f1, 0) >= FEATURE_PRIORITY.get(f2, 0) else f1

            df = df.drop(columns=[drop])

        return df

    def select_by_importance(
        self,
        features: pd.DataFrame,
        importance_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Keep top-N features by importance score.
        """
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: -x[1]
        )
        keep_cols = [f for f, _ in sorted_features[:self.max_features]]
        return features[[c for c in keep_cols if c in features.columns]]

    def full_pipeline(
        self,
        features: pd.DataFrame,
        importance_scores: Optional[Dict[str, float]] = None,
        keep_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Complete feature selection pipeline.

        Steps:
        1. Remove low variance
        2. Remove highly correlated
        3. Select top by importance (if scores provided)
        """
        df = self.remove_low_variance(features)
        df = self.remove_collinear(df, keep_features)

        if importance_scores:
            df = self.select_by_importance(df, importance_scores)

        return df
```

### 4.3 Known Collinear Groups

```python
# Pre-defined collinear groups (keep first in each list)
COLLINEAR_GROUPS = {
    # Momentum oscillators (keep RSI)
    'momentum_oscillators': ['rsi', 'stochastic_k', 'williams_r'],

    # Moving averages (keep EMA, drop SMA)
    'moving_averages': ['ema_21', 'sma_21', 'ema_20', 'sma_20'],

    # Volatility (keep ATR%, drop raw ATR for cross-asset)
    'volatility': ['atrp', 'atr', 'hv_21'],

    # Volume flow (keep CMF, it's bounded)
    'volume_flow': ['cmf', 'adl_pct_change', 'obv_pct_change'],

    # Trend strength
    'trend_strength': ['adx', 'aroon_oscillator'],

    # Bands (keep Bollinger %B)
    'bands': ['bb_percent_b', 'keltner_position', 'donchian_position'],
}

def get_representative_features() -> List[str]:
    """Get the representative feature from each collinear group."""
    return [group[0] for group in COLLINEAR_GROUPS.values()]
```

---

## 5. Indicator State Vector Design

### 5.1 State Vector Structure

```python
@dataclass
class IndicatorStateVector:
    """
    Complete state representation for a ticker at a point in time.
    """

    # Metadata
    ticker: str
    timestamp: datetime
    timeframe: str

    # Continuous features (normalized)
    continuous: Dict[str, float]

    # Categorical features (discretized)
    categorical: Dict[str, str]

    # Regime states
    regimes: Dict[str, str]

    # Signal states
    signals: Dict[str, int]  # -1, 0, 1

    # Derived meta-features
    meta_features: Dict[str, float]

    def to_flat_vector(self) -> np.ndarray:
        """Convert to flat numpy array for ML."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        ...
```

### 5.2 State Vector Components

```python
# Complete state vector specification
STATE_VECTOR_SPEC = {
    # =========================================
    # CONTINUOUS FEATURES (normalized to ~[-4, 4] or [0, 1])
    # =========================================
    'continuous': {
        # Trend (5 features)
        'ema_8_pct': 'price_relative',       # Price distance from EMA(8)
        'ema_21_pct': 'price_relative',      # Price distance from EMA(21)
        'ema_50_pct': 'price_relative',      # Price distance from EMA(50)
        'ema_200_pct': 'price_relative',     # Price distance from EMA(200)
        'adx_norm': 'min_max',               # ADX [0, 1]

        # Momentum (6 features)
        'rsi_norm': 'min_max',               # RSI [0, 1]
        'stoch_k_norm': 'min_max',           # Stochastic %K [0, 1]
        'macd_z': 'z_score',                 # MACD z-score
        'macd_hist_z': 'z_score',            # MACD histogram z-score
        'roc_5': 'raw',                      # 5-day return (%)
        'roc_21': 'raw',                     # 21-day return (%)

        # Volatility (4 features)
        'atrp': 'raw',                       # ATR as % of price
        'hv_21_pctl': 'percentile',          # 21-day HV percentile
        'bb_width_z': 'z_score',             # Bollinger width z-score
        'bb_pct_b': 'raw',                   # Bollinger %B [0, 1]

        # Volume (4 features)
        'volume_ratio': 'raw',               # Volume / 20-day avg
        'cmf': 'raw',                        # Chaikin Money Flow [-1, 1]
        'mfi_norm': 'min_max',               # MFI [0, 1]
        'obv_roc_21': 'z_score',             # OBV 21-day change z-score

        # Structure (3 features)
        'range_position': 'raw',             # Position in 20-day range [0, 1]
        'pivot_distance': 'price_relative',  # Distance to nearest pivot
        'atr_distance_high': 'raw',          # ATRs from 20-day high

        # Relative Strength (3 features)
        'rs_vs_spy_21': 'raw',               # 21-day return vs SPY
        'rs_vs_spy_63': 'raw',               # 63-day return vs SPY
        'sector_rank_pctl': 'percentile',    # Rank within sector

        # Fundamentals (optional, 4 features)
        'pe_z': 'robust_z',                  # P/E robust z-score
        'pb_z': 'robust_z',                  # P/B robust z-score
        'dividend_yield': 'raw',             # Dividend yield (%)
        'earnings_surprise_avg': 'raw',      # Avg surprise last 4 quarters
    },

    # =========================================
    # CATEGORICAL FEATURES (one-hot encoded)
    # =========================================
    'categorical': {
        # Regime states
        'trend_regime': ['STRONG_UPTREND', 'UPTREND', 'RANGE', 'DOWNTREND', 'STRONG_DOWNTREND'],
        'volatility_regime': ['LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL', 'EXTREME_VOL'],
        'momentum_regime': ['OB_ACC', 'OB_DEC', 'NEUTRAL', 'OS_DEC', 'OS_ACC'],

        # Discretized indicators
        'rsi_zone': ['EXTREME_OS', 'OVERSOLD', 'NEUTRAL_BEAR', 'NEUTRAL_BULL', 'OVERBOUGHT', 'EXTREME_OB'],
        'adx_zone': ['NO_TREND', 'WEAK', 'MODERATE', 'STRONG', 'EXTREME'],
        'volume_zone': ['VERY_LOW', 'LOW', 'NORMAL', 'ELEVATED', 'HIGH', 'EXTREME'],
    },

    # =========================================
    # SIGNAL STATES (ternary: -1, 0, 1)
    # =========================================
    'signals': {
        # Crossover signals
        'macd_signal': 'MACD vs Signal line cross',
        'ema_8_21_cross': 'EMA(8) vs EMA(21) cross',
        'price_ema_50_cross': 'Price vs EMA(50) cross',

        # Threshold signals
        'rsi_signal': 'RSI OB/OS signal',
        'stoch_signal': 'Stochastic OB/OS signal',

        # Pattern signals
        'higher_high': 'Made new 20-day high',
        'lower_low': 'Made new 20-day low',

        # Volume signals
        'volume_breakout': 'Volume > 2x average',
    },

    # =========================================
    # META-FEATURES (derived from above)
    # =========================================
    'meta_features': {
        # Consensus measures
        'bullish_signal_count': 'Count of bullish signals',
        'bearish_signal_count': 'Count of bearish signals',
        'signal_consensus': 'Net bullish - bearish',

        # Trend alignment
        'ema_alignment': 'EMA stack alignment score',
        'trend_consistency': 'Multi-TF trend agreement',

        # Composite scores
        'momentum_composite': 'Weighted momentum score',
        'quality_composite': 'Weighted quality score',
    }
}
```

### 5.3 State Vector Builder

```python
class StateVectorBuilder:
    """
    Build complete state vectors from raw indicators.
    """

    def __init__(
        self,
        normalizers: Dict[str, Callable],
        discretizers: Dict[str, Callable],
        spec: Dict = STATE_VECTOR_SPEC
    ):
        self.normalizers = normalizers
        self.discretizers = discretizers
        self.spec = spec

    def build(
        self,
        raw_indicators: Dict[str, float],
        price: float,
        benchmark_data: Optional[Dict] = None
    ) -> IndicatorStateVector:
        """
        Build state vector from raw indicator values.

        Args:
            raw_indicators: Dict of indicator_name -> raw value
            price: Current price (for price-relative features)
            benchmark_data: Optional benchmark data for RS

        Returns:
            Complete IndicatorStateVector
        """
        # 1. Normalize continuous features
        continuous = self._build_continuous(raw_indicators, price)

        # 2. Discretize into categories
        categorical = self._build_categorical(raw_indicators)

        # 3. Classify regimes
        regimes = self._build_regimes(raw_indicators)

        # 4. Compute signals
        signals = self._build_signals(raw_indicators)

        # 5. Compute meta-features
        meta_features = self._build_meta_features(continuous, signals)

        return IndicatorStateVector(
            ticker=raw_indicators.get('ticker', ''),
            timestamp=raw_indicators.get('timestamp', datetime.now()),
            timeframe=raw_indicators.get('timeframe', '1d'),
            continuous=continuous,
            categorical=categorical,
            regimes=regimes,
            signals=signals,
            meta_features=meta_features
        )

    def _build_continuous(
        self,
        raw: Dict[str, float],
        price: float
    ) -> Dict[str, float]:
        """Build normalized continuous features."""
        result = {}

        for feature_name, norm_type in self.spec['continuous'].items():
            raw_name = self._get_raw_indicator_name(feature_name)
            raw_value = raw.get(raw_name)

            if raw_value is None:
                result[feature_name] = np.nan
                continue

            if norm_type == 'price_relative':
                result[feature_name] = (price - raw_value) / price
            elif norm_type == 'min_max':
                result[feature_name] = self.normalizers['min_max'].transform(raw_value)
            elif norm_type == 'z_score':
                result[feature_name] = self.normalizers['z_score'].transform(raw_value)
            elif norm_type == 'percentile':
                result[feature_name] = self.normalizers['percentile'].transform(raw_value)
            elif norm_type == 'raw':
                result[feature_name] = raw_value
            else:
                result[feature_name] = raw_value

        return result

    def _build_categorical(
        self,
        raw: Dict[str, float]
    ) -> Dict[str, str]:
        """Build discretized categorical features."""
        result = {}

        for feature_name, categories in self.spec['categorical'].items():
            raw_name = self._get_raw_indicator_name(feature_name)
            raw_value = raw.get(raw_name)

            if raw_value is None:
                result[feature_name] = 'UNKNOWN'
                continue

            discretizer = self.discretizers.get(feature_name)
            if discretizer:
                result[feature_name] = discretizer.transform(raw_value)
            else:
                result[feature_name] = 'UNKNOWN'

        return result

    def _build_signals(
        self,
        raw: Dict[str, float]
    ) -> Dict[str, int]:
        """Compute signal states (-1, 0, 1)."""
        signals = {}

        # MACD signal
        macd = raw.get('macd', 0)
        macd_signal = raw.get('macd_signal', 0)
        signals['macd_signal'] = 1 if macd > macd_signal else (-1 if macd < macd_signal else 0)

        # RSI signal
        rsi = raw.get('rsi', 50)
        if rsi > 70:
            signals['rsi_signal'] = -1  # Overbought = bearish signal
        elif rsi < 30:
            signals['rsi_signal'] = 1   # Oversold = bullish signal
        else:
            signals['rsi_signal'] = 0

        # EMA crossover
        ema_8 = raw.get('ema_8', 0)
        ema_21 = raw.get('ema_21', 0)
        signals['ema_8_21_cross'] = 1 if ema_8 > ema_21 else -1

        # Price vs EMA(50)
        price = raw.get('close', 0)
        ema_50 = raw.get('ema_50', 0)
        signals['price_ema_50_cross'] = 1 if price > ema_50 else -1

        # Volume breakout
        volume_ratio = raw.get('volume_ratio', 1)
        signals['volume_breakout'] = 1 if volume_ratio > 2.0 else 0

        return signals

    def _build_meta_features(
        self,
        continuous: Dict[str, float],
        signals: Dict[str, int]
    ) -> Dict[str, float]:
        """Compute derived meta-features."""
        meta = {}

        # Signal consensus
        bullish = sum(1 for s in signals.values() if s == 1)
        bearish = sum(1 for s in signals.values() if s == -1)
        meta['bullish_signal_count'] = bullish
        meta['bearish_signal_count'] = bearish
        meta['signal_consensus'] = bullish - bearish

        # EMA alignment (-1 to 1)
        # Perfect uptrend: price > ema8 > ema21 > ema50 > ema200
        ema_pcts = [
            continuous.get('ema_8_pct', 0),
            continuous.get('ema_21_pct', 0),
            continuous.get('ema_50_pct', 0),
            continuous.get('ema_200_pct', 0),
        ]
        # Check if sorted descending (bullish stack)
        alignment = 0
        for i in range(len(ema_pcts) - 1):
            if ema_pcts[i] > ema_pcts[i + 1]:
                alignment += 1
            elif ema_pcts[i] < ema_pcts[i + 1]:
                alignment -= 1
        meta['ema_alignment'] = alignment / (len(ema_pcts) - 1)

        return meta

    def _get_raw_indicator_name(self, feature_name: str) -> str:
        """Map feature name to raw indicator name."""
        mapping = {
            'ema_8_pct': 'ema_8',
            'ema_21_pct': 'ema_21',
            'rsi_norm': 'rsi',
            'rsi_zone': 'rsi',
            'adx_norm': 'adx',
            'adx_zone': 'adx',
            # ... etc
        }
        return mapping.get(feature_name, feature_name.replace('_norm', '').replace('_z', '').replace('_pctl', ''))
```

### 5.4 State Vector Encoding for Storage

```python
class StateVectorEncoder:
    """
    Encode/decode state vectors for efficient storage.
    """

    def to_dense_array(
        self,
        state: IndicatorStateVector
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert to dense numpy array with metadata.

        Returns:
            (array, column_mapping)
        """
        values = []
        columns = {}
        idx = 0

        # Continuous features
        for name, value in state.continuous.items():
            values.append(value)
            columns[name] = idx
            idx += 1

        # One-hot encode categoricals
        for name, category in state.categorical.items():
            all_categories = STATE_VECTOR_SPEC['categorical'][name]
            for cat in all_categories:
                values.append(1.0 if category == cat else 0.0)
                columns[f"{name}_{cat}"] = idx
                idx += 1

        # Signals
        for name, value in state.signals.items():
            values.append(float(value))
            columns[f"signal_{name}"] = idx
            idx += 1

        # Meta-features
        for name, value in state.meta_features.items():
            values.append(value)
            columns[f"meta_{name}"] = idx
            idx += 1

        return np.array(values, dtype=np.float32), columns

    def to_sparse_dict(
        self,
        state: IndicatorStateVector
    ) -> Dict[str, Any]:
        """
        Convert to sparse dictionary (only non-zero values).
        Efficient for storage in document databases.
        """
        result = {
            'ticker': state.ticker,
            'timestamp': state.timestamp.isoformat(),
            'timeframe': state.timeframe,
            'c': {},  # continuous
            'd': {},  # discrete (categorical)
            's': {},  # signals
            'm': {},  # meta
        }

        # Only store non-null continuous
        for name, value in state.continuous.items():
            if not np.isnan(value):
                result['c'][name] = round(value, 6)

        # Store categorical as indices
        for name, category in state.categorical.items():
            categories = STATE_VECTOR_SPEC['categorical'][name]
            result['d'][name] = categories.index(category) if category in categories else -1

        # Store non-zero signals
        for name, value in state.signals.items():
            if value != 0:
                result['s'][name] = value

        # Store meta-features
        for name, value in state.meta_features.items():
            if not np.isnan(value):
                result['m'][name] = round(value, 4)

        return result
```

---

## 6. Multi-Timeframe Feature Handling

### 6.1 Timeframe Hierarchy

```python
# Supported timeframes and their relationships
TIMEFRAME_HIERARCHY = {
    '1m':  {'parent': '5m',  'bars_per_parent': 5},
    '5m':  {'parent': '15m', 'bars_per_parent': 3},
    '15m': {'parent': '1h',  'bars_per_parent': 4},
    '1h':  {'parent': '4h',  'bars_per_parent': 4},
    '4h':  {'parent': '1d',  'bars_per_parent': ~6},
    '1d':  {'parent': '1wk', 'bars_per_parent': 5},
    '1wk': {'parent': '1mo', 'bars_per_parent': ~4},
}

# Standard multi-timeframe combinations
MTF_COMBINATIONS = {
    'intraday': ['5m', '15m', '1h', '4h'],
    'swing': ['1h', '4h', '1d', '1wk'],
    'position': ['1d', '1wk', '1mo'],
    'default': ['1d'],  # Daily only
}
```

### 6.2 Multi-Timeframe Feature Builder

```python
class MultiTimeframeFeatureBuilder:
    """
    Build features across multiple timeframes.
    """

    def __init__(
        self,
        timeframes: List[str],
        indicators: List[str]
    ):
        self.timeframes = timeframes
        self.indicators = indicators
        self.primary_tf = timeframes[-1]  # Highest timeframe is primary

    def build_mtf_features(
        self,
        indicator_values: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Build multi-timeframe features.

        Args:
            indicator_values: {timeframe: {indicator: value}}

        Returns:
            Flattened feature dict with MTF suffix
        """
        features = {}

        for tf in self.timeframes:
            suffix = f"_{tf}" if tf != self.primary_tf else ""

            for indicator in self.indicators:
                value = indicator_values.get(tf, {}).get(indicator)
                if value is not None:
                    features[f"{indicator}{suffix}"] = value

        # Add MTF alignment features
        features.update(self._compute_alignment_features(indicator_values))

        return features

    def _compute_alignment_features(
        self,
        indicator_values: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute cross-timeframe alignment features.
        """
        alignment = {}

        # Trend alignment across timeframes
        trend_signals = []
        for tf in self.timeframes:
            tf_data = indicator_values.get(tf, {})
            # Use EMA slope as trend signal
            ema_slope = tf_data.get('ema_21_slope', 0)
            trend_signals.append(1 if ema_slope > 0 else -1)

        alignment['trend_alignment'] = sum(trend_signals) / len(trend_signals)

        # Momentum alignment
        momentum_signals = []
        for tf in self.timeframes:
            tf_data = indicator_values.get(tf, {})
            rsi = tf_data.get('rsi', 50)
            momentum_signals.append(1 if rsi > 50 else -1)

        alignment['momentum_alignment'] = sum(momentum_signals) / len(momentum_signals)

        # Volatility consistency
        volatility_values = []
        for tf in self.timeframes:
            tf_data = indicator_values.get(tf, {})
            hv_pctl = tf_data.get('hv_percentile', 50)
            volatility_values.append(hv_pctl)

        if len(volatility_values) >= 2:
            alignment['volatility_consistency'] = 1 - np.std(volatility_values) / 50

        return alignment

    def get_feature_names(self) -> List[str]:
        """Get all feature names including MTF variants."""
        names = []

        for tf in self.timeframes:
            suffix = f"_{tf}" if tf != self.primary_tf else ""
            for indicator in self.indicators:
                names.append(f"{indicator}{suffix}")

        # Alignment features
        names.extend([
            'trend_alignment',
            'momentum_alignment',
            'volatility_consistency'
        ])

        return names
```

### 6.3 Timeframe Synchronization

```python
class TimeframeSynchronizer:
    """
    Align data across timeframes for consistent feature computation.
    """

    def __init__(self, timeframes: List[str]):
        self.timeframes = sorted(
            timeframes,
            key=lambda x: TIMEFRAME_MINUTES.get(x, 0)
        )

    def synchronize(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Align timestamps across timeframes.

        Higher timeframe data is forward-filled to lower timeframe timestamps.
        This prevents lookahead bias.
        """
        # Get base timestamps from lowest timeframe
        base_tf = self.timeframes[0]
        base_timestamps = data[base_tf].index

        synchronized = {}

        for tf in self.timeframes:
            df = data[tf].copy()

            if tf == base_tf:
                synchronized[tf] = df
            else:
                # Reindex to base timestamps, forward fill
                df_reindexed = df.reindex(base_timestamps, method='ffill')
                synchronized[tf] = df_reindexed

        return synchronized

    def get_latest_complete_bar(
        self,
        timestamp: datetime,
        timeframe: str
    ) -> datetime:
        """
        Get the timestamp of the most recent complete bar.
        Prevents using incomplete bar data.
        """
        minutes = TIMEFRAME_MINUTES[timeframe]
        # Round down to nearest complete bar
        return timestamp - timedelta(minutes=timestamp.minute % minutes)

# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
    '1wk': 10080,
}
```

### 6.4 MTF Feature Naming Convention

```python
# Feature naming pattern: {indicator}_{normalization}_{timeframe}
# Examples:
#   rsi_norm           (primary timeframe, normalized)
#   rsi_norm_1h        (1-hour timeframe)
#   rsi_norm_4h        (4-hour timeframe)
#   macd_z_1d          (daily timeframe, z-scored)

MTF_FEATURE_PATTERN = re.compile(
    r'^(?P<indicator>[a-z_]+)_(?P<norm>norm|z|pct|raw)(?:_(?P<timeframe>\d+[mhdw]))?$'
)

def parse_mtf_feature_name(name: str) -> Dict[str, str]:
    """Parse MTF feature name into components."""
    match = MTF_FEATURE_PATTERN.match(name)
    if match:
        return match.groupdict()
    return {'indicator': name, 'norm': 'raw', 'timeframe': None}
```

### 6.5 MTF State Vector Extension

```python
@dataclass
class MTFIndicatorStateVector:
    """
    Multi-timeframe state vector.
    """

    ticker: str
    timestamp: datetime
    timeframes: List[str]

    # Per-timeframe state vectors
    states: Dict[str, IndicatorStateVector]

    # Cross-timeframe features
    alignment_features: Dict[str, float]

    # Composite signals considering all timeframes
    mtf_signals: Dict[str, int]

    def get_primary_state(self) -> IndicatorStateVector:
        """Get state for primary (highest) timeframe."""
        primary_tf = max(
            self.timeframes,
            key=lambda x: TIMEFRAME_MINUTES.get(x, 0)
        )
        return self.states[primary_tf]

    def get_all_continuous(self) -> Dict[str, float]:
        """Flatten all continuous features with TF suffix."""
        result = {}
        for tf, state in self.states.items():
            suffix = '' if tf == self.timeframes[-1] else f'_{tf}'
            for name, value in state.continuous.items():
                result[f"{name}{suffix}"] = value
        result.update(self.alignment_features)
        return result
```

---

## 7. Feature Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RAW INDICATORS                                                             │
│  ──────────────                                                             │
│  RSI=35, MACD=1.5, ATR=2.3, EMA_50=148.5, Volume=5M, ...                   │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. NORMALIZATION                                                    │   │
│  │    - Bounded → Min-Max [0,1]                                        │   │
│  │    - Unbounded → Rolling Z-Score                                    │   │
│  │    - Price-based → Price-Relative                                   │   │
│  │    - Cross-sectional → Percentile Rank                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. DISCRETIZATION                                                   │   │
│  │    - Fixed thresholds (RSI → OVERSOLD/NEUTRAL/OVERBOUGHT)           │   │
│  │    - Quantile bins (MACD → Q1/Q2/Q3/Q4/Q5)                          │   │
│  │    - Regime classification (UPTREND/RANGE/DOWNTREND)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. COLLINEARITY REDUCTION                                           │   │
│  │    - Remove highly correlated (>0.85)                               │   │
│  │    - Keep representative from each group                            │   │
│  │    - Preserve interpretable features                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 4. STATE VECTOR ASSEMBLY                                            │   │
│  │    - Continuous features (~25 dims)                                 │   │
│  │    - Categorical features (one-hot, ~20 dims)                       │   │
│  │    - Signal states (~8 dims)                                        │   │
│  │    - Meta-features (~5 dims)                                        │   │
│  │    = ~58 total dimensions per timeframe                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 5. MULTI-TIMEFRAME EXTENSION                                        │   │
│  │    - Replicate for each timeframe                                   │   │
│  │    - Add alignment features                                         │   │
│  │    - Synchronize timestamps                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  FINAL STATE VECTOR                                                         │
│  ──────────────────                                                         │
│  [0.35, 0.72, 1.24, -0.5, ..., 1, 0, -1, ..., 0.75, ...]                   │
│  Ready for scoring, screening, or ML                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Assumptions

1. **Rolling Statistics:** Z-scores and percentiles use 252-day (1 year) lookback as default
2. **Cross-Sectional Data:** Universe-wide percentiles require daily universe computation
3. **Stationarity:** Normalization transforms assume reasonable stationarity over lookback window
4. **Missing Data:** NaN handling via forward-fill for short gaps, exclusion for long gaps
5. **Timeframe Independence:** Each timeframe computed independently, then aligned
6. **Primary Timeframe:** Daily (1d) is the default primary timeframe
7. **Feature Stability:** Feature definitions are versioned for reproducibility

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
