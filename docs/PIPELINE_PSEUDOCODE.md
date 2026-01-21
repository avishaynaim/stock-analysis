# Full Pipeline Pseudocode

## 1. Overview

This document provides end-to-end pseudocode for the stock analysis system's major execution flows. The pseudocode is implementation-ready and covers all critical paths including error handling.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM PIPELINE OVERVIEW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           ┌─────────────────┐                               │
│                           │   USER INPUT    │                               │
│                           │  (CLI Command)  │                               │
│                           └────────┬────────┘                               │
│                                    │                                        │
│                    ┌───────────────┼───────────────┐                        │
│                    │               │               │                        │
│                    ▼               ▼               ▼                        │
│           ┌────────────┐   ┌────────────┐   ┌────────────┐                  │
│           │  ANALYZE   │   │    SCAN    │   │  BACKTEST  │                  │
│           │  (single)  │   │ (universe) │   │   (hist)   │                  │
│           └─────┬──────┘   └─────┬──────┘   └─────┬──────┘                  │
│                 │                │                │                         │
│                 └────────────────┼────────────────┘                         │
│                                  │                                          │
│                                  ▼                                          │
│                    ┌─────────────────────────┐                              │
│                    │     CORE PIPELINE       │                              │
│                    │  ┌─────────────────┐    │                              │
│                    │  │   DATA LAYER    │    │                              │
│                    │  └────────┬────────┘    │                              │
│                    │           ▼             │                              │
│                    │  ┌─────────────────┐    │                              │
│                    │  │   INDICATORS    │    │                              │
│                    │  └────────┬────────┘    │                              │
│                    │           ▼             │                              │
│                    │  ┌─────────────────┐    │                              │
│                    │  │    FEATURES     │    │                              │
│                    │  └────────┬────────┘    │                              │
│                    │           ▼             │                              │
│                    │  ┌─────────────────┐    │                              │
│                    │  │     MODELS      │    │                              │
│                    │  └────────┬────────┘    │                              │
│                    │           ▼             │                              │
│                    │  ┌─────────────────┐    │                              │
│                    │  │    SCORING      │    │                              │
│                    │  └────────┬────────┘    │                              │
│                    │           ▼             │                              │
│                    │  ┌─────────────────┐    │                              │
│                    │  │     OUTPUT      │    │                              │
│                    │  └─────────────────┘    │                              │
│                    └─────────────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Universe Scan Flow

### 2.1 High-Level Scan Flow

```python
def scan_universe(
    universe: str,
    min_score: float,
    max_results: int,
    horizon: str,
    filters: FilterConfig,
    as_of_date: datetime = None
) -> ScanResult:
    """
    UNIVERSE SCAN - Main Entry Point

    Scans entire universe for high-scoring opportunities.
    Returns ranked list of candidates meeting criteria.
    """

    # =========================================================================
    # PHASE 1: INITIALIZATION
    # =========================================================================

    logger.info(f"Starting universe scan: {universe}")

    # Set analysis date (default: latest available)
    if as_of_date is None:
        as_of_date = get_latest_trading_date()

    # Validate date is not in future
    if as_of_date > datetime.now():
        raise ValidationError("Cannot analyze future date")

    # Load configuration
    config = Config.get_all()

    # Initialize components
    data_provider = DataProvider(config.data)
    indicator_engine = IndicatorEngine(config.indicators)
    feature_builder = FeatureBuilder(config.features)
    model_ensemble = load_models(config.models)
    scoring_engine = ScoringEngine(config.scoring)

    # Initialize parallel executor
    executor = ParallelExecutor(config.execution.parallel)

    # =========================================================================
    # PHASE 2: UNIVERSE CONSTRUCTION
    # =========================================================================

    logger.info("Building universe...")

    # Get universe members as of date (point-in-time)
    try:
        raw_universe = data_provider.get_universe_members(
            universe=universe,
            as_of_date=as_of_date
        )
    except DataError as e:
        logger.error(f"Failed to get universe: {e}")
        raise

    logger.info(f"Raw universe size: {len(raw_universe)}")

    # Apply filters
    filtered_universe = apply_universe_filters(
        tickers=raw_universe,
        filters=filters,
        data_provider=data_provider,
        as_of_date=as_of_date
    )

    logger.info(f"Filtered universe size: {len(filtered_universe)}")

    if len(filtered_universe) == 0:
        logger.warning("No tickers passed filters")
        return ScanResult(candidates=[], metadata={...})

    # =========================================================================
    # PHASE 3: DATA PREFETCH
    # =========================================================================

    logger.info("Prefetching data...")

    # Batch prefetch for efficiency
    try:
        data_provider.prefetch_batch(
            tickers=filtered_universe,
            start_date=as_of_date - timedelta(days=756),  # 3 years lookback
            end_date=as_of_date,
            data_types=['prices', 'fundamentals', 'earnings']
        )
    except DataError as e:
        logger.warning(f"Prefetch partially failed: {e}")
        # Continue - individual ticker analysis will handle missing data

    # =========================================================================
    # PHASE 4: PARALLEL ANALYSIS
    # =========================================================================

    logger.info(f"Analyzing {len(filtered_universe)} tickers...")

    # Create analysis tasks
    tasks = [
        AnalysisTask(
            ticker=ticker,
            as_of_date=as_of_date,
            horizon=horizon,
            include_probability=True
        )
        for ticker in filtered_universe
    ]

    # Execute in parallel with progress tracking
    results = []
    errors = []

    with ProgressBar(total=len(tasks), desc="Scanning") as progress:

        # Process in chunks for memory efficiency
        for chunk in chunked(tasks, size=config.execution.parallel.chunk_size):

            chunk_results = executor.map(
                func=lambda task: analyze_single_ticker_safe(
                    task=task,
                    data_provider=data_provider,
                    indicator_engine=indicator_engine,
                    feature_builder=feature_builder,
                    model_ensemble=model_ensemble,
                    scoring_engine=scoring_engine
                ),
                items=chunk
            )

            for result in chunk_results:
                if isinstance(result, AnalysisError):
                    errors.append(result)
                else:
                    results.append(result)

                progress.update(1)

    logger.info(f"Analysis complete: {len(results)} success, {len(errors)} errors")

    # =========================================================================
    # PHASE 5: FILTERING & RANKING
    # =========================================================================

    logger.info("Filtering and ranking results...")

    # Filter by minimum score
    candidates = [r for r in results if r.score >= min_score]

    logger.info(f"Candidates meeting threshold: {len(candidates)}")

    # Sort by score (descending)
    candidates.sort(key=lambda x: x.score, reverse=True)

    # Limit results
    candidates = candidates[:max_results]

    # =========================================================================
    # PHASE 6: RESULT ASSEMBLY
    # =========================================================================

    scan_result = ScanResult(
        candidates=candidates,
        metadata={
            'universe': universe,
            'as_of_date': as_of_date,
            'horizon': horizon,
            'raw_universe_size': len(raw_universe),
            'filtered_universe_size': len(filtered_universe),
            'analyzed_count': len(results),
            'error_count': len(errors),
            'candidates_count': len(candidates),
            'min_score_threshold': min_score,
            'scan_duration_seconds': elapsed_time,
            'errors': [e.to_dict() for e in errors[:10]]  # First 10 errors
        }
    )

    logger.info(f"Scan complete: {len(candidates)} candidates found")

    return scan_result
```

### 2.2 Universe Filter Application

```python
def apply_universe_filters(
    tickers: List[str],
    filters: FilterConfig,
    data_provider: DataProvider,
    as_of_date: datetime
) -> List[str]:
    """
    Apply filters to universe members.

    Filters applied:
    1. Minimum price
    2. Minimum volume
    3. Minimum market cap
    4. Sector inclusion/exclusion
    5. Data availability
    """

    filtered = []
    filter_stats = defaultdict(int)

    for ticker in tickers:

        # ─────────────────────────────────────────────────────────────────────
        # CHECK: Data Availability
        # ─────────────────────────────────────────────────────────────────────

        try:
            price_data = data_provider.get_latest_price(ticker, as_of_date)
        except DataNotFoundError:
            filter_stats['no_data'] += 1
            continue

        if price_data is None:
            filter_stats['no_data'] += 1
            continue

        # ─────────────────────────────────────────────────────────────────────
        # CHECK: Minimum Price
        # ─────────────────────────────────────────────────────────────────────

        if filters.min_price is not None:
            if price_data.close < filters.min_price:
                filter_stats['below_min_price'] += 1
                continue

        # ─────────────────────────────────────────────────────────────────────
        # CHECK: Minimum Volume
        # ─────────────────────────────────────────────────────────────────────

        if filters.min_volume is not None:
            avg_volume = data_provider.get_average_dollar_volume(
                ticker=ticker,
                as_of_date=as_of_date,
                lookback_days=20
            )

            if avg_volume < filters.min_volume:
                filter_stats['below_min_volume'] += 1
                continue

        # ─────────────────────────────────────────────────────────────────────
        # CHECK: Minimum Market Cap
        # ─────────────────────────────────────────────────────────────────────

        if filters.min_market_cap is not None:
            market_cap = data_provider.get_market_cap(ticker, as_of_date)

            if market_cap is None or market_cap < filters.min_market_cap:
                filter_stats['below_min_mcap'] += 1
                continue

        # ─────────────────────────────────────────────────────────────────────
        # CHECK: Sector Filter
        # ─────────────────────────────────────────────────────────────────────

        if filters.sectors is not None and len(filters.sectors) > 0:
            sector = data_provider.get_sector(ticker)

            if filters.sector_mode == 'include':
                if sector not in filters.sectors:
                    filter_stats['sector_excluded'] += 1
                    continue
            else:  # exclude mode
                if sector in filters.sectors:
                    filter_stats['sector_excluded'] += 1
                    continue

        # ─────────────────────────────────────────────────────────────────────
        # CHECK: Minimum History
        # ─────────────────────────────────────────────────────────────────────

        if filters.min_history_days is not None:
            history_days = data_provider.get_history_length(ticker, as_of_date)

            if history_days < filters.min_history_days:
                filter_stats['insufficient_history'] += 1
                continue

        # ─────────────────────────────────────────────────────────────────────
        # PASSED ALL FILTERS
        # ─────────────────────────────────────────────────────────────────────

        filtered.append(ticker)

    # Log filter statistics
    logger.debug(f"Filter statistics: {dict(filter_stats)}")

    return filtered
```

---

## 3. Single Ticker Flow

### 3.1 Complete Single Ticker Analysis

```python
def analyze_single_ticker(
    ticker: str,
    as_of_date: datetime,
    horizons: List[str] = ['5d', '21d', '63d'],
    include_indicators: bool = True,
    include_probability: bool = True,
    include_risk_analysis: bool = True,
    benchmark: str = None
) -> TickerAnalysis:
    """
    SINGLE TICKER ANALYSIS - Complete Flow

    Performs comprehensive analysis of a single ticker.
    """

    logger.info(f"Analyzing {ticker} as of {as_of_date}")

    # =========================================================================
    # PHASE 1: INITIALIZATION & VALIDATION
    # =========================================================================

    # Validate ticker format
    if not is_valid_ticker(ticker):
        raise ValidationError(f"Invalid ticker format: {ticker}")

    # Load components
    config = Config.get_all()
    data_provider = DataProvider(config.data)
    indicator_engine = IndicatorEngine(config.indicators)
    feature_builder = FeatureBuilder(config.features)
    model_ensemble = load_models(config.models)
    scoring_engine = ScoringEngine(config.scoring)
    probability_engine = ProbabilityEngine(config.probability)

    # =========================================================================
    # PHASE 2: DATA RETRIEVAL
    # =========================================================================

    logger.debug(f"[{ticker}] Loading data...")

    # Calculate required lookback
    max_lookback_days = 756  # 3 years for all indicators
    start_date = as_of_date - timedelta(days=max_lookback_days)

    # Load price data
    try:
        prices = data_provider.get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=as_of_date
        )
    except DataNotFoundError:
        raise AnalysisError(
            ticker=ticker,
            phase='data_retrieval',
            message=f"No price data found for {ticker}"
        )

    # Validate sufficient data
    if len(prices) < config.data.quality.min_history_days:
        raise AnalysisError(
            ticker=ticker,
            phase='data_validation',
            message=f"Insufficient history: {len(prices)} days"
        )

    # Load supplementary data (non-critical)
    fundamentals = safe_load(
        lambda: data_provider.get_fundamentals(ticker, as_of_date),
        default=None
    )

    earnings = safe_load(
        lambda: data_provider.get_earnings(ticker, as_of_date),
        default=None
    )

    # Load benchmark if requested
    benchmark_prices = None
    if benchmark:
        benchmark_prices = safe_load(
            lambda: data_provider.get_prices(benchmark, start_date, as_of_date),
            default=None
        )

    # =========================================================================
    # PHASE 3: INDICATOR COMPUTATION
    # =========================================================================

    logger.debug(f"[{ticker}] Computing indicators...")

    indicators = {}

    for timeframe in config.indicators.timeframes:

        # Resample prices to timeframe
        tf_prices = resample_prices(prices, timeframe)

        if len(tf_prices) < 50:  # Minimum for indicators
            logger.warning(f"[{ticker}] Insufficient data for {timeframe}")
            continue

        # Compute all indicator groups
        tf_indicators = {}

        for group in INDICATOR_GROUPS:

            if not config.indicators.groups.get(group, True):
                continue  # Group disabled

            try:
                group_values = indicator_engine.compute_group(
                    group=group,
                    prices=tf_prices,
                    fundamentals=fundamentals,
                    as_of_date=as_of_date
                )
                tf_indicators[group] = group_values

            except IndicatorError as e:
                logger.warning(f"[{ticker}] Indicator error ({group}): {e}")
                tf_indicators[group] = None

        indicators[timeframe] = tf_indicators

    # =========================================================================
    # PHASE 4: FEATURE ENGINEERING
    # =========================================================================

    logger.debug(f"[{ticker}] Building features...")

    # Build feature vector
    try:
        features = feature_builder.build(
            indicators=indicators,
            prices=prices,
            fundamentals=fundamentals,
            as_of_date=as_of_date
        )
    except FeatureError as e:
        raise AnalysisError(
            ticker=ticker,
            phase='feature_engineering',
            message=str(e)
        )

    # Normalize features
    features_normalized = feature_builder.normalize(features)

    # Build state vector for probability engine
    state_vector = feature_builder.build_state_vector(features_normalized)

    # =========================================================================
    # PHASE 5: MODEL INFERENCE
    # =========================================================================

    logger.debug(f"[{ticker}] Running models...")

    model_outputs = {}

    # Run each model in ensemble
    for model_name, model in model_ensemble.items():

        try:
            output = model.predict(features_normalized)
            model_outputs[model_name] = output

        except ModelError as e:
            logger.warning(f"[{ticker}] Model error ({model_name}): {e}")
            model_outputs[model_name] = None

    # Aggregate model outputs
    ensemble_output = aggregate_model_outputs(
        outputs=model_outputs,
        weights=config.models.ensemble.weights
    )

    # =========================================================================
    # PHASE 6: PROBABILITY ESTIMATION
    # =========================================================================

    probabilities = {}

    if include_probability:

        logger.debug(f"[{ticker}] Estimating probabilities...")

        for horizon in horizons:

            try:
                prob_estimate = probability_engine.estimate(
                    ticker=ticker,
                    state_vector=state_vector,
                    horizon=horizon,
                    as_of_date=as_of_date
                )
                probabilities[horizon] = prob_estimate

            except ProbabilityError as e:
                logger.warning(f"[{ticker}] Probability error ({horizon}): {e}")
                probabilities[horizon] = ProbabilityEstimate(
                    probability=0.5,
                    confidence='low',
                    sample_size=0
                )

    # =========================================================================
    # PHASE 7: SCORING
    # =========================================================================

    logger.debug(f"[{ticker}] Computing scores...")

    # Compute subscores
    subscores = scoring_engine.compute_subscores(
        indicators=indicators,
        features=features,
        model_output=ensemble_output
    )

    # Compute edge score from probabilities
    edge_score = scoring_engine.compute_edge_score(
        probabilities=probabilities,
        horizon_weights=config.scoring.horizon_weights
    )
    subscores['edge'] = edge_score

    # Compute risk penalties
    risk_penalties = scoring_engine.compute_risk_penalties(
        prices=prices,
        indicators=indicators,
        as_of_date=as_of_date
    )

    # Compute final score
    final_score = scoring_engine.compute_final_score(
        subscores=subscores,
        risk_penalties=risk_penalties,
        weights=config.scoring.weights
    )

    # Get risk flags
    risk_flags = scoring_engine.get_risk_flags(
        risk_penalties=risk_penalties,
        indicators=indicators
    )

    # =========================================================================
    # PHASE 8: BENCHMARK COMPARISON (Optional)
    # =========================================================================

    relative_metrics = None

    if benchmark_prices is not None:

        logger.debug(f"[{ticker}] Computing relative metrics...")

        relative_metrics = compute_relative_metrics(
            ticker_prices=prices,
            benchmark_prices=benchmark_prices,
            as_of_date=as_of_date
        )

    # =========================================================================
    # PHASE 9: RESULT ASSEMBLY
    # =========================================================================

    analysis = TickerAnalysis(
        ticker=ticker,
        as_of_date=as_of_date,

        # Core results
        score=final_score.score,
        score_label=final_score.label,
        subscores=subscores,

        # Probability
        probabilities=probabilities if include_probability else None,

        # Risk
        risk_penalties=risk_penalties,
        risk_flags=risk_flags,

        # Details
        indicators=indicators if include_indicators else None,
        features=features if include_indicators else None,
        model_outputs=model_outputs,

        # Relative
        benchmark=benchmark,
        relative_metrics=relative_metrics,

        # Metadata
        metadata={
            'analysis_timestamp': datetime.now(),
            'data_start': prices.index[0],
            'data_end': prices.index[-1],
            'data_points': len(prices),
            'model_version': config.models.version
        }
    )

    logger.info(f"[{ticker}] Analysis complete: Score = {final_score.score:.1f}")

    return analysis
```

### 3.2 Safe Single Ticker Analysis (For Parallel Execution)

```python
def analyze_single_ticker_safe(
    task: AnalysisTask,
    data_provider: DataProvider,
    indicator_engine: IndicatorEngine,
    feature_builder: FeatureBuilder,
    model_ensemble: Dict[str, Model],
    scoring_engine: ScoringEngine
) -> Union[TickerAnalysis, AnalysisError]:
    """
    Safe wrapper for single ticker analysis.

    Catches all exceptions and returns AnalysisError instead of raising.
    Used for parallel execution where we don't want one failure to stop all.
    """

    try:
        # Run full analysis
        return analyze_single_ticker_internal(
            ticker=task.ticker,
            as_of_date=task.as_of_date,
            horizons=[task.horizon] if task.horizon != 'all' else ['5d', '21d', '63d'],
            include_probability=task.include_probability,
            data_provider=data_provider,
            indicator_engine=indicator_engine,
            feature_builder=feature_builder,
            model_ensemble=model_ensemble,
            scoring_engine=scoring_engine
        )

    except AnalysisError as e:
        # Already wrapped error
        return e

    except DataNotFoundError as e:
        return AnalysisError(
            ticker=task.ticker,
            phase='data_retrieval',
            message=str(e),
            recoverable=False
        )

    except IndicatorError as e:
        return AnalysisError(
            ticker=task.ticker,
            phase='indicator_computation',
            message=str(e),
            recoverable=True
        )

    except ModelError as e:
        return AnalysisError(
            ticker=task.ticker,
            phase='model_inference',
            message=str(e),
            recoverable=True
        )

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error analyzing {task.ticker}: {e}")
        return AnalysisError(
            ticker=task.ticker,
            phase='unknown',
            message=str(e),
            exception=e,
            recoverable=False
        )
```

---

## 4. Data → Features → Models → Score → Output Pipeline

### 4.1 Complete Pipeline Flow

```python
class AnalysisPipeline:
    """
    Complete analysis pipeline from raw data to final output.

    Pipeline stages:
    1. DATA      - Load and validate raw market data
    2. INDICATORS - Compute technical/fundamental indicators
    3. FEATURES   - Engineer and normalize features
    4. MODELS     - Run ML models for prediction
    5. SCORING    - Compute final scores
    6. OUTPUT     - Format and deliver results
    """

    def __init__(self, config: Config):
        self.config = config
        self._init_components()

    def _init_components(self):
        """Initialize pipeline components."""

        # Stage 1: Data
        self.data_provider = DataProvider(self.config.data)
        self.data_validator = DataValidator(self.config.data.quality)

        # Stage 2: Indicators
        self.indicator_engine = IndicatorEngine(self.config.indicators)

        # Stage 3: Features
        self.feature_builder = FeatureBuilder(self.config.features)
        self.normalizer = FeatureNormalizer(self.config.features.normalization)

        # Stage 4: Models
        self.model_loader = ModelLoader(self.config.models)
        self.models = self.model_loader.load_all()

        # Stage 5: Scoring
        self.scoring_engine = ScoringEngine(self.config.scoring)
        self.probability_engine = ProbabilityEngine(self.config.probability)

        # Stage 6: Output
        self.formatter = OutputFormatter(self.config.reporting)

    # =========================================================================
    # STAGE 1: DATA
    # =========================================================================

    def stage_data(
        self,
        ticker: str,
        as_of_date: datetime
    ) -> PipelineData:
        """
        Stage 1: Load and validate all required data.
        """

        logger.debug(f"[PIPELINE:{ticker}] Stage 1: DATA")

        # Calculate lookback period
        lookback_days = self._calculate_lookback()
        start_date = as_of_date - timedelta(days=lookback_days)

        # ─────────────────────────────────────────────────────────────────────
        # 1.1 Load Price Data (REQUIRED)
        # ─────────────────────────────────────────────────────────────────────

        prices = self.data_provider.get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=as_of_date
        )

        if prices is None or len(prices) == 0:
            raise DataError(f"No price data for {ticker}")

        # Validate price data
        validation = self.data_validator.validate_prices(prices)
        if not validation.is_valid:
            if validation.severity == 'critical':
                raise DataError(f"Invalid price data: {validation.issues}")
            else:
                logger.warning(f"Data quality issues: {validation.issues}")

        # Apply corporate action adjustments
        prices = self._adjust_for_corporate_actions(ticker, prices, as_of_date)

        # ─────────────────────────────────────────────────────────────────────
        # 1.2 Load Fundamental Data (OPTIONAL)
        # ─────────────────────────────────────────────────────────────────────

        fundamentals = None
        try:
            fundamentals = self.data_provider.get_fundamentals(
                ticker=ticker,
                as_of_date=as_of_date
            )
        except DataError as e:
            logger.debug(f"No fundamentals for {ticker}: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # 1.3 Load Earnings Data (OPTIONAL)
        # ─────────────────────────────────────────────────────────────────────

        earnings = None
        try:
            earnings = self.data_provider.get_earnings(
                ticker=ticker,
                as_of_date=as_of_date,
                lookback_quarters=8
            )
        except DataError as e:
            logger.debug(f"No earnings for {ticker}: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # 1.4 Load Benchmark Data (OPTIONAL)
        # ─────────────────────────────────────────────────────────────────────

        benchmark = None
        benchmark_ticker = self._get_benchmark_for_ticker(ticker)
        try:
            benchmark = self.data_provider.get_prices(
                ticker=benchmark_ticker,
                start_date=start_date,
                end_date=as_of_date
            )
        except DataError as e:
            logger.debug(f"No benchmark data: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # 1.5 Assemble Pipeline Data
        # ─────────────────────────────────────────────────────────────────────

        return PipelineData(
            ticker=ticker,
            as_of_date=as_of_date,
            prices=prices,
            fundamentals=fundamentals,
            earnings=earnings,
            benchmark=benchmark,
            metadata={
                'data_start': prices.index[0],
                'data_end': prices.index[-1],
                'data_points': len(prices),
                'has_fundamentals': fundamentals is not None,
                'has_earnings': earnings is not None,
                'has_benchmark': benchmark is not None
            }
        )

    # =========================================================================
    # STAGE 2: INDICATORS
    # =========================================================================

    def stage_indicators(
        self,
        data: PipelineData
    ) -> PipelineIndicators:
        """
        Stage 2: Compute all technical and fundamental indicators.
        """

        logger.debug(f"[PIPELINE:{data.ticker}] Stage 2: INDICATORS")

        indicators = {}
        indicator_errors = []

        # Process each timeframe
        for timeframe in self.config.indicators.timeframes:

            # ─────────────────────────────────────────────────────────────────
            # 2.1 Resample to Timeframe
            # ─────────────────────────────────────────────────────────────────

            tf_prices = self._resample_to_timeframe(data.prices, timeframe)

            if len(tf_prices) < self.config.indicators.min_periods:
                logger.warning(
                    f"[{data.ticker}] Skipping {timeframe}: "
                    f"only {len(tf_prices)} periods"
                )
                continue

            # ─────────────────────────────────────────────────────────────────
            # 2.2 Compute Indicator Groups
            # ─────────────────────────────────────────────────────────────────

            tf_indicators = {}

            # GROUP A: Trend Indicators
            if self.config.indicators.groups.get('trend', True):
                try:
                    tf_indicators['trend'] = self.indicator_engine.compute_trend(
                        prices=tf_prices
                    )
                except IndicatorError as e:
                    indicator_errors.append(('trend', timeframe, str(e)))

            # GROUP B: Momentum Indicators
            if self.config.indicators.groups.get('momentum', True):
                try:
                    tf_indicators['momentum'] = self.indicator_engine.compute_momentum(
                        prices=tf_prices
                    )
                except IndicatorError as e:
                    indicator_errors.append(('momentum', timeframe, str(e)))

            # GROUP C: Volatility Indicators
            if self.config.indicators.groups.get('volatility', True):
                try:
                    tf_indicators['volatility'] = self.indicator_engine.compute_volatility(
                        prices=tf_prices
                    )
                except IndicatorError as e:
                    indicator_errors.append(('volatility', timeframe, str(e)))

            # GROUP D: Volume Indicators
            if self.config.indicators.groups.get('volume', True):
                try:
                    tf_indicators['volume'] = self.indicator_engine.compute_volume(
                        prices=tf_prices
                    )
                except IndicatorError as e:
                    indicator_errors.append(('volume', timeframe, str(e)))

            # GROUP E: Microstructure Indicators
            if self.config.indicators.groups.get('microstructure', True):
                try:
                    tf_indicators['microstructure'] = self.indicator_engine.compute_microstructure(
                        prices=tf_prices
                    )
                except IndicatorError as e:
                    indicator_errors.append(('microstructure', timeframe, str(e)))

            # GROUP F: Regime Indicators
            if self.config.indicators.groups.get('regime', True):
                try:
                    tf_indicators['regime'] = self.indicator_engine.compute_regime(
                        prices=tf_prices,
                        benchmark=data.benchmark
                    )
                except IndicatorError as e:
                    indicator_errors.append(('regime', timeframe, str(e)))

            # GROUP G: Relative Strength Indicators
            if self.config.indicators.groups.get('relative_strength', True):
                try:
                    tf_indicators['relative_strength'] = self.indicator_engine.compute_relative_strength(
                        prices=tf_prices,
                        benchmark=data.benchmark
                    )
                except IndicatorError as e:
                    indicator_errors.append(('relative_strength', timeframe, str(e)))

            # GROUP H: Structure Indicators
            if self.config.indicators.groups.get('structure', True):
                try:
                    tf_indicators['structure'] = self.indicator_engine.compute_structure(
                        prices=tf_prices
                    )
                except IndicatorError as e:
                    indicator_errors.append(('structure', timeframe, str(e)))

            # GROUP I: Fundamental Indicators
            if self.config.indicators.groups.get('fundamentals', True) and data.fundamentals:
                try:
                    tf_indicators['fundamentals'] = self.indicator_engine.compute_fundamentals(
                        fundamentals=data.fundamentals,
                        earnings=data.earnings,
                        prices=tf_prices
                    )
                except IndicatorError as e:
                    indicator_errors.append(('fundamentals', timeframe, str(e)))

            indicators[timeframe] = tf_indicators

        # ─────────────────────────────────────────────────────────────────────
        # 2.3 Validate Minimum Indicators
        # ─────────────────────────────────────────────────────────────────────

        total_indicators = sum(
            len(tf.keys()) for tf in indicators.values()
        )

        if total_indicators < self.config.indicators.min_required:
            raise IndicatorError(
                f"Insufficient indicators: {total_indicators} < "
                f"{self.config.indicators.min_required}"
            )

        return PipelineIndicators(
            indicators=indicators,
            errors=indicator_errors,
            metadata={
                'timeframes': list(indicators.keys()),
                'groups_computed': total_indicators,
                'errors_count': len(indicator_errors)
            }
        )

    # =========================================================================
    # STAGE 3: FEATURES
    # =========================================================================

    def stage_features(
        self,
        data: PipelineData,
        indicators: PipelineIndicators
    ) -> PipelineFeatures:
        """
        Stage 3: Engineer and normalize features for model input.
        """

        logger.debug(f"[PIPELINE:{data.ticker}] Stage 3: FEATURES")

        # ─────────────────────────────────────────────────────────────────────
        # 3.1 Extract Raw Features
        # ─────────────────────────────────────────────────────────────────────

        raw_features = {}

        for timeframe, tf_indicators in indicators.indicators.items():

            tf_prefix = f"{timeframe}_"

            for group, group_values in tf_indicators.items():
                if group_values is None:
                    continue

                for indicator_name, value in group_values.items():
                    feature_name = f"{tf_prefix}{group}_{indicator_name}"
                    raw_features[feature_name] = value

        # ─────────────────────────────────────────────────────────────────────
        # 3.2 Add Derived Features
        # ─────────────────────────────────────────────────────────────────────

        derived_features = self.feature_builder.compute_derived(
            raw_features=raw_features,
            prices=data.prices,
            as_of_date=data.as_of_date
        )

        raw_features.update(derived_features)

        # ─────────────────────────────────────────────────────────────────────
        # 3.3 Handle Missing Values
        # ─────────────────────────────────────────────────────────────────────

        features_filled = self.feature_builder.handle_missing(
            features=raw_features,
            strategy=self.config.features.missing_strategy
        )

        # ─────────────────────────────────────────────────────────────────────
        # 3.4 Normalize Features
        # ─────────────────────────────────────────────────────────────────────

        features_normalized = {}

        for name, value in features_filled.items():

            # Get normalization method for this feature
            norm_method = self.normalizer.get_method(name)

            # Apply normalization
            normalized_value = self.normalizer.normalize(
                value=value,
                method=norm_method,
                feature_name=name
            )

            features_normalized[name] = normalized_value

        # ─────────────────────────────────────────────────────────────────────
        # 3.5 Build State Vector
        # ─────────────────────────────────────────────────────────────────────

        state_vector = self.feature_builder.build_state_vector(
            features=features_normalized,
            include_discretized=True
        )

        # ─────────────────────────────────────────────────────────────────────
        # 3.6 Convert to Model Input Format
        # ─────────────────────────────────────────────────────────────────────

        model_input = self.feature_builder.to_model_input(
            features=features_normalized,
            feature_order=self.config.features.feature_order
        )

        return PipelineFeatures(
            raw_features=raw_features,
            normalized_features=features_normalized,
            state_vector=state_vector,
            model_input=model_input,
            metadata={
                'raw_feature_count': len(raw_features),
                'normalized_feature_count': len(features_normalized),
                'state_vector_dim': len(state_vector),
                'missing_filled': len(raw_features) - len(features_filled)
            }
        )

    # =========================================================================
    # STAGE 4: MODELS
    # =========================================================================

    def stage_models(
        self,
        features: PipelineFeatures
    ) -> PipelineModelOutputs:
        """
        Stage 4: Run all models and aggregate predictions.
        """

        logger.debug(f"[PIPELINE] Stage 4: MODELS")

        model_outputs = {}
        model_errors = []

        # ─────────────────────────────────────────────────────────────────────
        # 4.1 Run Individual Models
        # ─────────────────────────────────────────────────────────────────────

        for model_name, model in self.models.items():

            try:
                # Check if model is ready
                if not model.is_ready():
                    logger.warning(f"Model {model_name} not ready, skipping")
                    continue

                # Run prediction
                output = model.predict(features.model_input)

                # Validate output
                if not self._validate_model_output(output, model_name):
                    logger.warning(f"Invalid output from {model_name}")
                    continue

                model_outputs[model_name] = output

            except ModelError as e:
                logger.warning(f"Model {model_name} failed: {e}")
                model_errors.append((model_name, str(e)))

        # ─────────────────────────────────────────────────────────────────────
        # 4.2 Validate Minimum Models
        # ─────────────────────────────────────────────────────────────────────

        if len(model_outputs) < self.config.models.min_required:
            raise ModelError(
                f"Insufficient model outputs: {len(model_outputs)} < "
                f"{self.config.models.min_required}"
            )

        # ─────────────────────────────────────────────────────────────────────
        # 4.3 Aggregate Model Outputs
        # ─────────────────────────────────────────────────────────────────────

        ensemble_output = self._aggregate_models(
            outputs=model_outputs,
            method=self.config.models.ensemble.method,
            weights=self.config.models.ensemble.weights
        )

        return PipelineModelOutputs(
            individual_outputs=model_outputs,
            ensemble_output=ensemble_output,
            errors=model_errors,
            metadata={
                'models_run': len(model_outputs),
                'models_failed': len(model_errors),
                'ensemble_method': self.config.models.ensemble.method
            }
        )

    # =========================================================================
    # STAGE 5: SCORING
    # =========================================================================

    def stage_scoring(
        self,
        data: PipelineData,
        indicators: PipelineIndicators,
        features: PipelineFeatures,
        model_outputs: PipelineModelOutputs,
        horizons: List[str]
    ) -> PipelineScores:
        """
        Stage 5: Compute final scores from all inputs.
        """

        logger.debug(f"[PIPELINE:{data.ticker}] Stage 5: SCORING")

        # ─────────────────────────────────────────────────────────────────────
        # 5.1 Compute Subscores
        # ─────────────────────────────────────────────────────────────────────

        subscores = {}

        # Trend subscore
        subscores['trend'] = self.scoring_engine.compute_trend_score(
            indicators=indicators.indicators
        )

        # Momentum subscore
        subscores['momentum'] = self.scoring_engine.compute_momentum_score(
            indicators=indicators.indicators
        )

        # Volume subscore
        subscores['volume'] = self.scoring_engine.compute_volume_score(
            indicators=indicators.indicators
        )

        # Relative strength subscore
        subscores['relative_strength'] = self.scoring_engine.compute_rs_score(
            indicators=indicators.indicators
        )

        # Fundamental subscore
        subscores['fundamental'] = self.scoring_engine.compute_fundamental_score(
            indicators=indicators.indicators,
            fundamentals=data.fundamentals
        )

        # ─────────────────────────────────────────────────────────────────────
        # 5.2 Compute Probability Estimates
        # ─────────────────────────────────────────────────────────────────────

        probabilities = {}

        for horizon in horizons:

            prob = self.probability_engine.estimate(
                state_vector=features.state_vector,
                horizon=horizon,
                model_output=model_outputs.ensemble_output
            )

            probabilities[horizon] = prob

        # ─────────────────────────────────────────────────────────────────────
        # 5.3 Compute Edge Score
        # ─────────────────────────────────────────────────────────────────────

        subscores['edge'] = self.scoring_engine.compute_edge_score(
            probabilities=probabilities,
            horizon_weights=self.config.scoring.horizon_weights
        )

        # ─────────────────────────────────────────────────────────────────────
        # 5.4 Compute Risk Penalties
        # ─────────────────────────────────────────────────────────────────────

        risk_penalties = {}

        # Volatility penalty
        risk_penalties['volatility'] = self.scoring_engine.compute_volatility_penalty(
            prices=data.prices,
            threshold=self.config.scoring.risk_penalties.volatility.threshold
        )

        # Drawdown penalty
        risk_penalties['drawdown'] = self.scoring_engine.compute_drawdown_penalty(
            prices=data.prices,
            threshold=self.config.scoring.risk_penalties.drawdown.threshold
        )

        # Liquidity penalty
        risk_penalties['liquidity'] = self.scoring_engine.compute_liquidity_penalty(
            prices=data.prices,
            threshold=self.config.scoring.risk_penalties.liquidity.threshold
        )

        # Gap risk penalty
        risk_penalties['gap'] = self.scoring_engine.compute_gap_penalty(
            prices=data.prices,
            threshold=self.config.scoring.risk_penalties.gap_risk.threshold
        )

        # Total penalty (capped)
        total_penalty = min(
            sum(risk_penalties.values()),
            self.config.scoring.risk_penalties.max_total
        )

        # ─────────────────────────────────────────────────────────────────────
        # 5.5 Compute Final Score
        # ─────────────────────────────────────────────────────────────────────

        # Weighted subscore sum
        weighted_sum = sum(
            subscores[category] * self.config.scoring.weights[category]
            for category in subscores.keys()
        )

        # Apply penalty
        raw_score = weighted_sum - total_penalty

        # Clamp to 0-10
        final_score = max(0.0, min(10.0, raw_score))

        # Get label
        score_label = self._get_score_label(final_score)

        # ─────────────────────────────────────────────────────────────────────
        # 5.6 Identify Risk Flags
        # ─────────────────────────────────────────────────────────────────────

        risk_flags = self.scoring_engine.identify_risk_flags(
            indicators=indicators.indicators,
            risk_penalties=risk_penalties,
            prices=data.prices
        )

        return PipelineScores(
            subscores=subscores,
            probabilities=probabilities,
            risk_penalties=risk_penalties,
            total_penalty=total_penalty,
            final_score=final_score,
            score_label=score_label,
            risk_flags=risk_flags,
            metadata={
                'weighted_sum': weighted_sum,
                'weights_used': self.config.scoring.weights
            }
        )

    # =========================================================================
    # STAGE 6: OUTPUT
    # =========================================================================

    def stage_output(
        self,
        data: PipelineData,
        indicators: PipelineIndicators,
        features: PipelineFeatures,
        model_outputs: PipelineModelOutputs,
        scores: PipelineScores,
        output_format: str = 'full'
    ) -> TickerAnalysis:
        """
        Stage 6: Assemble and format final output.
        """

        logger.debug(f"[PIPELINE:{data.ticker}] Stage 6: OUTPUT")

        # ─────────────────────────────────────────────────────────────────────
        # 6.1 Assemble Analysis Object
        # ─────────────────────────────────────────────────────────────────────

        analysis = TickerAnalysis(
            ticker=data.ticker,
            as_of_date=data.as_of_date,

            # Core scores
            score=scores.final_score,
            score_label=scores.score_label,
            subscores=scores.subscores,

            # Probabilities
            probabilities=scores.probabilities,

            # Risk
            risk_penalties=scores.risk_penalties,
            risk_flags=scores.risk_flags,

            # Detail level depends on output format
            indicators=indicators.indicators if output_format in ['full', 'detailed'] else None,
            features=features.normalized_features if output_format == 'full' else None,
            model_outputs=model_outputs.individual_outputs if output_format == 'full' else None,

            # Metadata
            metadata=self._build_metadata(data, indicators, features, model_outputs, scores)
        )

        # ─────────────────────────────────────────────────────────────────────
        # 6.2 Validate Output
        # ─────────────────────────────────────────────────────────────────────

        if not self._validate_analysis(analysis):
            raise OutputError(f"Invalid analysis output for {data.ticker}")

        return analysis

    # =========================================================================
    # FULL PIPELINE EXECUTION
    # =========================================================================

    def run(
        self,
        ticker: str,
        as_of_date: datetime,
        horizons: List[str] = ['5d', '21d', '63d'],
        output_format: str = 'full'
    ) -> TickerAnalysis:
        """
        Run complete pipeline from data to output.
        """

        logger.info(f"[PIPELINE] Running full pipeline for {ticker}")

        try:
            # Stage 1: Data
            data = self.stage_data(ticker, as_of_date)

            # Stage 2: Indicators
            indicators = self.stage_indicators(data)

            # Stage 3: Features
            features = self.stage_features(data, indicators)

            # Stage 4: Models
            model_outputs = self.stage_models(features)

            # Stage 5: Scoring
            scores = self.stage_scoring(
                data, indicators, features, model_outputs, horizons
            )

            # Stage 6: Output
            analysis = self.stage_output(
                data, indicators, features, model_outputs, scores, output_format
            )

            logger.info(
                f"[PIPELINE] Complete: {ticker} = {analysis.score:.1f} "
                f"({analysis.score_label})"
            )

            return analysis

        except PipelineError as e:
            logger.error(f"[PIPELINE] Failed for {ticker}: {e}")
            raise
```

### 4.2 Pipeline Stage Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA → FEATURES → MODELS → SCORE → OUTPUT                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: DATA                                                       │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ Input:  ticker, as_of_date                                          │   │
│  │                                                                     │   │
│  │ Process:                                                            │   │
│  │   1. Load price data (OHLCV) ─────────────────┐                     │   │
│  │   2. Validate data quality                     │                    │   │
│  │   3. Adjust for corporate actions              ├──► PipelineData    │   │
│  │   4. Load fundamentals (optional)              │                    │   │
│  │   5. Load earnings (optional)                  │                    │   │
│  │   6. Load benchmark (optional) ───────────────┘                     │   │
│  │                                                                     │   │
│  │ Output: PipelineData { prices, fundamentals, earnings, benchmark }  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: INDICATORS                                                 │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ Input:  PipelineData                                                │   │
│  │                                                                     │   │
│  │ Process:                                                            │   │
│  │   For each timeframe (1d, 1w, 1mo):                                 │   │
│  │     1. Resample prices ────────────────────────┐                    │   │
│  │     2. Compute trend indicators                │                    │   │
│  │     3. Compute momentum indicators             │                    │   │
│  │     4. Compute volatility indicators           ├──► PipelineIndicators│ │
│  │     5. Compute volume indicators               │                    │   │
│  │     6. Compute microstructure indicators       │                    │   │
│  │     7. Compute regime indicators               │                    │   │
│  │     8. Compute relative strength indicators    │                    │   │
│  │     9. Compute structure indicators            │                    │   │
│  │    10. Compute fundamental indicators ─────────┘                    │   │
│  │                                                                     │   │
│  │ Output: PipelineIndicators { indicators[timeframe][group] }         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: FEATURES                                                   │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ Input:  PipelineData, PipelineIndicators                            │   │
│  │                                                                     │   │
│  │ Process:                                                            │   │
│  │   1. Extract raw features from indicators ─────┐                    │   │
│  │   2. Compute derived features                  │                    │   │
│  │   3. Handle missing values                     ├──► PipelineFeatures│   │
│  │   4. Normalize features                        │                    │   │
│  │   5. Build state vector                        │                    │   │
│  │   6. Convert to model input format ────────────┘                    │   │
│  │                                                                     │   │
│  │ Output: PipelineFeatures { raw, normalized, state_vector, input }   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: MODELS                                                     │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ Input:  PipelineFeatures                                            │   │
│  │                                                                     │   │
│  │ Process:                                                            │   │
│  │   For each model in ensemble:                                       │   │
│  │     1. Validate model ready ───────────────────┐                    │   │
│  │     2. Run prediction                          │                    │   │
│  │     3. Validate output                         ├──► PipelineModelOutputs│
│  │   4. Aggregate outputs (weighted average) ─────┘                    │   │
│  │                                                                     │   │
│  │ Output: PipelineModelOutputs { individual, ensemble }               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 5: SCORING                                                    │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ Input:  PipelineData, PipelineIndicators, PipelineFeatures,         │   │
│  │         PipelineModelOutputs                                        │   │
│  │                                                                     │   │
│  │ Process:                                                            │   │
│  │   1. Compute subscores ────────────────────────┐                    │   │
│  │      • Trend (18%)                             │                    │   │
│  │      • Momentum (18%)                          │                    │   │
│  │      • Volume (12%)                            │                    │   │
│  │      • Relative Strength (12%)                 │                    │   │
│  │      • Fundamental (8%)                        │                    │   │
│  │   2. Estimate probabilities per horizon        │                    │   │
│  │   3. Compute edge score (32%)                  ├──► PipelineScores  │   │
│  │   4. Compute risk penalties                    │                    │   │
│  │      • Volatility                              │                    │   │
│  │      • Drawdown                                │                    │   │
│  │      • Liquidity                               │                    │   │
│  │      • Gap risk                                │                    │   │
│  │   5. Compute final score (weighted - penalty)  │                    │   │
│  │   6. Identify risk flags ──────────────────────┘                    │   │
│  │                                                                     │   │
│  │ Output: PipelineScores { subscores, probabilities, final_score }    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 6: OUTPUT                                                     │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ Input:  All pipeline stages                                         │   │
│  │                                                                     │   │
│  │ Process:                                                            │   │
│  │   1. Assemble TickerAnalysis object ───────────┐                    │   │
│  │   2. Include detail level per format           ├──► TickerAnalysis  │   │
│  │   3. Build metadata                            │                    │   │
│  │   4. Validate output ──────────────────────────┘                    │   │
│  │                                                                     │   │
│  │ Output: TickerAnalysis { score, subscores, probabilities, ... }     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Failure Handling

### 5.1 Error Hierarchy

```python
class StockAnalysisError(Exception):
    """Base exception for all stock analysis errors."""

    def __init__(
        self,
        message: str,
        code: str = None,
        recoverable: bool = False,
        details: dict = None
    ):
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable
        self.details = details or {}


class DataError(StockAnalysisError):
    """Data loading or validation error."""
    pass


class DataNotFoundError(DataError):
    """Requested data does not exist."""
    pass


class DataValidationError(DataError):
    """Data failed validation checks."""
    pass


class IndicatorError(StockAnalysisError):
    """Indicator computation error."""
    pass


class FeatureError(StockAnalysisError):
    """Feature engineering error."""
    pass


class ModelError(StockAnalysisError):
    """Model loading or inference error."""
    pass


class ModelNotFoundError(ModelError):
    """Model file not found."""
    pass


class ScoringError(StockAnalysisError):
    """Score computation error."""
    pass


class PipelineError(StockAnalysisError):
    """Pipeline execution error."""

    def __init__(
        self,
        message: str,
        stage: str,
        ticker: str = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.stage = stage
        self.ticker = ticker


class ConfigError(StockAnalysisError):
    """Configuration error."""
    pass


class ValidationError(StockAnalysisError):
    """Input validation error."""
    pass
```

### 5.2 Error Handling Patterns

```python
# =============================================================================
# PATTERN 1: Fail-Fast for Critical Errors
# =============================================================================

def load_critical_data(ticker: str, as_of_date: datetime) -> PriceData:
    """
    Load data that is absolutely required.
    Fails immediately if data unavailable.
    """

    try:
        data = data_provider.get_prices(ticker, as_of_date)

        if data is None or len(data) == 0:
            raise DataNotFoundError(
                f"No price data for {ticker}",
                code='DATA_NOT_FOUND',
                recoverable=False
            )

        return data

    except ConnectionError as e:
        raise DataError(
            f"Failed to connect to data source: {e}",
            code='CONNECTION_FAILED',
            recoverable=True  # Can retry
        )


# =============================================================================
# PATTERN 2: Graceful Degradation for Optional Data
# =============================================================================

def load_optional_data(
    ticker: str,
    as_of_date: datetime,
    data_type: str
) -> Optional[Any]:
    """
    Load optional data with graceful fallback.
    Returns None if unavailable, does not raise.
    """

    try:
        if data_type == 'fundamentals':
            return data_provider.get_fundamentals(ticker, as_of_date)
        elif data_type == 'earnings':
            return data_provider.get_earnings(ticker, as_of_date)
        elif data_type == 'benchmark':
            return data_provider.get_benchmark(as_of_date)
        else:
            logger.warning(f"Unknown data type: {data_type}")
            return None

    except DataNotFoundError:
        logger.debug(f"Optional data not found: {data_type} for {ticker}")
        return None

    except DataError as e:
        logger.warning(f"Failed to load {data_type} for {ticker}: {e}")
        return None


# =============================================================================
# PATTERN 3: Retry with Backoff
# =============================================================================

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_errors: tuple = (ConnectionError, TimeoutError)
) -> Any:
    """
    Retry function with exponential backoff.
    """

    delay = initial_delay
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return func()

        except retryable_errors as e:
            last_error = e

            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_retries + 1} attempts failed")

        except Exception as e:
            # Non-retryable error
            raise

    raise last_error


# Usage:
def fetch_with_retry(ticker: str) -> PriceData:
    return retry_with_backoff(
        func=lambda: data_provider.get_prices(ticker),
        max_retries=3,
        retryable_errors=(ConnectionError, TimeoutError, RateLimitError)
    )


# =============================================================================
# PATTERN 4: Partial Success Handling
# =============================================================================

@dataclass
class BatchResult:
    """Result of batch operation with partial success."""
    successful: List[Any]
    failed: List[Tuple[Any, Exception]]

    @property
    def success_rate(self) -> float:
        total = len(self.successful) + len(self.failed)
        return len(self.successful) / total if total > 0 else 0


def process_batch_with_partial_success(
    items: List[Any],
    processor: Callable,
    min_success_rate: float = 0.5,
    fail_fast_threshold: int = None
) -> BatchResult:
    """
    Process batch allowing partial failures.

    Args:
        items: Items to process
        processor: Function to apply to each item
        min_success_rate: Minimum acceptable success rate
        fail_fast_threshold: Stop if this many consecutive failures
    """

    successful = []
    failed = []
    consecutive_failures = 0

    for item in items:
        try:
            result = processor(item)
            successful.append(result)
            consecutive_failures = 0

        except Exception as e:
            failed.append((item, e))
            consecutive_failures += 1

            # Check fail-fast threshold
            if fail_fast_threshold and consecutive_failures >= fail_fast_threshold:
                logger.error(
                    f"Fail-fast triggered after {consecutive_failures} "
                    f"consecutive failures"
                )
                break

    result = BatchResult(successful=successful, failed=failed)

    # Check minimum success rate
    if result.success_rate < min_success_rate:
        raise PipelineError(
            f"Batch success rate {result.success_rate:.1%} below "
            f"minimum {min_success_rate:.1%}",
            stage='batch_processing'
        )

    return result


# =============================================================================
# PATTERN 5: Fallback Chain
# =============================================================================

def get_data_with_fallbacks(
    ticker: str,
    as_of_date: datetime,
    fallback_chain: List[Callable]
) -> Any:
    """
    Try multiple data sources in order until one succeeds.
    """

    errors = []

    for i, fallback in enumerate(fallback_chain):
        try:
            data = fallback(ticker, as_of_date)

            if data is not None:
                if i > 0:
                    logger.info(f"Using fallback #{i} for {ticker}")
                return data

        except Exception as e:
            errors.append((i, str(e)))
            logger.debug(f"Fallback #{i} failed: {e}")

    # All fallbacks failed
    raise DataError(
        f"All {len(fallback_chain)} data sources failed for {ticker}",
        details={'errors': errors}
    )


# Usage:
def get_prices(ticker: str, as_of_date: datetime) -> PriceData:
    return get_data_with_fallbacks(
        ticker=ticker,
        as_of_date=as_of_date,
        fallback_chain=[
            lambda t, d: cache.get_prices(t, d),           # 1. Local cache
            lambda t, d: database.get_prices(t, d),        # 2. Database
            lambda t, d: api_primary.get_prices(t, d),     # 3. Primary API
            lambda t, d: api_secondary.get_prices(t, d),   # 4. Secondary API
        ]
    )


# =============================================================================
# PATTERN 6: Circuit Breaker
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = 'CLOSED'
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""

        # Check if circuit is open
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                self.half_open_calls = 0
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker is open, retry after "
                    f"{self._time_until_reset():.0f}s"
                )

        # Check half-open limit
        if self.state == 'HALF_OPEN':
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpen("Half-open call limit reached")
            self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            logger.info("Circuit breaker closed after successful recovery")
        self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _time_until_reset(self) -> float:
        """Time remaining until reset attempt."""
        if self.last_failure_time is None:
            return 0
        elapsed = time.time() - self.last_failure_time
        return max(0, self.recovery_timeout - elapsed)


# =============================================================================
# PATTERN 7: Comprehensive Error Context
# =============================================================================

@dataclass
class AnalysisError:
    """Detailed error information for analysis failures."""

    ticker: str
    phase: str  # data, indicators, features, models, scoring
    message: str
    exception: Optional[Exception] = None
    recoverable: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'phase': self.phase,
            'message': self.message,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp.isoformat(),
            'exception_type': type(self.exception).__name__ if self.exception else None,
            'context': self.context
        }

    def __str__(self):
        return f"[{self.ticker}:{self.phase}] {self.message}"


def wrap_analysis_error(
    ticker: str,
    phase: str,
    context: dict = None
) -> Callable:
    """
    Decorator to wrap exceptions in AnalysisError.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AnalysisError:
                raise  # Already wrapped
            except Exception as e:
                raise AnalysisError(
                    ticker=ticker,
                    phase=phase,
                    message=str(e),
                    exception=e,
                    recoverable=isinstance(e, (TimeoutError, ConnectionError)),
                    context=context or {}
                )
        return wrapper
    return decorator
```

### 5.3 Pipeline Error Recovery

```python
class PipelineErrorHandler:
    """
    Centralized error handling for pipeline execution.
    """

    def __init__(self, config: Config):
        self.config = config
        self.error_log: List[AnalysisError] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def handle_stage_error(
        self,
        error: Exception,
        stage: str,
        ticker: str,
        context: dict = None
    ) -> Union[None, Any]:
        """
        Handle error from a pipeline stage.

        Returns:
            None to skip ticker, or fallback value if available
        """

        # Wrap in AnalysisError if needed
        if not isinstance(error, AnalysisError):
            error = AnalysisError(
                ticker=ticker,
                phase=stage,
                message=str(error),
                exception=error,
                context=context or {}
            )

        # Log error
        self.error_log.append(error)
        logger.error(f"Pipeline error: {error}")

        # Determine recovery action
        recovery_action = self._determine_recovery(error, stage)

        if recovery_action == 'skip':
            return None

        elif recovery_action == 'fallback':
            return self._get_fallback(stage, ticker)

        elif recovery_action == 'retry':
            # Return special marker for retry
            return RetryMarker(error)

        elif recovery_action == 'abort':
            raise PipelineError(
                f"Aborting pipeline due to critical error in {stage}",
                stage=stage,
                ticker=ticker
            )

    def _determine_recovery(
        self,
        error: AnalysisError,
        stage: str
    ) -> str:
        """Determine recovery action based on error and stage."""

        # Critical stages - abort on failure
        if stage in ['data'] and not error.recoverable:
            return 'abort' if self.config.strict_mode else 'skip'

        # Stages with fallbacks
        if stage in ['indicators', 'features']:
            if error.recoverable:
                return 'retry'
            return 'fallback'

        # Model stage - can continue with partial models
        if stage == 'models':
            return 'fallback'

        # Scoring - critical
        if stage == 'scoring':
            return 'skip'

        return 'skip'

    def _get_fallback(self, stage: str, ticker: str) -> Any:
        """Get fallback value for stage."""

        if stage == 'indicators':
            # Return empty indicators
            return PipelineIndicators(indicators={}, errors=[])

        elif stage == 'features':
            # Return minimal features
            return PipelineFeatures(
                raw_features={},
                normalized_features={},
                state_vector=np.zeros(58),
                model_input=np.zeros(100)
            )

        elif stage == 'models':
            # Return neutral prediction
            return PipelineModelOutputs(
                individual_outputs={},
                ensemble_output={'prediction': 0.5, 'confidence': 'low'}
            )

        return None

    def get_error_summary(self) -> dict:
        """Get summary of all errors."""

        by_stage = defaultdict(list)
        by_ticker = defaultdict(list)

        for error in self.error_log:
            by_stage[error.phase].append(error)
            by_ticker[error.ticker].append(error)

        return {
            'total_errors': len(self.error_log),
            'by_stage': {k: len(v) for k, v in by_stage.items()},
            'by_ticker': {k: len(v) for k, v in by_ticker.items()},
            'recoverable_count': sum(1 for e in self.error_log if e.recoverable),
            'recent_errors': [e.to_dict() for e in self.error_log[-10:]]
        }
```

### 5.4 Error Handling Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ERROR HANDLING FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────┐                                     │
│                         │   ERROR     │                                     │
│                         │  OCCURS     │                                     │
│                         └──────┬──────┘                                     │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │  Classify Error Type  │                                │
│                    └───────────┬───────────┘                                │
│                                │                                            │
│            ┌───────────────────┼───────────────────┐                        │
│            │                   │                   │                        │
│            ▼                   ▼                   ▼                        │
│    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐               │
│    │   CRITICAL    │   │  RECOVERABLE  │   │    WARNING    │               │
│    │ (Data Missing │   │ (Timeout,     │   │ (Optional     │               │
│    │  Model Fail)  │   │  Rate Limit)  │   │  Data Missing)│               │
│    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘               │
│            │                   │                   │                        │
│            ▼                   ▼                   ▼                        │
│    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐               │
│    │ Check Stage   │   │ Check Retry   │   │    Log &      │               │
│    │ Criticality   │   │    Count      │   │   Continue    │               │
│    └───────┬───────┘   └───────┬───────┘   └───────────────┘               │
│            │                   │                                            │
│     ┌──────┴──────┐     ┌──────┴──────┐                                     │
│     │             │     │             │                                     │
│     ▼             ▼     ▼             ▼                                     │
│ ┌───────┐    ┌───────┐ ┌───────┐ ┌───────┐                                 │
│ │ ABORT │    │ SKIP  │ │ RETRY │ │ FAIL  │                                 │
│ │(strict│    │ticker │ │ with  │ │ after │                                 │
│ │ mode) │    │       │ │backoff│ │ max   │                                 │
│ └───┬───┘    └───┬───┘ └───┬───┘ └───┬───┘                                 │
│     │            │         │         │                                      │
│     ▼            │         │         │                                      │
│ ┌───────────┐    │         │         │                                      │
│ │  RAISE    │    │         ▼         │                                      │
│ │ Pipeline  │    │    ┌─────────┐    │                                      │
│ │   Error   │    │    │  Wait   │    │                                      │
│ └───────────┘    │    │  delay  │    │                                      │
│                  │    └────┬────┘    │                                      │
│                  │         │         │                                      │
│                  │         ▼         │                                      │
│                  │    ┌─────────┐    │                                      │
│                  │    │  Retry  │    │                                      │
│                  │    │ Request │    │                                      │
│                  │    └────┬────┘    │                                      │
│                  │         │         │                                      │
│                  │    ┌────┴────┐    │                                      │
│                  │    │         │    │                                      │
│                  │    ▼         ▼    ▼                                      │
│                  │ Success    Fail──►│                                      │
│                  │    │              │                                      │
│                  ▼    ▼              ▼                                      │
│            ┌──────────────────────────────┐                                 │
│            │   Log Error & Continue to    │                                 │
│            │       Next Ticker            │                                 │
│            └──────────────────────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PIPELINE PSEUDOCODE SUMMARY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  UNIVERSE SCAN FLOW                                                         │
│  ──────────────────                                                          │
│  1. Initialize → 2. Build Universe → 3. Apply Filters →                     │
│  4. Prefetch Data → 5. Parallel Analysis → 6. Filter & Rank → 7. Output    │
│                                                                             │
│  SINGLE TICKER FLOW                                                         │
│  ──────────────────                                                          │
│  1. Validate → 2. Load Data → 3. Compute Indicators →                       │
│  4. Build Features → 5. Run Models → 6. Estimate Probability →              │
│  7. Compute Scores → 8. Compare Benchmark → 9. Assemble Output             │
│                                                                             │
│  DATA → OUTPUT PIPELINE                                                     │
│  ──────────────────────                                                      │
│  Stage 1: DATA        Load prices, fundamentals, earnings, benchmark        │
│  Stage 2: INDICATORS  Compute 55 indicators across 3 timeframes             │
│  Stage 3: FEATURES    Engineer ~58 features, normalize, build state vector  │
│  Stage 4: MODELS      Run ensemble, aggregate predictions                   │
│  Stage 5: SCORING     Compute subscores, probabilities, penalties, final    │
│  Stage 6: OUTPUT      Assemble TickerAnalysis, validate                     │
│                                                                             │
│  FAILURE HANDLING                                                           │
│  ─────────────────                                                           │
│  Patterns:                                                                  │
│  • Fail-Fast for critical errors (missing price data)                       │
│  • Graceful Degradation for optional data (fundamentals)                    │
│  • Retry with Backoff for transient errors (network)                        │
│  • Partial Success for batch operations                                     │
│  • Fallback Chain for data sources                                          │
│  • Circuit Breaker for external services                                    │
│  • Comprehensive Error Context for debugging                                │
│                                                                             │
│  Error Categories:                                                          │
│  • CRITICAL → Abort/Skip                                                    │
│  • RECOVERABLE → Retry with backoff                                         │
│  • WARNING → Log and continue                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
