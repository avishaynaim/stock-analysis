# Persistence & Database Schema - Design Document

## 1. Overview

The persistence layer provides local analytical storage optimized for quantitative research workflows: backtesting, re-analysis, and reproducible research. This document defines the database choice, complete schema, and performance considerations.

---

## 2. Database Choice: DuckDB

### 2.1 Decision Matrix

| Criteria | SQLite | DuckDB | Winner |
|----------|--------|--------|--------|
| **Analytical Queries** | Row-oriented, slow aggregations | Column-oriented, fast OLAP | DuckDB |
| **Time-Series Performance** | Requires careful indexing | Native time-series optimizations | DuckDB |
| **Window Functions** | Supported, slower | Highly optimized | DuckDB |
| **Parquet Integration** | Requires extension | Native, zero-copy | DuckDB |
| **Concurrent Reads** | Good | Excellent | DuckDB |
| **Concurrent Writes** | Single writer | Single writer | Tie |
| **Memory Efficiency** | Good | Excellent (streaming) | DuckDB |
| **Python Integration** | sqlite3 built-in | duckdb native + pandas | DuckDB |
| **Maturity** | Very mature | Production-ready | SQLite |
| **File Size** | Smaller | Larger (columnar) | SQLite |

### 2.2 Justification

**Primary Choice: DuckDB** for the following reasons:

1. **Columnar Storage:** Financial analysis queries (aggregations, window functions, time-series) are 10-100x faster on columnar databases

2. **Native Parquet Support:** Zero-copy reads from Parquet files; can query cached OHLCV data directly without import

3. **Analytical SQL:** Rich window functions, ASOF joins for point-in-time queries, native time-series support

4. **Memory Efficiency:** Streaming execution allows processing datasets larger than RAM

5. **Pandas Integration:** Direct DataFrame ↔ DuckDB without serialization overhead

6. **Backtesting Workloads:** Designed for exactly our use case - analytical queries over historical data

### 2.3 Hybrid Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ DuckDB (Primary Analytical Store)                       │   │
│  │ ├── prices, features, indicator_states                  │   │
│  │ ├── runs, recommendations, edge_statistics              │   │
│  │ └── Optimized for: reads, aggregations, time-series     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Parquet Files (Large Time-Series)                       │   │
│  │ ├── Raw OHLCV data (queryable directly by DuckDB)       │   │
│  │ ├── Partitioned by ticker and year                      │   │
│  │ └── Optimized for: storage efficiency, portability      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SQLite (Metadata & Config)                              │   │
│  │ ├── Cache index, universe definitions                   │   │
│  │ ├── User settings, API keys (encrypted)                 │   │
│  │ └── Optimized for: small records, key-value access      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Complete Database Schema

### 3.1 Schema Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATABASE SCHEMA MAP                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MARKET DATA                 ANALYSIS                    EXECUTION          │
│  ───────────                 ────────                    ─────────          │
│  ┌──────────┐               ┌──────────────┐            ┌──────────┐       │
│  │ tickers  │───┐           │ indicator_   │            │  runs    │       │
│  └──────────┘   │           │ definitions  │            └────┬─────┘       │
│                 │           └──────────────┘                 │             │
│  ┌──────────┐   │                  │                         │             │
│  │ prices   │◄──┤                  │                         │             │
│  └──────────┘   │                  ▼                         ▼             │
│                 │           ┌──────────────┐            ┌──────────┐       │
│  ┌──────────┐   │           │ indicator_   │◄───────────│run_results│      │
│  │corporate_│◄──┤           │ states       │            └──────────┘       │
│  │ actions  │   │           └──────────────┘                 │             │
│  └──────────┘   │                  │                         │             │
│                 │                  │                         ▼             │
│  ┌──────────┐   │                  │                   ┌────────────┐      │
│  │benchmarks│◄──┘                  ▼                   │recommen-   │      │
│  └──────────┘               ┌──────────────┐           │ dations    │      │
│                             │  features    │           └────────────┘      │
│  ┌──────────┐               └──────────────┘                 │             │
│  │ earnings │                      │                         │             │
│  └──────────┘                      │                         ▼             │
│                                    │                   ┌────────────┐      │
│  ┌──────────┐                      └──────────────────►│   edge_    │      │
│  │fundament-│                                          │ statistics │      │
│  │   als    │                                          └────────────┘      │
│  └──────────┘                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Table Definitions

---

#### 3.2.1 `tickers` - Security Master

```sql
-- Security master table with metadata
CREATE TABLE tickers (
    ticker_id       INTEGER PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL UNIQUE,

    -- Identifiers
    company_name    VARCHAR(255),
    isin            VARCHAR(12),
    cusip           VARCHAR(9),
    figi            VARCHAR(12),

    -- Classification
    exchange        VARCHAR(20),           -- NYSE, NASDAQ, etc.
    asset_type      VARCHAR(20),           -- STOCK, ETF, ADR
    sector          VARCHAR(100),
    industry        VARCHAR(100),
    sub_industry    VARCHAR(100),

    -- Status
    is_active       BOOLEAN DEFAULT TRUE,
    listing_date    DATE,
    delisting_date  DATE,

    -- Metadata
    currency        VARCHAR(3) DEFAULT 'USD',
    country         VARCHAR(3) DEFAULT 'USA',

    -- Tracking
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_source     VARCHAR(50)
);

-- Indexes
CREATE INDEX idx_tickers_symbol ON tickers(symbol);
CREATE INDEX idx_tickers_sector ON tickers(sector);
CREATE INDEX idx_tickers_exchange ON tickers(exchange);
CREATE INDEX idx_tickers_active ON tickers(is_active) WHERE is_active = TRUE;
```

---

#### 3.2.2 `prices` - OHLCV Time-Series

```sql
-- Core price data table (partitioned by year internally)
CREATE TABLE prices (
    price_id        BIGINT PRIMARY KEY,
    ticker_id       INTEGER NOT NULL REFERENCES tickers(ticker_id),

    -- Time dimensions
    trade_date      DATE NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,  -- '1d', '1h', '5m', etc.

    -- OHLCV data
    open            DECIMAL(18,6) NOT NULL,
    high            DECIMAL(18,6) NOT NULL,
    low             DECIMAL(18,6) NOT NULL,
    close           DECIMAL(18,6) NOT NULL,
    adj_close       DECIMAL(18,6) NOT NULL,
    volume          BIGINT NOT NULL,

    -- Computed fields (for query efficiency)
    vwap            DECIMAL(18,6),         -- Volume-weighted average price
    trade_count     INTEGER,               -- Number of trades (if available)

    -- Data quality
    is_adjusted     BOOLEAN DEFAULT TRUE,
    adjustment_factor DECIMAL(18,10),
    data_source     VARCHAR(50),

    -- Versioning
    schema_version  INTEGER DEFAULT 1,
    fetched_at      TIMESTAMP,

    -- Composite unique constraint
    UNIQUE(ticker_id, trade_date, timeframe)
);

-- Critical indexes for time-series queries
CREATE INDEX idx_prices_ticker_date ON prices(ticker_id, trade_date DESC);
CREATE INDEX idx_prices_date ON prices(trade_date);
CREATE INDEX idx_prices_timeframe ON prices(timeframe, trade_date);
CREATE INDEX idx_prices_ticker_tf_date ON prices(ticker_id, timeframe, trade_date DESC);

-- Partition by year for large datasets (DuckDB syntax)
-- Actual partitioning handled via Parquet files for massive datasets
```

---

#### 3.2.3 `corporate_actions` - Splits & Dividends

```sql
-- Corporate actions affecting price adjustments
CREATE TABLE corporate_actions (
    action_id       INTEGER PRIMARY KEY,
    ticker_id       INTEGER NOT NULL REFERENCES tickers(ticker_id),

    -- Action details
    action_type     VARCHAR(20) NOT NULL,  -- 'SPLIT', 'DIVIDEND', 'SPINOFF'

    -- Dates
    ex_date         DATE NOT NULL,
    record_date     DATE,
    pay_date        DATE,
    declared_date   DATE,

    -- Split details
    split_from      DECIMAL(10,4),         -- e.g., 1 (for 4:1 split)
    split_to        DECIMAL(10,4),         -- e.g., 4 (for 4:1 split)
    split_ratio     DECIMAL(18,10),        -- e.g., 4.0

    -- Dividend details
    dividend_amount DECIMAL(18,6),
    dividend_type   VARCHAR(20),           -- 'REGULAR', 'SPECIAL', 'QUALIFIED'
    currency        VARCHAR(3) DEFAULT 'USD',

    -- Tracking
    data_source     VARCHAR(50),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(ticker_id, action_type, ex_date)
);

CREATE INDEX idx_corp_actions_ticker ON corporate_actions(ticker_id, ex_date DESC);
CREATE INDEX idx_corp_actions_date ON corporate_actions(ex_date);
CREATE INDEX idx_corp_actions_type ON corporate_actions(action_type, ex_date);
```

---

#### 3.2.4 `fundamentals` - Financial Snapshots

```sql
-- Point-in-time fundamental data snapshots
CREATE TABLE fundamentals (
    fundamental_id  INTEGER PRIMARY KEY,
    ticker_id       INTEGER NOT NULL REFERENCES tickers(ticker_id),

    -- Snapshot timing
    snapshot_date   DATE NOT NULL,         -- When this data was captured
    fiscal_period   VARCHAR(10),           -- 'Q1 2024', 'FY 2023'
    period_end_date DATE,                  -- End of fiscal period

    -- Valuation metrics
    market_cap      DECIMAL(20,2),
    enterprise_value DECIMAL(20,2),
    pe_ratio_ttm    DECIMAL(12,4),
    pe_ratio_forward DECIMAL(12,4),
    peg_ratio       DECIMAL(12,4),
    price_to_book   DECIMAL(12,4),
    price_to_sales  DECIMAL(12,4),
    ev_to_ebitda    DECIMAL(12,4),
    ev_to_revenue   DECIMAL(12,4),

    -- Profitability
    gross_margin    DECIMAL(8,4),
    operating_margin DECIMAL(8,4),
    profit_margin   DECIMAL(8,4),
    roe             DECIMAL(8,4),
    roa             DECIMAL(8,4),
    roic            DECIMAL(8,4),

    -- Growth (YoY)
    revenue_growth  DECIMAL(8,4),
    earnings_growth DECIMAL(8,4),
    ebitda_growth   DECIMAL(8,4),

    -- Financial health
    current_ratio   DECIMAL(8,4),
    quick_ratio     DECIMAL(8,4),
    debt_to_equity  DECIMAL(12,4),
    debt_to_assets  DECIMAL(8,4),
    interest_coverage DECIMAL(12,4),

    -- Cash flow
    free_cash_flow  DECIMAL(20,2),
    fcf_yield       DECIMAL(8,4),
    operating_cash_flow DECIMAL(20,2),

    -- Per share
    eps_ttm         DECIMAL(12,4),
    eps_forward     DECIMAL(12,4),
    book_value_per_share DECIMAL(12,4),
    revenue_per_share DECIMAL(12,4),

    -- Shares & float
    shares_outstanding BIGINT,
    float_shares    BIGINT,
    insider_ownership DECIMAL(8,4),
    institutional_ownership DECIMAL(8,4),
    short_interest  BIGINT,
    short_ratio     DECIMAL(8,4),

    -- Dividends
    dividend_yield  DECIMAL(8,4),
    dividend_rate   DECIMAL(12,4),
    payout_ratio    DECIMAL(8,4),
    ex_dividend_date DATE,

    -- Analyst estimates
    target_price_low DECIMAL(12,4),
    target_price_mean DECIMAL(12,4),
    target_price_high DECIMAL(12,4),
    analyst_count   INTEGER,
    recommendation_mean DECIMAL(4,2),      -- 1=Strong Buy, 5=Strong Sell

    -- Data quality
    data_source     VARCHAR(50),
    is_estimated    BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(ticker_id, snapshot_date)
);

CREATE INDEX idx_fundamentals_ticker ON fundamentals(ticker_id, snapshot_date DESC);
CREATE INDEX idx_fundamentals_date ON fundamentals(snapshot_date);
CREATE INDEX idx_fundamentals_pit ON fundamentals(ticker_id, snapshot_date)
    WHERE snapshot_date <= CURRENT_DATE;  -- For point-in-time queries
```

---

#### 3.2.5 `earnings` - Earnings Calendar & Results

```sql
-- Earnings events and results
CREATE TABLE earnings (
    earnings_id     INTEGER PRIMARY KEY,
    ticker_id       INTEGER NOT NULL REFERENCES tickers(ticker_id),

    -- Period identification
    fiscal_year     INTEGER NOT NULL,
    fiscal_quarter  VARCHAR(5) NOT NULL,   -- 'Q1', 'Q2', 'Q3', 'Q4', 'FY'

    -- Timing
    report_date     DATE NOT NULL,
    report_time     VARCHAR(10),           -- 'BMO', 'AMC', 'TNS'

    -- Estimates (captured before announcement)
    eps_estimate    DECIMAL(12,4),
    eps_estimate_high DECIMAL(12,4),
    eps_estimate_low DECIMAL(12,4),
    eps_num_estimates INTEGER,
    revenue_estimate DECIMAL(20,2),
    revenue_estimate_high DECIMAL(20,2),
    revenue_estimate_low DECIMAL(20,2),

    -- Actuals (filled after announcement)
    eps_actual      DECIMAL(12,4),
    revenue_actual  DECIMAL(20,2),

    -- Surprises (computed)
    eps_surprise    DECIMAL(12,4),
    eps_surprise_pct DECIMAL(8,4),
    revenue_surprise DECIMAL(20,2),
    revenue_surprise_pct DECIMAL(8,4),

    -- Guidance (if provided)
    guidance_eps_low DECIMAL(12,4),
    guidance_eps_high DECIMAL(12,4),
    guidance_revenue_low DECIMAL(20,2),
    guidance_revenue_high DECIMAL(20,2),

    -- Market reaction (1-day)
    price_before    DECIMAL(12,4),
    price_after     DECIMAL(12,4),
    price_change_pct DECIMAL(8,4),
    volume_ratio    DECIMAL(8,4),          -- Volume vs 20-day avg

    -- Status
    is_confirmed    BOOLEAN DEFAULT FALSE,
    data_source     VARCHAR(50),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(ticker_id, fiscal_year, fiscal_quarter)
);

CREATE INDEX idx_earnings_ticker ON earnings(ticker_id, report_date DESC);
CREATE INDEX idx_earnings_date ON earnings(report_date);
CREATE INDEX idx_earnings_upcoming ON earnings(report_date)
    WHERE report_date >= CURRENT_DATE;
```

---

#### 3.2.6 `benchmarks` - Index & Benchmark Data

```sql
-- Benchmark/index price data
CREATE TABLE benchmarks (
    benchmark_id    INTEGER PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL,

    -- Identification
    name            VARCHAR(100),
    benchmark_type  VARCHAR(30),           -- 'BROAD_MARKET', 'SECTOR', 'FACTOR'

    -- Time series
    trade_date      DATE NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,

    -- OHLCV
    open            DECIMAL(18,6),
    high            DECIMAL(18,6),
    low             DECIMAL(18,6),
    close           DECIMAL(18,6) NOT NULL,
    adj_close       DECIMAL(18,6) NOT NULL,
    volume          BIGINT,

    -- Tracking
    data_source     VARCHAR(50),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(symbol, trade_date, timeframe)
);

-- Pre-defined benchmarks
-- SPY (S&P 500), QQQ (NASDAQ 100), IWM (Russell 2000)
-- XLK (Tech), XLF (Financials), XLV (Healthcare), etc.

CREATE INDEX idx_benchmarks_symbol_date ON benchmarks(symbol, trade_date DESC);
CREATE INDEX idx_benchmarks_date ON benchmarks(trade_date);
```

---

#### 3.2.7 `indicator_definitions` - Indicator Registry

```sql
-- Metadata about available indicators
CREATE TABLE indicator_definitions (
    indicator_id    INTEGER PRIMARY KEY,

    -- Identification
    name            VARCHAR(100) NOT NULL UNIQUE,
    short_name      VARCHAR(20) NOT NULL UNIQUE,
    category        VARCHAR(50) NOT NULL,  -- 'MOMENTUM', 'TREND', 'VOLATILITY', etc.
    subcategory     VARCHAR(50),

    -- Description
    description     TEXT,
    formula_description TEXT,
    interpretation  TEXT,

    -- Parameters
    default_params  JSON,                  -- {"period": 14, "overbought": 70}
    param_schema    JSON,                  -- JSON Schema for validation

    -- Data requirements
    required_fields VARCHAR(200),          -- 'close' or 'high,low,close,volume'
    min_periods     INTEGER NOT NULL,      -- Minimum bars needed

    -- Output specification
    output_fields   JSON,                  -- [{"name": "rsi", "type": "float"}]
    output_range    JSON,                  -- {"min": 0, "max": 100}

    -- Classification
    is_bounded      BOOLEAN DEFAULT FALSE, -- RSI (0-100) vs MACD (unbounded)
    is_centered     BOOLEAN DEFAULT FALSE, -- Oscillates around zero?
    signal_type     VARCHAR(20),           -- 'CONTINUOUS', 'BINARY', 'CATEGORICAL'

    -- Versioning
    version         VARCHAR(20) DEFAULT '1.0.0',
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_indicator_defs_category ON indicator_definitions(category);
CREATE INDEX idx_indicator_defs_active ON indicator_definitions(is_active);
```

---

#### 3.2.8 `indicator_states` - Computed Indicator Values

```sql
-- Computed indicator values per ticker/date
CREATE TABLE indicator_states (
    state_id        BIGINT PRIMARY KEY,
    ticker_id       INTEGER NOT NULL REFERENCES tickers(ticker_id),
    indicator_id    INTEGER NOT NULL REFERENCES indicator_definitions(indicator_id),

    -- Time reference
    trade_date      DATE NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,

    -- Parameters used (for reproducibility)
    params          JSON NOT NULL,         -- {"period": 14}
    params_hash     VARCHAR(64),           -- SHA256 of params for indexing

    -- Computed values
    value           DECIMAL(18,6),         -- Primary indicator value
    value_json      JSON,                  -- Complex outputs: {"macd": 1.5, "signal": 1.2, "histogram": 0.3}

    -- Signal interpretation
    signal          VARCHAR(20),           -- 'BULLISH', 'BEARISH', 'NEUTRAL'
    signal_strength DECIMAL(5,4),          -- 0.0 to 1.0

    -- Percentile context (across universe)
    percentile_1d   DECIMAL(5,4),          -- Today's percentile
    percentile_20d  DECIMAL(5,4),          -- 20-day rolling percentile
    z_score         DECIMAL(8,4),          -- Standardized value

    -- Computation tracking
    run_id          INTEGER REFERENCES runs(run_id),
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    compute_time_ms INTEGER,

    -- Versioning
    indicator_version VARCHAR(20),
    schema_version  INTEGER DEFAULT 1,

    UNIQUE(ticker_id, indicator_id, trade_date, timeframe, params_hash)
);

-- Critical indexes for analysis queries
CREATE INDEX idx_ind_states_ticker_date ON indicator_states(ticker_id, trade_date DESC);
CREATE INDEX idx_ind_states_indicator ON indicator_states(indicator_id, trade_date DESC);
CREATE INDEX idx_ind_states_ticker_ind ON indicator_states(ticker_id, indicator_id, trade_date DESC);
CREATE INDEX idx_ind_states_signal ON indicator_states(indicator_id, signal, trade_date);
CREATE INDEX idx_ind_states_run ON indicator_states(run_id);

-- Partial index for latest values only
CREATE INDEX idx_ind_states_latest ON indicator_states(ticker_id, indicator_id, trade_date DESC)
    WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days';
```

---

#### 3.2.9 `features` - Derived Feature Store

```sql
-- Pre-computed features for analysis/ML
CREATE TABLE features (
    feature_id      BIGINT PRIMARY KEY,
    ticker_id       INTEGER NOT NULL REFERENCES tickers(ticker_id),

    -- Time reference
    trade_date      DATE NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,

    -- Feature identification
    feature_name    VARCHAR(100) NOT NULL,
    feature_category VARCHAR(50),          -- 'PRICE', 'VOLUME', 'TECHNICAL', 'FUNDAMENTAL'

    -- Feature value
    value           DECIMAL(18,6),
    value_normalized DECIMAL(18,6),        -- Standardized/normalized

    -- Transformations applied
    transformation  VARCHAR(50),           -- 'RAW', 'LOG', 'PCT_CHANGE', 'Z_SCORE'
    lookback_period INTEGER,               -- For rolling features

    -- Computation reference
    source_table    VARCHAR(50),           -- 'prices', 'indicator_states', etc.
    source_formula  TEXT,                  -- SQL or expression used

    -- Tracking
    run_id          INTEGER REFERENCES runs(run_id),
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(ticker_id, trade_date, timeframe, feature_name)
);

CREATE INDEX idx_features_ticker_date ON features(ticker_id, trade_date DESC);
CREATE INDEX idx_features_name ON features(feature_name, trade_date DESC);
CREATE INDEX idx_features_category ON features(feature_category, trade_date);
```

---

#### 3.2.10 `runs` - Execution Tracking

```sql
-- Track analysis/scan executions
CREATE TABLE runs (
    run_id          INTEGER PRIMARY KEY,

    -- Run identification
    run_type        VARCHAR(30) NOT NULL,  -- 'SINGLE_TICKER', 'UNIVERSE_SCAN', 'BACKTEST'
    run_name        VARCHAR(200),

    -- Scope
    universe_id     VARCHAR(50),           -- Universe scanned (if applicable)
    tickers         JSON,                  -- List of tickers (if explicit)
    ticker_count    INTEGER,

    -- Time range
    start_date      DATE,
    end_date        DATE,
    timeframe       VARCHAR(10),

    -- Configuration
    config          JSON NOT NULL,         -- Full run configuration
    config_hash     VARCHAR(64),           -- For deduplication
    indicators_used JSON,                  -- List of indicator names

    -- Execution
    status          VARCHAR(20) NOT NULL,  -- 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED'
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    duration_seconds DECIMAL(10,2),

    -- Results summary
    tickers_processed INTEGER,
    tickers_failed  INTEGER,
    errors          JSON,                  -- List of errors encountered

    -- Environment
    system_info     JSON,                  -- Python version, package versions
    git_commit      VARCHAR(40),           -- Code version

    -- Metadata
    created_by      VARCHAR(100),
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_runs_status ON runs(status, started_at DESC);
CREATE INDEX idx_runs_type ON runs(run_type, started_at DESC);
CREATE INDEX idx_runs_config ON runs(config_hash);
```

---

#### 3.2.11 `run_results` - Per-Ticker Run Output

```sql
-- Detailed results per ticker per run
CREATE TABLE run_results (
    result_id       BIGINT PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES runs(run_id),
    ticker_id       INTEGER NOT NULL REFERENCES tickers(ticker_id),

    -- Timing
    as_of_date      DATE NOT NULL,         -- Analysis date

    -- Composite scores
    overall_score   DECIMAL(5,4),          -- 0.0 to 1.0
    technical_score DECIMAL(5,4),
    fundamental_score DECIMAL(5,4),
    momentum_score  DECIMAL(5,4),

    -- Ranking
    rank_overall    INTEGER,
    rank_sector     INTEGER,
    percentile      DECIMAL(5,4),

    -- Signal summary
    signal          VARCHAR(20),           -- 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    signal_confidence DECIMAL(5,4),

    -- Indicator breakdown
    indicator_signals JSON,                -- {"rsi": {"value": 35, "signal": "BULLISH"}, ...}

    -- Key metrics snapshot
    price           DECIMAL(18,6),
    change_1d       DECIMAL(8,4),
    change_5d       DECIMAL(8,4),
    change_20d      DECIMAL(8,4),
    volume_ratio    DECIMAL(8,4),

    -- Processing
    compute_time_ms INTEGER,
    warnings        JSON,

    UNIQUE(run_id, ticker_id)
);

CREATE INDEX idx_run_results_run ON run_results(run_id, overall_score DESC);
CREATE INDEX idx_run_results_ticker ON run_results(ticker_id, as_of_date DESC);
CREATE INDEX idx_run_results_signal ON run_results(run_id, signal);
CREATE INDEX idx_run_results_rank ON run_results(run_id, rank_overall);
```

---

#### 3.2.12 `recommendations` - Analysis Recommendations

```sql
-- Generated recommendations from analysis
CREATE TABLE recommendations (
    recommendation_id INTEGER PRIMARY KEY,
    run_id          INTEGER REFERENCES runs(run_id),
    ticker_id       INTEGER NOT NULL REFERENCES tickers(ticker_id),

    -- Recommendation
    recommendation  VARCHAR(30) NOT NULL,  -- 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    confidence      DECIMAL(5,4) NOT NULL, -- 0.0 to 1.0

    -- Timing
    generated_at    TIMESTAMP NOT NULL,
    valid_from      DATE NOT NULL,
    valid_until     DATE,                  -- NULL = until next recommendation

    -- Context
    as_of_price     DECIMAL(18,6),
    target_price    DECIMAL(18,6),
    stop_loss_price DECIMAL(18,6),
    upside_pct      DECIMAL(8,4),
    risk_reward_ratio DECIMAL(8,4),

    -- Reasoning
    primary_factors JSON,                  -- Top factors driving recommendation
    supporting_indicators JSON,
    contrary_indicators JSON,
    narrative       TEXT,                  -- Human-readable explanation

    -- Thresholds triggered
    thresholds      JSON,                  -- {"rsi_oversold": true, "macd_crossover": true}

    -- Classification
    thesis_type     VARCHAR(50),           -- 'MOMENTUM', 'VALUE', 'MEAN_REVERSION', 'BREAKOUT'
    time_horizon    VARCHAR(20),           -- 'SHORT', 'MEDIUM', 'LONG'

    -- Status tracking
    status          VARCHAR(20) DEFAULT 'ACTIVE',  -- 'ACTIVE', 'EXPIRED', 'SUPERSEDED'
    superseded_by   INTEGER REFERENCES recommendations(recommendation_id),

    -- Outcome tracking (filled later)
    outcome_date    DATE,
    outcome_price   DECIMAL(18,6),
    outcome_return  DECIMAL(8,4),
    outcome_status  VARCHAR(20),           -- 'HIT_TARGET', 'HIT_STOP', 'EXPIRED', 'PENDING'

    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_recs_ticker ON recommendations(ticker_id, generated_at DESC);
CREATE INDEX idx_recs_run ON recommendations(run_id);
CREATE INDEX idx_recs_active ON recommendations(status, valid_from)
    WHERE status = 'ACTIVE';
CREATE INDEX idx_recs_outcome ON recommendations(outcome_status, generated_at);
```

---

#### 3.2.13 `edge_statistics` - Historical Performance Tracking

```sql
-- Track historical edge of indicators and strategies
CREATE TABLE edge_statistics (
    stat_id         INTEGER PRIMARY KEY,

    -- What we're measuring
    stat_type       VARCHAR(30) NOT NULL,  -- 'INDICATOR', 'SIGNAL', 'STRATEGY', 'RECOMMENDATION'
    entity_name     VARCHAR(100) NOT NULL, -- Indicator/strategy name
    entity_params   JSON,                  -- Parameters if applicable

    -- Scope
    universe        VARCHAR(50),           -- 'SP500', 'ALL', specific sector
    sector          VARCHAR(100),
    timeframe       VARCHAR(10),

    -- Time range of measurement
    measurement_start DATE NOT NULL,
    measurement_end DATE NOT NULL,
    sample_count    INTEGER NOT NULL,

    -- Signal distribution
    total_signals   INTEGER,
    bullish_signals INTEGER,
    bearish_signals INTEGER,
    neutral_signals INTEGER,

    -- Forward returns (after signal)
    return_1d_mean  DECIMAL(8,4),
    return_1d_median DECIMAL(8,4),
    return_1d_std   DECIMAL(8,4),

    return_5d_mean  DECIMAL(8,4),
    return_5d_median DECIMAL(8,4),
    return_5d_std   DECIMAL(8,4),

    return_20d_mean DECIMAL(8,4),
    return_20d_median DECIMAL(8,4),
    return_20d_std  DECIMAL(8,4),

    -- Win rates
    win_rate_1d     DECIMAL(5,4),          -- % of signals with positive return
    win_rate_5d     DECIMAL(5,4),
    win_rate_20d    DECIMAL(5,4),

    -- Risk-adjusted metrics
    sharpe_1d       DECIMAL(8,4),
    sharpe_5d       DECIMAL(8,4),
    sharpe_20d      DECIMAL(8,4),

    -- Information coefficient
    ic_1d           DECIMAL(8,4),          -- Correlation: signal strength vs return
    ic_5d           DECIMAL(8,4),
    ic_20d          DECIMAL(8,4),

    -- Conditional analysis
    edge_in_uptrend DECIMAL(8,4),          -- Return when market up
    edge_in_downtrend DECIMAL(8,4),        -- Return when market down

    -- Statistical significance
    t_stat_1d       DECIMAL(8,4),
    t_stat_5d       DECIMAL(8,4),
    t_stat_20d      DECIMAL(8,4),
    p_value_1d      DECIMAL(8,6),
    p_value_5d      DECIMAL(8,6),
    p_value_20d     DECIMAL(8,6),

    -- Decay analysis
    edge_decay_halflife_days INTEGER,      -- How quickly edge decays

    -- Metadata
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    computation_notes TEXT,

    UNIQUE(stat_type, entity_name, universe, timeframe, measurement_start, measurement_end)
);

CREATE INDEX idx_edge_stats_entity ON edge_statistics(entity_name, measurement_end DESC);
CREATE INDEX idx_edge_stats_type ON edge_statistics(stat_type, measurement_end DESC);
CREATE INDEX idx_edge_stats_universe ON edge_statistics(universe, stat_type);
```

---

## 4. Indexing & Performance Considerations

### 4.1 Index Strategy Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INDEXING STRATEGY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRIMARY ACCESS PATTERNS           INDEXES                                  │
│  ───────────────────────           ───────                                  │
│                                                                             │
│  1. Get price history for ticker   (ticker_id, trade_date DESC)            │
│                                                                             │
│  2. Get all prices for a date      (trade_date)                            │
│                                                                             │
│  3. Get indicator for ticker       (ticker_id, indicator_id, trade_date)   │
│                                                                             │
│  4. Find all bullish signals       (indicator_id, signal, trade_date)      │
│                                                                             │
│  5. Get run results ranked         (run_id, overall_score DESC)            │
│                                                                             │
│  6. Active recommendations         (status, valid_from) WHERE active       │
│                                                                             │
│  7. Recent data only               Partial index: date >= 30 days ago      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Query Optimization Patterns

```sql
-- Pattern 1: Point-in-Time Lookup (No Lookahead Bias)
-- Use ASOF join in DuckDB for efficient PIT queries
SELECT p.*, f.*
FROM prices p
ASOF JOIN fundamentals f
    ON p.ticker_id = f.ticker_id
    AND f.snapshot_date <= p.trade_date
WHERE p.ticker_id = ?
ORDER BY p.trade_date;

-- Pattern 2: Cross-Sectional Analysis (All tickers, one date)
SELECT t.symbol, r.*
FROM run_results r
JOIN tickers t ON r.ticker_id = t.ticker_id
WHERE r.run_id = ?
ORDER BY r.rank_overall;

-- Pattern 3: Time-Series Windowing
SELECT
    ticker_id,
    trade_date,
    close,
    AVG(close) OVER (
        PARTITION BY ticker_id
        ORDER BY trade_date
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) as sma_20
FROM prices
WHERE timeframe = '1d';

-- Pattern 4: Multi-Indicator Pivot
SELECT
    p.ticker_id,
    p.trade_date,
    MAX(CASE WHEN i.name = 'RSI' THEN s.value END) as rsi,
    MAX(CASE WHEN i.name = 'MACD' THEN s.value END) as macd,
    MAX(CASE WHEN i.name = 'SMA_50' THEN s.value END) as sma_50
FROM prices p
LEFT JOIN indicator_states s
    ON p.ticker_id = s.ticker_id
    AND p.trade_date = s.trade_date
LEFT JOIN indicator_definitions i
    ON s.indicator_id = i.indicator_id
GROUP BY p.ticker_id, p.trade_date;
```

### 4.3 Partitioning Strategy

```python
# Partitioning for large tables

PARTITION_STRATEGIES = {
    "prices": {
        "method": "by_year",
        "column": "trade_date",
        "implementation": "parquet_files",  # Separate parquet per year
        "naming": "prices_{ticker}_{year}.parquet"
    },

    "indicator_states": {
        "method": "by_month",
        "column": "trade_date",
        "implementation": "duckdb_partition",
        "retention_months": 24
    },

    "run_results": {
        "method": "by_run",
        "column": "run_id",
        "implementation": "duckdb_native",
        "archive_after_days": 90
    }
}
```

### 4.4 Data Retention & Archival

```yaml
# configs/retention.yaml

retention:
  prices:
    daily:
      hot_storage_years: 5      # Keep in DuckDB
      cold_storage_years: 20    # Archive to Parquet
      delete_after_years: null  # Never delete

    intraday:
      hot_storage_days: 90
      cold_storage_days: 365
      delete_after_days: 730    # 2 years

  indicator_states:
    hot_storage_months: 6
    cold_storage_months: 24
    delete_after_months: 36

  run_results:
    hot_storage_days: 90
    cold_storage_days: 365
    delete_after_days: null     # Keep for audit

  recommendations:
    keep_all: true              # Never delete (audit trail)

  edge_statistics:
    rolling_window_years: 5
    keep_annual_snapshots: true
```

### 4.5 Compression Settings

```python
# DuckDB compression settings for analytical workloads

COMPRESSION_CONFIG = {
    "default": "zstd",           # Best compression ratio

    "per_column": {
        # High cardinality, frequently accessed
        "trade_date": "dictionary",
        "ticker_id": "dictionary",
        "timeframe": "dictionary",

        # Numeric, range queries
        "close": "zstd",
        "volume": "zstd",

        # Low cardinality categorical
        "signal": "dictionary",
        "sector": "dictionary",

        # JSON fields
        "params": "zstd",
        "config": "zstd"
    }
}
```

### 4.6 Memory Management

```python
# DuckDB memory configuration

DUCKDB_CONFIG = {
    # Memory limits
    "memory_limit": "4GB",               # Max memory usage
    "threads": 4,                        # Parallel threads

    # Temp storage for large queries
    "temp_directory": "./data/temp",
    "max_temp_directory_size": "10GB",

    # Query optimization
    "enable_progress_bar": True,
    "enable_object_cache": True,

    # For time-series workloads
    "default_order": "ASC",              # Optimize for time-ordered data
}
```

---

## 5. File System Layout

```
data/
├── duckdb/
│   └── stock_analysis.duckdb     # Main analytical database
│
├── parquet/
│   ├── prices/
│   │   ├── daily/
│   │   │   ├── 2024/
│   │   │   │   ├── AAPL.parquet
│   │   │   │   ├── MSFT.parquet
│   │   │   │   └── ...
│   │   │   └── 2023/
│   │   │       └── ...
│   │   └── intraday/
│   │       └── ...
│   │
│   ├── fundamentals/
│   │   └── snapshots_2024.parquet
│   │
│   └── archive/
│       └── ...
│
├── sqlite/
│   ├── metadata.db               # Cache index, settings
│   └── universes.db              # Universe definitions
│
└── exports/
    └── ...
```

---

## 6. Database Migrations

```python
# Migration tracking table
CREATE TABLE schema_migrations (
    migration_id    INTEGER PRIMARY KEY,
    version         VARCHAR(20) NOT NULL UNIQUE,
    name            VARCHAR(200) NOT NULL,
    applied_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checksum        VARCHAR(64),
    execution_time_ms INTEGER
);

# Migration file naming: YYYYMMDD_HHMMSS_description.sql
# Example: 20240115_120000_add_fundamentals_roic.sql
```

---

## 7. Assumptions

1. **Single-User Access:** No concurrent write contention expected; single analyst workstation
2. **Read-Heavy Workload:** 90%+ reads; writes are batch operations
3. **Dataset Size:** Up to 5000 tickers × 20 years daily = ~36M price rows; fits in memory for most queries
4. **Disk Space:** Estimate ~5GB for full S&P 500 history with all indicators
5. **Query Patterns:** Mostly time-series and cross-sectional; rare ad-hoc exploration
6. **Reproducibility Priority:** Schema versioning and data lineage are critical
7. **DuckDB Stability:** DuckDB is production-ready for embedded analytical workloads
8. **Parquet Compatibility:** Parquet provides portable archival format

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
