# Stock Analysis System - Architecture Document

## 1. System Overview

A modular, offline-first stock analysis platform designed for quantitative research and screening. The system supports both single-ticker deep analysis and full-universe scanning capabilities.

**Primary Use Case:** Research-grade analysis (not trading execution)

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STOCK ANALYSIS SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   CLI/API    │    │  Web UI      │    │  Notebook    │                  │
│  │  Interface   │    │  (Future)    │    │  Integration │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                          │
│         └───────────────────┼───────────────────┘                          │
│                             ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ORCHESTRATION LAYER                            │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │   │
│  │  │ Analysis Runner │  │ Universe Scanner│  │ Report Generator│     │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                              │
│         ┌───────────────────┼───────────────────┐                          │
│         ▼                   ▼                   ▼                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│  │   DATA      │    │  ANALYSIS   │    │  SCORING    │                    │
│  │   LAYER     │    │  ENGINE     │    │  ENGINE     │                    │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                    │
│         │                  │                  │                           │
│         ▼                  ▼                  ▼                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│  │ Data Store  │    │ Indicator   │    │  Signal     │                    │
│  │ (Local DB)  │    │ Registry    │    │  Aggregator │                    │
│  └─────────────┘    └─────────────┘    └─────────────┘                    │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │ Yahoo Finance│   │   CSV/JSON  │    │  Future APIs │                    │
│  │   (yfinance) │   │   Files     │    │  (Pluggable) │                    │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Modules & Responsibilities

### 3.1 Data Layer (`data/`)

| Component | Responsibility |
|-----------|----------------|
| `data_fetcher.py` | Abstract interface for fetching OHLCV + fundamental data |
| `yahoo_provider.py` | Yahoo Finance implementation via yfinance |
| `csv_provider.py` | Local CSV file data provider |
| `data_cache.py` | Local caching layer (SQLite/Parquet) for offline operation |
| `universe_manager.py` | Manages ticker universes (S&P500, custom lists, etc.) |

**Key Design:** Provider pattern allows swapping data sources without changing analysis code.

---

### 3.2 Analysis Engine (`analysis/`)

| Component | Responsibility |
|-----------|----------------|
| `indicator_base.py` | Abstract base class for all indicators |
| `indicator_registry.py` | Dynamic registration and discovery of indicators |
| `technical/` | Technical indicator implementations |
| `fundamental/` | Fundamental analysis indicators |
| `composite/` | Multi-factor composite indicators |
| `analyzer.py` | Orchestrates indicator computation for a ticker |

**Key Design:** Indicators are self-contained, declarative, and registry-based for extensibility.

---

### 3.3 Scoring Engine (`scoring/`)

| Component | Responsibility |
|-----------|----------------|
| `signal.py` | Standardized signal output (bullish/bearish/neutral + strength) |
| `scorer.py` | Combines multiple indicator signals into composite scores |
| `weights_config.py` | Configurable weighting schemes |
| `ranking.py` | Cross-ticker ranking and percentile computation |

**Key Design:** Separation of indicator computation from signal interpretation.

---

### 3.4 Orchestration Layer (`core/`)

| Component | Responsibility |
|-----------|----------------|
| `analysis_runner.py` | Single-ticker deep analysis workflow |
| `universe_scanner.py` | Batch processing for full-universe scans |
| `job_queue.py` | Parallel execution management |
| `config.py` | System-wide configuration management |

---

### 3.5 Output Layer (`output/`)

| Component | Responsibility |
|-----------|----------------|
| `report_generator.py` | Structured report creation |
| `formatters/` | JSON, CSV, HTML, Markdown output formats |
| `visualizations.py` | Chart generation (matplotlib/plotly) |
| `export.py` | Export to external formats |

---

### 3.6 Interface Layer (`interfaces/`)

| Component | Responsibility |
|-----------|----------------|
| `cli.py` | Command-line interface |
| `api.py` | Programmatic Python API |
| `notebook_helpers.py` | Jupyter notebook utilities |

---

## 4. Data Flow

### 4.1 Single-Ticker Analysis Flow

```
User Request (ticker: "AAPL", period: "1Y")
         │
         ▼
┌─────────────────┐
│ Analysis Runner │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Data Cache    │◄────│  Data Fetcher   │◄──── External API
│   (check first) │     │  (if cache miss)│
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Indicator       │
│ Registry        │──► Get enabled indicators
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyzer        │──► Compute each indicator
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scoring Engine  │──► Generate signals + composite score
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Report Generator│──► Format output
└────────┬────────┘
         │
         ▼
    Analysis Report
```

### 4.2 Full-Universe Scan Flow

```
User Request (universe: "SP500", filters: {...})
         │
         ▼
┌─────────────────┐
│ Universe Scanner│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Universe Manager│──► Load ticker list
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Job Queue       │──► Parallel ticker processing
└────────┬────────┘
         │
         ├──► Ticker 1 ──► Analysis Runner ──► Results
         ├──► Ticker 2 ──► Analysis Runner ──► Results
         ├──► ...
         └──► Ticker N ──► Analysis Runner ──► Results
                                                  │
                                                  ▼
                                    ┌─────────────────┐
                                    │ Ranking Engine  │──► Cross-ticker ranking
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Filter + Sort   │
                                    └────────┬────────┘
                                             │
                                             ▼
                                      Screener Results
```

---

## 5. Execution Modes

### 5.1 Single-Ticker Analysis Mode

**Purpose:** Deep-dive analysis on one stock

**Characteristics:**
- All indicators computed
- Full historical context
- Detailed report with charts
- Signal explanations

**Example Usage:**
```python
from stock_analysis import analyze

result = analyze("AAPL", period="2Y", indicators="all")
result.to_report("html")
```

---

### 5.2 Full-Universe Scan Mode

**Purpose:** Screen entire market for opportunities

**Characteristics:**
- Batch processing with parallelization
- Configurable indicator subset (for speed)
- Ranking and filtering
- Summary output

**Example Usage:**
```python
from stock_analysis import scan

results = scan(
    universe="sp500",
    indicators=["rsi", "macd", "earnings_surprise"],
    filters={"min_volume": 1_000_000},
    top_n=20
)
```

---

## 6. Offline-First Design

### 6.1 Principles

1. **Cache Everything:** All fetched data stored locally
2. **Graceful Degradation:** System works without network
3. **Explicit Refresh:** User controls when to fetch fresh data
4. **Stale Data Awareness:** Reports show data freshness

### 6.2 Cache Strategy

```
┌─────────────────────────────────────────┐
│              CACHE LAYERS               │
├─────────────────────────────────────────┤
│                                         │
│  Layer 1: In-Memory (Session)           │
│  └── Hot data for current analysis      │
│                                         │
│  Layer 2: Local Database (SQLite)       │
│  └── OHLCV data, fundamentals           │
│  └── Configurable retention             │
│                                         │
│  Layer 3: Parquet Files                 │
│  └── Large historical datasets          │
│  └── Efficient columnar storage         │
│                                         │
└─────────────────────────────────────────┘
```

### 6.3 Data Freshness

| Data Type | Default TTL | Configurable |
|-----------|-------------|--------------|
| Intraday OHLCV | 15 minutes | Yes |
| Daily OHLCV | 24 hours | Yes |
| Fundamentals | 7 days | Yes |
| Universe Lists | 30 days | Yes |

---

## 7. Directory Structure

```
stock-analysis/
├── src/
│   └── stock_analysis/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── analysis_runner.py
│       │   ├── universe_scanner.py
│       │   └── job_queue.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── data_fetcher.py
│       │   ├── yahoo_provider.py
│       │   ├── csv_provider.py
│       │   ├── data_cache.py
│       │   └── universe_manager.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── indicator_base.py
│       │   ├── indicator_registry.py
│       │   ├── analyzer.py
│       │   ├── technical/
│       │   ├── fundamental/
│       │   └── composite/
│       ├── scoring/
│       │   ├── __init__.py
│       │   ├── signal.py
│       │   ├── scorer.py
│       │   ├── weights_config.py
│       │   └── ranking.py
│       ├── output/
│       │   ├── __init__.py
│       │   ├── report_generator.py
│       │   ├── formatters/
│       │   ├── visualizations.py
│       │   └── export.py
│       └── interfaces/
│           ├── __init__.py
│           ├── cli.py
│           ├── api.py
│           └── notebook_helpers.py
├── tests/
├── data/
│   ├── cache/
│   └── universes/
├── docs/
├── configs/
└── notebooks/
```

---

## 8. Key Design Principles

1. **Modularity:** Each component has single responsibility
2. **Extensibility:** New indicators via registry pattern
3. **Testability:** Pure functions where possible, dependency injection
4. **Performance:** Lazy loading, parallel processing, efficient caching
5. **Reproducibility:** Deterministic results, version-tracked configs
6. **Transparency:** Clear signal explanations, audit trail

---

## 9. Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Data Manipulation | pandas, numpy |
| Technical Analysis | ta-lib (optional), pandas-ta |
| Data Storage | SQLite, Parquet |
| Visualization | matplotlib, plotly |
| CLI | click or typer |
| Testing | pytest |
| Config | YAML + pydantic |

---

## 10. Constraints & Non-Goals

### In Scope
- Historical data analysis
- Technical and fundamental indicators
- Screening and ranking
- Report generation

### Out of Scope (v1)
- Real-time streaming data
- Order execution / trading
- Portfolio management
- Backtesting engine
- Machine learning models

---

## 11. Assumptions

1. **Data Availability:** Yahoo Finance provides sufficient data for analysis
2. **Single Machine:** No distributed computing required for v1
3. **Batch Processing:** Near-real-time is acceptable (not sub-second)
4. **User Expertise:** Users understand financial indicators
5. **Python Environment:** Users can run Python scripts/notebooks

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
