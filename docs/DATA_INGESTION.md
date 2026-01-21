# Data Ingestion Layer - Design Document

## 1. Overview

The Data Ingestion Layer is responsible for acquiring, normalizing, caching, and serving all financial data required by the analysis system. It abstracts data sources behind a unified interface and implements robust offline-first caching.

---

## 2. Supported Data Types

### 2.1 Core Data Types

| Data Type | Description | Source Priority | Required |
|-----------|-------------|-----------------|----------|
| OHLCV | Open, High, Low, Close, Volume | Yahoo Finance → CSV | ✅ Yes |
| Adjusted OHLCV | Split/dividend adjusted prices | Yahoo Finance | ✅ Yes |
| Corporate Actions | Splits, dividends, spinoffs | Yahoo Finance | ✅ Yes |
| Fundamental Snapshot | Key ratios, market cap, sector | Yahoo Finance | ✅ Yes |
| Earnings Calendar | EPS estimates, actuals, dates | Yahoo Finance | ⚠️ Optional |
| Benchmark Data | Index prices (SPY, QQQ, etc.) | Yahoo Finance | ⚠️ Optional |

### 2.2 Data Schemas

#### 2.2.1 OHLCV Schema

```python
@dataclass
class OHLCVBar:
    timestamp: datetime          # Bar timestamp (UTC)
    open: float                  # Open price
    high: float                  # High price
    low: float                   # Low price
    close: float                 # Close price (unadjusted)
    adj_close: float             # Adjusted close
    volume: int                  # Trading volume

@dataclass
class OHLCVSeries:
    ticker: str                  # Symbol
    timeframe: Timeframe         # 1m, 5m, 1h, 1d, 1wk, 1mo
    bars: List[OHLCVBar]         # Sorted by timestamp ascending
    source: str                  # Data provider name
    fetched_at: datetime         # When data was retrieved
    adjustment_type: str         # "split_only" | "split_and_dividend"
```

#### 2.2.2 Corporate Actions Schema

```python
@dataclass
class CorporateAction:
    ticker: str
    action_type: ActionType      # SPLIT, DIVIDEND, SPINOFF
    ex_date: date                # Ex-dividend/ex-split date
    record_date: Optional[date]
    pay_date: Optional[date]

@dataclass
class Split(CorporateAction):
    ratio: float                 # e.g., 4.0 for 4:1 split

@dataclass
class Dividend(CorporateAction):
    amount: float                # Per-share dividend
    currency: str                # USD, EUR, etc.
    dividend_type: str           # REGULAR, SPECIAL, QUALIFIED
```

#### 2.2.3 Fundamental Snapshot Schema

```python
@dataclass
class FundamentalSnapshot:
    ticker: str
    snapshot_date: date

    # Identifiers
    company_name: str
    sector: str
    industry: str
    exchange: str

    # Valuation
    market_cap: Optional[float]
    enterprise_value: Optional[float]
    pe_ratio_ttm: Optional[float]
    pe_ratio_forward: Optional[float]
    peg_ratio: Optional[float]
    price_to_book: Optional[float]
    price_to_sales: Optional[float]
    ev_to_ebitda: Optional[float]

    # Profitability
    profit_margin: Optional[float]
    operating_margin: Optional[float]
    roe: Optional[float]
    roa: Optional[float]

    # Growth
    revenue_growth_yoy: Optional[float]
    earnings_growth_yoy: Optional[float]

    # Financial Health
    current_ratio: Optional[float]
    debt_to_equity: Optional[float]

    # Trading
    avg_volume_10d: Optional[int]
    avg_volume_3mo: Optional[int]
    shares_outstanding: Optional[int]
    float_shares: Optional[int]
    short_ratio: Optional[float]

    # Dividends
    dividend_yield: Optional[float]
    payout_ratio: Optional[float]

    # Analyst
    target_price_mean: Optional[float]
    target_price_low: Optional[float]
    target_price_high: Optional[float]
    recommendation: Optional[str]  # BUY, HOLD, SELL
```

#### 2.2.4 Earnings Calendar Schema

```python
@dataclass
class EarningsEvent:
    ticker: str
    fiscal_quarter: str          # "Q1 2024"
    fiscal_year: int
    report_date: date
    report_time: str             # "BMO" (before), "AMC" (after), "TNS" (not specified)

    # Estimates
    eps_estimate: Optional[float]
    revenue_estimate: Optional[float]

    # Actuals (filled after release)
    eps_actual: Optional[float]
    revenue_actual: Optional[float]
    eps_surprise: Optional[float]
    eps_surprise_pct: Optional[float]
```

#### 2.2.5 Benchmark Data Schema

```python
@dataclass
class BenchmarkDefinition:
    symbol: str                  # SPY, QQQ, IWM, etc.
    name: str                    # S&P 500 ETF, etc.
    benchmark_type: str          # INDEX_ETF, SECTOR_ETF, FACTOR_ETF

# Benchmark OHLCV uses same OHLCVSeries schema
```

---

## 3. Multi-Timeframe Handling

### 3.1 Supported Timeframes

```python
class Timeframe(Enum):
    # Intraday (limited history)
    MINUTE_1 = "1m"      # 7 days history max
    MINUTE_5 = "5m"      # 60 days history max
    MINUTE_15 = "15m"    # 60 days history max
    MINUTE_30 = "30m"    # 60 days history max
    HOUR_1 = "1h"        # 730 days history max

    # Daily+ (full history)
    DAY_1 = "1d"         # Full history
    WEEK_1 = "1wk"       # Full history
    MONTH_1 = "1mo"      # Full history
```

### 3.2 Timeframe Configuration

```yaml
# configs/timeframes.yaml

timeframes:
  1m:
    max_history_days: 7
    cache_ttl_minutes: 5
    market_hours_only: true

  5m:
    max_history_days: 60
    cache_ttl_minutes: 15
    market_hours_only: true

  1h:
    max_history_days: 730
    cache_ttl_minutes: 60
    market_hours_only: false

  1d:
    max_history_days: null  # unlimited
    cache_ttl_minutes: 1440  # 24 hours
    market_hours_only: false

  1wk:
    max_history_days: null
    cache_ttl_minutes: 10080  # 7 days
    market_hours_only: false

  1mo:
    max_history_days: null
    cache_ttl_minutes: 43200  # 30 days
    market_hours_only: false
```

### 3.3 Timeframe Alignment

```python
class TimeframeAligner:
    """
    Ensures consistent bar alignment across timeframes.

    Rules:
    - Daily bars: Close at market close (16:00 ET for US)
    - Weekly bars: Close on Friday
    - Monthly bars: Close on last trading day
    - Intraday: Aligned to UTC, converted to market timezone
    """

    def align_timestamp(self, ts: datetime, timeframe: Timeframe) -> datetime:
        ...

    def get_expected_bar_count(
        self,
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
        market: str = "US"
    ) -> int:
        ...
```

### 3.4 Timeframe Resampling

```python
class TimeframeResampler:
    """
    Resample lower timeframe data to higher timeframes.

    Example: 1m → 5m → 15m → 1h → 1d

    Rules:
    - Open: First bar's open
    - High: Max of all highs
    - Low: Min of all lows
    - Close: Last bar's close
    - Volume: Sum of all volumes
    """

    def resample(
        self,
        data: OHLCVSeries,
        target_timeframe: Timeframe
    ) -> OHLCVSeries:
        ...
```

---

## 4. Rate Limiting, Retry & Error Handling

### 4.1 Rate Limit Strategy

```python
@dataclass
class RateLimitConfig:
    provider: str
    requests_per_second: float
    requests_per_minute: int
    requests_per_hour: int
    concurrent_requests: int

# Provider-specific limits
RATE_LIMITS = {
    "yahoo_finance": RateLimitConfig(
        provider="yahoo_finance",
        requests_per_second=2.0,
        requests_per_minute=100,
        requests_per_hour=2000,
        concurrent_requests=5
    ),
    "csv_local": RateLimitConfig(
        provider="csv_local",
        requests_per_second=float('inf'),
        requests_per_minute=float('inf'),
        requests_per_hour=float('inf'),
        concurrent_requests=50
    )
}
```

### 4.2 Rate Limiter Implementation

```python
class TokenBucketRateLimiter:
    """
    Token bucket algorithm for smooth rate limiting.

    Features:
    - Burst handling with token accumulation
    - Multiple time windows (second, minute, hour)
    - Per-provider configuration
    - Async-friendly with asyncio.Semaphore
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.requests_per_second
        self.last_update = time.monotonic()
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)

    async def acquire(self) -> None:
        """Block until rate limit allows request."""
        ...

    def get_wait_time(self) -> float:
        """Return seconds until next request allowed."""
        ...
```

### 4.3 Retry Strategy

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    # Retryable error types
    retry_on_status_codes: List[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )
    retry_on_exceptions: List[str] = field(
        default_factory=lambda: [
            "ConnectionError",
            "Timeout",
            "ChunkedEncodingError"
        ]
    )

class RetryHandler:
    """
    Exponential backoff with jitter.

    Delay formula: min(max_delay, base_delay * (exponential_base ^ attempt))
    Jitter: random value in [0, delay * 0.1]
    """

    def __init__(self, config: RetryConfig):
        self.config = config

    def calculate_delay(self, attempt: int) -> float:
        delay = min(
            self.config.max_delay_seconds,
            self.config.base_delay_seconds * (self.config.exponential_base ** attempt)
        )
        if self.config.jitter:
            delay += random.uniform(0, delay * 0.1)
        return delay

    def should_retry(self, error: Exception, attempt: int) -> bool:
        ...
```

### 4.4 Error Classification

```python
class DataErrorType(Enum):
    # Transient - will retry
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"

    # Permanent - will not retry
    TICKER_NOT_FOUND = "ticker_not_found"
    INVALID_DATE_RANGE = "invalid_date_range"
    INSUFFICIENT_DATA = "insufficient_data"
    AUTHENTICATION_FAILED = "auth_failed"

    # Data quality
    STALE_DATA = "stale_data"
    MISSING_FIELDS = "missing_fields"
    DATA_ANOMALY = "data_anomaly"

@dataclass
class DataError:
    error_type: DataErrorType
    message: str
    ticker: Optional[str]
    provider: str
    timestamp: datetime
    is_retryable: bool
    raw_error: Optional[Exception]
```

---

## 5. Caching Strategy

### 5.1 Cache Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CACHE HIERARCHY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ L1: In-Memory LRU Cache                                 │   │
│  │ ├── Capacity: 1000 items (configurable)                 │   │
│  │ ├── TTL: Session-based                                  │   │
│  │ └── Use: Hot data for current analysis                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │ miss                                │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ L2: SQLite Metadata + Parquet Data                      │   │
│  │ ├── SQLite: Index, metadata, small datasets             │   │
│  │ ├── Parquet: Large OHLCV series (columnar, compressed)  │   │
│  │ └── Use: Persistent local storage                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │ miss                                │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ L3: Remote Data Provider                                │   │
│  │ ├── Yahoo Finance API                                   │   │
│  │ └── Fetch → Cache in L2 → Return                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Cache Key Design

```python
@dataclass
class CacheKey:
    """Unique identifier for cached data."""

    data_type: str          # "ohlcv", "fundamental", "earnings", etc.
    ticker: str             # Normalized ticker symbol
    timeframe: Optional[str] # For OHLCV data
    start_date: Optional[date]
    end_date: Optional[date]
    provider: str           # Data source
    version: int            # Schema version

    def to_string(self) -> str:
        """Generate unique cache key string."""
        parts = [
            f"v{self.version}",
            self.data_type,
            self.ticker.upper(),
            self.provider
        ]
        if self.timeframe:
            parts.append(self.timeframe)
        if self.start_date:
            parts.append(self.start_date.isoformat())
        if self.end_date:
            parts.append(self.end_date.isoformat())
        return ":".join(parts)

    # Example: "v1:ohlcv:AAPL:yahoo:1d:2023-01-01:2024-01-01"
```

### 5.3 Cache TTL Configuration

```yaml
# configs/cache.yaml

cache:
  l1_memory:
    max_items: 1000
    max_memory_mb: 500

  l2_persistent:
    base_path: "./data/cache"
    sqlite_db: "cache_index.db"
    parquet_dir: "ohlcv/"

  ttl:
    ohlcv:
      1m: 5m          # 5 minutes
      5m: 15m
      15m: 30m
      1h: 1h
      1d: 24h
      1wk: 7d
      1mo: 30d

    fundamental:
      snapshot: 24h
      full_refresh: 7d

    earnings:
      upcoming: 6h
      historical: 30d

    corporate_actions:
      recent: 24h      # Last 30 days
      historical: 90d

    universe:
      sp500: 7d
      russell2000: 7d
      custom: never    # Manual refresh only

  staleness:
    warn_after_ttl_multiplier: 2.0   # Warn if data > 2x TTL
    error_after_ttl_multiplier: 10.0 # Error if data > 10x TTL
```

### 5.4 Cache Operations

```python
class DataCache:
    """Unified cache interface."""

    async def get(
        self,
        key: CacheKey,
        max_staleness: Optional[timedelta] = None
    ) -> Optional[CacheEntry]:
        """
        Retrieve from cache.

        Args:
            key: Cache lookup key
            max_staleness: Override default TTL

        Returns:
            CacheEntry if found and fresh, None otherwise
        """
        ...

    async def put(
        self,
        key: CacheKey,
        data: Any,
        metadata: Optional[Dict] = None
    ) -> None:
        """Store in cache with metadata."""
        ...

    async def invalidate(
        self,
        pattern: str  # Supports wildcards: "ohlcv:AAPL:*"
    ) -> int:
        """Invalidate matching entries. Returns count."""
        ...

    async def get_or_fetch(
        self,
        key: CacheKey,
        fetcher: Callable[[], Awaitable[Any]],
        max_staleness: Optional[timedelta] = None
    ) -> CacheEntry:
        """Get from cache or fetch if missing/stale."""
        ...

@dataclass
class CacheEntry:
    key: CacheKey
    data: Any
    cached_at: datetime
    expires_at: datetime
    source: str          # "memory", "disk", "network"
    is_stale: bool
    staleness_seconds: float
```

### 5.5 Cache Storage Layout

```
data/
└── cache/
    ├── cache_index.db          # SQLite: metadata, small data
    ├── ohlcv/
    │   ├── daily/
    │   │   ├── AAPL.parquet
    │   │   ├── MSFT.parquet
    │   │   └── ...
    │   ├── hourly/
    │   │   └── ...
    │   └── intraday/
    │       └── ...
    ├── fundamentals/
    │   └── snapshots.parquet   # All tickers, partitioned by date
    ├── earnings/
    │   └── calendar.parquet
    └── corporate_actions/
        └── actions.parquet
```

---

## 6. Universe Definition Mechanism

### 6.1 Universe Types

```python
class UniverseType(Enum):
    INDEX = "index"           # S&P 500, Russell 2000, etc.
    SECTOR = "sector"         # Technology, Healthcare, etc.
    CUSTOM = "custom"         # User-defined list
    SCREEN = "screen"         # Dynamic filter-based
    COMPOSITE = "composite"   # Union/intersection of universes
```

### 6.2 Built-in Universes

```yaml
# configs/universes.yaml

universes:
  # Major Indices
  sp500:
    type: index
    name: "S&P 500"
    source: wikipedia
    source_url: "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    refresh_interval: 7d
    ticker_count: ~500

  sp100:
    type: index
    name: "S&P 100"
    source: wikipedia
    refresh_interval: 7d
    ticker_count: ~100

  nasdaq100:
    type: index
    name: "NASDAQ 100"
    source: wikipedia
    refresh_interval: 7d
    ticker_count: ~100

  russell2000:
    type: index
    name: "Russell 2000"
    source: file
    file_path: "data/universes/russell2000.csv"
    refresh_interval: 30d
    ticker_count: ~2000

  russell1000:
    type: index
    name: "Russell 1000"
    source: file
    file_path: "data/universes/russell1000.csv"
    refresh_interval: 30d
    ticker_count: ~1000

  dow30:
    type: index
    name: "Dow Jones Industrial Average"
    source: wikipedia
    refresh_interval: 30d
    ticker_count: 30

  # Sector Universes (derived from S&P 500)
  sector_technology:
    type: sector
    name: "Technology Sector"
    parent: sp500
    sector_filter: "Technology"

  sector_healthcare:
    type: sector
    name: "Healthcare Sector"
    parent: sp500
    sector_filter: "Health Care"

  sector_financials:
    type: sector
    name: "Financials Sector"
    parent: sp500
    sector_filter: "Financials"

  # Special Universes
  mag7:
    type: custom
    name: "Magnificent 7"
    tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

  faang:
    type: custom
    name: "FAANG"
    tickers: ["META", "AAPL", "AMZN", "NFLX", "GOOGL"]
```

### 6.3 Universe Definition Schema

```python
@dataclass
class UniverseDefinition:
    """Complete universe specification."""

    id: str                      # Unique identifier
    name: str                    # Display name
    universe_type: UniverseType
    description: Optional[str]

    # Source configuration
    source: str                  # "wikipedia", "file", "api", "derived"
    source_config: Dict[str, Any]

    # Refresh settings
    refresh_interval: timedelta
    last_refreshed: Optional[datetime]

    # Filtering (for derived universes)
    parent_universe: Optional[str]
    filters: Optional[Dict[str, Any]]

    # Metadata
    expected_count: Optional[int]
    created_at: datetime
    updated_at: datetime

@dataclass
class Universe:
    """Materialized universe with tickers."""

    definition: UniverseDefinition
    tickers: List[str]
    metadata: Dict[str, Dict]    # Per-ticker metadata (sector, etc.)
    materialized_at: datetime
    is_stale: bool
```

### 6.4 Custom Universe Creation

```python
class UniverseManager:
    """Manage universe definitions and materialization."""

    def create_custom_universe(
        self,
        name: str,
        tickers: List[str],
        description: Optional[str] = None
    ) -> UniverseDefinition:
        """Create universe from explicit ticker list."""
        ...

    def create_from_file(
        self,
        name: str,
        file_path: Path,
        ticker_column: str = "ticker"
    ) -> UniverseDefinition:
        """Create universe from CSV/Excel file."""
        ...

    def create_screen_universe(
        self,
        name: str,
        parent: str,
        filters: Dict[str, Any]
    ) -> UniverseDefinition:
        """
        Create dynamic universe from filters.

        Example filters:
        {
            "market_cap_min": 10_000_000_000,
            "sector": ["Technology", "Healthcare"],
            "pe_ratio_max": 30,
            "avg_volume_min": 1_000_000
        }
        """
        ...

    def create_composite_universe(
        self,
        name: str,
        operation: str,  # "union", "intersection", "difference"
        universes: List[str]
    ) -> UniverseDefinition:
        """Combine multiple universes."""
        ...

    async def materialize(
        self,
        universe_id: str,
        force_refresh: bool = False
    ) -> Universe:
        """Fetch/compute actual ticker list."""
        ...

    def list_universes(self) -> List[UniverseDefinition]:
        """List all available universes."""
        ...
```

### 6.5 Universe Storage

```
data/
└── universes/
    ├── definitions.yaml        # Universe definitions
    ├── sp500.csv               # Cached ticker list
    ├── russell2000.csv
    ├── custom/
    │   ├── my_watchlist.csv
    │   └── earnings_plays.csv
    └── metadata/
        └── ticker_sectors.parquet  # Sector/industry mappings
```

---

## 7. Data Versioning Strategy

### 7.1 Version Dimensions

```python
@dataclass
class DataVersion:
    """Multi-dimensional versioning."""

    # Schema version - changes when data structure changes
    schema_version: int          # e.g., 1, 2, 3

    # Data snapshot - when data was captured
    snapshot_timestamp: datetime

    # Provider version - track provider API changes
    provider_version: str        # e.g., "yfinance-0.2.28"

    # Adjustment version - for price adjustments
    adjustment_date: date        # Last adjustment applied

    def to_string(self) -> str:
        return f"s{self.schema_version}_{self.snapshot_timestamp.strftime('%Y%m%d%H%M%S')}"
```

### 7.2 Schema Versioning

```python
# Schema version history
SCHEMA_VERSIONS = {
    1: {
        "ohlcv": ["timestamp", "open", "high", "low", "close", "volume"],
        "valid_from": "2024-01-01",
        "valid_to": "2024-06-01"
    },
    2: {
        "ohlcv": ["timestamp", "open", "high", "low", "close", "adj_close", "volume"],
        "valid_from": "2024-06-01",
        "valid_to": None  # Current
    }
}

CURRENT_SCHEMA_VERSION = 2

class SchemaRegistry:
    """Track and migrate between schema versions."""

    def get_current_version(self, data_type: str) -> int:
        ...

    def is_compatible(self, data_version: int, current_version: int) -> bool:
        ...

    def migrate(
        self,
        data: Any,
        from_version: int,
        to_version: int
    ) -> Any:
        """Migrate data between schema versions."""
        ...
```

### 7.3 Point-in-Time Data Access

```python
class PointInTimeData:
    """
    Access data as it existed at a specific point in time.
    Critical for reproducible research.
    """

    async def get_ohlcv_as_of(
        self,
        ticker: str,
        as_of_date: date,
        lookback_days: int = 365
    ) -> OHLCVSeries:
        """
        Get OHLCV data as it would have appeared on as_of_date.

        Handles:
        - Corporate actions (splits, dividends) up to that date
        - No future data leakage
        """
        ...

    async def get_fundamental_as_of(
        self,
        ticker: str,
        as_of_date: date
    ) -> FundamentalSnapshot:
        """Get fundamentals known as of a specific date."""
        ...
```

### 7.4 Data Lineage Tracking

```python
@dataclass
class DataLineage:
    """Track data provenance for reproducibility."""

    data_id: str                 # Unique identifier
    ticker: str
    data_type: str

    # Source information
    provider: str
    provider_version: str
    fetch_timestamp: datetime
    api_endpoint: Optional[str]

    # Transformation history
    transformations: List[Dict]  # List of applied transformations

    # Versioning
    schema_version: int
    adjustment_version: str

    # Checksums
    row_count: int
    checksum: str                # MD5/SHA256 of data

    def to_dict(self) -> Dict:
        ...
```

### 7.5 Version Storage

```python
class VersionedDataStore:
    """Store multiple versions of data."""

    def __init__(self, base_path: Path):
        self.base_path = base_path

    def store(
        self,
        data: Any,
        version: DataVersion,
        lineage: DataLineage
    ) -> str:
        """Store data with version info. Returns data_id."""
        ...

    def retrieve(
        self,
        data_id: str,
        version: Optional[DataVersion] = None
    ) -> Tuple[Any, DataLineage]:
        """Retrieve specific version of data."""
        ...

    def list_versions(
        self,
        ticker: str,
        data_type: str
    ) -> List[DataVersion]:
        """List all available versions."""
        ...

    def cleanup_old_versions(
        self,
        keep_versions: int = 5,
        keep_days: int = 90
    ) -> int:
        """Remove old versions. Returns count deleted."""
        ...
```

---

## 8. Data Provider Interface

### 8.1 Abstract Provider

```python
from abc import ABC, abstractmethod

class DataProvider(ABC):
    """Abstract base class for all data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        ...

    @property
    @abstractmethod
    def supported_data_types(self) -> List[str]:
        """List of supported data types."""
        ...

    @abstractmethod
    async def fetch_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: date,
        end: date
    ) -> OHLCVSeries:
        ...

    @abstractmethod
    async def fetch_fundamentals(
        self,
        ticker: str
    ) -> FundamentalSnapshot:
        ...

    @abstractmethod
    async def fetch_corporate_actions(
        self,
        ticker: str,
        start: date,
        end: date
    ) -> List[CorporateAction]:
        ...

    @abstractmethod
    async def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker exists and is valid."""
        ...

    @abstractmethod
    def get_rate_limit_config(self) -> RateLimitConfig:
        ...
```

### 8.2 Provider Registry

```python
class ProviderRegistry:
    """Registry of available data providers."""

    _providers: Dict[str, DataProvider] = {}
    _priority: List[str] = []

    def register(
        self,
        provider: DataProvider,
        priority: int = 0
    ) -> None:
        ...

    def get(self, name: str) -> DataProvider:
        ...

    def get_for_data_type(self, data_type: str) -> List[DataProvider]:
        """Get providers supporting this data type, by priority."""
        ...

    def list_providers(self) -> List[str]:
        ...
```

---

## 9. Configuration

### 9.1 Master Configuration

```yaml
# configs/data_ingestion.yaml

data_ingestion:
  # Default provider priority
  providers:
    - yahoo_finance
    - csv_local

  # Default timeframe for analysis
  default_timeframe: 1d

  # Default lookback period
  default_lookback_days: 365

  # Parallel fetching
  max_concurrent_fetches: 10
  batch_size: 50

  # Offline mode
  offline_mode: false
  offline_fail_on_cache_miss: false

  # Data quality
  min_data_points: 20
  max_gap_days: 5

  # Logging
  log_fetches: true
  log_cache_hits: false
```

---

## 10. Assumptions

1. **Yahoo Finance Availability:** Primary data source; system degrades gracefully if unavailable
2. **US Market Focus:** Initial focus on US equities; timezone handling assumes ET
3. **Daily Data Primary:** Most analysis uses daily data; intraday is supplementary
4. **Single Currency:** USD only for v1; no FX conversion
5. **Historical Data:** No real-time streaming; batch/delayed data only
6. **Internet Connectivity:** Required for initial data fetch; offline after caching
7. **Storage Availability:** Sufficient local disk for cache (estimate: ~1GB per 1000 tickers/year)
8. **Data Accuracy:** Provider data assumed accurate; no external validation

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
