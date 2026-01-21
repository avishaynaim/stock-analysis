"""
Data models for the stock analysis system.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class OHLCVBar:
    """Single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

    def __post_init__(self) -> None:
        if self.adjusted_close is None:
            self.adjusted_close = self.close


@dataclass
class PriceData:
    """Price data container with helper methods."""

    ticker: str
    data: pd.DataFrame  # Columns: open, high, low, close, volume, adj_close
    timeframe: str = "1d"

    def __post_init__(self) -> None:
        """Ensure proper column names and index."""
        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        # Sort by date
        self.data = self.data.sort_index()

        # Standardize column names
        column_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
            "Adjusted Close": "adj_close",
        }
        self.data = self.data.rename(columns=column_map)

        # Ensure adj_close exists
        if "adj_close" not in self.data.columns:
            self.data["adj_close"] = self.data["close"]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def start_date(self) -> datetime:
        """Get first date in data."""
        return self.data.index[0].to_pydatetime()

    @property
    def end_date(self) -> datetime:
        """Get last date in data."""
        return self.data.index[-1].to_pydatetime()

    @property
    def close(self) -> pd.Series:
        """Get close prices."""
        return self.data["adj_close"]

    @property
    def returns(self) -> pd.Series:
        """Get daily returns."""
        return self.close.pct_change()

    @property
    def log_returns(self) -> pd.Series:
        """Get log returns."""
        return np.log(self.close / self.close.shift(1))

    def get_latest(self) -> OHLCVBar:
        """Get the most recent bar."""
        row = self.data.iloc[-1]
        return OHLCVBar(
            timestamp=self.data.index[-1].to_pydatetime(),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=int(row["volume"]),
            adjusted_close=row["adj_close"],
        )

    def get_as_of(self, as_of_date: datetime) -> "PriceData":
        """Get data up to and including as_of_date."""
        mask = self.data.index <= as_of_date
        return PriceData(
            ticker=self.ticker,
            data=self.data[mask].copy(),
            timeframe=self.timeframe,
        )

    def resample(self, timeframe: str) -> "PriceData":
        """Resample to different timeframe."""
        if timeframe == self.timeframe:
            return self

        # Map timeframe to pandas offset
        offset_map = {
            "1d": "D",
            "1w": "W",
            "1mo": "ME",
            "1M": "ME",
        }
        offset = offset_map.get(timeframe, timeframe)

        # Resample OHLCV
        resampled = self.data.resample(offset).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "adj_close": "last",
            }
        ).dropna()

        return PriceData(
            ticker=self.ticker,
            data=resampled,
            timeframe=timeframe,
        )


@dataclass
class FundamentalData:
    """Fundamental data for a ticker."""

    ticker: str
    as_of_date: datetime

    # Valuation
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    ev_to_ebitda: Optional[float] = None

    # Profitability
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None

    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None

    # Financial health
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    total_debt: Optional[float] = None
    total_cash: Optional[float] = None
    free_cash_flow: Optional[float] = None

    # Per share
    eps_ttm: Optional[float] = None
    eps_forward: Optional[float] = None
    book_value_per_share: Optional[float] = None
    revenue_per_share: Optional[float] = None

    # Dividends
    dividend_yield: Optional[float] = None
    dividend_rate: Optional[float] = None
    payout_ratio: Optional[float] = None
    ex_dividend_date: Optional[date] = None

    # Other
    beta: Optional[float] = None
    shares_outstanding: Optional[float] = None
    float_shares: Optional[float] = None
    short_ratio: Optional[float] = None
    short_percent_of_float: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class EarningsData:
    """Earnings data for a ticker."""

    ticker: str
    earnings_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Columns: date, eps_actual, eps_estimate, surprise, surprise_pct

    next_earnings_date: Optional[date] = None
    eps_estimate_next: Optional[float] = None
    revenue_estimate_next: Optional[float] = None

    @property
    def latest_surprise(self) -> Optional[float]:
        """Get most recent earnings surprise percentage."""
        if len(self.earnings_history) == 0:
            return None
        return self.earnings_history["surprise_pct"].iloc[-1]

    @property
    def avg_surprise(self) -> Optional[float]:
        """Get average earnings surprise percentage."""
        if len(self.earnings_history) == 0:
            return None
        return self.earnings_history["surprise_pct"].mean()

    @property
    def beat_rate(self) -> Optional[float]:
        """Get percentage of earnings beats."""
        if len(self.earnings_history) == 0:
            return None
        return (self.earnings_history["surprise_pct"] > 0).mean()


@dataclass
class CorporateAction:
    """Corporate action record."""

    ticker: str
    date: datetime
    action_type: str  # 'SPLIT', 'DIVIDEND', 'MERGER', 'SPINOFF'
    value: float  # Split ratio or dividend amount
    description: Optional[str] = None


@dataclass
class TickerInfo:
    """Basic ticker information."""

    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "USD"
    country: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None

    # Classification
    market_cap_category: Optional[str] = None  # 'mega', 'large', 'mid', 'small', 'micro'
    is_etf: bool = False
    is_adr: bool = False

    def __str__(self) -> str:
        return f"{self.ticker} - {self.name}"
