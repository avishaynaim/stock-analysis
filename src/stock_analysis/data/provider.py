"""
Data provider for fetching and managing market data.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from stock_analysis.core.exceptions import DataError, DataNotFoundError
from stock_analysis.core.logging import get_logger
from stock_analysis.data.models import (
    EarningsData,
    FundamentalData,
    PriceData,
    TickerInfo,
)
from stock_analysis.data.cache import DataCache
from stock_analysis.data.storage import LocalStorage, get_storage

logger = get_logger("data.provider")


class DataProvider:
    """
    Main data provider for the stock analysis system.

    Handles data fetching from various sources with caching.
    Uses local storage to persist all data and only fetch new/missing data.
    """

    def __init__(
        self,
        cache: Optional[DataCache] = None,
        storage: Optional[LocalStorage] = None,
        rate_limit: int = 5,
        use_local_storage: bool = True,
    ):
        self.cache = cache or DataCache()
        self.storage = storage or (get_storage() if use_local_storage else None)
        self.rate_limit = rate_limit
        self.use_local_storage = use_local_storage
        self._ticker_cache: dict[str, yf.Ticker] = {}

    def _get_yf_ticker(self, ticker: str) -> yf.Ticker:
        """Get or create yfinance Ticker object."""
        if ticker not in self._ticker_cache:
            self._ticker_cache[ticker] = yf.Ticker(ticker)
        return self._ticker_cache[ticker]

    def get_prices(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1d",
    ) -> PriceData:
        """
        Get price data for a ticker.

        Uses local storage to persist data and only fetch new/missing data.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (default: 3 years ago)
            end_date: End date (default: today)
            timeframe: Data timeframe ('1d', '1w', '1mo')

        Returns:
            PriceData object

        Raises:
            DataNotFoundError: If no data available
        """
        ticker = ticker.upper()

        # Set defaults
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=756)  # 3 years

        # Try to load from local storage first (for daily data)
        if self.use_local_storage and self.storage and timeframe == "1d":
            df = self._get_prices_with_storage(ticker, start_date, end_date)
            if df is not None and len(df) > 0:
                # Normalize column names
                df = self._normalize_columns(df)
                price_data = PriceData(
                    ticker=ticker,
                    data=df,
                    timeframe=timeframe,
                )
                return price_data

        # Fall back to cache check
        cache_key = f"prices:{ticker}:{timeframe}:{start_date.date()}:{end_date.date()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Fetch from yfinance
        logger.info(f"Fetching prices for {ticker} from {start_date.date()} to {end_date.date()}")

        try:
            yf_ticker = self._get_yf_ticker(ticker)
            df = yf_ticker.history(
                start=start_date,
                end=end_date + timedelta(days=1),  # yfinance end is exclusive
                interval=timeframe,
                auto_adjust=False,
            )
        except Exception as e:
            raise DataError(f"Failed to fetch prices for {ticker}: {e}")

        if df is None or len(df) == 0:
            raise DataNotFoundError(ticker, "price")

        # Normalize column names
        df = self._normalize_columns(df)

        # Save to local storage (for daily data)
        if self.use_local_storage and self.storage and timeframe == "1d":
            self.storage.save_prices(ticker, df, merge=True)

        # Create PriceData
        price_data = PriceData(
            ticker=ticker,
            data=df,
            timeframe=timeframe,
        )

        # Cache result
        self.cache.set(cache_key, price_data, ttl_hours=24)

        return price_data

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and timezone to be consistent."""
        df = df.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        # Ensure adj_close exists
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]
        # Remove timezone info from index for consistent comparison
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def _get_prices_with_storage(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Get prices using local storage, fetching only missing data."""
        # Check what ranges we're missing
        missing_ranges = self.storage.get_missing_range(ticker, start_date, end_date)

        # Fetch missing ranges
        for range_start, range_end in missing_ranges:
            logger.info(f"Fetching missing data for {ticker}: {range_start.date()} to {range_end.date()}")
            try:
                yf_ticker = self._get_yf_ticker(ticker)
                df = yf_ticker.history(
                    start=range_start,
                    end=range_end + timedelta(days=1),
                    interval="1d",
                    auto_adjust=False,
                )
                if df is not None and len(df) > 0:
                    df = self._normalize_columns(df)
                    self.storage.save_prices(ticker, df, merge=True)
            except Exception as e:
                logger.warning(f"Failed to fetch missing data for {ticker}: {e}")

        # Load from storage
        return self.storage.load_prices(ticker, start_date, end_date)

    def get_latest_price(
        self,
        ticker: str,
        as_of_date: Optional[datetime] = None,
    ) -> float:
        """Get the latest closing price."""
        if as_of_date is None:
            as_of_date = datetime.now()

        prices = self.get_prices(
            ticker,
            start_date=as_of_date - timedelta(days=10),
            end_date=as_of_date,
        )

        return prices.close.iloc[-1]

    def get_fundamentals(
        self,
        ticker: str,
        as_of_date: Optional[datetime] = None,
    ) -> FundamentalData:
        """
        Get fundamental data for a ticker.

        Note: yfinance returns current fundamentals, not point-in-time.
        For backtesting, a proper fundamental data provider with
        point-in-time data is recommended.
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        cache_key = f"fundamentals:{ticker}:{as_of_date.date()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        logger.debug(f"Fetching fundamentals for {ticker}")

        try:
            yf_ticker = self._get_yf_ticker(ticker)
            info = yf_ticker.info
        except Exception as e:
            raise DataError(f"Failed to fetch fundamentals for {ticker}: {e}")

        if not info:
            raise DataNotFoundError(ticker, "fundamental")

        fundamentals = FundamentalData(
            ticker=ticker,
            as_of_date=as_of_date,
            # Valuation
            market_cap=info.get("marketCap"),
            enterprise_value=info.get("enterpriseValue"),
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            peg_ratio=info.get("pegRatio"),
            price_to_book=info.get("priceToBook"),
            price_to_sales=info.get("priceToSalesTrailing12Months"),
            ev_to_ebitda=info.get("enterpriseToEbitda"),
            # Profitability
            profit_margin=info.get("profitMargins"),
            operating_margin=info.get("operatingMargins"),
            gross_margin=info.get("grossMargins"),
            return_on_equity=info.get("returnOnEquity"),
            return_on_assets=info.get("returnOnAssets"),
            # Growth
            revenue_growth=info.get("revenueGrowth"),
            earnings_growth=info.get("earningsGrowth"),
            # Financial health
            current_ratio=info.get("currentRatio"),
            quick_ratio=info.get("quickRatio"),
            debt_to_equity=info.get("debtToEquity"),
            total_debt=info.get("totalDebt"),
            total_cash=info.get("totalCash"),
            free_cash_flow=info.get("freeCashflow"),
            # Per share
            eps_ttm=info.get("trailingEps"),
            eps_forward=info.get("forwardEps"),
            book_value_per_share=info.get("bookValue"),
            # Dividends
            dividend_yield=info.get("dividendYield"),
            dividend_rate=info.get("dividendRate"),
            payout_ratio=info.get("payoutRatio"),
            # Other
            beta=info.get("beta"),
            shares_outstanding=info.get("sharesOutstanding"),
            float_shares=info.get("floatShares"),
            short_ratio=info.get("shortRatio"),
            short_percent_of_float=info.get("shortPercentOfFloat"),
        )

        self.cache.set(cache_key, fundamentals, ttl_hours=24)

        return fundamentals

    def get_earnings(
        self,
        ticker: str,
        as_of_date: Optional[datetime] = None,
        lookback_quarters: int = 8,
    ) -> EarningsData:
        """Get earnings data for a ticker."""
        if as_of_date is None:
            as_of_date = datetime.now()

        cache_key = f"earnings:{ticker}:{as_of_date.date()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        logger.debug(f"Fetching earnings for {ticker}")

        try:
            yf_ticker = self._get_yf_ticker(ticker)

            # Get earnings history
            earnings_hist = yf_ticker.earnings_history
            if earnings_hist is not None and len(earnings_hist) > 0:
                earnings_df = earnings_hist.tail(lookback_quarters).copy()
                earnings_df = earnings_df.rename(
                    columns={
                        "epsActual": "eps_actual",
                        "epsEstimate": "eps_estimate",
                        "epsDifference": "surprise",
                        "surprisePercent": "surprise_pct",
                    }
                )
            else:
                earnings_df = pd.DataFrame()

            # Get next earnings date
            calendar = yf_ticker.calendar
            next_date = None
            if calendar is not None and "Earnings Date" in calendar:
                dates = calendar["Earnings Date"]
                if len(dates) > 0:
                    next_date = dates[0].date() if hasattr(dates[0], "date") else dates[0]

        except Exception as e:
            logger.warning(f"Failed to fetch earnings for {ticker}: {e}")
            earnings_df = pd.DataFrame()
            next_date = None

        earnings = EarningsData(
            ticker=ticker,
            earnings_history=earnings_df,
            next_earnings_date=next_date,
        )

        self.cache.set(cache_key, earnings, ttl_hours=24)

        return earnings

    def get_ticker_info(self, ticker: str) -> TickerInfo:
        """Get basic ticker information."""
        cache_key = f"info:{ticker}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        logger.debug(f"Fetching info for {ticker}")

        try:
            yf_ticker = self._get_yf_ticker(ticker)
            info = yf_ticker.info
        except Exception as e:
            raise DataError(f"Failed to fetch info for {ticker}: {e}")

        if not info:
            raise DataNotFoundError(ticker, "info")

        # Determine market cap category
        market_cap = info.get("marketCap", 0)
        if market_cap >= 200_000_000_000:
            cap_category = "mega"
        elif market_cap >= 10_000_000_000:
            cap_category = "large"
        elif market_cap >= 2_000_000_000:
            cap_category = "mid"
        elif market_cap >= 300_000_000:
            cap_category = "small"
        else:
            cap_category = "micro"

        ticker_info = TickerInfo(
            ticker=ticker,
            name=info.get("longName", info.get("shortName", ticker)),
            sector=info.get("sector"),
            industry=info.get("industry"),
            exchange=info.get("exchange"),
            currency=info.get("currency", "USD"),
            country=info.get("country"),
            website=info.get("website"),
            description=info.get("longBusinessSummary"),
            market_cap_category=cap_category,
            is_etf=info.get("quoteType") == "ETF",
        )

        self.cache.set(cache_key, ticker_info, ttl_hours=168)  # 1 week

        return ticker_info

    def get_average_dollar_volume(
        self,
        ticker: str,
        as_of_date: Optional[datetime] = None,
        lookback_days: int = 20,
    ) -> float:
        """Get average daily dollar volume."""
        if as_of_date is None:
            as_of_date = datetime.now()

        prices = self.get_prices(
            ticker,
            start_date=as_of_date - timedelta(days=lookback_days + 10),
            end_date=as_of_date,
        )

        # Dollar volume = close * volume
        dollar_volume = prices.data["close"] * prices.data["volume"]

        return dollar_volume.tail(lookback_days).mean()

    def get_market_cap(
        self,
        ticker: str,
        as_of_date: Optional[datetime] = None,
    ) -> Optional[float]:
        """Get market capitalization."""
        try:
            fundamentals = self.get_fundamentals(ticker, as_of_date)
            return fundamentals.market_cap
        except DataError:
            return None

    def get_sector(self, ticker: str) -> Optional[str]:
        """Get ticker sector."""
        try:
            info = self.get_ticker_info(ticker)
            return info.sector
        except DataError:
            return None

    def get_history_length(
        self,
        ticker: str,
        as_of_date: Optional[datetime] = None,
    ) -> int:
        """Get number of trading days of available history."""
        if as_of_date is None:
            as_of_date = datetime.now()

        try:
            prices = self.get_prices(
                ticker,
                start_date=as_of_date - timedelta(days=3650),  # 10 years
                end_date=as_of_date,
            )
            return len(prices)
        except DataError:
            return 0

    def prefetch_batch(
        self,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime,
        data_types: Optional[list[str]] = None,
    ) -> dict[str, bool]:
        """
        Prefetch data for multiple tickers.

        Returns dict of ticker -> success status.
        """
        if data_types is None:
            data_types = ["prices"]

        results = {}

        for ticker in tickers:
            try:
                if "prices" in data_types:
                    self.get_prices(ticker, start_date, end_date)
                if "fundamentals" in data_types:
                    self.get_fundamentals(ticker, end_date)
                if "earnings" in data_types:
                    self.get_earnings(ticker, end_date)
                results[ticker] = True
            except DataError as e:
                logger.debug(f"Prefetch failed for {ticker}: {e}")
                results[ticker] = False

        return results
