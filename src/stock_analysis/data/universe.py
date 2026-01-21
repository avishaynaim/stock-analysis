"""
Universe management for stock selection.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from stock_analysis.core.logging import get_logger
from stock_analysis.data.cache import DataCache

logger = get_logger("data.universe")


class UniverseManager:
    """
    Manage stock universes (S&P 500, Russell, custom lists).
    """

    # Wikipedia URLs for index constituents
    WIKIPEDIA_URLS = {
        "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "SP400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "SP600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        "NASDAQ100": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "DOW30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    }

    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()
        self._custom_universes: dict[str, list[str]] = {}

    def get_universe(
        self,
        universe: str,
        as_of_date: Optional[datetime] = None,
    ) -> list[str]:
        """
        Get list of tickers in a universe.

        Args:
            universe: Universe name (SP500, NASDAQ100, custom, etc.)
            as_of_date: Point-in-time date (not fully supported yet)

        Returns:
            List of ticker symbols
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        # Check for custom universe
        if universe in self._custom_universes:
            return self._custom_universes[universe]

        # Check cache
        cache_key = f"universe:{universe}:{as_of_date.date()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Fetch from source
        if universe.upper() in self.WIKIPEDIA_URLS:
            tickers = self._fetch_from_wikipedia(universe.upper())
        else:
            raise ValueError(f"Unknown universe: {universe}")

        # Cache result
        self.cache.set(cache_key, tickers, ttl_hours=168)  # 1 week

        return tickers

    def _fetch_from_wikipedia(self, universe: str) -> list[str]:
        """Fetch universe constituents from Wikipedia."""
        url = self.WIKIPEDIA_URLS[universe]

        logger.info(f"Fetching {universe} constituents from Wikipedia")

        try:
            # Fetch with proper headers to avoid 403 Forbidden
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Read tables from HTML content
            tables = pd.read_html(response.text)

            # Find the table with ticker symbols
            tickers = []

            for table in tables:
                # Look for common column names
                for col in ["Symbol", "Ticker", "Ticker symbol", "Stock symbol"]:
                    if col in table.columns:
                        tickers = table[col].tolist()
                        break

                if tickers:
                    break

            if not tickers:
                # Try first column if it looks like tickers
                first_table = tables[0]
                first_col = first_table.columns[0]
                potential_tickers = first_table[first_col].tolist()

                # Filter to look like tickers (1-5 uppercase letters)
                tickers = [
                    t for t in potential_tickers
                    if isinstance(t, str) and 1 <= len(t) <= 5 and t.isalpha() and t.isupper()
                ]

            # Clean tickers
            cleaned = []
            for ticker in tickers:
                if isinstance(ticker, str):
                    # Remove any class designations (e.g., BRK.B -> BRK-B)
                    ticker = ticker.replace(".", "-").strip()
                    if ticker and ticker not in cleaned:
                        cleaned.append(ticker)

            logger.info(f"Found {len(cleaned)} tickers in {universe}")

            return cleaned

        except Exception as e:
            logger.error(f"Failed to fetch {universe} from Wikipedia: {e}")
            raise

    def register_custom_universe(
        self,
        name: str,
        tickers: list[str],
    ) -> None:
        """Register a custom universe."""
        self._custom_universes[name] = [t.upper() for t in tickers]
        logger.info(f"Registered custom universe '{name}' with {len(tickers)} tickers")

    def load_custom_universe(
        self,
        name: str,
        file_path: str,
        ticker_column: str = "ticker",
    ) -> list[str]:
        """Load custom universe from CSV file."""
        df = pd.read_csv(file_path)

        if ticker_column not in df.columns:
            # Try to find ticker column
            for col in df.columns:
                if "ticker" in col.lower() or "symbol" in col.lower():
                    ticker_column = col
                    break
            else:
                # Use first column
                ticker_column = df.columns[0]

        tickers = df[ticker_column].dropna().astype(str).str.upper().tolist()

        self.register_custom_universe(name, tickers)

        return tickers

    def get_available_universes(self) -> list[str]:
        """Get list of available universes."""
        standard = list(self.WIKIPEDIA_URLS.keys())
        custom = list(self._custom_universes.keys())
        return standard + custom

    def filter_universe(
        self,
        tickers: list[str],
        filters: dict,
        data_provider: "DataProvider",  # type: ignore
        as_of_date: Optional[datetime] = None,
    ) -> list[str]:
        """
        Apply filters to universe.

        Filters:
        - min_price: Minimum stock price
        - min_volume: Minimum average daily dollar volume
        - min_market_cap: Minimum market cap
        - sectors: List of sectors to include/exclude
        - sector_mode: 'include' or 'exclude'
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        filtered = []

        for ticker in tickers:
            try:
                # Price filter
                if "min_price" in filters:
                    price = data_provider.get_latest_price(ticker, as_of_date)
                    if price < filters["min_price"]:
                        continue

                # Volume filter
                if "min_volume" in filters:
                    volume = data_provider.get_average_dollar_volume(
                        ticker, as_of_date
                    )
                    if volume < filters["min_volume"]:
                        continue

                # Market cap filter
                if "min_market_cap" in filters:
                    market_cap = data_provider.get_market_cap(ticker, as_of_date)
                    if market_cap is None or market_cap < filters["min_market_cap"]:
                        continue

                # Sector filter
                if "sectors" in filters and filters["sectors"]:
                    sector = data_provider.get_sector(ticker)
                    mode = filters.get("sector_mode", "include")

                    if mode == "include":
                        if sector not in filters["sectors"]:
                            continue
                    else:  # exclude
                        if sector in filters["sectors"]:
                            continue

                filtered.append(ticker)

            except Exception as e:
                logger.debug(f"Filter error for {ticker}: {e}")
                continue

        return filtered
