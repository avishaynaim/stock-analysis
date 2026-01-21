"""
Persistent local storage for price data.

Stores all historical data locally and only fetches new/missing data.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from stock_analysis.core.logging import get_logger

logger = get_logger("data.storage")


class LocalStorage:
    """
    Persistent local storage for stock price data.

    Stores data in Parquet format for efficient read/write.
    Only fetches missing data from the network.
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
    ):
        """Initialize local storage.

        Args:
            storage_dir: Directory for storing data files
        """
        self.storage_dir = storage_dir or Path.home() / ".stock-analysis" / "data"
        self.prices_dir = self.storage_dir / "prices"
        self.metadata_file = self.storage_dir / "metadata.json"

        # Create directories
        self.prices_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {"tickers": {}}

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def _get_price_path(self, ticker: str) -> Path:
        """Get path for ticker price data."""
        return self.prices_dir / f"{ticker.upper()}.parquet"

    def has_data(self, ticker: str) -> bool:
        """Check if we have any data for ticker."""
        return self._get_price_path(ticker).exists()

    def get_data_range(self, ticker: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range of stored data for ticker."""
        ticker = ticker.upper()
        if ticker in self._metadata.get("tickers", {}):
            meta = self._metadata["tickers"][ticker]
            start = datetime.fromisoformat(meta["start_date"]) if meta.get("start_date") else None
            end = datetime.fromisoformat(meta["end_date"]) if meta.get("end_date") else None
            return start, end
        return None, None

    def load_prices(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Load price data from local storage.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            DataFrame with price data or None if not found
        """
        path = self._get_price_path(ticker)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)

            # Ensure index is datetime and timezone-naive for consistent comparison
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Remove timezone info if present
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Filter by date range (convert input dates to naive timestamps)
            if start_date is not None:
                start_ts = pd.Timestamp(start_date)
                if start_ts.tz is not None:
                    start_ts = start_ts.tz_localize(None)
                df = df[df.index >= start_ts]
            if end_date is not None:
                end_ts = pd.Timestamp(end_date)
                if end_ts.tz is not None:
                    end_ts = end_ts.tz_localize(None)
                df = df[df.index <= end_ts]

            return df if len(df) > 0 else None

        except Exception as e:
            logger.warning(f"Failed to load prices for {ticker}: {e}")
            return None

    def save_prices(
        self,
        ticker: str,
        data: pd.DataFrame,
        merge: bool = True,
    ) -> None:
        """Save price data to local storage.

        Args:
            ticker: Stock ticker symbol
            data: DataFrame with price data
            merge: If True, merge with existing data
        """
        ticker = ticker.upper()
        path = self._get_price_path(ticker)

        # Ensure index is datetime and timezone-naive
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Merge with existing data if requested
        if merge and path.exists():
            try:
                existing = pd.read_parquet(path)
                if not isinstance(existing.index, pd.DatetimeIndex):
                    existing.index = pd.to_datetime(existing.index)
                if existing.index.tz is not None:
                    existing.index = existing.index.tz_localize(None)

                # Combine, keeping new data where there's overlap
                combined = pd.concat([existing, data])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                data = combined
            except Exception as e:
                logger.warning(f"Failed to merge existing data for {ticker}: {e}")

        # Save to parquet
        try:
            data.to_parquet(path, engine='pyarrow')

            # Update metadata
            if "tickers" not in self._metadata:
                self._metadata["tickers"] = {}

            self._metadata["tickers"][ticker] = {
                "start_date": str(data.index.min()),
                "end_date": str(data.index.max()),
                "rows": len(data),
                "last_updated": str(datetime.now()),
            }
            self._save_metadata()

            logger.debug(f"Saved {len(data)} rows for {ticker}")

        except Exception as e:
            logger.error(f"Failed to save prices for {ticker}: {e}")

    def get_missing_range(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Get date ranges that need to be fetched.

        Args:
            ticker: Stock ticker symbol
            start_date: Desired start date
            end_date: Desired end date

        Returns:
            List of (start, end) tuples for missing ranges
        """
        stored_start, stored_end = self.get_data_range(ticker)

        if stored_start is None or stored_end is None:
            # No data stored, fetch everything
            return [(start_date, end_date)]

        # Normalize all datetimes to naive (remove timezone info for comparison)
        def to_naive(dt):
            if dt is None:
                return None
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                return dt.replace(tzinfo=None)
            return dt

        start_date = to_naive(start_date)
        end_date = to_naive(end_date)
        stored_start = to_naive(stored_start)
        stored_end = to_naive(stored_end)

        missing = []

        # Check if we need earlier data
        if start_date < stored_start:
            missing.append((start_date, stored_start - timedelta(days=1)))

        # Check if we need more recent data
        if end_date > stored_end:
            missing.append((stored_end + timedelta(days=1), end_date))

        return missing

    def list_tickers(self) -> list[str]:
        """List all tickers with stored data."""
        tickers = []
        for path in self.prices_dir.glob("*.parquet"):
            tickers.append(path.stem.upper())
        return sorted(tickers)

    def get_stats(self) -> dict:
        """Get storage statistics."""
        total_size = 0
        total_rows = 0
        ticker_count = 0

        for path in self.prices_dir.glob("*.parquet"):
            ticker_count += 1
            total_size += path.stat().st_size

        for ticker, meta in self._metadata.get("tickers", {}).items():
            total_rows += meta.get("rows", 0)

        return {
            "ticker_count": ticker_count,
            "total_rows": total_rows,
            "total_size_mb": total_size / (1024 * 1024),
            "storage_dir": str(self.storage_dir),
        }

    def delete_ticker(self, ticker: str) -> bool:
        """Delete all data for a ticker."""
        ticker = ticker.upper()
        path = self._get_price_path(ticker)

        if path.exists():
            try:
                path.unlink()
                if ticker in self._metadata.get("tickers", {}):
                    del self._metadata["tickers"][ticker]
                    self._save_metadata()
                return True
            except Exception as e:
                logger.error(f"Failed to delete {ticker}: {e}")
        return False

    def clear_all(self) -> None:
        """Delete all stored data."""
        for path in self.prices_dir.glob("*.parquet"):
            try:
                path.unlink()
            except Exception:
                pass

        self._metadata = {"tickers": {}}
        self._save_metadata()


# Global storage instance
_storage: Optional[LocalStorage] = None


def get_storage() -> LocalStorage:
    """Get global storage instance."""
    global _storage
    if _storage is None:
        _storage = LocalStorage()
    return _storage
