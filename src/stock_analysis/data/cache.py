"""
Caching layer for data storage.
"""

import hashlib
import pickle
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from stock_analysis.core.logging import get_logger

logger = get_logger("data.cache")


class CacheEntry:
    """Cache entry with expiration."""

    def __init__(self, value: Any, expires_at: Optional[datetime] = None):
        self.value = value
        self.expires_at = expires_at
        self.created_at = datetime.now()

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class DataCache:
    """
    Multi-level cache for data.

    Level 1: In-memory cache
    Level 2: Disk cache (pickle files)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_items: int = 1000,
        enable_disk_cache: bool = True,
    ):
        self.cache_dir = cache_dir or Path.home() / ".stock-analysis" / "cache"
        self.max_memory_items = max_memory_items
        self.enable_disk_cache = enable_disk_cache

        self._memory_cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self._lock = threading.Lock()

        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_disk_path(self, key: str) -> Path:
        """Get disk path for cache key."""
        # Hash the key to create a valid filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Checks memory first, then disk.
        """
        # Check memory cache
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if not entry.is_expired():
                    # Move to end of access order (LRU)
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                    return entry.value
                else:
                    # Remove expired entry
                    del self._memory_cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)

        # Check disk cache
        if self.enable_disk_cache:
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                try:
                    with open(disk_path, "rb") as f:
                        entry = pickle.load(f)
                    if not entry.is_expired():
                        # Promote to memory cache
                        self._set_memory(key, entry)
                        return entry.value
                    else:
                        # Remove expired disk entry
                        disk_path.unlink()
                except Exception as e:
                    logger.debug(f"Failed to load disk cache for {key}: {e}")

        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_hours: Optional[float] = None,
    ) -> None:
        """
        Set value in cache.

        Stores in both memory and disk (if enabled).
        """
        expires_at = None
        if ttl_hours is not None:
            expires_at = datetime.now() + timedelta(hours=ttl_hours)

        entry = CacheEntry(value, expires_at)

        # Set in memory
        self._set_memory(key, entry)

        # Set on disk
        if self.enable_disk_cache:
            try:
                disk_path = self._get_disk_path(key)
                with open(disk_path, "wb") as f:
                    pickle.dump(entry, f)
            except Exception as e:
                logger.debug(f"Failed to write disk cache for {key}: {e}")

    def _set_memory(self, key: str, entry: CacheEntry) -> None:
        """Set value in memory cache with LRU eviction."""
        with self._lock:
            # Evict if at capacity
            while len(self._memory_cache) >= self.max_memory_items:
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    if oldest_key in self._memory_cache:
                        del self._memory_cache[oldest_key]
                else:
                    break

            self._memory_cache[key] = entry

            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    def delete(self, key: str) -> None:
        """Delete entry from cache."""
        with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

        if self.enable_disk_cache:
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                disk_path.unlink()

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            self._access_order.clear()

        if self.enable_disk_cache:
            for path in self.cache_dir.glob("*.pkl"):
                try:
                    path.unlink()
                except Exception:
                    pass

    def clear_expired(self) -> int:
        """Clear expired entries. Returns count of cleared entries."""
        cleared = 0

        # Clear memory
        with self._lock:
            expired_keys = [
                key
                for key, entry in self._memory_cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._memory_cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                cleared += 1

        # Clear disk
        if self.enable_disk_cache:
            for path in self.cache_dir.glob("*.pkl"):
                try:
                    with open(path, "rb") as f:
                        entry = pickle.load(f)
                    if entry.is_expired():
                        path.unlink()
                        cleared += 1
                except Exception:
                    pass

        return cleared

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            memory_count = len(self._memory_cache)

        disk_count = 0
        disk_size = 0
        if self.enable_disk_cache:
            for path in self.cache_dir.glob("*.pkl"):
                disk_count += 1
                disk_size += path.stat().st_size

        return {
            "memory_entries": memory_count,
            "memory_max": self.max_memory_items,
            "disk_entries": disk_count,
            "disk_size_mb": disk_size / (1024 * 1024),
            "disk_enabled": self.enable_disk_cache,
        }
