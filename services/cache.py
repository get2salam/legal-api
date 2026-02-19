"""
In-memory LRU cache with TTL for search results.

Provides a lightweight, async-compatible caching layer that avoids
hitting the database for repeated queries within a configurable time
window.  Designed for read-heavy legal search workloads where the
same query patterns recur frequently.

Features
--------
- Thread-safe LRU eviction with configurable max entries.
- Per-entry TTL — stale results are lazily purged on access.
- Deterministic cache keys derived from normalised query parameters.
- Hit/miss/eviction counters for observability.
- Manual ``invalidate`` and ``clear`` methods.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class CacheEntry:
    """Single cached value with metadata."""

    value: Any
    created_at: float
    ttl: float
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) >= self.ttl


@dataclass
class CacheStats:
    """Counters exposed via the ``/cache/stats`` endpoint."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    def to_dict(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "current_size": self.current_size,
            "max_size": self.max_size,
        }


class SearchCache:
    """
    LRU cache with TTL for search query results.

    Parameters
    ----------
    max_size : int
        Maximum number of cached entries before eviction.
    default_ttl : float
        Default time-to-live in seconds for each entry.
    """

    def __init__(self, max_size: int = 256, default_ttl: float = 300.0) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._stats = CacheStats(max_size=max_size)

    # ── Public API ────────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value by *key*, or ``None`` on miss/expiry."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry.is_expired:
                del self._store[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.current_size = len(self._store)
                return None
            # Move to end (most-recently used)
            self._store.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Insert or update *key* with *value*."""
        ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            if key in self._store:
                # Update existing — refresh timestamp
                self._store[key] = CacheEntry(value=value, created_at=time.monotonic(), ttl=ttl)
                self._store.move_to_end(key)
            else:
                # Evict oldest if at capacity
                while len(self._store) >= self._max_size:
                    self._store.popitem(last=False)
                    self._stats.evictions += 1
                self._store[key] = CacheEntry(value=value, created_at=time.monotonic(), ttl=ttl)
            self._stats.current_size = len(self._store)

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry.  Returns ``True`` if the key existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._stats.current_size = len(self._store)
                return True
            return False

    def clear(self) -> int:
        """Flush every entry.  Returns the number of entries removed."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._stats.current_size = 0
            return count

    def stats(self) -> CacheStats:
        """Return a *snapshot* of current counters."""
        with self._lock:
            self._stats.current_size = len(self._store)
            # Return a copy so callers don't hold the lock
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                current_size=self._stats.current_size,
                max_size=self._stats.max_size,
            )

    def purge_expired(self) -> int:
        """Eagerly remove all expired entries.  Returns count removed."""
        with self._lock:
            expired_keys = [k for k, v in self._store.items() if v.is_expired]
            for k in expired_keys:
                del self._store[k]
            self._stats.evictions += len(expired_keys)
            self._stats.current_size = len(self._store)
            return len(expired_keys)

    # ── Key helpers ───────────────────────────────────────────────────────

    @staticmethod
    def make_key(**params: Any) -> str:
        """
        Build a deterministic cache key from query parameters.

        Keys are SHA-256 hashes of the sorted, JSON-serialised parameter
        dict — safe, fixed-length, and collision-resistant.
        """
        # Drop None values so missing optional filters don't change the key
        cleaned = {k: v for k, v in sorted(params.items()) if v is not None}
        raw = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    # ── Dunder helpers ────────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            entry = self._store.get(key)
            return not (entry is None or entry.is_expired)

    def __repr__(self) -> str:
        return (
            f"SearchCache(max_size={self._max_size}, current={len(self)}, ttl={self._default_ttl}s)"
        )


# ── Module-level singleton ────────────────────────────────────────────────────

#: Global search cache instance used across the application.
search_cache = SearchCache(max_size=512, default_ttl=300.0)
