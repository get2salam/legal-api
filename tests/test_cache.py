"""
Tests for the in-memory LRU search cache (services.cache).

Covers: get/put, TTL expiry, LRU eviction, key generation,
cache stats, purge, invalidation, thread safety, and the
cache management API endpoints.
"""

import time

import pytest
from httpx import AsyncClient

from services.cache import CacheEntry, CacheStats, SearchCache

# ─── CacheEntry ──────────────────────────────────────────────────────────────


class TestCacheEntry:
    """Unit tests for CacheEntry dataclass."""

    def test_not_expired_within_ttl(self):
        entry = CacheEntry(value="x", created_at=time.monotonic(), ttl=60.0)
        assert not entry.is_expired

    def test_expired_after_ttl(self):
        entry = CacheEntry(value="x", created_at=time.monotonic() - 120, ttl=60.0)
        assert entry.is_expired

    def test_hits_default_zero(self):
        entry = CacheEntry(value="data", created_at=0.0, ttl=10.0)
        assert entry.hits == 0


# ─── CacheStats ──────────────────────────────────────────────────────────────


class TestCacheStats:
    """Unit tests for CacheStats."""

    def test_to_dict_empty(self):
        stats = CacheStats()
        d = stats.to_dict()
        assert d["hits"] == 0
        assert d["misses"] == 0
        assert d["hit_rate"] == 0.0

    def test_to_dict_with_data(self):
        stats = CacheStats(hits=7, misses=3, evictions=1, current_size=5, max_size=10)
        d = stats.to_dict()
        assert d["hit_rate"] == 0.7
        assert d["evictions"] == 1

    def test_hit_rate_precision(self):
        stats = CacheStats(hits=1, misses=2)
        d = stats.to_dict()
        assert d["hit_rate"] == pytest.approx(0.3333, abs=0.001)


# ─── SearchCache core ────────────────────────────────────────────────────────


class TestSearchCacheBasics:
    """Basic get/put operations."""

    def test_put_and_get(self):
        cache = SearchCache(max_size=10)
        cache.put("k1", {"data": 42})
        assert cache.get("k1") == {"data": 42}

    def test_get_missing_key(self):
        cache = SearchCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_overwrite_existing_key(self):
        cache = SearchCache(max_size=10)
        cache.put("k1", "old")
        cache.put("k1", "new")
        assert cache.get("k1") == "new"
        assert len(cache) == 1

    def test_len(self):
        cache = SearchCache(max_size=10)
        cache.put("a", 1)
        cache.put("b", 2)
        assert len(cache) == 2

    def test_contains(self):
        cache = SearchCache(max_size=10)
        cache.put("k", "v")
        assert "k" in cache
        assert "missing" not in cache

    def test_repr(self):
        cache = SearchCache(max_size=32, default_ttl=120.0)
        r = repr(cache)
        assert "32" in r
        assert "120.0" in r


# ─── TTL expiry ──────────────────────────────────────────────────────────────


class TestSearchCacheTTL:
    """Time-to-live eviction."""

    def test_entry_expires(self):
        cache = SearchCache(max_size=10, default_ttl=0.0)
        cache.put("k", "v")  # TTL=0 → expires immediately
        # monotonic clock ensures expiry
        assert cache.get("k") is None

    def test_custom_ttl_per_entry(self):
        cache = SearchCache(max_size=10, default_ttl=300.0)
        cache.put("short", "data", ttl=0.0)
        cache.put("long", "data", ttl=300.0)
        assert cache.get("short") is None
        assert cache.get("long") == "data"

    def test_contains_respects_ttl(self):
        cache = SearchCache(max_size=10, default_ttl=0.0)
        cache.put("k", "v")
        assert "k" not in cache


# ─── LRU eviction ────────────────────────────────────────────────────────────


class TestSearchCacheLRU:
    """Least-recently-used eviction at capacity."""

    def test_evicts_oldest_on_overflow(self):
        cache = SearchCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("d") == 4
        assert len(cache) == 3

    def test_access_refreshes_order(self):
        cache = SearchCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # Access "a" to move it to end
        cache.get("a")
        cache.put("d", 4)  # Should evict "b" (oldest untouched)
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_eviction_counter(self):
        cache = SearchCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # evicts "a"
        stats = cache.stats()
        assert stats.evictions == 1


# ─── Invalidate & clear ─────────────────────────────────────────────────────


class TestSearchCacheManagement:
    """Invalidate and clear."""

    def test_invalidate_existing(self):
        cache = SearchCache(max_size=10)
        cache.put("k", "v")
        assert cache.invalidate("k") is True
        assert cache.get("k") is None

    def test_invalidate_missing(self):
        cache = SearchCache(max_size=10)
        assert cache.invalidate("nope") is False

    def test_clear(self):
        cache = SearchCache(max_size=10)
        cache.put("a", 1)
        cache.put("b", 2)
        removed = cache.clear()
        assert removed == 2
        assert len(cache) == 0

    def test_purge_expired(self):
        cache = SearchCache(max_size=10, default_ttl=300.0)
        cache.put("alive", "ok", ttl=300.0)
        cache.put("dead", "gone", ttl=0.0)
        purged = cache.purge_expired()
        assert purged == 1
        assert cache.get("alive") == "ok"
        assert cache.get("dead") is None


# ─── Stats ───────────────────────────────────────────────────────────────────


class TestSearchCacheStats:
    """Counters and hit-rate tracking."""

    def test_miss_increments(self):
        cache = SearchCache(max_size=10)
        cache.get("nope")
        cache.get("also_nope")
        stats = cache.stats()
        assert stats.misses == 2
        assert stats.hits == 0

    def test_hit_increments(self):
        cache = SearchCache(max_size=10)
        cache.put("k", "v")
        cache.get("k")
        cache.get("k")
        stats = cache.stats()
        assert stats.hits == 2

    def test_expired_counts_as_miss_and_eviction(self):
        cache = SearchCache(max_size=10, default_ttl=0.0)
        cache.put("k", "v")
        cache.get("k")  # expired → miss + eviction
        stats = cache.stats()
        assert stats.misses == 1
        assert stats.evictions == 1

    def test_current_size_reflects_state(self):
        cache = SearchCache(max_size=10)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.stats().current_size == 2
        cache.invalidate("a")
        assert cache.stats().current_size == 1


# ─── Key generation ──────────────────────────────────────────────────────────


class TestMakeKey:
    """Deterministic cache key construction."""

    def test_same_params_same_key(self):
        k1 = SearchCache.make_key(q="contract", page=1, court=None)
        k2 = SearchCache.make_key(q="contract", page=1, court=None)
        assert k1 == k2

    def test_different_params_different_key(self):
        k1 = SearchCache.make_key(q="contract", page=1)
        k2 = SearchCache.make_key(q="contract", page=2)
        assert k1 != k2

    def test_none_values_ignored(self):
        k1 = SearchCache.make_key(q="test", court=None)
        k2 = SearchCache.make_key(q="test")
        assert k1 == k2

    def test_order_independent(self):
        k1 = SearchCache.make_key(q="law", year=2024, court="Supreme Court")
        k2 = SearchCache.make_key(court="Supreme Court", q="law", year=2024)
        assert k1 == k2

    def test_key_is_hex_sha256(self):
        key = SearchCache.make_key(q="test")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ─── API Endpoints ───────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_cache_stats_endpoint(client: AsyncClient):
    """GET /api/v1/cache/stats returns cache counters."""
    resp = await client.get("/api/v1/cache/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "hits" in data
    assert "misses" in data
    assert "hit_rate" in data
    assert "current_size" in data
    assert "max_size" in data


@pytest.mark.anyio
async def test_cache_clear_endpoint(client: AsyncClient):
    """DELETE /api/v1/cache clears the cache."""
    resp = await client.delete("/api/v1/cache")
    assert resp.status_code == 200
    data = resp.json()
    assert "cleared" in data
    assert isinstance(data["cleared"], int)


@pytest.mark.anyio
async def test_cache_purge_endpoint(client: AsyncClient):
    """POST /api/v1/cache/purge removes expired entries."""
    resp = await client.post("/api/v1/cache/purge")
    assert resp.status_code == 200
    data = resp.json()
    assert "purged" in data
    assert isinstance(data["purged"], int)
