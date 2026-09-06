"""In-memory exact-match response cache with TTL and optional tenant namespaces."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class _CacheEntry:
    """Stored cache value with absolute expiry time."""

    value: object
    expires_at: float


class ResponseCache:
    """Exact-match response cache keyed by model, messages, and temperature.

    Keys may be optionally namespaced by tenant so multi-tenant gateways do not
    share completions across customers. Entries expire after ``ttl_seconds``.
    """

    def __init__(self, ttl_seconds: float = 300.0, *, enabled: bool = True) -> None:
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for each entry in seconds.
            enabled: When False, get always misses and set is a no-op.
        """
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")
        self._ttl_seconds = float(ttl_seconds)
        self._enabled = enabled
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        """Return whether the cache accepts lookups and writes."""
        return self._enabled

    @property
    def ttl_seconds(self) -> float:
        """Return the configured TTL in seconds."""
        return self._ttl_seconds

    @staticmethod
    def make_key(
        model: str | None,
        messages: Sequence[object],
        temperature: float,
        tenant: str | None = None,
    ) -> str:
        """Build a stable SHA-256 cache key.

        Args:
            model: Requested model id (or None when the router will choose).
            messages: Chat messages as dicts or objects with role/content.
            temperature: Sampling temperature included in the fingerprint.
            tenant: Optional tenant namespace for isolation.

        Returns:
            Hex digest cache key.
        """
        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role", "user"))
                content = str(message.get("content", ""))
            else:
                role = str(getattr(message, "role", "user"))
                content = str(getattr(message, "content", ""))
            normalized_messages.append({"role": role, "content": content})
        payload = {
            "tenant": tenant or "",
            "model": model or "",
            "messages": normalized_messages,
            "temperature": float(temperature),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> object | None:
        """Return a cached value when present and unexpired.

        Args:
            key: Cache key from :meth:`make_key`.

        Returns:
            Cached value, or None on miss / disabled / expiry.
        """
        if not self._enabled:
            self.misses += 1
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                self.misses += 1
                return None
            self.hits += 1
            return entry.value

    def set(self, key: str, value: object) -> None:
        """Store a value under ``key`` with the configured TTL.

        Args:
            key: Cache key from :meth:`make_key`.
            value: Value to store (typically a RouterResponse).
        """
        if not self._enabled:
            return
        expires_at = time.monotonic() + self._ttl_seconds
        with self._lock:
            self._entries[key] = _CacheEntry(value=value, expires_at=expires_at)

    def clear(self) -> None:
        """Drop all entries and reset hit/miss counters."""
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        """Return the number of in-memory entries (including expired)."""
        with self._lock:
            return len(self._entries)
