"""Semantic fuzzy response cache using character-trigram Jaccard similarity."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FuzzyCacheHit:
    """A near-match cache hit with Jaccard similarity score."""

    key: str
    value: object
    similarity: float


@dataclass(slots=True)
class _FuzzyEntry:
    """Stored fuzzy-cache value with text fingerprint and expiry."""

    key: str
    namespace: str
    text: str
    trigrams: frozenset[str]
    value: object
    expires_at: float


def _normalize_messages(messages: Sequence[object]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
        else:
            role = str(getattr(message, "role", "user"))
            content = str(getattr(message, "content", ""))
        parts.append(f"{role}:{content}")
    return "\n".join(parts).strip().lower()


def _char_trigrams(text: str) -> frozenset[str]:
    padded = f"  {text}  "
    if len(padded) < 3:
        return frozenset({padded})
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


class SemanticFuzzyCache:
    """Near-duplicate response cache via character-trigram Jaccard similarity.

    Distinct from exact-match ``ResponseCache`` and from the ``semantic-cache``
    *routing* strategy. Closes the Portkey/LiteLLM semantic response-cache gap
    without an embeddings dependency.
    """

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        *,
        threshold: float = 0.92,
        max_entries: int = 256,
        enabled: bool = True,
    ) -> None:
        """Initialize the fuzzy cache.

        Args:
            ttl_seconds: Time-to-live for each entry in seconds.
            threshold: Minimum Jaccard similarity required for a hit.
            max_entries: Soft cap; oldest entries are dropped on overflow.
            enabled: When False, get always misses and set is a no-op.
        """
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._ttl_seconds = float(ttl_seconds)
        self._threshold = float(threshold)
        self._max_entries = int(max_entries)
        self._enabled = enabled
        self._entries: dict[str, _FuzzyEntry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        """Return whether the cache accepts lookups and writes."""
        return self._enabled

    @property
    def threshold(self) -> float:
        """Return the configured Jaccard similarity threshold."""
        return self._threshold

    @property
    def ttl_seconds(self) -> float:
        """Return the configured TTL in seconds."""
        return self._ttl_seconds

    @staticmethod
    def _namespace(tenant: str | None, model: str | None) -> str:
        return f"{tenant or ''}|{model or ''}"

    def _purge_expired_locked(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            del self._entries[key]

    def _evict_overflow_locked(self) -> None:
        while len(self._entries) > self._max_entries:
            oldest_key = min(self._entries, key=lambda k: self._entries[k].expires_at)
            del self._entries[oldest_key]

    def get(
        self,
        messages: Sequence[object],
        *,
        model: str | None = None,
        tenant: str | None = None,
    ) -> FuzzyCacheHit | None:
        """Return the best near-match hit at or above the similarity threshold.

        Args:
            messages: Chat messages as dicts or objects with role/content.
            model: Optional model namespace.
            tenant: Optional tenant namespace.

        Returns:
            ``FuzzyCacheHit`` on success, or None on miss / disabled / expiry.
        """
        if not self._enabled:
            self.misses += 1
            return None
        text = _normalize_messages(messages)
        trigrams = _char_trigrams(text)
        namespace = self._namespace(tenant, model)
        now = time.monotonic()
        best: FuzzyCacheHit | None = None
        with self._lock:
            self._purge_expired_locked(now)
            for entry in self._entries.values():
                if entry.namespace != namespace:
                    continue
                similarity = _jaccard(trigrams, entry.trigrams)
                if similarity < self._threshold:
                    continue
                if best is None or similarity > best.similarity:
                    best = FuzzyCacheHit(key=entry.key, value=entry.value, similarity=similarity)
            if best is None:
                self.misses += 1
                return None
            self.hits += 1
            return best

    def set(
        self,
        messages: Sequence[object],
        value: object,
        *,
        model: str | None = None,
        tenant: str | None = None,
    ) -> str:
        """Store a value under a fuzzy fingerprint and return the entry key.

        Args:
            messages: Chat messages used to build the trigram fingerprint.
            value: Value to store (typically a RouterResponse).
            model: Optional model namespace.
            tenant: Optional tenant namespace.

        Returns:
            Stable entry key string (empty when disabled).
        """
        if not self._enabled:
            return ""
        text = _normalize_messages(messages)
        trigrams = _char_trigrams(text)
        namespace = self._namespace(tenant, model)
        key = hashlib.sha256(f"{namespace}\n{text}\n{uuid.uuid4().hex}".encode()).hexdigest()
        expires_at = time.monotonic() + self._ttl_seconds
        entry = _FuzzyEntry(
            key=key,
            namespace=namespace,
            text=text,
            trigrams=trigrams,
            value=value,
            expires_at=expires_at,
        )
        with self._lock:
            self._purge_expired_locked(time.monotonic())
            self._entries[key] = entry
            self._evict_overflow_locked()
        return key

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
