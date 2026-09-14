"""In-memory identical-request fingerprint deduper (TTL window)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FingerprintObservation:
    """Result of observing one request fingerprint inside the dedup window."""

    fingerprint: str
    is_duplicate: bool
    prior_seen_at: float | None
    age_seconds: float | None
    hits_in_window: int


@dataclass
class _Entry:
    """Mutable window state for one fingerprint."""

    first_seen_at: float
    last_seen_at: float
    hits: int


class RequestFingerprintDeduper:
    """Detect identical request fingerprints inside a short TTL window.

    Closes the LiteLLM / Portkey / OpenRouter *content-hash* duplicate-burst
    gap for Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2: when clients retry or fan-out the same payload without an
    ``Idempotency-Key``, callers can ``fingerprint(...)`` the body and
    ``observe(...)`` to learn whether an identical hash already landed in the
    configured window.

    Distinct from ``IdempotencyStore`` (durable client-key → response replay).
    This module never stores response bodies and never rejects traffic —
    callers decide whether to coalesce, shed, or proceed on ``is_duplicate``.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 60.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize an empty in-memory fingerprint window.

        Args:
            window_seconds: Dedup window length in seconds (``> 0``).
            clock: Injectable monotonic clock (defaults to ``time.monotonic``).

        Raises:
            ValueError: If ``window_seconds`` is not strictly positive.
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._window_seconds = float(window_seconds)
        self._clock = clock or time.monotonic
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    @property
    def window_seconds(self) -> float:
        """Return the configured dedup window length in seconds."""
        return self._window_seconds

    @staticmethod
    def fingerprint(
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        temperature: float | None = None,
        tenant: str | None = None,
    ) -> str:
        """Return a stable SHA-256 hex digest for the request payload.

        Args:
            model: Model identifier (non-empty).
            messages: Chat messages (must be non-empty).
            temperature: Optional sampling temperature included in the hash.
            tenant: Optional tenant namespace included in the hash.

        Returns:
            64-character lowercase hex digest.

        Raises:
            ValueError: If ``model`` is empty or ``messages`` is empty.
        """
        if not model:
            raise ValueError("model must be non-empty")
        if not messages:
            raise ValueError("messages must be non-empty")
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "tenant": tenant,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _evict_expired_locked(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if (now - entry.last_seen_at) > self._window_seconds
        ]
        for key in expired:
            del self._entries[key]

    def observe(self, fingerprint: str) -> FingerprintObservation:
        """Record ``fingerprint`` and report whether it is a window duplicate.

        Expired entries are evicted before the lookup. The first sighting
        (or first after expiry) returns ``is_duplicate=False``; later
        sightings inside the window return ``is_duplicate=True``.

        Args:
            fingerprint: Hex digest from ``fingerprint(...)`` (non-empty).

        Returns:
            ``FingerprintObservation`` describing duplicate state and hit count.

        Raises:
            ValueError: If ``fingerprint`` is empty.
        """
        if not fingerprint:
            raise ValueError("fingerprint must be non-empty")
        now = float(self._clock())
        with self._lock:
            self._evict_expired_locked(now)
            existing = self._entries.get(fingerprint)
            if existing is None:
                self._entries[fingerprint] = _Entry(
                    first_seen_at=now,
                    last_seen_at=now,
                    hits=1,
                )
                return FingerprintObservation(
                    fingerprint=fingerprint,
                    is_duplicate=False,
                    prior_seen_at=None,
                    age_seconds=None,
                    hits_in_window=1,
                )
            prior = existing.first_seen_at
            existing.hits += 1
            existing.last_seen_at = now
            return FingerprintObservation(
                fingerprint=fingerprint,
                is_duplicate=True,
                prior_seen_at=prior,
                age_seconds=now - prior,
                hits_in_window=existing.hits,
            )

    def size(self) -> int:
        """Return the number of fingerprints currently inside the window."""
        now = float(self._clock())
        with self._lock:
            self._evict_expired_locked(now)
            return len(self._entries)

    def clear(self) -> None:
        """Drop every fingerprint from the in-memory window."""
        with self._lock:
            self._entries.clear()
