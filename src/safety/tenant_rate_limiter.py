"""Hard per-tenant rate limiter (token bucket keyed by tenant_id)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


class TenantRateLimitExceededError(RuntimeError):
    """Raised when a tenant exceeds its request rate limit."""


@dataclass
class _TenantBucket:
    """Mutable token-bucket state for one tenant."""

    tokens: float
    updated_at: float


class TenantRateLimiter:
    """Enforce hard per-tenant token-bucket rate limits.

    Distinct from ``TokenBucketRateLimiter`` (keyed by ``api_key_id``) and from
    the ``token-bucket-tenant`` *routing* strategy (which soft-sheds to a cheaper
    model). This limiter hard-rejects excess GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2 traffic for a tenant before dispatch.
    """

    def __init__(self, capacity: int, refill_per_second: float) -> None:
        """Initialize per-tenant token-bucket settings.

        Args:
            capacity: Maximum tokens per tenant bucket.
            refill_per_second: Token refill rate for each tenant.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be > 0")
        self._capacity = float(capacity)
        self._refill_per_second = float(refill_per_second)
        self._buckets: dict[str, _TenantBucket] = {}
        self._lock = threading.Lock()

    @property
    def capacity(self) -> float:
        """Return the configured per-tenant bucket capacity."""
        return self._capacity

    @property
    def refill_per_second(self) -> float:
        """Return the configured refill rate."""
        return self._refill_per_second

    def _refill_locked(self, tenant_id: str, now: float) -> _TenantBucket:
        bucket = self._buckets.setdefault(
            tenant_id,
            _TenantBucket(tokens=self._capacity, updated_at=now),
        )
        elapsed_seconds = max(0.0, now - bucket.updated_at)
        bucket.tokens = min(
            self._capacity,
            bucket.tokens + elapsed_seconds * self._refill_per_second,
        )
        bucket.updated_at = now
        return bucket

    def assert_allowed(self, tenant_id: str, tokens: int = 1) -> None:
        """Consume tokens or raise when the tenant is rate limited.

        Args:
            tenant_id: Tenant / customer identifier.
            tokens: Tokens to consume for this request.

        Raises:
            TenantRateLimitExceededError: If the bucket has insufficient tokens.
            ValueError: If ``tenant_id`` is empty or ``tokens`` is non-positive.
        """
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if tokens <= 0:
            raise ValueError("tokens must be > 0")
        now = time.monotonic()
        with self._lock:
            bucket = self._refill_locked(tenant_id, now)
            if bucket.tokens < tokens:
                raise TenantRateLimitExceededError(f"tenant rate limit exceeded for {tenant_id}")
            bucket.tokens -= tokens

    def remaining(self, tenant_id: str) -> float:
        """Return the number of tokens currently available for ``tenant_id``.

        Args:
            tenant_id: Tenant / customer identifier.

        Returns:
            Available tokens after applying refill (does not consume).
        """
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        now = time.monotonic()
        with self._lock:
            bucket = self._refill_locked(tenant_id, now)
            return float(bucket.tokens)
