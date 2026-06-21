"""Token-bucket rate limiter."""

import time
from dataclasses import dataclass


class RateLimitExceededError(RuntimeError):
    """Raised when an API key exceeds its rate limit."""


@dataclass
class Bucket:
    """Mutable token bucket state."""

    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    """Enforce per-key token-bucket rate limits."""

    def __init__(self, capacity: int, refill_per_second: float) -> None:
        """Initialize token bucket settings.

        Args:
            capacity: Maximum tokens per bucket.
            refill_per_second: Token refill rate.
        """
        self._capacity = float(capacity)
        self._refill_per_second = refill_per_second
        self._buckets: dict[str, Bucket] = {}

    def assert_allowed(self, api_key_id: str, tokens: int = 1) -> None:
        """Consume tokens or raise when the key is rate limited.

        Args:
            api_key_id: API key identifier.
            tokens: Tokens to consume.

        Raises:
            RateLimitExceededError: If the bucket has insufficient tokens.
        """
        now = time.monotonic()
        bucket = self._buckets.setdefault(api_key_id, Bucket(tokens=self._capacity, updated_at=now))
        elapsed_seconds = max(0.0, now - bucket.updated_at)
        bucket.tokens = min(
            self._capacity,
            bucket.tokens + elapsed_seconds * self._refill_per_second,
        )
        bucket.updated_at = now
        if bucket.tokens < tokens:
            raise RateLimitExceededError(f"rate limit exceeded for {api_key_id}")
        bucket.tokens -= tokens
