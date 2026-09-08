"""Tests for hard per-tenant rate limiter."""

from __future__ import annotations

import time

import pytest

from safety.tenant_rate_limiter import (
    TenantRateLimiter,
    TenantRateLimitExceededError,
)


def test_assert_allowed_consumes_tokens_then_raises() -> None:
    """Capacity tokens succeed; the next request hard-rejects."""
    limiter = TenantRateLimiter(capacity=2, refill_per_second=0.01)
    limiter.assert_allowed("acme")
    limiter.assert_allowed("acme")
    with pytest.raises(TenantRateLimitExceededError, match="acme"):
        limiter.assert_allowed("acme")


def test_tenants_are_isolated() -> None:
    """Exhausting one tenant does not affect another tenant's budget."""
    limiter = TenantRateLimiter(capacity=1, refill_per_second=0.01)
    limiter.assert_allowed("acme")
    with pytest.raises(TenantRateLimitExceededError):
        limiter.assert_allowed("acme")
    limiter.assert_allowed("globex")
    assert limiter.remaining("globex") == pytest.approx(0.0)


def test_remaining_reports_refill(monkeypatch: pytest.MonkeyPatch) -> None:
    """remaining reflects refill without consuming tokens."""
    limiter = TenantRateLimiter(capacity=10, refill_per_second=5.0)
    clock = {"now": 100.0}

    def fake_monotonic() -> float:
        return clock["now"]

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    limiter.assert_allowed("acme", tokens=4)
    assert limiter.remaining("acme") == pytest.approx(6.0)
    clock["now"] = 101.0  # +5 tokens from refill
    assert limiter.remaining("acme") == pytest.approx(10.0)  # capped at capacity


def test_refill_allows_traffic_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """After refill, assert_allowed succeeds again."""
    limiter = TenantRateLimiter(capacity=1, refill_per_second=1.0)
    clock = {"now": 50.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    limiter.assert_allowed("acme")
    with pytest.raises(TenantRateLimitExceededError):
        limiter.assert_allowed("acme")
    clock["now"] = 51.5
    limiter.assert_allowed("acme")


def test_rejects_empty_tenant_and_nonpositive_tokens() -> None:
    """Invalid arguments raise ValueError."""
    limiter = TenantRateLimiter(capacity=5, refill_per_second=1.0)
    with pytest.raises(ValueError, match="tenant_id"):
        limiter.assert_allowed("")
    with pytest.raises(ValueError, match="tokens"):
        limiter.assert_allowed("acme", tokens=0)
    with pytest.raises(ValueError, match="capacity"):
        TenantRateLimiter(capacity=0, refill_per_second=1.0)


def test_distinct_from_api_key_limiter_import() -> None:
    """TenantRateLimiter is a separate class from TokenBucketRateLimiter."""
    from safety.rate_limiter import TokenBucketRateLimiter

    assert TenantRateLimiter is not TokenBucketRateLimiter
    tenant = TenantRateLimiter(capacity=3, refill_per_second=1.0)
    api_key = TokenBucketRateLimiter(capacity=3, refill_per_second=1.0)
    tenant.assert_allowed("tenant-a")
    api_key.assert_allowed("api-key-a")
    assert tenant.remaining("tenant-a") == pytest.approx(2.0)
