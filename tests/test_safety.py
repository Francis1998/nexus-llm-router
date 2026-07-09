"""Tests for safety controls."""

import pytest

from safety.budget import BudgetExceededError, BudgetGuardrail
from safety.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError
from safety.pii import PiiScrubber
from safety.rate_limiter import RateLimitExceededError, TokenBucketRateLimiter


def test_budget_guardrail_hard_rejects_excess_spend() -> None:
    """Budget guardrail should reject spend above the configured cap."""
    guardrail = BudgetGuardrail(cap_usd=1.0)
    guardrail.record_spend("user-a", 0.75)
    with pytest.raises(BudgetExceededError):
        guardrail.assert_can_spend("user-a", 0.30)


def test_circuit_breaker_opens_after_three_failures() -> None:
    """Circuit breaker should open after three consecutive failures."""
    circuit_breakers = CircuitBreakerRegistry(failure_threshold=3, recovery_window_seconds=60.0)
    circuit_breakers.record_failure("openai")
    circuit_breakers.record_failure("openai")
    circuit_breakers.record_failure("openai")
    with pytest.raises(CircuitOpenError):
        circuit_breakers.assert_available("openai")


def test_circuit_breaker_is_available_reports_open_state_without_mutation() -> None:
    """``is_available`` reflects an open circuit and never resets its state."""
    circuit_breakers = CircuitBreakerRegistry(failure_threshold=2, recovery_window_seconds=60.0)

    assert circuit_breakers.is_available("openai") is True

    circuit_breakers.record_failure("openai")
    circuit_breakers.record_failure("openai")

    assert circuit_breakers.is_available("openai") is False
    # The read must not recover the circuit, so assert_available still trips.
    with pytest.raises(CircuitOpenError):
        circuit_breakers.assert_available("openai")


def test_pii_scrubber_redacts_email_and_phone() -> None:
    """PII scrubber should redact email addresses and US phone numbers."""
    scrubber = PiiScrubber(enabled=True)
    redacted = scrubber.scrub_text("Email jane@example.com or call 415-555-1212.")
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_pii_scrubber_fully_redacts_parenthesized_and_prefixed_phone() -> None:
    """Numbers starting with ``(`` or ``+`` must be redacted whole, with no leak.

    A leading ``\\b`` could not match before the ``(`` of ``(415) 555-1234`` or
    the ``+`` of ``+1 415 555 1234``, so the redaction started mid-number and
    left the ``(``/``+`` dangling outside the placeholder. The whole number,
    including its leading boundary character, must be replaced.
    """
    scrubber = PiiScrubber(enabled=True)

    assert scrubber.scrub_text("call (415) 555-1234 now") == "call [REDACTED_PHONE] now"
    assert scrubber.scrub_text("intl +1 415 555 1234 line") == "intl [REDACTED_PHONE] line"


def test_rate_limiter_rejects_empty_bucket() -> None:
    """Token bucket should reject when insufficient tokens remain."""
    limiter = TokenBucketRateLimiter(capacity=1, refill_per_second=0.0)
    limiter.assert_allowed("key-a")
    with pytest.raises(RateLimitExceededError):
        limiter.assert_allowed("key-a")
