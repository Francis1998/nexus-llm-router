"""Tests for JsonSchemaRetryBudgetGuard ok/advisory/exhausted retry budgets."""

from __future__ import annotations

import pytest

from safety.json_schema_retry import (
    JsonSchemaRetryBudgetExceededError,
    JsonSchemaRetryBudgetGuard,
    JsonSchemaRetrySnapshot,
)


def test_ok_band_before_advisory_limit() -> None:
    """First retries under advisory_limit stay ok and allow further retries."""
    guard = JsonSchemaRetryBudgetGuard(advisory_limit=2, hard_limit=5)
    snap = guard.record_retry("gpt-5.5-req")
    assert isinstance(snap, JsonSchemaRetrySnapshot)
    assert snap.band == "ok"
    assert snap.retries == 1
    assert snap.blocked is False
    assert guard.allow_retry("gpt-5.5-req") is True
    assert "ok" in snap.advisory.lower()


def test_advisory_band_at_advisory_limit() -> None:
    """Retries at/above advisory_limit but under hard_limit are advisory."""
    guard = JsonSchemaRetryBudgetGuard(advisory_limit=2, hard_limit=5)
    guard.record_retry("claude-sonnet-4.6-req")
    snap = guard.record_retry("claude-sonnet-4.6-req")
    assert snap.band == "advisory"
    assert snap.retries == 2
    assert snap.blocked is False
    assert "advisory" in snap.advisory.lower()


def test_exhausted_band_at_hard_limit_with_hard_gate() -> None:
    """Retries at hard_limit exhaust the budget; hard_gate raises."""
    guard = JsonSchemaRetryBudgetGuard(advisory_limit=2, hard_limit=3, hard_gate=True)
    for _ in range(3):
        snap = guard.record_retry("gemini-3-req")
    assert snap.band == "exhausted"
    assert snap.blocked is True
    assert snap.remaining_hard == 0
    assert guard.allow_retry("gemini-3-req") is False
    with pytest.raises(JsonSchemaRetryBudgetExceededError):
        guard.assert_within_budget("gemini-3-req")


def test_frontier_models_share_retry_budgets() -> None:
    """Same budgets apply across GPT-5.5 / Sonnet / Gemini / Kimi request ids."""
    guard = JsonSchemaRetryBudgetGuard(advisory_limit=1, hard_limit=3)
    for request_id in (
        "gpt-5.5",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "kimi-k2",
    ):
        snap = guard.record_retry(request_id)
        assert snap.band == "advisory"
        assert snap.request_id == request_id
        assert guard.remaining(request_id) == 2


def test_rejects_invalid_inputs_and_clear() -> None:
    """Constructor validates limits; clear resets counters; empty ids rejected."""
    with pytest.raises(ValueError, match="advisory_limit"):
        JsonSchemaRetryBudgetGuard(advisory_limit=0, hard_limit=5)
    with pytest.raises(ValueError, match="hard_limit"):
        JsonSchemaRetryBudgetGuard(advisory_limit=3, hard_limit=3)
    guard = JsonSchemaRetryBudgetGuard(advisory_limit=2, hard_limit=4)
    with pytest.raises(ValueError, match="request_id"):
        guard.record_retry("")
    guard.record_retry("kimi-k2-req")
    assert guard.snapshot("kimi-k2-req") is not None
    guard.clear("kimi-k2-req")
    assert guard.snapshot("kimi-k2-req") is None
    assert guard.allow_retry("kimi-k2-req") is True
    # Distinct from OutputJsonSchemaGuard — this tracks retries, not JSON keys.
    assert "kimi-k2-req" not in guard.request_ids()
