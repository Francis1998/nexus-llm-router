"""Tests for RequestFingerprintDeduper identical-request window dedup."""

from __future__ import annotations

import pytest

from safety.request_fingerprint import FingerprintObservation, RequestFingerprintDeduper


class _Clock:
    """Injectable monotonic clock."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_fingerprint_stable_for_identical_payload() -> None:
    """Same model/messages/temperature yield the same fingerprint."""
    deduper = RequestFingerprintDeduper(window_seconds=30.0)
    messages = [{"role": "user", "content": "summarize GPT-5.5 vs Claude Sonnet 4.6"}]
    first = deduper.fingerprint(
        model="gpt-5.5",
        messages=messages,
        temperature=0.2,
        tenant="acme",
    )
    second = deduper.fingerprint(
        model="gpt-5.5",
        messages=messages,
        temperature=0.2,
        tenant="acme",
    )
    assert first == second
    assert len(first) == 64


def test_observe_flags_duplicate_within_window() -> None:
    """Second observe of the same fingerprint inside the window is a duplicate."""
    clock = _Clock(100.0)
    deduper = RequestFingerprintDeduper(window_seconds=10.0, clock=clock)
    fp = deduper.fingerprint(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
    )
    first = deduper.observe(fp)
    assert isinstance(first, FingerprintObservation)
    assert first.is_duplicate is False
    assert first.hits_in_window == 1
    clock.now = 105.0
    second = deduper.observe(fp)
    assert second.is_duplicate is True
    assert second.hits_in_window == 2
    assert second.age_seconds == pytest.approx(5.0)
    assert second.prior_seen_at == pytest.approx(100.0)


def test_window_expiry_clears_duplicate_state() -> None:
    """After the window elapses, the fingerprint is treated as new again."""
    clock = _Clock(0.0)
    deduper = RequestFingerprintDeduper(window_seconds=5.0, clock=clock)
    fp = deduper.fingerprint(
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "ping"}],
        tenant="t1",
    )
    assert deduper.observe(fp).is_duplicate is False
    clock.now = 6.0
    again = deduper.observe(fp)
    assert again.is_duplicate is False
    assert again.hits_in_window == 1


def test_different_tenants_or_models_do_not_collide() -> None:
    """Tenant and model participate in the fingerprint namespace."""
    deduper = RequestFingerprintDeduper(window_seconds=60.0)
    messages = [{"role": "user", "content": "shared prompt"}]
    a = deduper.fingerprint(model="kimi-k2", messages=messages, tenant="a")
    b = deduper.fingerprint(model="kimi-k2", messages=messages, tenant="b")
    c = deduper.fingerprint(model="gpt-5.5", messages=messages, tenant="a")
    assert a != b
    assert a != c
    assert deduper.observe(a).is_duplicate is False
    assert deduper.observe(b).is_duplicate is False
    assert deduper.observe(c).is_duplicate is False


def test_size_clear_and_rejects_invalid_inputs() -> None:
    """size/clear work; invalid window/fingerprint/messages raise ValueError."""
    with pytest.raises(ValueError, match="window_seconds"):
        RequestFingerprintDeduper(window_seconds=0)
    clock = _Clock(1.0)
    deduper = RequestFingerprintDeduper(window_seconds=2.0, clock=clock)
    with pytest.raises(ValueError, match="model"):
        deduper.fingerprint(model="", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(ValueError, match="messages"):
        deduper.fingerprint(model="gpt-5.5", messages=[])
    with pytest.raises(ValueError, match="fingerprint"):
        deduper.observe("")
    fp = deduper.fingerprint(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "x"}],
    )
    deduper.observe(fp)
    assert deduper.size() == 1
    deduper.clear()
    assert deduper.size() == 0
