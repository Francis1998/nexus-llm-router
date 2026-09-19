"""Tests for TenantConcurrencySlotGuard ok/near/full bands."""

from __future__ import annotations

import pytest

from safety.tenant_concurrency import (
    TenantConcurrencyExceededError,
    TenantConcurrencySlotGuard,
)


def test_ok_band() -> None:
    guard = TenantConcurrencySlotGuard(max_slots=4, near_ratio=0.75)
    snap = guard.acquire("tenant-gpt-5.5")
    assert snap.band == "ok"
    assert snap.in_flight == 1
    assert snap.blocked is False


def test_near_and_full_bands() -> None:
    guard = TenantConcurrencySlotGuard(max_slots=4, near_ratio=0.75)
    guard.acquire("t1")
    guard.acquire("t1")
    near = guard.acquire("t1")
    assert near.band == "near"
    full = guard.acquire("t1")
    assert full.band == "full"
    assert full.blocked is True
    assert full.remaining == 0


def test_hard_gate_raises() -> None:
    guard = TenantConcurrencySlotGuard(max_slots=1, hard_gate=True)
    guard.acquire("t-hard")
    with pytest.raises(TenantConcurrencyExceededError):
        guard.acquire("t-hard")


def test_release_decrements() -> None:
    guard = TenantConcurrencySlotGuard(max_slots=2)
    guard.acquire("t2")
    guard.acquire("t2")
    snap = guard.release("t2")
    assert snap is not None
    assert snap.in_flight == 1
    assert snap.band == "ok"


def test_empty_tenant_raises() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        TenantConcurrencySlotGuard().acquire("")
