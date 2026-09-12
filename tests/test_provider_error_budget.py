"""Tests for ProviderErrorBudgetShed sliding-window error-budget gate."""

from __future__ import annotations

import pytest

from safety.provider_error_budget import (
    ProviderErrorBudgetExceededError,
    ProviderErrorBudgetShed,
    ProviderErrorBudgetSnapshot,
)


class _Clock:
    """Injectable monotonic clock."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_should_shed_when_error_rate_exceeds_budget() -> None:
    """Rolling error rate above the budget flips shed / should_shed."""
    shedder = ProviderErrorBudgetShed(
        window_size=10,
        error_budget_rate=0.20,
        min_samples=5,
    )
    for _ in range(4):
        shedder.record_success("openai")
    snap = shedder.record_failure("openai")
    assert isinstance(snap, ProviderErrorBudgetSnapshot)
    # 1/5 = 0.20 — shed only when strictly greater than budget.
    assert snap.error_rate == pytest.approx(0.20)
    assert snap.shed is False
    assert shedder.should_shed("openai") is False

    snap = shedder.record_failure("openai")
    # 2/6 ≈ 0.333 > 0.20
    assert snap.shed is True
    assert shedder.should_shed("openai") is True
    assert snap.successes == 4
    assert snap.failures == 2


def test_cold_start_below_min_samples_never_sheds() -> None:
    """Providers with fewer than min_samples stay routable."""
    shedder = ProviderErrorBudgetShed(
        window_size=20,
        error_budget_rate=0.10,
        min_samples=5,
    )
    for _ in range(4):
        shedder.record_failure("anthropic")
    snap = shedder.snapshot("anthropic")
    assert snap is not None
    assert snap.error_rate == 1.0
    assert snap.shed is False
    assert shedder.should_shed("anthropic") is False


def test_window_evicts_oldest_and_recovers() -> None:
    """Sliding window eviction can bring a provider back under budget."""
    shedder = ProviderErrorBudgetShed(
        window_size=4,
        error_budget_rate=0.25,
        min_samples=4,
    )
    for _ in range(4):
        shedder.record_failure("google")
    assert shedder.should_shed("google") is True
    for _ in range(4):
        shedder.record_success("google")
    snap = shedder.snapshot("google")
    assert snap is not None
    assert snap.successes == 4
    assert snap.failures == 0
    assert snap.shed is False


def test_providers_isolated_and_hard_gate() -> None:
    """Per-provider windows stay isolated; hard_gate raises when shed."""
    clock = _Clock(1.0)
    shedder = ProviderErrorBudgetShed(
        window_size=10,
        error_budget_rate=0.15,
        min_samples=3,
        hard_gate=True,
        clock=clock,
    )
    for _ in range(3):
        shedder.record_success("moonshot")
    assert shedder.should_shed("moonshot") is False
    for _ in range(3):
        shedder.record_failure("openai")
    assert shedder.should_shed("openai") is True
    assert shedder.should_shed("moonshot") is False
    with pytest.raises(ProviderErrorBudgetExceededError) as exc_info:
        shedder.assert_within_budget("openai")
    assert exc_info.value.snapshot.provider == "openai"
    assert shedder.assert_within_budget("moonshot") is not None
    assert shedder.providers() == ["moonshot", "openai"]
    shedder.clear("openai")
    assert shedder.should_shed("openai") is False
    shedder.clear()
    assert shedder.providers() == []


def test_rejects_invalid_inputs() -> None:
    """Construction and record validate window, budget, and provider."""
    with pytest.raises(ValueError, match="window_size"):
        ProviderErrorBudgetShed(window_size=0)
    with pytest.raises(ValueError, match="error_budget_rate"):
        ProviderErrorBudgetShed(error_budget_rate=1.5)
    with pytest.raises(ValueError, match="min_samples"):
        ProviderErrorBudgetShed(min_samples=0)
    shedder = ProviderErrorBudgetShed()
    with pytest.raises(ValueError, match="provider"):
        shedder.record_success("")
    with pytest.raises(ValueError, match="provider"):
        shedder.record_failure("")
    assert shedder.snapshot("missing") is None
