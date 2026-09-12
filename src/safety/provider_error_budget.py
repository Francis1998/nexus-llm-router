"""Sliding-window provider error-budget shed (advisory / hard gate)."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderErrorBudgetSnapshot:
    """Rolling error-budget view for one provider."""

    provider: str
    successes: int
    failures: int
    error_rate: float
    shed: bool


class ProviderErrorBudgetShed:
    """Sliding-window error-budget % shed for provider routing decisions.

    Closes the LiteLLM / OpenRouter error-budget gap as a reusable **safety**
    library (not buried in ``strategies.py``): record successes/failures per
    provider and ask ``should_shed(provider)`` when the rolling error rate
    exceeds the configured budget.

    Distinct from ``CircuitBreakerRegistry`` (consecutive-failure open/close)
    and from the ``provider-error-budget-shed`` *routing strategy* (which reads
    shared engine ``SuccessStats``). This module owns its own sliding window
    and can act as an advisory flag or a hard pre-dispatch gate for
    GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.
    """

    def __init__(
        self,
        *,
        window_size: int = 100,
        error_budget_rate: float = 0.15,
        min_samples: int = 5,
        hard_gate: bool = False,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize an empty per-provider error-budget tracker.

        Args:
            window_size: Maximum outcome samples retained per provider (``>= 1``).
            error_budget_rate: Maximum acceptable rolling error rate in
                ``[0.0, 1.0]``. Shed when ``error_rate > error_budget_rate``.
            min_samples: Minimum observations before shedding can trigger
                (cold-start protection; ``>= 1``).
            hard_gate: When ``True``, ``assert_within_budget`` raises if shed.
            clock: Injectable clock (defaults to ``time.monotonic``).

        Raises:
            ValueError: If window, budget, or min_samples are invalid.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 <= error_budget_rate <= 1.0:
            raise ValueError("error_budget_rate must be within [0.0, 1.0]")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        self._window_size = int(window_size)
        self._error_budget_rate = float(error_budget_rate)
        self._min_samples = int(min_samples)
        self._hard_gate = bool(hard_gate)
        self._clock = clock or time.monotonic
        self._outcomes: dict[str, deque[bool]] = {}
        self._lock = threading.Lock()

    @property
    def window_size(self) -> int:
        """Return the configured per-provider sample window size."""
        return self._window_size

    @property
    def error_budget_rate(self) -> float:
        """Return the configured maximum acceptable error rate."""
        return self._error_budget_rate

    @property
    def min_samples(self) -> int:
        """Return the cold-start sample floor before shedding."""
        return self._min_samples

    @property
    def hard_gate(self) -> bool:
        """Return whether ``assert_within_budget`` raises on shed."""
        return self._hard_gate

    def _snapshot_locked(self, provider: str, window: deque[bool]) -> ProviderErrorBudgetSnapshot:
        successes = sum(1 for ok in window if ok)
        failures = len(window) - successes
        total = len(window)
        error_rate = (failures / total) if total else 0.0
        shed = total >= self._min_samples and error_rate > self._error_budget_rate
        return ProviderErrorBudgetSnapshot(
            provider=provider,
            successes=successes,
            failures=failures,
            error_rate=error_rate,
            shed=shed,
        )

    def _record(self, provider: str, success: bool) -> ProviderErrorBudgetSnapshot:
        if not provider:
            raise ValueError("provider must be non-empty")
        _ = float(self._clock())
        with self._lock:
            window = self._outcomes.setdefault(provider, deque(maxlen=self._window_size))
            window.append(bool(success))
            return self._snapshot_locked(provider, window)

    def record_success(self, provider: str) -> ProviderErrorBudgetSnapshot:
        """Record a successful provider call and return the updated snapshot.

        Args:
            provider: Provider name (for example ``openai``).

        Returns:
            Updated ``ProviderErrorBudgetSnapshot``.

        Raises:
            ValueError: If ``provider`` is empty.
        """
        return self._record(provider, True)

    def record_failure(self, provider: str) -> ProviderErrorBudgetSnapshot:
        """Record a failed provider call and return the updated snapshot.

        Args:
            provider: Provider name.

        Returns:
            Updated ``ProviderErrorBudgetSnapshot``.

        Raises:
            ValueError: If ``provider`` is empty.
        """
        return self._record(provider, False)

    def snapshot(self, provider: str) -> ProviderErrorBudgetSnapshot | None:
        """Return the current snapshot for ``provider``, or ``None`` if unknown."""
        if not provider:
            raise ValueError("provider must be non-empty")
        with self._lock:
            window = self._outcomes.get(provider)
            if window is None or not window:
                return None
            return self._snapshot_locked(provider, window)

    def should_shed(self, provider: str) -> bool:
        """Return whether ``provider`` exceeds the error budget.

        Unknown / cold providers (fewer than ``min_samples``) never shed.

        Args:
            provider: Provider name.

        Returns:
            ``True`` when the rolling error rate exceeds the budget.
        """
        snap = self.snapshot(provider)
        return bool(snap is not None and snap.shed)

    def assert_within_budget(self, provider: str) -> ProviderErrorBudgetSnapshot | None:
        """Hard-gate helper: raise when ``hard_gate`` and provider should shed.

        Args:
            provider: Provider name.

        Returns:
            Current snapshot, or ``None`` when the provider is unknown.

        Raises:
            ProviderErrorBudgetExceededError: When ``hard_gate`` is enabled and
                ``should_shed`` is true.
            ValueError: If ``provider`` is empty.
        """
        snap = self.snapshot(provider)
        if self._hard_gate and snap is not None and snap.shed:
            raise ProviderErrorBudgetExceededError(snap)
        return snap

    def providers(self) -> list[str]:
        """Return known provider names in insertion order."""
        with self._lock:
            return list(self._outcomes.keys())

    def clear(self, provider: str | None = None) -> None:
        """Clear one provider window, or every window when omitted."""
        with self._lock:
            if provider is None:
                self._outcomes.clear()
                return
            self._outcomes.pop(provider, None)


class ProviderErrorBudgetExceededError(RuntimeError):
    """Raised by ``assert_within_budget`` when a hard gate sheds a provider."""

    def __init__(self, snapshot: ProviderErrorBudgetSnapshot) -> None:
        """Attach the exceeding snapshot to the exception.

        Args:
            snapshot: Error-budget state at the moment of rejection.
        """
        self.snapshot = snapshot
        super().__init__(
            f"provider error budget exceeded for {snapshot.provider}: "
            f"error_rate={snapshot.error_rate:.4f} "
            f"successes={snapshot.successes} failures={snapshot.failures}"
        )
