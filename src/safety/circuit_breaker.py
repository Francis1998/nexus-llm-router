"""Circuit breaker safety control."""

import time
from dataclasses import dataclass


class CircuitOpenError(RuntimeError):
    """Raised when a provider circuit is open."""


@dataclass
class CircuitState:
    """Mutable circuit state for one provider."""

    consecutive_failures: int = 0
    opened_at: float | None = None


class CircuitBreakerRegistry:
    """Manage per-provider circuit breakers."""

    def __init__(self, failure_threshold: int = 3, recovery_window_seconds: float = 60.0) -> None:
        """Initialize circuit breaker settings.

        Args:
            failure_threshold: Failures before opening a circuit.
            recovery_window_seconds: Seconds before a circuit may recover.
        """
        self._failure_threshold = failure_threshold
        self._recovery_window_seconds = recovery_window_seconds
        self._states: dict[str, CircuitState] = {}

    def assert_available(self, provider: str) -> None:
        """Raise if a provider circuit is open.

        Args:
            provider: Provider name.

        Raises:
            CircuitOpenError: If the provider circuit is open.
        """
        state = self._states.setdefault(provider, CircuitState())
        if state.opened_at is None:
            return
        elapsed_seconds = time.monotonic() - state.opened_at
        if elapsed_seconds < self._recovery_window_seconds:
            raise CircuitOpenError(f"provider circuit is open: {provider}")
        state.consecutive_failures = 0
        state.opened_at = None

    def record_success(self, provider: str) -> None:
        """Record a successful provider call.

        Args:
            provider: Provider name.
        """
        self._states[provider] = CircuitState()

    def record_failure(self, provider: str) -> None:
        """Record a failed provider call and open the circuit when needed.

        Args:
            provider: Provider name.
        """
        state = self._states.setdefault(provider, CircuitState())
        state.consecutive_failures += 1
        if state.consecutive_failures >= self._failure_threshold:
            state.opened_at = time.monotonic()
