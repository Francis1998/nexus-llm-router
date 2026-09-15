"""Advisory rolling provider health scoreboard (success/error bands)."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Literal

HealthBand = Literal["healthy", "degraded", "unhealthy"]


@dataclass(frozen=True, slots=True)
class ProviderHealthScoreAdvice:
    """Structured advisory health view for one provider."""

    provider: str
    successes: int
    failures: int
    samples: int
    success_rate: float
    error_rate: float
    health_score: float
    band: HealthBand
    advisory: str


class ProviderHealthScoreboard:
    """Classify providers into healthy / degraded / unhealthy bands.

    Closes the Portkey / Helicone / LiteLLM *provider health dashboard* gap
    for offline Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2: operators need a process-local rolling success /
    error scoreboard with advisory bands, not only fallback ranking.

    Distinct from ``ProviderFallbackScoreboard`` (EWMA score used to *rank*
    fallbacks inside ``NexusRouter``), ``ProviderFallbackChainPlanner``
    (static preference chains), ``FirstTokenLatencySloAdvisor`` (TTFT SLO),
    and ``CircuitBreakerRegistry`` (consecutive-failure open/close). This
    scoreboard never rejects traffic and never reorders routes — callers
    log or alert on ``band``.
    """

    def __init__(
        self,
        *,
        window_size: int = 100,
        healthy_min_success_rate: float = 0.95,
        degraded_min_success_rate: float = 0.80,
        min_samples: int = 5,
    ) -> None:
        """Initialize an empty per-provider rolling health window.

        Args:
            window_size: Max outcome samples retained per provider (``>= 1``).
            healthy_min_success_rate: Success-rate floor for ``healthy`` in
                ``(0.0, 1.0]``. Must be ``> degraded_min_success_rate``.
            degraded_min_success_rate: Success-rate floor for ``degraded`` in
                ``[0.0, 1.0)``. Below this band is ``unhealthy``.
            min_samples: Cold-start floor before bands leave ``healthy``
                (``>= 1``). Until then, ``band`` stays ``healthy``.

        Raises:
            ValueError: If window, rates, or min_samples are invalid.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 < healthy_min_success_rate <= 1.0:
            raise ValueError("healthy_min_success_rate must be in (0.0, 1.0]")
        if not 0.0 <= degraded_min_success_rate < 1.0:
            raise ValueError("degraded_min_success_rate must be in [0.0, 1.0)")
        if healthy_min_success_rate <= degraded_min_success_rate:
            raise ValueError("healthy_min_success_rate must be > degraded_min_success_rate")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        self._window_size = int(window_size)
        self._healthy_min = float(healthy_min_success_rate)
        self._degraded_min = float(degraded_min_success_rate)
        self._min_samples = int(min_samples)
        self._outcomes: dict[str, deque[bool]] = {}
        self._lock = threading.Lock()

    @property
    def window_size(self) -> int:
        """Return the configured per-provider sample window size."""
        return self._window_size

    @property
    def healthy_min_success_rate(self) -> float:
        """Return the success-rate floor for the healthy band."""
        return self._healthy_min

    @property
    def degraded_min_success_rate(self) -> float:
        """Return the success-rate floor for the degraded band."""
        return self._degraded_min

    @property
    def min_samples(self) -> int:
        """Return the cold-start sample floor before non-healthy bands."""
        return self._min_samples

    def _advise_locked(self, provider: str, window: deque[bool]) -> ProviderHealthScoreAdvice:
        successes = sum(1 for ok in window if ok)
        samples = len(window)
        failures = samples - successes
        success_rate = (successes / samples) if samples else 1.0
        error_rate = 1.0 - success_rate
        health_score = float(success_rate)
        if samples < self._min_samples:
            band: HealthBand = "healthy"
            advisory = (
                f"healthy: cold-start samples={samples} < min_samples="
                f"{self._min_samples}; health_score={health_score:.3f}"
            )
        elif success_rate >= self._healthy_min:
            band = "healthy"
            advisory = (
                f"healthy: success_rate={success_rate:.3f} >= "
                f"healthy_min={self._healthy_min:.3f} "
                f"(health_score={health_score:.3f})"
            )
        elif success_rate >= self._degraded_min:
            band = "degraded"
            advisory = (
                f"degraded: success_rate={success_rate:.3f} < "
                f"healthy_min={self._healthy_min:.3f} but >= "
                f"degraded_min={self._degraded_min:.3f} "
                f"(health_score={health_score:.3f})"
            )
        else:
            band = "unhealthy"
            advisory = (
                f"unhealthy: success_rate={success_rate:.3f} < "
                f"degraded_min={self._degraded_min:.3f} "
                f"(health_score={health_score:.3f}, error_rate={error_rate:.3f})"
            )
        return ProviderHealthScoreAdvice(
            provider=provider,
            successes=successes,
            failures=failures,
            samples=samples,
            success_rate=float(success_rate),
            error_rate=float(error_rate),
            health_score=health_score,
            band=band,
            advisory=advisory,
        )

    def _record(self, provider: str, success: bool) -> ProviderHealthScoreAdvice:
        if not provider:
            raise ValueError("provider must be non-empty")
        with self._lock:
            window = self._outcomes.setdefault(provider, deque(maxlen=self._window_size))
            window.append(bool(success))
            return self._advise_locked(provider, window)

    def record_success(self, provider: str) -> ProviderHealthScoreAdvice:
        """Record a successful call and return the updated advisory."""
        return self._record(provider, True)

    def record_failure(self, provider: str) -> ProviderHealthScoreAdvice:
        """Record a failed call and return the updated advisory."""
        return self._record(provider, False)

    def record_outcome(self, provider: str, *, success: bool) -> ProviderHealthScoreAdvice:
        """Record an outcome and return the updated advisory."""
        return self._record(provider, bool(success))

    def snapshot(self, provider: str) -> ProviderHealthScoreAdvice | None:
        """Return the latest advisory for ``provider``, or ``None``."""
        if not provider:
            raise ValueError("provider must be non-empty")
        with self._lock:
            window = self._outcomes.get(provider)
            if window is None or not window:
                return None
            return self._advise_locked(provider, window)

    def scoreboard(self) -> list[ProviderHealthScoreAdvice]:
        """Return sorted advisories for every known provider."""
        with self._lock:
            result = [
                self._advise_locked(provider, window)
                for provider, window in self._outcomes.items()
                if window
            ]
            result.sort(key=lambda item: (-item.health_score, item.provider))
            return result

    def unhealthy(self) -> list[ProviderHealthScoreAdvice]:
        """Return sorted advisories whose band is ``unhealthy``."""
        return [item for item in self.scoreboard() if item.band == "unhealthy"]

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
