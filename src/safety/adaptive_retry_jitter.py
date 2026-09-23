"""Adaptive retry jitter advisor.

Advises low/medium/high jitter bands from retry attempt index. Closes the
LiteLLM / Portkey / OpenRouter retry-jitter gap. Distinct from
``ProviderErrorBudgetGuard`` (error budgets) and ``RequestHedgingAdvisor``
(hedged replicas). Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2. Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryJitterAdvice:
    """Jitter advice for one retry attempt."""

    attempt: int
    base_delay_ms: float
    jitter_ms: float
    band: str


class AdaptiveRetryJitterAdvisor:
    """Advise retry jitter magnitude from attempt index."""

    def advise(
        self,
        *,
        attempt: int,
        base_delay_ms: float = 100.0,
    ) -> RetryJitterAdvice:
        """Return jitter band for ``attempt``.

        Args:
            attempt: Zero-based attempt index (``>= 0``).
            base_delay_ms: Base delay in ms (``> 0``).

        Returns:
            RetryJitterAdvice with band ``low`` / ``medium`` / ``high``.
        """

        if attempt < 0:
            raise ValueError("attempt must be >= 0")
        if base_delay_ms <= 0:
            raise ValueError("base_delay_ms must be > 0")

        if attempt <= 1:
            band = "low"
            factor = 0.25
        elif attempt <= 3:
            band = "medium"
            factor = 0.5
        else:
            band = "high"
            factor = 1.0

        jitter = round(base_delay_ms * (2**attempt) * factor, 3)
        return RetryJitterAdvice(
            attempt=int(attempt),
            base_delay_ms=float(base_delay_ms),
            jitter_ms=float(jitter),
            band=band,
        )
