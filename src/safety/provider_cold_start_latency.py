"""Provider cold-start latency advisor.

Advises warm/cooling/cold bands from measured first-token cold-start latency.
Closes the OpenRouter / LiteLLM / Portkey cold-start routing gap.
Distinct from ``FirstTokenLatencySloAdvisor`` and ``ProviderWarmPoolAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderColdStartLatencyAdvice:
    """Provider cold-start latency advice."""

    provider: str
    cold_start_ms: float
    warm_threshold_ms: float
    cold_threshold_ms: float
    band: str


class ProviderColdStartLatencyAdvisor:
    """Advise provider cold-start latency bands."""

    def advise(
        self,
        *,
        provider: str,
        cold_start_ms: float,
        warm_threshold_ms: float = 250.0,
        cold_threshold_ms: float = 1500.0,
    ) -> ProviderColdStartLatencyAdvice:
        """Return band for measured cold-start latency.

        Args:
            provider: Non-empty provider id.
            cold_start_ms: Measured cold-start latency ms (``>= 0``).
            warm_threshold_ms: Warm upper bound ms (``> 0``).
            cold_threshold_ms: Cold lower bound ms (``>= warm_threshold_ms``).

        Returns:
            ProviderColdStartLatencyAdvice with ``warm`` / ``cooling`` / ``cold``.
        """

        name = provider.strip()
        if not name:
            raise ValueError("provider must be non-empty")
        if cold_start_ms < 0:
            raise ValueError("cold_start_ms must be >= 0")
        if warm_threshold_ms <= 0:
            raise ValueError("warm_threshold_ms must be > 0")
        if cold_threshold_ms < warm_threshold_ms:
            raise ValueError("cold_threshold_ms must be >= warm_threshold_ms")

        if cold_start_ms >= cold_threshold_ms:
            band = "cold"
        elif cold_start_ms >= warm_threshold_ms:
            band = "cooling"
        else:
            band = "warm"
        return ProviderColdStartLatencyAdvice(
            provider=name,
            cold_start_ms=float(cold_start_ms),
            warm_threshold_ms=float(warm_threshold_ms),
            cold_threshold_ms=float(cold_threshold_ms),
            band=band,
        )
