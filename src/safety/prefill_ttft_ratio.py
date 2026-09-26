"""Prefill vs TTFT ratio advisor.

Advises routing bands from prefill_ms / ttft_ms ratios.
Closes the OpenRouter / LiteLLM / Portkey prefill-heavy routing gap.
Distinct from ``FirstTokenLatencySloAdvisor`` and ``ProviderColdStartLatencyAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefillTtftRatioAdvice:
    """Prefill/TTFT ratio advice."""

    provider: str
    prefill_ms: float
    ttft_ms: float
    ratio: float
    band: str


class PrefillTtftRatioAdvisor:
    """Advise prefill-heavy vs balanced TTFT bands."""

    def advise(
        self,
        *,
        provider: str,
        prefill_ms: float,
        ttft_ms: float,
    ) -> PrefillTtftRatioAdvice:
        """Return band for prefill/TTFT ratio.

        Args:
            provider: Non-empty provider id.
            prefill_ms: Prefill latency ms (``>= 0``).
            ttft_ms: Time-to-first-token ms (``> 0``).

        Returns:
            PrefillTtftRatioAdvice with ``balanced`` / ``prefill_heavy`` / ``pathological``.
        """

        name = provider.strip()
        if not name:
            raise ValueError("provider must be non-empty")
        if prefill_ms < 0:
            raise ValueError("prefill_ms must be >= 0")
        if ttft_ms <= 0:
            raise ValueError("ttft_ms must be > 0")

        ratio = round(prefill_ms / ttft_ms, 4)
        if ratio >= 0.9:
            band = "pathological"
        elif ratio >= 0.6:
            band = "prefill_heavy"
        else:
            band = "balanced"
        return PrefillTtftRatioAdvice(
            provider=name,
            prefill_ms=float(prefill_ms),
            ttft_ms=float(ttft_ms),
            ratio=float(ratio),
            band=band,
        )
