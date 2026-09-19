"""Provider quota-remaining advisor (ok / low / exhausted)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

QuotaBand = Literal["ok", "low", "exhausted"]


@dataclass(frozen=True, slots=True)
class ProviderQuotaAdvice:
    """Quota-remaining advisory for one provider."""

    provider: str
    remaining: int
    capacity: int
    band: QuotaBand
    remaining_ratio: float
    advisory: str


class ProviderQuotaRemainingAdvisor:
    """Classify provider quota remaining into ok/low/exhausted bands.

    Closes the OpenRouter / LiteLLM / Helicone *remaining quota* dashboard gap
    for offline Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x
    / Kimi K2. Distinct from ``ProviderErrorBudgetShed`` (error rate) and
    ``TenantSpendQuotaGuard`` (tenant spend). Advisory only — never rejects.
    """

    def __init__(self, *, low_ratio: float = 0.2) -> None:
        if not 0.0 < low_ratio < 1.0:
            raise ValueError("low_ratio must be in (0, 1)")
        self._low_ratio = float(low_ratio)
        self._state: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def observe(self, provider: str, *, remaining: int, capacity: int) -> ProviderQuotaAdvice:
        """Record remaining/capacity for ``provider`` and return advice."""
        if not provider:
            raise ValueError("provider must be non-empty")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if remaining < 0:
            raise ValueError("remaining must be >= 0")
        if remaining > capacity:
            raise ValueError("remaining must be <= capacity")
        with self._lock:
            self._state[provider] = (int(remaining), int(capacity))
        return self._advise(provider, remaining, capacity)

    def _advise(self, provider: str, remaining: int, capacity: int) -> ProviderQuotaAdvice:
        ratio = remaining / float(capacity)
        if remaining == 0:
            band: QuotaBand = "exhausted"
            advisory = f"exhausted: remaining=0/{capacity} for provider={provider}"
        elif ratio <= self._low_ratio:
            band = "low"
            advisory = (
                f"low: remaining={remaining}/{capacity} "
                f"(ratio={ratio:.3f} <= low_ratio={self._low_ratio}) "
                f"for provider={provider}"
            )
        else:
            band = "ok"
            advisory = f"ok: remaining={remaining}/{capacity} for provider={provider}"
        return ProviderQuotaAdvice(
            provider=provider,
            remaining=remaining,
            capacity=capacity,
            band=band,
            remaining_ratio=round(ratio, 4),
            advisory=advisory,
        )

    def snapshot(self, provider: str) -> ProviderQuotaAdvice | None:
        if not provider:
            raise ValueError("provider must be non-empty")
        with self._lock:
            if provider not in self._state:
                return None
            remaining, capacity = self._state[provider]
            return self._advise(provider, remaining, capacity)
