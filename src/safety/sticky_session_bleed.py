"""Sticky session bleed advisor.

Advises cross-tenant sticky-session bleed bands. Closes the vLLM /
OpenRouter / LiteLLM sticky-session tenant bleed gap.
Distinct from ``StickyProviderAffinity`` and ``TenantConcurrencySlotGuard``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StickySessionBleedAdvice:
    """Sticky session bleed advice."""

    request_id: str
    foreign_hit_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class StickySessionBleedAdvisor:
    """Advise sticky-session cross-tenant bleed bands."""

    def advise(
        self,
        *,
        request_id: str,
        foreign_hit_rate: float,
        soft_limit: float = 0.05,
        hard_limit: float = 0.2,
    ) -> StickySessionBleedAdvice:
        """Return bleed band.

        Args:
            request_id: Non-empty request id.
            foreign_hit_rate: Fraction of sticky hits from other tenants (``0..1``).
            soft_limit: Soft bleed threshold (``> 0``).
            hard_limit: Hard bleed threshold (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if foreign_hit_rate < 0 or foreign_hit_rate > 1:
            raise ValueError("foreign_hit_rate must be in [0, 1]")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if foreign_hit_rate >= hard_limit:
            band = "breach"
        elif foreign_hit_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return StickySessionBleedAdvice(
            request_id=rid,
            foreign_hit_rate=float(foreign_hit_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
