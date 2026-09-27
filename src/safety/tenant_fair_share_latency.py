"""Tenant fair-share latency SLO advisor.

Advises fairness bands from a tenant p95 latency vs fleet fair-share SLO.
Closes the Helicone / Portkey / LiteLLM multi-tenant latency fairness gap.
Distinct from ``FirstTokenLatencySloAdvisor`` and ``TenantConcurrencySlotGuard``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TenantFairShareLatencyAdvice:
    """Tenant fair-share latency advice."""

    tenant_id: str
    tenant_p95_ms: float
    fair_share_slo_ms: float
    ratio: float
    band: str


class TenantFairShareLatencySloAdvisor:
    """Advise tenant latency fairness vs a fair-share SLO."""

    def advise(
        self,
        *,
        tenant_id: str,
        tenant_p95_ms: float,
        fair_share_slo_ms: float,
    ) -> TenantFairShareLatencyAdvice:
        """Return band for tenant p95 vs fair-share SLO.

        Args:
            tenant_id: Non-empty tenant id.
            tenant_p95_ms: Tenant p95 latency ms (``>= 0``).
            fair_share_slo_ms: Fair-share SLO ms (``> 0``).

        Returns:
            TenantFairShareLatencyAdvice with ``ok`` / ``watch`` / ``breach``.
        """

        tid = tenant_id.strip()
        if not tid:
            raise ValueError("tenant_id must be non-empty")
        if tenant_p95_ms < 0:
            raise ValueError("tenant_p95_ms must be >= 0")
        if fair_share_slo_ms <= 0:
            raise ValueError("fair_share_slo_ms must be > 0")

        ratio = round(tenant_p95_ms / fair_share_slo_ms, 4)
        if ratio >= 1.5:
            band = "breach"
        elif ratio >= 1.1:
            band = "watch"
        else:
            band = "ok"
        return TenantFairShareLatencyAdvice(
            tenant_id=tid,
            tenant_p95_ms=float(tenant_p95_ms),
            fair_share_slo_ms=float(fair_share_slo_ms),
            ratio=float(ratio),
            band=band,
        )
