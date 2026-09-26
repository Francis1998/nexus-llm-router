"""Cost spike anomaly advisor.

Advises anomaly bands from current spend vs rolling baseline.
Closes the Helicone / Portkey / LiteLLM cost-anomaly routing gap.
Distinct from ``CostForecastAdvisor`` and ``TenantSpendQuotaGuard``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostSpikeAnomalyAdvice:
    """Cost spike anomaly advice."""

    tenant_id: str
    current_spend_usd: float
    baseline_spend_usd: float
    spike_ratio: float
    band: str


class CostSpikeAnomalyAdvisor:
    """Advise cost-spike anomaly bands vs a rolling baseline."""

    def advise(
        self,
        *,
        tenant_id: str,
        current_spend_usd: float,
        baseline_spend_usd: float,
    ) -> CostSpikeAnomalyAdvice:
        """Return band for spend spike ratio.

        Args:
            tenant_id: Non-empty tenant id.
            current_spend_usd: Current window spend (``>= 0``).
            baseline_spend_usd: Rolling baseline spend (``> 0``).

        Returns:
            CostSpikeAnomalyAdvice with ``normal`` / ``elevated`` / ``anomalous``.
        """

        tid = tenant_id.strip()
        if not tid:
            raise ValueError("tenant_id must be non-empty")
        if current_spend_usd < 0:
            raise ValueError("current_spend_usd must be >= 0")
        if baseline_spend_usd <= 0:
            raise ValueError("baseline_spend_usd must be > 0")

        ratio = round(current_spend_usd / baseline_spend_usd, 4)
        if ratio >= 3.0:
            band = "anomalous"
        elif ratio >= 1.5:
            band = "elevated"
        else:
            band = "normal"
        return CostSpikeAnomalyAdvice(
            tenant_id=tid,
            current_spend_usd=float(current_spend_usd),
            baseline_spend_usd=float(baseline_spend_usd),
            spike_ratio=float(ratio),
            band=band,
        )
