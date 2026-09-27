"""Queue wait SLO advisor.

Advises queue-wait bands from seconds spent waiting before dispatch vs SLO.
Closes the OpenRouter / LiteLLM / Portkey queue-wait SLO gap.
Distinct from ``RequestPriorityAgingAdvisor`` and ``TenantFairShareLatencySloAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueueWaitSloAdvice:
    """Queue wait SLO advice."""

    request_id: str
    wait_s: float
    slo_s: float
    ratio: float
    band: str


class QueueWaitSloAdvisor:
    """Advise queue-wait seconds vs SLO bands."""

    def advise(
        self,
        *,
        request_id: str,
        wait_s: float,
        slo_s: float,
    ) -> QueueWaitSloAdvice:
        """Return band for queue wait vs SLO.

        Args:
            request_id: Non-empty request id.
            wait_s: Seconds waiting in queue (``>= 0``).
            slo_s: Queue-wait SLO seconds (``> 0``).

        Returns:
            QueueWaitSloAdvice with ``fresh`` / ``aging`` / ``late``.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if wait_s < 0:
            raise ValueError("wait_s must be >= 0")
        if slo_s <= 0:
            raise ValueError("slo_s must be > 0")

        ratio = round(wait_s / slo_s, 4)
        if ratio >= 1.0:
            band = "late"
        elif ratio >= 0.7:
            band = "aging"
        else:
            band = "fresh"
        return QueueWaitSloAdvice(
            request_id=rid,
            wait_s=float(wait_s),
            slo_s=float(slo_s),
            ratio=float(ratio),
            band=band,
        )
