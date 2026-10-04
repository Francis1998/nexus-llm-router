"""ToolChoiceStickyAffinity advisor.

Advises affinity_drift vs budgets.
Closes the OpenRouter/LiteLLM/vLLM tool-choice sticky affinity advisors gap. Distinct from
``StickyProviderRouter`` and ``StickySessionBleedAdvisor``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolChoiceStickyAffinityAdvice:
    """ToolChoiceStickyAffinityAdvisor advice."""

    request_id: str
    affinity_drift: float
    soft_limit: float
    hard_limit: float
    band: str


class ToolChoiceStickyAffinityAdvisor:
    """Advise affinity_drift bands."""

    def advise(
        self,
        *,
        request_id: str,
        affinity_drift: float,
        soft_limit: float = 0.15,
        hard_limit: float = 0.4,
    ) -> ToolChoiceStickyAffinityAdvice:
        """Return affinity_drift band.

        Args:
            request_id: Non-empty request id.
            affinity_drift: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if affinity_drift < 0:
            raise ValueError("affinity_drift must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if affinity_drift >= hard_limit:
            band = "breach"
        elif affinity_drift >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ToolChoiceStickyAffinityAdvice(
            request_id=rid,
            affinity_drift=float(affinity_drift),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
