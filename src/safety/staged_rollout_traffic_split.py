"""StagedRolloutTrafficSplit guard.

Advises canary share vs budgets.
Closes the OpenRouter/LiteLLM/BentoML staged rollout traffic-split guards gap. Distinct from
``ProviderCanaryRolloutGuard and ShadowTrafficMirrorGuard``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StagedRolloutTrafficSplitAdvice:
    """StagedRolloutTrafficSplitGuard advice."""

    request_id: str
    canary_share: float
    soft_limit: float
    hard_limit: float
    band: str


class StagedRolloutTrafficSplitGuard:
    """Advise canary share bands."""

    def check(
        self,
        *,
        request_id: str,
        canary_share: float,
        soft_limit: float = 0.1,
        hard_limit: float = 0.35,
    ) -> StagedRolloutTrafficSplitAdvice:
        """Return canary_share band.

        Args:
            request_id: Non-empty request id.
            canary_share: Observed ratio (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if canary_share < 0:
            raise ValueError("canary_share must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if canary_share >= hard_limit:
            band = "breach"
        elif canary_share >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return StagedRolloutTrafficSplitAdvice(
            request_id=rid,
            canary_share=float(canary_share),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
