"""WebGroundingInjectionGateAdvisor.

Advises injection_score vs budgets.
Closes the Perplexity/Bing/OpenAI web-grounding injection gates gap. Distinct from
`PromptInjectionGateway` and `ReasoningTraceLeakAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WebGroundingInjectionGateAdvice:
    """WebGroundingInjectionGateAdvisor advice."""

    request_id: str
    injection_score: float
    soft_limit: float
    hard_limit: float
    band: str


class WebGroundingInjectionGateAdvisor:
    """Advise injection_score bands."""

    def advise(
        self,
        *,
        request_id: str,
        injection_score: float,
        soft_limit: float = 0.25,
        hard_limit: float = 0.6,
    ) -> WebGroundingInjectionGateAdvice:
        """Return injection_score band.

        Args:
            request_id: Non-empty request id.
            injection_score: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if injection_score < 0:
            raise ValueError("injection_score must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if injection_score >= hard_limit:
            band = "breach"
        elif injection_score >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return WebGroundingInjectionGateAdvice(
            request_id=rid,
            injection_score=float(injection_score),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
