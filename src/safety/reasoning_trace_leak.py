"""ReasoningTraceLeakAdvisor.

Advises leak_score vs budgets.
Closes the vLLM/SGLang/OpenAI-compatible reasoning-trace leak gates gap. Distinct from
`ReasoningTokenBudgetAdvisor` and `PromptInjectionGateway`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReasoningTraceLeakAdvice:
    """ReasoningTraceLeakAdvisor advice."""

    request_id: str
    leak_score: float
    soft_limit: float
    hard_limit: float
    band: str


class ReasoningTraceLeakAdvisor:
    """Advise leak_score bands."""

    def advise(
        self,
        *,
        request_id: str,
        leak_score: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.6,
    ) -> ReasoningTraceLeakAdvice:
        """Return leak_score band.

        Args:
            request_id: Non-empty request id.
            leak_score: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if leak_score < 0:
            raise ValueError("leak_score must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if leak_score >= hard_limit:
            band = "breach"
        elif leak_score >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ReasoningTraceLeakAdvice(
            request_id=rid,
            leak_score=float(leak_score),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
