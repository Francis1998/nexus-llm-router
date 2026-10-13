"""PromptInjectionCanaryFloorAdvisor.

Advises canary_hit_rate vs budgets.
Closes the LiteLLM/vLLM/Guardrails prompt-injection canary-floor monitors gap. Distinct from
`PromptInjectionGateway` / `ProviderCanaryRolloutGuard`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptInjectionCanaryFloorAdvice:
    """PromptInjectionCanaryFloorAdvisor advice."""

    request_id: str
    canary_hit_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class PromptInjectionCanaryFloorAdvisor:
    """Advise canary_hit_rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        canary_hit_rate: float,
        soft_limit: float = 0.02,
        hard_limit: float = 0.1,
    ) -> PromptInjectionCanaryFloorAdvice:
        """Return canary_hit_rate band.

        Args:
            request_id: Non-empty request id.
            canary_hit_rate: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if canary_hit_rate < 0:
            raise ValueError("canary_hit_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if canary_hit_rate >= hard_limit:
            band = "breach"
        elif canary_hit_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return PromptInjectionCanaryFloorAdvice(
            request_id=rid,
            canary_hit_rate=float(canary_hit_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
