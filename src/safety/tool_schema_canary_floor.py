"""ToolSchemaCanaryFloorAdvisor.

Advises canary_miss_ratio vs budgets.
Closes the vLLM/SGLang/LiteLLM tool-schema canary floor monitors gap. Distinct from
`PromptInjectionCanaryFloorAdvisor` / `ProviderCanaryRolloutGuard`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolSchemaCanaryFloorAdvice:
    """ToolSchemaCanaryFloorAdvisor advice."""

    request_id: str
    canary_miss_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class ToolSchemaCanaryFloorAdvisor:
    """Advise canary_miss_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        canary_miss_ratio: float,
        soft_limit: float = 0.15,
        hard_limit: float = 0.4,
    ) -> ToolSchemaCanaryFloorAdvice:
        """Return canary_miss_ratio band.

        Args:
            request_id: Non-empty request id.
            canary_miss_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if canary_miss_ratio < 0:
            raise ValueError("canary_miss_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if canary_miss_ratio >= hard_limit:
            band = "breach"
        elif canary_miss_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ToolSchemaCanaryFloorAdvice(
            request_id=rid,
            canary_miss_ratio=float(canary_miss_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
