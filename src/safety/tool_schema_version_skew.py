"""ToolSchemaVersionSkewAdvisor.

Advises skew_score vs budgets.
Closes the LiteLLM/OpenAI/Anthropic tool-schema version-skew monitors gap. Distinct from
`ToolSchemaDriftAdvisor` / `ToolLoopDepthGateAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolSchemaVersionSkewAdvice:
    """ToolSchemaVersionSkewAdvisor advice."""

    request_id: str
    skew_score: float
    soft_limit: float
    hard_limit: float
    band: str


class ToolSchemaVersionSkewAdvisor:
    """Advise skew_score bands."""

    def advise(
        self,
        *,
        request_id: str,
        skew_score: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.7,
    ) -> ToolSchemaVersionSkewAdvice:
        """Return skew_score band.

        Args:
            request_id: Non-empty request id.
            skew_score: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if skew_score < 0:
            raise ValueError("skew_score must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if skew_score >= hard_limit:
            band = "breach"
        elif skew_score >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ToolSchemaVersionSkewAdvice(
            request_id=rid,
            skew_score=float(skew_score),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
