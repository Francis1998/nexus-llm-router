"""JsonSchemaStrictnessBandAdvisor.
Advises strictness_score vs budgets.
Closes the OpenAI/Instructor/Outlines JSON-schema strictness advisors gap. Distinct from
`JsonSchemaRetryAdvisor` and `OutputSchemaGuard`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JsonSchemaStrictnessBandAdvice:
    """JsonSchemaStrictnessBandAdvisor advice."""

    request_id: str
    strictness_score: float
    soft_limit: float
    hard_limit: float
    band: str


class JsonSchemaStrictnessBandAdvisor:
    """Advise strictness_score bands."""

    def advise(
        self,
        *,
        request_id: str,
        strictness_score: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.6,
    ) -> JsonSchemaStrictnessBandAdvice:
        """Return strictness_score band.

        Args:
            request_id: Non-empty request id.
            strictness_score: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if strictness_score < 0:
            raise ValueError("strictness_score must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if strictness_score >= hard_limit:
            band = "breach"
        elif strictness_score >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return JsonSchemaStrictnessBandAdvice(
            request_id=rid,
            strictness_score=float(strictness_score),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
