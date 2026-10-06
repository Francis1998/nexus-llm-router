"""StructuredOutputRetryBand advisor.

Advises structured-output retry_count vs budgets.
Closes the OpenAI/Anthropic/Gemini structured-output retry planners gap. Distinct from
``JsonSchemaRetryBudgetGuard`` and ``AdaptiveRetryJitterAdvisor``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructuredOutputRetryBandAdvice:
    """StructuredOutputRetryBandAdvisor advice."""

    request_id: str
    retry_count: float
    soft_limit: float
    hard_limit: float
    band: str


class StructuredOutputRetryBandAdvisor:
    """Advise structured-output retry_count bands."""

    def advise(
        self,
        *,
        request_id: str,
        retry_count: float,
        soft_limit: float = 2.0,
        hard_limit: float = 5.0,
    ) -> StructuredOutputRetryBandAdvice:
        """Return retry_count band.

        Args:
            request_id: Non-empty request id.
            retry_count: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if retry_count < 0:
            raise ValueError("retry_count must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if retry_count >= hard_limit:
            band = "breach"
        elif retry_count >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return StructuredOutputRetryBandAdvice(
            request_id=rid,
            retry_count=float(retry_count),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
