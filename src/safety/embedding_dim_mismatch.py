"""EmbeddingDimMismatch advisor.

Advises dim_delta vs budgets.
Closes the LiteLLM/OpenRouter/vLLM embedding dimension mismatch advisors gap. Distinct from
``TokenizerMismatchAdvisor`` and ``EmbeddingBatchSkewAdvisor``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingDimMismatchAdvice:
    """EmbeddingDimMismatchAdvisor advice."""

    request_id: str
    dim_delta: float
    soft_limit: float
    hard_limit: float
    band: str


class EmbeddingDimMismatchAdvisor:
    """Advise dim_delta bands."""

    def advise(
        self,
        *,
        request_id: str,
        dim_delta: float,
        soft_limit: float = 1.0,
        hard_limit: float = 8.0,
    ) -> EmbeddingDimMismatchAdvice:
        """Return dim_delta band.

        Args:
            request_id: Non-empty request id.
            dim_delta: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if dim_delta < 0:
            raise ValueError("dim_delta must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if dim_delta >= hard_limit:
            band = "breach"
        elif dim_delta >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return EmbeddingDimMismatchAdvice(
            request_id=rid,
            dim_delta=float(dim_delta),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
