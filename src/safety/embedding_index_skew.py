"""EmbeddingIndexSkewAdvisor.

Advises index_skew_ratio vs budgets.
Closes the vLLM/SGLang/embedding-serving index-skew monitors gap. Distinct from
`EmbeddingBatchSkewAdvisor` / `EmbeddingDimMismatchAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingIndexSkewAdvice:
    """EmbeddingIndexSkewAdvisor advice."""

    request_id: str
    index_skew_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class EmbeddingIndexSkewAdvisor:
    """Advise index_skew_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        index_skew_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> EmbeddingIndexSkewAdvice:
        """Return index_skew_ratio band.

        Args:
            request_id: Non-empty request id.
            index_skew_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if index_skew_ratio < 0:
            raise ValueError("index_skew_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if index_skew_ratio >= hard_limit:
            band = "breach"
        elif index_skew_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return EmbeddingIndexSkewAdvice(
            request_id=rid,
            index_skew_ratio=float(index_skew_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
