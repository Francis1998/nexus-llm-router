"""EmbeddingCacheThrashAdvisor.

Advises thrash_ratio vs budgets.
Closes the Redis/vLLM/embedding-cache thrash monitors gap. Distinct from
`TokenizerVocabularyDriftAdvisor` and `StickySessionBleedAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingCacheThrashAdvice:
    """EmbeddingCacheThrashAdvisor advice."""

    request_id: str
    thrash_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class EmbeddingCacheThrashAdvisor:
    """Advise thrash_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        thrash_ratio: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.7,
    ) -> EmbeddingCacheThrashAdvice:
        """Return thrash_ratio band.

        Args:
            request_id: Non-empty request id.
            thrash_ratio: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if thrash_ratio < 0:
            raise ValueError("thrash_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if thrash_ratio >= hard_limit:
            band = "breach"
        elif thrash_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return EmbeddingCacheThrashAdvice(
            request_id=rid,
            thrash_ratio=float(thrash_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
