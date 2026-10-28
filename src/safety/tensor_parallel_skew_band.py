"""TensorParallelSkewBandAdvisor.

Advises skew_ratio vs budgets.
Closes the vLLM/SGLang/TGI tensor-parallel skew monitors gap. Distinct from
`DisaggPrefillDecodeSkewBandAdvisor` / `TokenizerVocabSkewBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorParallelSkewBandAdvice:
    """TensorParallelSkewBandAdvisor advice."""

    request_id: str
    skew_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class TensorParallelSkewBandAdvisor:
    """Advise skew_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        skew_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> TensorParallelSkewBandAdvice:
        """Return skew_ratio band.

        Args:
            request_id: Non-empty request id.
            skew_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if skew_ratio < 0:
            raise ValueError("skew_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if skew_ratio >= hard_limit:
            band = "breach"
        elif skew_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return TensorParallelSkewBandAdvice(
            request_id=rid,
            skew_ratio=float(skew_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
