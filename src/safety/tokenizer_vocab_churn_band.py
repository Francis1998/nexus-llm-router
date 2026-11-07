"""TokenizerVocabChurnBandAdvisor.

Advises churn_ratio vs budgets.
Closes the vLLM/SGLang/TGI tokenizer vocabulary-churn monitors gap. Distinct from
`TokenizerVocabSkewBandAdvisor` / `TokenizerBpeMergeDriftAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizerVocabChurnBandAdvice:
    """TokenizerVocabChurnBandAdvisor advice."""

    request_id: str
    churn_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class TokenizerVocabChurnBandAdvisor:
    """Advise churn_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        churn_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> TokenizerVocabChurnBandAdvice:
        """Return churn_ratio band.

        Args:
            request_id: Non-empty request id.
            churn_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if churn_ratio < 0:
            raise ValueError("churn_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if churn_ratio >= hard_limit:
            band = "breach"
        elif churn_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return TokenizerVocabChurnBandAdvice(
            request_id=rid,
            churn_ratio=float(churn_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
