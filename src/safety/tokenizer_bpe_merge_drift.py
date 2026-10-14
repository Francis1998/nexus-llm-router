"""TokenizerBpeMergeDriftAdvisor.

Advises bpe_merge_drift_score vs budgets.
Closes the vLLM/TGI/HuggingFace tokenizer BPE merge-drift monitors gap. Distinct from
`TokenizerVocabularyDriftAdvisor` / `TokenizerMismatchAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizerBpeMergeDriftAdvice:
    """TokenizerBpeMergeDriftAdvisor advice."""

    request_id: str
    bpe_merge_drift_score: float
    soft_limit: float
    hard_limit: float
    band: str


class TokenizerBpeMergeDriftAdvisor:
    """Advise bpe_merge_drift_score bands."""

    def advise(
        self,
        *,
        request_id: str,
        bpe_merge_drift_score: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> TokenizerBpeMergeDriftAdvice:
        """Return bpe_merge_drift_score band.

        Args:
            request_id: Non-empty request id.
            bpe_merge_drift_score: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if bpe_merge_drift_score < 0:
            raise ValueError("bpe_merge_drift_score must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if bpe_merge_drift_score >= hard_limit:
            band = "breach"
        elif bpe_merge_drift_score >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return TokenizerBpeMergeDriftAdvice(
            request_id=rid,
            bpe_merge_drift_score=float(bpe_merge_drift_score),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
