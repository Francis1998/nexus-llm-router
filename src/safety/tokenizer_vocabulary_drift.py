"""TokenizerVocabularyDrift advisor.

Advises drift_ratio vs budgets.
Closes the vLLM/HuggingFace/OpenAI tokenizer vocabulary-drift advisors gap. Distinct from
``TokenizerMismatchAdvisor`` and ``EmbeddingDimMismatchAdvisor``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizerVocabularyDriftAdvice:
    """TokenizerVocabularyDriftAdvisor advice."""

    request_id: str
    drift_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class TokenizerVocabularyDriftAdvisor:
    """Advise drift_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        drift_ratio: float,
        soft_limit: float = 0.05,
        hard_limit: float = 0.15,
    ) -> TokenizerVocabularyDriftAdvice:
        """Return drift_ratio band.

        Args:
            request_id: Non-empty request id.
            drift_ratio: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if drift_ratio < 0:
            raise ValueError("drift_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if drift_ratio >= hard_limit:
            band = "breach"
        elif drift_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return TokenizerVocabularyDriftAdvice(
            request_id=rid,
            drift_ratio=float(drift_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
