"""Embedding batch skew advisor.

Advises embedding-batch size skew vs a target band. Closes the
vLLM / OpenRouter / LiteLLM embedding-batch skew gap.
Distinct from ``MultimodalTokenTaxAdvisor`` and
``PromptCompressionRatioAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingBatchSkewAdvice:
    """Embedding batch skew advice."""

    request_id: str
    batch_size: int
    target_size: int
    skew_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class EmbeddingBatchSkewAdvisor:
    """Advise embedding-batch skew bands."""

    def advise(
        self,
        *,
        request_id: str,
        batch_size: int,
        target_size: int,
        soft_limit: float = 1.5,
        hard_limit: float = 3.0,
    ) -> EmbeddingBatchSkewAdvice:
        """Return skew band.

        Args:
            request_id: Non-empty request id.
            batch_size: Observed batch size (``>= 1``).
            target_size: Target batch size (``>= 1``).
            soft_limit: Soft size/target ratio (``> 0``).
            hard_limit: Hard size/target ratio (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if target_size < 1:
            raise ValueError("target_size must be >= 1")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        skew_ratio = round(batch_size / target_size, 4)
        if skew_ratio >= hard_limit:
            band = "breach"
        elif skew_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return EmbeddingBatchSkewAdvice(
            request_id=rid,
            batch_size=int(batch_size),
            target_size=int(target_size),
            skew_ratio=float(skew_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
