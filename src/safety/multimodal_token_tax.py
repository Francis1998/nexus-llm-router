"""Multimodal token tax advisor.

Advises vision-token tax vs text budget bands. Closes the vLLM /
OpenRouter / LiteLLM multimodal vision-token tax gap.
Distinct from ``MultimodalPayloadSizeGate`` and ``OutputTokenCeilingGuard``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MultimodalTokenTaxAdvice:
    """Multimodal token tax advice."""

    request_id: str
    vision_tokens: int
    text_tokens: int
    tax_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class MultimodalTokenTaxAdvisor:
    """Advise multimodal vision-token tax bands."""

    def advise(
        self,
        *,
        request_id: str,
        vision_tokens: int,
        text_tokens: int,
        soft_limit: float = 2.0,
        hard_limit: float = 8.0,
    ) -> MultimodalTokenTaxAdvice:
        """Return tax band.

        Args:
            request_id: Non-empty request id.
            vision_tokens: Vision/image tokens (``>= 0``).
            text_tokens: Text tokens (``>= 0``).
            soft_limit: Soft vision/text ratio (``> 0``).
            hard_limit: Hard vision/text ratio (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if vision_tokens < 0:
            raise ValueError("vision_tokens must be >= 0")
        if text_tokens < 0:
            raise ValueError("text_tokens must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        denom = max(text_tokens, 1)
        tax_ratio = round(vision_tokens / denom, 4)
        if tax_ratio >= hard_limit:
            band = "breach"
        elif tax_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return MultimodalTokenTaxAdvice(
            request_id=rid,
            vision_tokens=int(vision_tokens),
            text_tokens=int(text_tokens),
            tax_ratio=float(tax_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
