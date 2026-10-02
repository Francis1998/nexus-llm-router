"""Tokenizer mismatch advisor.

Advises router-vs-provider tokenizer mismatch ratio vs budgets.
Closes the vLLM / OpenRouter / LiteLLM router-vs-provider tokenizer mismatch bands gap. Di
stinct from
``ToolSchemaDriftGate`` and ``MultimodalTokenTaxAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizerMismatchAdvice:
    """Tokenizer mismatch advice."""

    request_id: str
    mismatch_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class TokenizerMismatchAdvisor:
    """Advise tokenizer mismatch-ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        mismatch_ratio: float,
        soft_limit: float = 0.05,
        hard_limit: float = 0.15,
    ) -> TokenizerMismatchAdvice:
        """Return mismatch-ratio band.

        Args:
            request_id: Non-empty request id.
            mismatch_ratio: |router_tokens - provider_tokens| / max (``>= 0``).
            soft_limit: Soft mismatch budget (``> 0``).
            hard_limit: Hard mismatch budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if mismatch_ratio < 0:
            raise ValueError("mismatch_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if mismatch_ratio >= hard_limit:
            band = "breach"
        elif mismatch_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return TokenizerMismatchAdvice(
            request_id=rid,
            mismatch_ratio=float(mismatch_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
