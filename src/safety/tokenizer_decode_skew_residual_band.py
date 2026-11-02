"""TokenizerDecodeSkewResidualBandAdvisor.

Advises decode_skew_ratio vs budgets.
Closes the vLLM/SGLang/TGI tokenizer decode-skew residual monitors gap. Distinct from
`AttentionSinkResidualBandAdvisor` / `SpeculativeAcceptanceEntropyBandAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizerDecodeSkewResidualBandAdvice:
    """TokenizerDecodeSkewResidualBandAdvisor advice."""

    request_id: str
    decode_skew_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class TokenizerDecodeSkewResidualBandAdvisor:
    """Advise decode_skew_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        decode_skew_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> TokenizerDecodeSkewResidualBandAdvice:
        """Return decode_skew_ratio band.

        Args:
            request_id: Non-empty request id.
            decode_skew_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if decode_skew_ratio < 0:
            raise ValueError("decode_skew_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if decode_skew_ratio >= hard_limit:
            band = "breach"
        elif decode_skew_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return TokenizerDecodeSkewResidualBandAdvice(
            request_id=rid,
            decode_skew_ratio=float(decode_skew_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
