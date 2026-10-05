"""PrefillDecodeTokenSkew advisor.

Advises skew_ratio vs budgets.
Closes the vLLM/SGLang/TensorRT-LLM prefill-vs-decode token-skew advisors gap. Distinct from
``PrefillTtftRatioAdvisor`` and ``SpeculativeDecodeBudgetAdvisor``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefillDecodeTokenSkewAdvice:
    """PrefillDecodeTokenSkewAdvisor advice."""

    request_id: str
    skew_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class PrefillDecodeTokenSkewAdvisor:
    """Advise skew_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        skew_ratio: float,
        soft_limit: float = 2.0,
        hard_limit: float = 5.0,
    ) -> PrefillDecodeTokenSkewAdvice:
        """Return skew_ratio band.

        Args:
            request_id: Non-empty request id.
            skew_ratio: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
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
        return PrefillDecodeTokenSkewAdvice(
            request_id=rid,
            skew_ratio=float(skew_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
