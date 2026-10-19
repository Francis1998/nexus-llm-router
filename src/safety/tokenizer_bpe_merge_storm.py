"""TokenizerBpeMergeStormAdvisor.

Advises bpe_merge_storm_rate vs budgets.
Closes the vLLM/SGLang/TGI tokenizer BPE merge-storm monitors gap. Distinct from
`TokenizerBpeMergeDriftAdvisor` / `PrefixCacheHitFloorAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizerBpeMergeStormAdvice:
    """TokenizerBpeMergeStormAdvisor advice."""

    request_id: str
    bpe_merge_storm_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class TokenizerBpeMergeStormAdvisor:
    """Advise bpe_merge_storm_rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        bpe_merge_storm_rate: float,
        soft_limit: float = 0.25,
        hard_limit: float = 0.6,
    ) -> TokenizerBpeMergeStormAdvice:
        """Return bpe_merge_storm_rate band.

        Args:
            request_id: Non-empty request id.
            bpe_merge_storm_rate: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if bpe_merge_storm_rate < 0:
            raise ValueError("bpe_merge_storm_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if bpe_merge_storm_rate >= hard_limit:
            band = "breach"
        elif bpe_merge_storm_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return TokenizerBpeMergeStormAdvice(
            request_id=rid,
            bpe_merge_storm_rate=float(bpe_merge_storm_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
