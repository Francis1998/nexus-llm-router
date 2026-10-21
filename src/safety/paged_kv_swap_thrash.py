"""PagedKvSwapThrashAdvisor.

Advises swap_thrash_ratio vs budgets.
Closes the vLLM/SGLang/TGI paged-KV swap-thrash monitors gap. Distinct from
`PagedAttentionSlotFragmentationAdvisor` / `KvBlockStealAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PagedKvSwapThrashAdvice:
    """PagedKvSwapThrashAdvisor advice."""

    request_id: str
    swap_thrash_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class PagedKvSwapThrashAdvisor:
    """Advise swap_thrash_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        swap_thrash_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> PagedKvSwapThrashAdvice:
        """Return swap_thrash_ratio band.

        Args:
            request_id: Non-empty request id.
            swap_thrash_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if swap_thrash_ratio < 0:
            raise ValueError("swap_thrash_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if swap_thrash_ratio >= hard_limit:
            band = "breach"
        elif swap_thrash_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return PagedKvSwapThrashAdvice(
            request_id=rid,
            swap_thrash_ratio=float(swap_thrash_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
