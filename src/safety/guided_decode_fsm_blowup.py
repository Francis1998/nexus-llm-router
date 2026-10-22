"""GuidedDecodeFsmBlowupAdvisor.

Advises fsm_blowup_ratio vs budgets.
Closes the vLLM/Outlines/Guidance guided-decode FSM blowup monitors gap. Distinct from
`DecodeSlotFragmentationBandAdvisor` / `PagedKvSwapThrashAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GuidedDecodeFsmBlowupAdvice:
    """GuidedDecodeFsmBlowupAdvisor advice."""

    request_id: str
    fsm_blowup_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class GuidedDecodeFsmBlowupAdvisor:
    """Advise fsm_blowup_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        fsm_blowup_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> GuidedDecodeFsmBlowupAdvice:
        """Return fsm_blowup_ratio band.

        Args:
            request_id: Non-empty request id.
            fsm_blowup_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if fsm_blowup_ratio < 0:
            raise ValueError("fsm_blowup_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if fsm_blowup_ratio >= hard_limit:
            band = "breach"
        elif fsm_blowup_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return GuidedDecodeFsmBlowupAdvice(
            request_id=rid,
            fsm_blowup_ratio=float(fsm_blowup_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
