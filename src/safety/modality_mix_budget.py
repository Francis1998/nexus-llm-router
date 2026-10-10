"""ModalityMixBudgetAdvisor.

Advises mix_ratio vs budgets.
Closes the LiteLLM/OpenRouter multimodal modality-mix budget monitors gap. Distinct from
`AudioTokenBudgetAdvisor` and `ReasoningTokenBudgetAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModalityMixBudgetAdvice:
    """ModalityMixBudgetAdvisor advice."""

    request_id: str
    mix_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class ModalityMixBudgetAdvisor:
    """Advise mix_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        mix_ratio: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.7,
    ) -> ModalityMixBudgetAdvice:
        """Return mix_ratio band.

        Args:
            request_id: Non-empty request id.
            mix_ratio: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if mix_ratio < 0:
            raise ValueError("mix_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if mix_ratio >= hard_limit:
            band = "breach"
        elif mix_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ModalityMixBudgetAdvice(
            request_id=rid,
            mix_ratio=float(mix_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
