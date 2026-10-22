"""MoeExpertDropoutStormAdvisor.

Advises dropout_storm_ratio vs budgets.
Closes the vLLM/SGLang/TGI MoE expert-dropout storm monitors gap. Distinct from
`MoeExpertLoadImbalanceAdvisor` / `MoeRoutingCollapseAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MoeExpertDropoutStormAdvice:
    """MoeExpertDropoutStormAdvisor advice."""

    request_id: str
    dropout_storm_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class MoeExpertDropoutStormAdvisor:
    """Advise dropout_storm_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        dropout_storm_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> MoeExpertDropoutStormAdvice:
        """Return dropout_storm_ratio band.

        Args:
            request_id: Non-empty request id.
            dropout_storm_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if dropout_storm_ratio < 0:
            raise ValueError("dropout_storm_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if dropout_storm_ratio >= hard_limit:
            band = "breach"
        elif dropout_storm_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return MoeExpertDropoutStormAdvice(
            request_id=rid,
            dropout_storm_ratio=float(dropout_storm_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
