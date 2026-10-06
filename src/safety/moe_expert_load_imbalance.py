"""MoeExpertLoadImbalance advisor.

Advises MoE expert imbalance_ratio vs budgets.
Closes the vLLM/TGI/SGLang MoE expert-load imbalance advisors gap. Distinct from
``ProviderExplorationEpsilonAdvisor`` and ``PrefillDecodeTokenSkewAdvisor``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MoeExpertLoadImbalanceAdvice:
    """MoeExpertLoadImbalanceAdvisor advice."""

    request_id: str
    imbalance_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class MoeExpertLoadImbalanceAdvisor:
    """Advise MoE expert imbalance_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        imbalance_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> MoeExpertLoadImbalanceAdvice:
        """Return imbalance_ratio band.

        Args:
            request_id: Non-empty request id.
            imbalance_ratio: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if imbalance_ratio < 0:
            raise ValueError("imbalance_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if imbalance_ratio >= hard_limit:
            band = "breach"
        elif imbalance_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return MoeExpertLoadImbalanceAdvice(
            request_id=rid,
            imbalance_ratio=float(imbalance_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
