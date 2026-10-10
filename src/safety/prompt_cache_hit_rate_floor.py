"""PromptCacheHitRateFloorAdvisor.

Advises hit_rate vs budgets.
Closes the Anthropic/OpenAI prompt-cache hit-rate floor monitors gap. Distinct from
`PromptCacheHitAdvisor` and `PrefixCacheThrashAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptCacheHitRateFloorAdvice:
    """PromptCacheHitRateFloorAdvisor advice."""

    request_id: str
    hit_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class PromptCacheHitRateFloorAdvisor:
    """Advise hit_rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        hit_rate: float,
        soft_limit: float = 0.4,
        hard_limit: float = 0.7,
    ) -> PromptCacheHitRateFloorAdvice:
        """Return hit_rate band.

        Args:
            request_id: Non-empty request id.
            hit_rate: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if hit_rate < 0:
            raise ValueError("hit_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        # Floor semantics: higher hit_rate is healthier.
        if hit_rate >= hard_limit:
            band = "within"
        elif hit_rate >= soft_limit:
            band = "soft"
        else:
            band = "breach"
        return PromptCacheHitRateFloorAdvice(
            request_id=rid,
            hit_rate=float(hit_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
