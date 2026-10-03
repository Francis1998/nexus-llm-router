"""PromptCacheKeyCollision advisor.

Advises collision rate vs budgets.
Closes the vLLM/OpenRouter/LiteLLM prompt-cache key collision controls gap. Distinct from
``PromptCacheHitRateAdvisor and PrefixCacheThrashAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptCacheKeyCollisionAdvice:
    """PromptCacheKeyCollisionAdvisor advice."""

    request_id: str
    collision_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class PromptCacheKeyCollisionAdvisor:
    """Advise collision rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        collision_rate: float,
        soft_limit: float = 0.02,
        hard_limit: float = 0.1,
    ) -> PromptCacheKeyCollisionAdvice:
        """Return collision_rate band.

        Args:
            request_id: Non-empty request id.
            collision_rate: Observed ratio (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if collision_rate < 0:
            raise ValueError("collision_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if collision_rate >= hard_limit:
            band = "breach"
        elif collision_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return PromptCacheKeyCollisionAdvice(
            request_id=rid,
            collision_rate=float(collision_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
