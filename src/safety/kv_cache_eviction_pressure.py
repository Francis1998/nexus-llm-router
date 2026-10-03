"""KvCacheEvictionPressure advisor.

Advises eviction rate vs budgets.
Closes the vLLM/TensorRT-LLM/SGLang KV-cache eviction pressure advisors gap. Distinct from
``KvCacheHitRateAdvisor and PrefixCacheThrashAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KvCacheEvictionPressureAdvice:
    """KvCacheEvictionPressureAdvisor advice."""

    request_id: str
    eviction_rate: float
    soft_limit: float
    hard_limit: float
    band: str


class KvCacheEvictionPressureAdvisor:
    """Advise eviction rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        eviction_rate: float,
        soft_limit: float = 0.05,
        hard_limit: float = 0.2,
    ) -> KvCacheEvictionPressureAdvice:
        """Return eviction_rate band.

        Args:
            request_id: Non-empty request id.
            eviction_rate: Observed ratio (``>= 0``).
            soft_limit: Soft budget (``> 0``).
            hard_limit: Hard budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if eviction_rate < 0:
            raise ValueError("eviction_rate must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if eviction_rate >= hard_limit:
            band = "breach"
        elif eviction_rate >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return KvCacheEvictionPressureAdvice(
            request_id=rid,
            eviction_rate=float(eviction_rate),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
