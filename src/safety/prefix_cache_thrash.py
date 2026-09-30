"""Prefix cache thrash advisor.

Advises prefix-cache thrash (evict/hit) bands. Closes the vLLM /
OpenRouter / LiteLLM prefix-cache thrash gap.
Distinct from ``PromptCacheHitRateAdvisor`` and ``KvCacheHitRateAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefixCacheThrashAdvice:
    """Prefix cache thrash advice."""

    request_id: str
    evict_count: int
    hit_count: int
    thrash_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class PrefixCacheThrashAdvisor:
    """Advise prefix-cache thrash bands."""

    def advise(
        self,
        *,
        request_id: str,
        evict_count: int,
        hit_count: int,
        soft_limit: float = 0.5,
        hard_limit: float = 2.0,
    ) -> PrefixCacheThrashAdvice:
        """Return thrash band.

        Args:
            request_id: Non-empty request id.
            evict_count: Prefix cache evictions (``>= 0``).
            hit_count: Prefix cache hits (``>= 0``).
            soft_limit: Soft evict/hit ratio (``> 0``).
            hard_limit: Hard evict/hit ratio (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if evict_count < 0:
            raise ValueError("evict_count must be >= 0")
        if hit_count < 0:
            raise ValueError("hit_count must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        denom = max(hit_count, 1)
        thrash_ratio = round(evict_count / denom, 4)
        if thrash_ratio >= hard_limit:
            band = "breach"
        elif thrash_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return PrefixCacheThrashAdvice(
            request_id=rid,
            evict_count=int(evict_count),
            hit_count=int(hit_count),
            thrash_ratio=float(thrash_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
