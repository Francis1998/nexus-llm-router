"""KV-cache hit-rate advisor.

Advises KV-cache hit-rate bands vs a target ratio. Closes the vLLM /
TensorRT-LLM / OpenRouter prompt-cache hit-rate gap. Distinct from
``PromptCacheHitAdvisor`` (binary hit) and ``PrefillTtftRatioAdvisor``
(TTFT). Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KvCacheHitRateAdvice:
    """KV-cache hit-rate advice."""

    request_id: str
    hit_rate: float
    target_rate: float
    ratio: float
    band: str


class KvCacheHitRateAdvisor:
    """Advise KV-cache hit rate vs target bands."""

    def advise(
        self,
        *,
        request_id: str,
        hit_rate: float,
        target_rate: float,
    ) -> KvCacheHitRateAdvice:
        """Return band for hit rate vs target.

        Args:
            request_id: Non-empty request id.
            hit_rate: Observed hit rate in ``[0, 1]``.
            target_rate: Target hit rate in ``(0, 1]``.

        Returns:
            KvCacheHitRateAdvice with ``healthy`` / ``soft`` / ``cold``.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if not 0.0 <= hit_rate <= 1.0:
            raise ValueError("hit_rate must be in [0, 1]")
        if not 0.0 < target_rate <= 1.0:
            raise ValueError("target_rate must be in (0, 1]")

        ratio = round(hit_rate / target_rate, 4)
        if ratio >= 1.0:
            band = "healthy"
        elif ratio >= 0.7:
            band = "soft"
        else:
            band = "cold"
        return KvCacheHitRateAdvice(
            request_id=rid,
            hit_rate=float(hit_rate),
            target_rate=float(target_rate),
            ratio=float(ratio),
            band=band,
        )
