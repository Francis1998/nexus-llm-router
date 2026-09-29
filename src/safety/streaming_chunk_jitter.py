"""Streaming chunk-jitter SLO advisor.

Advises inter-chunk jitter bands vs an SLO budget. Closes the vLLM /
OpenRouter / LiteLLM streaming chunk-jitter gap. Distinct from
``FirstTokenLatencySloAdvisor`` (TTFT) and ``StreamingBackpressureAdvisor``
(backpressure). Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2. Never performs network I/O.
"""


from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamingChunkJitterAdvice:
    """Streaming chunk-jitter advice."""

    request_id: str
    jitter_ms: float
    slo_ms: float
    ratio: float
    band: str


class StreamingChunkJitterSloAdvisor:
    """Advise streaming inter-chunk jitter vs SLO bands."""

    def advise(
        self,
        *,
        request_id: str,
        jitter_ms: float,
        slo_ms: float,
    ) -> StreamingChunkJitterAdvice:
        """Return band for jitter vs SLO.

        Args:
            request_id: Non-empty request id.
            jitter_ms: Observed inter-chunk jitter milliseconds (``>= 0``).
            slo_ms: Jitter SLO milliseconds (``> 0``).

        Returns:
            StreamingChunkJitterAdvice with ``within`` / ``soft`` / ``breach``.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if jitter_ms < 0:
            raise ValueError("jitter_ms must be >= 0")
        if slo_ms <= 0:
            raise ValueError("slo_ms must be > 0")

        ratio = round(jitter_ms / slo_ms, 4)
        if ratio <= 1.0:
            band = "within"
        elif ratio <= 1.5:
            band = "soft"
        else:
            band = "breach"
        return StreamingChunkJitterAdvice(
            request_id=rid,
            jitter_ms=float(jitter_ms),
            slo_ms=float(slo_ms),
            ratio=float(ratio),
            band=band,
        )
