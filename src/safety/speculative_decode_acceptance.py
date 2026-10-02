"""Speculative-decode acceptance-band advisor.

Advises speculative-decode acceptance rate vs soft/hard floors.
Closes the vLLM / TensorRT-LLM / OpenRouter speculative-decode acceptance-rate bands gap.
Distinct from
``SpeculativeDecodeAbortAdvisor`` and ``SpeculativeDecodeBudgetAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeDecodeAcceptanceAdvice:
    """Speculative-decode acceptance advice."""

    request_id: str
    acceptance_rate: float
    soft_floor: float
    hard_floor: float
    band: str


class SpeculativeDecodeAcceptanceBandAdvisor:
    """Advise speculative-decode acceptance-rate bands."""

    def advise(
        self,
        *,
        request_id: str,
        acceptance_rate: float,
        soft_floor: float = 0.7,
        hard_floor: float = 0.4,
    ) -> SpeculativeDecodeAcceptanceAdvice:
        """Return acceptance-rate band.

        Args:
            request_id: Non-empty request id.
            acceptance_rate: Accepted draft tokens / proposed (``0..1``).
            soft_floor: Soft min acceptance (``0 < soft_floor <= 1``).
            hard_floor: Hard min acceptance (``0 <= hard_floor < soft_floor``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if acceptance_rate < 0.0 or acceptance_rate > 1.0:
            raise ValueError("acceptance_rate must be in [0, 1]")
        if not 0.0 < soft_floor <= 1.0:
            raise ValueError("soft_floor must be in (0, 1]")
        if not 0.0 <= hard_floor < soft_floor:
            raise ValueError("hard_floor must be in [0, soft_floor)")

        if acceptance_rate >= soft_floor:
            band = "within"
        elif acceptance_rate >= hard_floor:
            band = "soft"
        else:
            band = "breach"
        return SpeculativeDecodeAcceptanceAdvice(
            request_id=rid,
            acceptance_rate=float(acceptance_rate),
            soft_floor=float(soft_floor),
            hard_floor=float(hard_floor),
            band=band,
        )
