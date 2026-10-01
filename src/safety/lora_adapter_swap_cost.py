"""LoRA adapter swap-cost advisor.

Advises LoRA adapter swap cost (ms) vs soft/hard budgets.
Closes the vLLM / TensorRT-LLM / OpenRouter LoRA adapter swap
cost gap. Distinct from ``PrefixCacheThrashAdvisor`` and
``ProviderColdStartLatencyAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoraAdapterSwapCostAdvice:
    """LoRA adapter swap-cost advice."""

    request_id: str
    swap_ms: float
    soft_limit_ms: float
    hard_limit_ms: float
    band: str


class LoraAdapterSwapCostAdvisor:
    """Advise LoRA adapter swap-cost bands."""

    def advise(
        self,
        *,
        request_id: str,
        swap_ms: float,
        soft_limit_ms: float = 50.0,
        hard_limit_ms: float = 200.0,
    ) -> LoraAdapterSwapCostAdvice:
        """Return swap-cost band.

        Args:
            request_id: Non-empty request id.
            swap_ms: Observed/forecast swap latency (``>= 0``).
            soft_limit_ms: Soft swap budget (``> 0``).
            hard_limit_ms: Hard swap budget (``> soft_limit_ms``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if swap_ms < 0:
            raise ValueError("swap_ms must be >= 0")
        if soft_limit_ms <= 0:
            raise ValueError("soft_limit_ms must be > 0")
        if hard_limit_ms <= soft_limit_ms:
            raise ValueError("hard_limit_ms must be > soft_limit_ms")

        if swap_ms >= hard_limit_ms:
            band = "breach"
        elif swap_ms >= soft_limit_ms:
            band = "soft"
        else:
            band = "within"
        return LoraAdapterSwapCostAdvice(
            request_id=rid,
            swap_ms=float(swap_ms),
            soft_limit_ms=float(soft_limit_ms),
            hard_limit_ms=float(hard_limit_ms),
            band=band,
        )
