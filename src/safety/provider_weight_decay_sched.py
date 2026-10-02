"""Provider weight-decay scheduler advisor.

Advises provider exploration weight-decay step vs soft/hard budgets.
Closes the OpenRouter / LiteLLM / Portkey provider weight-decay scheduler bands gap. Distinct from
``ProviderExplorationEpsilonAdvisor`` and ``ProviderHealthHysteresisAdvisor``.
Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderWeightDecaySchedAdvice:
    """Provider weight-decay scheduler advice."""

    request_id: str
    decay_step: float
    soft_limit: float
    hard_limit: float
    band: str


class ProviderWeightDecaySchedulerAdvisor:
    """Advise provider weight-decay step bands."""

    def advise(
        self,
        *,
        request_id: str,
        decay_step: float,
        soft_limit: float = 0.05,
        hard_limit: float = 0.2,
    ) -> ProviderWeightDecaySchedAdvice:
        """Return decay-step band.

        Args:
            request_id: Non-empty request id.
            decay_step: Absolute weight decay applied this tick (``>= 0``).
            soft_limit: Soft decay budget (``> 0``).
            hard_limit: Hard decay budget (``> soft_limit``).
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if decay_step < 0:
            raise ValueError("decay_step must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")

        if decay_step >= hard_limit:
            band = "breach"
        elif decay_step >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ProviderWeightDecaySchedAdvice(
            request_id=rid,
            decay_step=float(decay_step),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
