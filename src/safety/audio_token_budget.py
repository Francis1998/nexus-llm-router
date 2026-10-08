"""AudioTokenBudgetAdvisor.
Advises audio_token_ratio vs budgets.
Closes the OpenAI Realtime/Gemini audio token budgets gap. Distinct from
`MultimodalTokenTaxAdvisor` and `OutputTokenCeilingAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioTokenBudgetAdvice:
    """AudioTokenBudgetAdvisor advice."""

    request_id: str
    audio_token_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class AudioTokenBudgetAdvisor:
    """Advise audio_token_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        audio_token_ratio: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.6,
    ) -> AudioTokenBudgetAdvice:
        """Return audio_token_ratio band.

        Args:
            request_id: Non-empty request id.
            audio_token_ratio: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if audio_token_ratio < 0:
            raise ValueError("audio_token_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if audio_token_ratio >= hard_limit:
            band = "breach"
        elif audio_token_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return AudioTokenBudgetAdvice(
            request_id=rid,
            audio_token_ratio=float(audio_token_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
