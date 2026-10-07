"""ProviderAuthTokenExpiry advisor.

Advises minutes_to_expiry vs budgets.
Closes the OpenAI/Anthropic/Gemini provider auth-token expiry advisors gap. Distinct from
``ProviderQuotaRemainingAdvisor`` and ``VirtualKeysAdvisor``.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderAuthTokenExpiryAdvice:
    """ProviderAuthTokenExpiryAdvisor advice."""

    request_id: str
    minutes_to_expiry: float
    soft_limit: float
    hard_limit: float
    band: str


class ProviderAuthTokenExpiryAdvisor:
    """Advise minutes_to_expiry bands."""

    def advise(
        self,
        *,
        request_id: str,
        minutes_to_expiry: float,
        soft_limit: float = 60.0,
        hard_limit: float = 15.0,
    ) -> ProviderAuthTokenExpiryAdvice:
        """Return minutes_to_expiry band.

        Args:
            request_id: Non-empty request id.
            minutes_to_expiry: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if minutes_to_expiry < 0:
            raise ValueError("minutes_to_expiry must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= 0:
            raise ValueError("hard_limit must be > 0")
        if hard_limit >= soft_limit:
            raise ValueError("hard_limit must be < soft_limit")
        if minutes_to_expiry >= soft_limit:
            band = "within"
        elif minutes_to_expiry >= hard_limit:
            band = "soft"
        else:
            band = "breach"
        return ProviderAuthTokenExpiryAdvice(
            request_id=rid,
            minutes_to_expiry=float(minutes_to_expiry),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
