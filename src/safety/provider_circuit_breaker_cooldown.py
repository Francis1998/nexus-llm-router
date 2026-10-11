"""ProviderCircuitBreakerCooldownAdvisor.

Advises cooldown_ratio vs budgets.
Closes the LiteLLM/Portkey/Helicone provider circuit-breaker cooldown monitors gap. Distinct from
`ProviderHealthHysteresisAdvisor` and `ProviderQuotaRemainingAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderCircuitBreakerCooldownAdvice:
    """ProviderCircuitBreakerCooldownAdvisor advice."""

    request_id: str
    cooldown_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class ProviderCircuitBreakerCooldownAdvisor:
    """Advise cooldown_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        cooldown_ratio: float,
        soft_limit: float = 0.3,
        hard_limit: float = 0.7,
    ) -> ProviderCircuitBreakerCooldownAdvice:
        """Return cooldown_ratio band.

        Args:
            request_id: Non-empty request id.
            cooldown_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if cooldown_ratio < 0:
            raise ValueError("cooldown_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if cooldown_ratio >= hard_limit:
            band = "breach"
        elif cooldown_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ProviderCircuitBreakerCooldownAdvice(
            request_id=rid,
            cooldown_ratio=float(cooldown_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
