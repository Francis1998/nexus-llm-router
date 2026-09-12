"""Offline request USD cost forecast from token counts and $/1M rates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CostForecast:
    """Estimated USD cost breakdown for one prospective request."""

    estimated_usd: float
    prompt_cost: float
    completion_cost: float
    assumptions: dict[str, Any]


class RequestCostForecastAdvisor:
    """Offline estimate of request USD cost before dispatch.

    Closes the LiteLLM / OpenRouter pre-request cost-estimate gap for Nexus
    gateways serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2:
    given ``prompt_tokens``, ``max_completion_tokens``, and $/1M input/output
    rates, return a structured ``CostForecast`` callers can log, surface in
    UI, or feed into soft budget advisors.

    Distinct from ``SpendLedger`` (historical durable spend) and
    ``TenantSpendQuotaEnforcer`` (hard monthly USD reject). This advisor never
    mutates state and never rejects traffic.
    """

    def __init__(
        self,
        *,
        default_input_usd_per_1m: float | None = None,
        default_output_usd_per_1m: float | None = None,
    ) -> None:
        """Optionally pin default $/1M rates used when ``forecast`` omits them.

        Args:
            default_input_usd_per_1m: Default prompt rate ($ per 1M tokens).
            default_output_usd_per_1m: Default completion rate ($ per 1M tokens).

        Raises:
            ValueError: If a provided default rate is negative.
        """
        if default_input_usd_per_1m is not None and default_input_usd_per_1m < 0:
            raise ValueError("default_input_usd_per_1m must be >= 0")
        if default_output_usd_per_1m is not None and default_output_usd_per_1m < 0:
            raise ValueError("default_output_usd_per_1m must be >= 0")
        self._default_input = (
            None if default_input_usd_per_1m is None else float(default_input_usd_per_1m)
        )
        self._default_output = (
            None if default_output_usd_per_1m is None else float(default_output_usd_per_1m)
        )

    @property
    def default_input_usd_per_1m(self) -> float | None:
        """Return the configured default prompt $/1M rate, if any."""
        return self._default_input

    @property
    def default_output_usd_per_1m(self) -> float | None:
        """Return the configured default completion $/1M rate, if any."""
        return self._default_output

    def forecast(
        self,
        *,
        prompt_tokens: int,
        max_completion_tokens: int,
        input_usd_per_1m: float | None = None,
        output_usd_per_1m: float | None = None,
        model: str | None = None,
    ) -> CostForecast:
        """Estimate USD cost assuming the completion uses the full max tokens.

        Args:
            prompt_tokens: Counted (or estimated) prompt tokens (``>= 0``).
            max_completion_tokens: Planned completion ceiling (``>= 0``).
            input_usd_per_1m: Prompt price in USD per 1M tokens. Falls back to
                the advisor default when omitted.
            output_usd_per_1m: Completion price in USD per 1M tokens. Falls
                back to the advisor default when omitted.
            model: Optional model id recorded in ``assumptions`` only.

        Returns:
            Immutable ``CostForecast`` with prompt/completion split and
            explicit assumptions (rates, token counts, full-max utilization).

        Raises:
            ValueError: If token counts are negative, rates are missing /
                negative, or defaults were never configured.
        """
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        if max_completion_tokens < 0:
            raise ValueError("max_completion_tokens must be >= 0")

        in_rate = float(input_usd_per_1m) if input_usd_per_1m is not None else self._default_input
        out_rate = (
            float(output_usd_per_1m) if output_usd_per_1m is not None else self._default_output
        )
        if in_rate is None:
            raise ValueError("input_usd_per_1m is required (no default configured)")
        if out_rate is None:
            raise ValueError("output_usd_per_1m is required (no default configured)")
        if in_rate < 0:
            raise ValueError("input_usd_per_1m must be >= 0")
        if out_rate < 0:
            raise ValueError("output_usd_per_1m must be >= 0")

        prompt_cost = (prompt_tokens / 1_000_000.0) * in_rate
        completion_cost = (max_completion_tokens / 1_000_000.0) * out_rate
        estimated = prompt_cost + completion_cost
        assumptions: dict[str, Any] = {
            "prompt_tokens": int(prompt_tokens),
            "max_completion_tokens": int(max_completion_tokens),
            "input_usd_per_1m": float(in_rate),
            "output_usd_per_1m": float(out_rate),
            "assumes_full_max_completion": True,
            "formula": "(prompt_tokens/1e6)*input_rate + (max_completion_tokens/1e6)*output_rate",
        }
        if model is not None:
            assumptions["model"] = model
        return CostForecast(
            estimated_usd=estimated,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            assumptions=assumptions,
        )
