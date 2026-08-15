"""Tests for the provider-token-fair-share routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    MOONSHOT_BALANCED_MODEL,
)
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    InflightStats,
    LatencyStats,
    ProviderTokenFairShareStrategy,
    TokenRpmWindow,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(request_id: str = "req-token-fair") -> RouterRequest:
    return RouterRequest(request_id=request_id, messages=[ChatMessage(content="hello")])


def _signals(prompt_tokens: int = 100) -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=prompt_tokens,
    )


def _strategy(
    *,
    ceiling: int = 1000,
    window: TokenRpmWindow | None = None,
) -> ProviderTokenFairShareStrategy:
    return ProviderTokenFairShareStrategy(
        default_model_catalog(),
        window or TokenRpmWindow(),
        token_fair_share_ceiling=ceiling,
    )


def test_provider_token_fair_share_enum_parses() -> None:
    assert (
        RoutingStrategyName("provider-token-fair-share")
        is RoutingStrategyName.PROVIDER_TOKEN_FAIR_SHARE
    )


def test_provider_token_fair_share_cold_start_keeps_quality_leader() -> None:
    decision = _strategy().choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.PROVIDER_TOKEN_FAIR_SHARE
    assert "highest remaining quota" in decision.rationale


def test_provider_token_fair_share_sheds_low_remaining_provider() -> None:
    window = TokenRpmWindow()
    window.record("anthropic", 950)
    strategy = _strategy(ceiling=1000, window=window)

    decision = strategy.choose(_request(), _signals(prompt_tokens=100))

    assert decision.provider != "anthropic"
    assert "remaining quota" in decision.rationale


def test_provider_token_fair_share_rotates_by_remaining_quota() -> None:
    window = TokenRpmWindow()
    strategy = _strategy(ceiling=1000, window=window)

    first = strategy.choose(_request("req-token-fair-1"), _signals(prompt_tokens=200))
    window.record(first.provider, 200)
    second = strategy.choose(_request("req-token-fair-2"), _signals(prompt_tokens=200))
    window.record(second.provider, 200)
    third = strategy.choose(_request("req-token-fair-3"), _signals(prompt_tokens=200))

    assert first.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert len({first.provider, second.provider, third.provider}) >= 2


def test_provider_token_fair_share_falls_back_when_all_over_ceiling() -> None:
    window = TokenRpmWindow()
    for provider, tokens in (
        ("anthropic", 1000),
        ("openai", 990),
        ("google", 980),
        ("moonshot", 970),
    ):
        window.record(provider, tokens)
    strategy = _strategy(ceiling=1000, window=window)

    decision = strategy.choose(_request(), _signals(prompt_tokens=100))

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "fell back to least-loaded" in decision.rationale


def test_provider_token_fair_share_rejects_non_positive_ceiling() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        ProviderTokenFairShareStrategy(
            default_model_catalog(),
            TokenRpmWindow(),
            token_fair_share_ceiling=0,
        )


def test_provider_token_fair_share_registered_by_strategy_factory() -> None:
    settings = RouterSettings(provider_token_fair_share_ceiling=42_000)
    strategies = build_strategies(
        default_model_catalog(),
        LatencyStats(),
        InflightStats(),
        settings.quality_floor,
        settings.ab_model_a,
        settings.ab_model_b,
        settings.ab_model_a_weight,
        CircuitBreakerRegistry(),
        settings.blend_quality_weight,
        settings.blend_cost_weight,
        settings.blend_latency_weight,
        settings.request_cost_ceiling_usd,
        settings.canary_stable_model,
        settings.canary_model,
        settings.canary_weight,
        settings.latency_sla_ms,
        provider_token_fair_share_ceiling=settings.provider_token_fair_share_ceiling,
        token_rpm_window=TokenRpmWindow(),
    )

    strategy = strategies[RoutingStrategyName.PROVIDER_TOKEN_FAIR_SHARE]
    assert isinstance(strategy, ProviderTokenFairShareStrategy)
    assert strategy._token_fair_share_ceiling == 42_000  # noqa: SLF001
    assert RouterSettings().provider_token_fair_share_ceiling == 100_000
