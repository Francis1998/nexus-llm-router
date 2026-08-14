"""Tests for the token-rpm-ceiling routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
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
    TokenRpmCeilingStrategy,
    TokenRpmWindow,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    return RouterRequest(request_id="req-token-rpm", messages=[ChatMessage(content="hello")])


def _signals(prompt_tokens: int = 100) -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=prompt_tokens,
    )


def test_token_rpm_ceiling_enum_parses() -> None:
    assert RoutingStrategyName("token-rpm-ceiling") is RoutingStrategyName.TOKEN_RPM_CEILING


def test_token_rpm_ceiling_cold_start_keeps_quality_leader() -> None:
    strategy = TokenRpmCeilingStrategy(default_model_catalog(), TokenRpmWindow(), 100_000)

    decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.TOKEN_RPM_CEILING
    assert "100/100000" in decision.rationale


def test_token_rpm_ceiling_sheds_provider_that_would_cross_limit() -> None:
    window = TokenRpmWindow()
    window.record("anthropic", 99_950)
    strategy = TokenRpmCeilingStrategy(default_model_catalog(), window, 100_000)

    decision = strategy.choose(_request(), _signals(100))

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert decision.provider == "openai"
    assert "rolling 60s window" in decision.rationale


def test_token_rpm_ceiling_allows_projected_total_equal_to_limit() -> None:
    window = TokenRpmWindow()
    window.record("anthropic", 99_900)
    strategy = TokenRpmCeilingStrategy(default_model_catalog(), window, 100_000)

    decision = strategy.choose(_request(), _signals(100))

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "100000/100000" in decision.rationale


def test_token_rpm_ceiling_all_over_uses_least_loaded_provider() -> None:
    window = TokenRpmWindow()
    for provider, tokens in {
        "anthropic": 200,
        "openai": 300,
        "google": 400,
        "moonshot": 100,
    }.items():
        window.record(provider, tokens)
    strategy = TokenRpmCeilingStrategy(default_model_catalog(), window, 50)

    decision = strategy.choose(_request(), _signals(10))

    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert "every eligible provider over" in decision.rationale


def test_token_rpm_window_prunes_samples_older_than_sixty_seconds() -> None:
    window = TokenRpmWindow(window_seconds=60.0)
    window.record("openai", 500, now=0.0)
    window.record("openai", 200, now=61.0)

    assert window.provider_tokens("openai", now=61.0) == 200
    assert window.would_exceed("openai", 300, 500, now=61.0) is False


def test_token_rpm_ceiling_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        TokenRpmCeilingStrategy(default_model_catalog(), TokenRpmWindow(), 0)


def test_token_rpm_ceiling_registered_by_strategy_factory() -> None:
    settings = RouterSettings(token_rpm_ceiling=42_000)
    window = TokenRpmWindow()
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
        token_rpm_ceiling=settings.token_rpm_ceiling,
        token_rpm_window=window,
    )

    strategy = strategies[RoutingStrategyName.TOKEN_RPM_CEILING]
    assert isinstance(strategy, TokenRpmCeilingStrategy)
    assert strategy._token_rpm_ceiling == 42_000  # noqa: SLF001
    assert RouterSettings().token_rpm_ceiling == 100_000
