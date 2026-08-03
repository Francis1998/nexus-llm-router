"""Tests for the prompt-length-tier-shed routing strategy."""

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    OPENAI_BALANCED_MODEL,
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
    ModelTier,
    PromptLengthTierShedStrategy,
    SuccessStats,
    build_strategies,
    infer_model_tier,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request() -> RouterRequest:
    """Build a minimal router request for strategy tests."""
    return RouterRequest(request_id="req-prompt-tier", messages=[ChatMessage(content="hello")])


def _signals(
    prompt_tokens: int,
    domain_tag: DomainTag = DomainTag.GENERAL,
) -> TaskSignals:
    """Build task signals with a prompt token estimate."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=prompt_tokens,
    )


def test_prompt_length_tier_shed_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("prompt-length-tier-shed")
        is RoutingStrategyName.PROMPT_LENGTH_TIER_SHED
    )


def test_prompt_length_tier_shed_keeps_frontier_for_short_prompts() -> None:
    """Short prompts keep highest-quality (frontier) selection."""
    strategy = PromptLengthTierShedStrategy(default_model_catalog(), prompt_length_tier_tokens=8000)

    decision = strategy.choose(_request(), _signals(120))

    assert decision.routing_strategy is RoutingStrategyName.PROMPT_LENGTH_TIER_SHED
    assert infer_model_tier(decision.chosen_model) is ModelTier.FRONTIER
    assert "within tier gate" in decision.rationale


def test_prompt_length_tier_shed_sheds_frontier_for_long_prompts() -> None:
    """Long prompts shed frontier tiers when mid/economy alternatives exist."""
    strategy = PromptLengthTierShedStrategy(default_model_catalog(), prompt_length_tier_tokens=8000)

    decision = strategy.choose(_request(), _signals(12_000))

    assert decision.routing_strategy is RoutingStrategyName.PROMPT_LENGTH_TIER_SHED
    assert infer_model_tier(decision.chosen_model) is not ModelTier.FRONTIER
    assert "above tier gate" in decision.rationale
    assert "shed frontier" in decision.rationale
    assert decision.chosen_model in {
        OPENAI_BALANCED_MODEL,
        GEMINI_FLASH_MODEL,
        "claude-haiku-4-5",
        "kimi-k2",
    }


def test_prompt_length_tier_shed_respects_domain_support() -> None:
    """Only medical-capable models are considered for a medical prompt."""
    strategy = PromptLengthTierShedStrategy(default_model_catalog(), prompt_length_tier_tokens=8000)

    decision = strategy.choose(_request(), _signals(100, DomainTag.MEDICAL))

    candidate = default_model_catalog()[decision.chosen_model]
    assert DomainTag.MEDICAL in candidate.supports_domains
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_prompt_length_tier_shed_rejects_non_positive_threshold() -> None:
    """A non-positive token threshold fails fast at construction."""
    with pytest.raises(ValueError, match=">= 1"):
        PromptLengthTierShedStrategy(default_model_catalog(), prompt_length_tier_tokens=0)


def test_prompt_length_tier_shed_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose prompt-length-tier-shed."""
    catalog = default_model_catalog()
    settings = RouterSettings(prompt_length_tier_tokens=4000)
    strategies = build_strategies(
        catalog,
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
        settings.prompt_prefix_cache_min_chars,
        settings.epsilon,
        settings.availability_slo,
        SuccessStats(),
        settings.failover_priority,
        settings.health_blend_success_weight,
        settings.health_blend_latency_weight,
        settings.health_blend_quality_weight,
        settings.health_blend_cost_weight,
        settings.concurrency_cap,
        provider_family_cost_ceiling_usd=settings.provider_family_cost_ceiling_usd,
        prompt_length_tier_tokens=settings.prompt_length_tier_tokens,
    )

    strategy = strategies[RoutingStrategyName.PROMPT_LENGTH_TIER_SHED]
    assert isinstance(strategy, PromptLengthTierShedStrategy)
    assert strategy.strategy_name is RoutingStrategyName.PROMPT_LENGTH_TIER_SHED
