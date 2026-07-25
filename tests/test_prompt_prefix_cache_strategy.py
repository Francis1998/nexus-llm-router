"""Tests for the prompt-prefix-cache routing strategy."""

import pytest

from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_FAST_MODEL,
    OPENAI_BALANCED_MODEL,
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
    CostOptimalStrategy,
    LatencyStats,
    PromptPrefixCacheStrategy,
    build_strategies,
)


class _AlwaysHealthy:
    """Provider health stub for building all strategies."""

    def is_available(self, provider: str) -> bool:
        """Return all providers as available."""
        return True


def _request(system_prompt: str | None, *, request_id: str = "req-prefix") -> RouterRequest:
    """Build a router request with an optional system prompt."""
    messages = []
    if system_prompt is not None:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.append(ChatMessage(role="user", content="Summarize the policy update."))
    return RouterRequest(request_id=request_id, messages=messages)


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for prompt-prefix-cache tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=512,
    )


def test_prompt_prefix_cache_pins_shared_long_prefix_to_same_model() -> None:
    """Requests sharing a long system prefix should route to one provider/model."""
    strategy = PromptPrefixCacheStrategy(
        default_model_catalog(),
        quality_floor=0.72,
        min_prefix_chars=64,
    )
    shared_prefix = "You are the Acme support policy agent. " * 3

    first = strategy.choose(
        _request(shared_prefix + "Handle billing escalations.", request_id="req-1"),
        _signals(),
    )
    second = strategy.choose(
        _request(shared_prefix + "Handle shipping escalations.", request_id="req-2"),
        _signals(),
    )

    assert first.chosen_model == second.chosen_model
    assert first.provider == second.provider
    assert first.routing_strategy is RoutingStrategyName.PROMPT_PREFIX_CACHE
    assert "prompt-prefix-cache routed prefix" in first.rationale


def test_prompt_prefix_cache_uses_prefix_not_request_id_for_affinity() -> None:
    """Changing request_id must not change a long-prefix affinity decision."""
    strategy = PromptPrefixCacheStrategy(
        default_model_catalog(),
        quality_floor=0.72,
        min_prefix_chars=48,
    )
    system_prompt = "You are a reusable product-doc copilot. " * 3

    chosen_models = set()
    for index in range(5):
        decision = strategy.choose(_request(system_prompt, request_id=f"req-{index}"), _signals())
        chosen_models.add(decision.chosen_model)

    assert len(chosen_models) == 1


def test_prompt_prefix_cache_short_system_prompt_falls_back_to_cost_optimal() -> None:
    """Short prefixes carry no KV-cache affinity signal and use cost-optimal."""
    catalog = default_model_catalog()
    strategy = PromptPrefixCacheStrategy(catalog, quality_floor=0.72, min_prefix_chars=128)
    cost_optimal = CostOptimalStrategy(catalog, quality_floor=0.72)
    request, signals = _request("Short system prompt."), _signals()

    decision = strategy.choose(request, signals)
    expected = cost_optimal.choose(request, signals)

    assert decision.chosen_model == expected.chosen_model
    assert decision.routing_strategy is RoutingStrategyName.PROMPT_PREFIX_CACHE
    assert "no system prompt prefix >=128 chars" in decision.rationale


def test_prompt_prefix_cache_respects_domain_support() -> None:
    """Prefix affinity should only bucket across domain-eligible models."""
    catalog = default_model_catalog()
    strategy = PromptPrefixCacheStrategy(catalog, quality_floor=0.72, min_prefix_chars=40)
    system_prompt = "You are a hospital triage safety assistant. " * 3

    for index in range(20):
        decision = strategy.choose(
            _request(system_prompt + str(index), request_id=f"med-{index}"),
            _signals(DomainTag.MEDICAL),
        )
        assert DomainTag.MEDICAL in catalog[decision.chosen_model].supports_domains


def test_prompt_prefix_cache_rejects_non_positive_minimum() -> None:
    """A non-positive prefix length would make every system prompt cacheable."""
    with pytest.raises(ValueError, match="min_prefix_chars must be positive"):
        PromptPrefixCacheStrategy(default_model_catalog(), quality_floor=0.72, min_prefix_chars=0)


def test_prompt_prefix_cache_strategy_name_parses_header_value() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert RoutingStrategyName("prompt-prefix-cache") is RoutingStrategyName.PROMPT_PREFIX_CACHE


def test_prompt_prefix_cache_is_registered_by_strategy_builder() -> None:
    """The central strategy factory should expose prompt-prefix-cache."""
    strategies = build_strategies(
        default_model_catalog(),
        LatencyStats(),
        0.72,
        OPENAI_BALANCED_MODEL,
        ANTHROPIC_FAST_MODEL,
        0.5,
        _AlwaysHealthy(),
        0.5,
        0.3,
        0.2,
        0.05,
        OPENAI_BALANCED_MODEL,
        OPENAI_FRONTIER_MODEL,
        0.1,
        750.0,
        64,
    )

    assert isinstance(
        strategies[RoutingStrategyName.PROMPT_PREFIX_CACHE],
        PromptPrefixCacheStrategy,
    )
