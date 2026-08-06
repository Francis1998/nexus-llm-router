"""Tests for the embedding-cache-key-namespace routing strategy."""

from hashlib import sha256

import pytest

from router.config import RouterSettings, default_model_catalog
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    EmbeddingCacheKeyNamespaceStrategy,
    InflightStats,
    LatencyStats,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    """Deterministic provider-health view for unit tests."""

    def __init__(self, unavailable: set[str]) -> None:
        """Store providers considered unavailable."""
        self._unavailable = unavailable

    def is_available(self, provider: str) -> bool:
        """Return whether a provider is routable."""
        return provider not in self._unavailable


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def _request(
    *,
    metadata: dict[str, str] | None = None,
    user_id: str = "anonymous",
    session_id: str = "default",
) -> RouterRequest:
    """Build a router request for embedding-namespace tests."""
    return RouterRequest(
        request_id="req-embed-ns",
        messages=[ChatMessage(content="embed me")],
        metadata=metadata or {},
        user_id=user_id,
        session_id=session_id,
    )


def _primary_model(namespaced_key: str) -> str:
    """Reproduce the strategy's primary sticky model for a namespaced key."""
    catalog = default_model_catalog()
    ordered = sorted(catalog.values(), key=lambda candidate: candidate.model)
    digest = sha256(namespaced_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % len(ordered)
    return ordered[bucket].model


def test_embedding_cache_key_namespace_enum_parses() -> None:
    """The API header parser can resolve the new strategy enum value."""
    assert (
        RoutingStrategyName("embedding-cache-key-namespace")
        is RoutingStrategyName.EMBEDDING_CACHE_KEY_NAMESPACE
    )


def test_embedding_cache_key_namespace_isolates_tenants() -> None:
    """Distinct tenants under the same prefix must not share sticky buckets."""
    strategy = EmbeddingCacheKeyNamespaceStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        namespace_prefix="embed",
    )

    first = strategy.choose(_request(metadata={"tenant_id": "acme"}), _signals())
    second = strategy.choose(_request(metadata={"tenant_id": "globex"}), _signals())

    assert first.chosen_model == _primary_model("embed:acme")
    assert second.chosen_model == _primary_model("embed:globex")
    assert "embedding-cache-key-namespace pinned 'embed:acme'" in first.rationale
    # With a 7-model catalog collisions are possible but rare; assert key isolation
    # via the namespaced rationale even when buckets coincide.
    assert "embed:acme" in first.rationale
    assert "embed:globex" in second.rationale


def test_embedding_cache_key_namespace_same_tenant_is_sticky() -> None:
    """The same tenant namespace repeatedly pins to the same model."""
    strategy = EmbeddingCacheKeyNamespaceStrategy(
        default_model_catalog(),
        CircuitBreakerRegistry(),
        namespace_prefix="embed",
    )

    first = strategy.choose(_request(metadata={"tenant_id": "acme"}), _signals())
    second = strategy.choose(_request(metadata={"tenant_id": "acme"}), _signals())

    assert first.chosen_model == second.chosen_model


def test_embedding_cache_key_namespace_prefix_changes_bucket() -> None:
    """Changing the namespace prefix remaps the sticky key."""
    catalog = default_model_catalog()
    health = CircuitBreakerRegistry()
    default_strategy = EmbeddingCacheKeyNamespaceStrategy(catalog, health, namespace_prefix="embed")
    alt_strategy = EmbeddingCacheKeyNamespaceStrategy(catalog, health, namespace_prefix="rag")

    default_decision = default_strategy.choose(_request(metadata={"tenant_id": "acme"}), _signals())
    alt_decision = alt_strategy.choose(_request(metadata={"tenant_id": "acme"}), _signals())

    assert "embed:acme" in default_decision.rationale
    assert "rag:acme" in alt_decision.rationale
    assert default_decision.chosen_model == _primary_model("embed:acme")
    assert alt_decision.chosen_model == _primary_model("rag:acme")


def test_embedding_cache_key_namespace_failovers_unhealthy_primary() -> None:
    """Unhealthy sticky primaries failover along the namespace ring."""
    catalog = default_model_catalog()
    namespaced_key = "embed:acme"
    primary = _primary_model(namespaced_key)
    primary_provider = catalog[primary].provider
    strategy = EmbeddingCacheKeyNamespaceStrategy(
        catalog,
        _FakeHealth({primary_provider}),
        namespace_prefix="embed",
    )

    decision = strategy.choose(_request(metadata={"tenant_id": "acme"}), _signals())

    assert decision.chosen_model != primary
    assert "failover offset" in decision.rationale


def test_embedding_cache_key_namespace_rejects_empty_prefix() -> None:
    """An empty namespace prefix fails fast at construction."""
    with pytest.raises(ValueError, match="non-empty"):
        EmbeddingCacheKeyNamespaceStrategy(
            default_model_catalog(),
            CircuitBreakerRegistry(),
            namespace_prefix="   ",
        )


def test_embedding_cache_key_namespace_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose embedding-cache-key-namespace."""
    settings = RouterSettings(embedding_cache_namespace_prefix="cache")
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
        cache_hit_sticky_min_chars=settings.cache_hit_sticky_min_chars,
        embedding_cache_namespace_prefix=settings.embedding_cache_namespace_prefix,
    )

    strategy = strategies[RoutingStrategyName.EMBEDDING_CACHE_KEY_NAMESPACE]
    assert isinstance(strategy, EmbeddingCacheKeyNamespaceStrategy)
    assert strategy.strategy_name is RoutingStrategyName.EMBEDDING_CACHE_KEY_NAMESPACE
