"""Tests for the token-bucket-tenant routing strategy."""

from unittest.mock import patch

import pytest

from router.config import RouterSettings, default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_PRO_MODEL,
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
    SuccessStats,
    TenantTokenBucketStats,
    TokenBucketTenantStrategy,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


def _request(
    *,
    tenant_id: str = "acme",
    session_id: str = "session-default",
) -> RouterRequest:
    """Build a request carrying tenant identity in metadata."""
    return RouterRequest(
        request_id=f"req-{tenant_id}-{session_id}",
        messages=[ChatMessage(content="Route within my tenant budget.")],
        metadata={"tenant_id": tenant_id},
        session_id=session_id,
    )


def _signals(domain_tag: DomainTag = DomainTag.GENERAL) -> TaskSignals:
    """Build task signals for strategy tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=128,
    )


def _strategy(rate_per_second: float = 1.0) -> TokenBucketTenantStrategy:
    """Build token-bucket-tenant with fresh state."""
    return TokenBucketTenantStrategy(
        default_model_catalog(),
        TenantTokenBucketStats(rate_per_second),
    )


def test_token_bucket_tenant_enum_parses() -> None:
    """The API header parser can resolve the strategy value."""
    assert RoutingStrategyName("token-bucket-tenant") is RoutingStrategyName.TOKEN_BUCKET_TENANT


def test_tenant_token_bucket_rejects_non_positive_rate() -> None:
    """Tenant token refill rate must be positive."""
    with pytest.raises(ValueError, match="rate_per_second must be positive"):
        TenantTokenBucketStats(0.0)


def test_tenant_token_bucket_refills_over_time() -> None:
    """A depleted tenant regains one request token after one second at rate one."""
    stats = TenantTokenBucketStats(1.0)
    with patch("router.strategies.time.monotonic", side_effect=[0.0, 0.0, 1.0]):
        assert stats.try_consume("acme")
        assert stats.available_tokens("acme") == pytest.approx(0.0)
        assert stats.available_tokens("acme") == pytest.approx(1.0)


def test_token_bucket_tenant_in_budget_picks_highest_quality() -> None:
    """The first request in a tenant bucket should retain quality-first routing."""
    strategy = _strategy()
    with patch("router.strategies.time.monotonic", return_value=0.0):
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.TOKEN_BUCKET_TENANT
    assert "within budget" in decision.rationale


def test_token_bucket_tenant_over_budget_sheds_to_cheapest() -> None:
    """A tenant without a quota token should route to the cheapest eligible model."""
    strategy = _strategy()
    with patch("router.strategies.time.monotonic", return_value=0.0):
        strategy.choose(_request(), _signals())
        decision = strategy.choose(_request(), _signals())

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
    assert "over 1.00/s budget" in decision.rationale
    assert "shed to cheapest eligible" in decision.rationale


def test_token_bucket_tenant_isolates_tenant_budgets() -> None:
    """Exhausting one tenant bucket must not affect another tenant."""
    strategy = _strategy()
    with patch("router.strategies.time.monotonic", return_value=0.0):
        strategy.choose(_request(tenant_id="acme"), _signals())
        acme_over_budget = strategy.choose(_request(tenant_id="acme"), _signals())
        globex_first = strategy.choose(_request(tenant_id="globex"), _signals())

    assert acme_over_budget.chosen_model == OPENAI_BALANCED_MODEL
    assert globex_first.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_token_bucket_tenant_prefers_metadata_identity_over_session() -> None:
    """The same metadata tenant should share a bucket across sessions."""
    strategy = _strategy()
    with patch("router.strategies.time.monotonic", return_value=0.0):
        first = strategy.choose(_request(session_id="session-a"), _signals())
        second = strategy.choose(_request(session_id="session-b"), _signals())

    assert first.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert second.chosen_model == OPENAI_BALANCED_MODEL
    assert "tenant 'acme'" in second.rationale


def test_token_bucket_tenant_cheapest_shed_respects_domain() -> None:
    """Over-budget shedding must retain domain eligibility."""
    strategy = _strategy()
    with patch("router.strategies.time.monotonic", return_value=0.0):
        strategy.choose(_request(), _signals(DomainTag.MEDICAL))
        decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert DomainTag.MEDICAL in default_model_catalog()[decision.chosen_model].supports_domains


def test_token_bucket_tenant_registered_by_strategy_factory() -> None:
    """The built-in strategy map should expose token-bucket-tenant."""
    settings = RouterSettings(token_bucket_tenant_rate=3.0)
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
        token_bucket_tenant_rate=settings.token_bucket_tenant_rate,
    )

    strategy = strategies[RoutingStrategyName.TOKEN_BUCKET_TENANT]
    assert isinstance(strategy, TokenBucketTenantStrategy)


def test_token_bucket_tenant_settings_default() -> None:
    """RouterSettings expose the per-tenant request-token rate."""
    assert RouterSettings().token_bucket_tenant_rate == 5.0
