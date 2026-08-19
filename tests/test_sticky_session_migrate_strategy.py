"""Tests for sticky-session-migrate routing."""

import pytest

from router.config import RouterSettings
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    ModelCandidate,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
    InflightStats,
    LatencyStats,
    StickySessionMigrateStats,
    StickySessionMigrateStrategy,
    SuccessStats,
    build_strategies,
)
from safety.circuit_breaker import CircuitBreakerRegistry


class _FakeHealth:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self.unavailable = unavailable or set()

    def is_available(self, provider: str) -> bool:
        return provider not in self.unavailable


def _catalog() -> dict[str, ModelCandidate]:
    return {
        "quality-primary": ModelCandidate(
            model="quality-primary",
            provider="openai",
            quality_score=0.95,
            input_cost_per_1k=0.006,
            output_cost_per_1k=0.018,
            supports_domains={DomainTag.GENERAL},
        ),
        "healthy-alternate": ModelCandidate(
            model="healthy-alternate",
            provider="google",
            quality_score=0.85,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.01,
            supports_domains={DomainTag.GENERAL},
        ),
    }


def _request(request_id: str = "req-sticky-migrate") -> RouterRequest:
    return RouterRequest(
        request_id=request_id,
        session_id="session-a",
        messages=[ChatMessage(content="Keep this conversation continuous")],
    )


def _signals() -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=100,
    )


def _strategy(
    success_stats: SuccessStats | None = None,
    migrate_stats: StickySessionMigrateStats | None = None,
    health: _FakeHealth | None = None,
    *,
    threshold: float = 0.8,
) -> StickySessionMigrateStrategy:
    return StickySessionMigrateStrategy(
        _catalog(),
        health or _FakeHealth(),
        success_stats or SuccessStats(),
        migrate_stats or StickySessionMigrateStats(),
        success_threshold=threshold,
    )


def test_sticky_session_migrate_enum_parses() -> None:
    assert (
        RoutingStrategyName("sticky-session-migrate") is RoutingStrategyName.STICKY_SESSION_MIGRATE
    )


def test_sticky_session_migrate_initializes_and_reuses_session_pin() -> None:
    migrate_stats = StickySessionMigrateStats()
    strategy = _strategy(migrate_stats=migrate_stats)

    first = strategy.choose(_request("req-1"), _signals())
    second = strategy.choose(_request("req-2"), _signals())

    assert first.chosen_model == second.chosen_model
    assert migrate_stats.pinned_model("session-a") == first.chosen_model
    assert migrate_stats.migration_count("session-a") == 0
    assert "initialized session" in first.rationale
    assert "retained session" in second.rationale


def test_sticky_session_migrate_retains_pin_at_success_threshold() -> None:
    success_stats = SuccessStats()
    for success in [True, True, True, True, False]:
        success_stats.observe("openai", success=success)
    migrate_stats = StickySessionMigrateStats()
    migrate_stats.pin("session-a", "quality-primary")

    decision = _strategy(success_stats, migrate_stats).choose(_request(), _signals())

    assert decision.chosen_model == "quality-primary"
    assert "success 80.00% >= 80.00%" in decision.rationale
    assert migrate_stats.migration_count("session-a") == 0


def test_sticky_session_migrate_moves_to_healthier_provider_and_stays_there() -> None:
    success_stats = SuccessStats()
    success_stats.observe("openai", success=False)
    migrate_stats = StickySessionMigrateStats()
    migrate_stats.pin("session-a", "quality-primary")
    strategy = _strategy(success_stats, migrate_stats)

    migrated = strategy.choose(_request("req-migrate"), _signals())
    for _ in range(10):
        success_stats.observe("openai", success=True)
    retained = strategy.choose(_request("req-after-recovery"), _signals())

    assert migrated.chosen_model == "healthy-alternate"
    assert retained.chosen_model == "healthy-alternate"
    assert migrate_stats.pinned_model("session-a") == "healthy-alternate"
    assert migrate_stats.migration_count("session-a") == 1
    assert "moved session" in migrated.rationale


def test_sticky_session_migrate_moves_when_pinned_provider_is_unavailable() -> None:
    migrate_stats = StickySessionMigrateStats()
    migrate_stats.pin("session-a", "quality-primary")

    decision = _strategy(
        migrate_stats=migrate_stats,
        health=_FakeHealth({"openai"}),
    ).choose(_request(), _signals())

    assert decision.chosen_model == "healthy-alternate"
    assert "provider unavailable" in decision.rationale
    assert migrate_stats.migration_count("session-a") == 1


def test_sticky_session_migrate_preserves_pin_when_no_healthy_target_exists() -> None:
    migrate_stats = StickySessionMigrateStats()
    migrate_stats.pin("session-a", "quality-primary")

    decision = _strategy(
        migrate_stats=migrate_stats,
        health=_FakeHealth({"openai", "google"}),
    ).choose(_request(), _signals())

    assert decision.chosen_model == "quality-primary"
    assert "no healthy target" in decision.rationale
    assert migrate_stats.migration_count("session-a") == 0


def test_sticky_session_migrate_rejects_invalid_success_threshold() -> None:
    with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
        _strategy(threshold=1.1)


def test_sticky_session_migrate_registered_by_strategy_factory() -> None:
    settings = RouterSettings(sticky_session_migrate_success_threshold=0.85)
    catalog = _catalog()
    success_stats = SuccessStats()
    migrate_stats = StickySessionMigrateStats()
    strategies = build_strategies(
        catalog,
        LatencyStats(),
        InflightStats(),
        settings.quality_floor,
        "quality-primary",
        "healthy-alternate",
        settings.ab_model_a_weight,
        CircuitBreakerRegistry(),
        settings.blend_quality_weight,
        settings.blend_cost_weight,
        settings.blend_latency_weight,
        settings.request_cost_ceiling_usd,
        "quality-primary",
        "healthy-alternate",
        settings.canary_weight,
        settings.latency_sla_ms,
        success_stats=success_stats,
        sticky_session_migrate_stats=migrate_stats,
        sticky_session_migrate_success_threshold=(
            settings.sticky_session_migrate_success_threshold
        ),
    )

    strategy = strategies[RoutingStrategyName.STICKY_SESSION_MIGRATE]
    assert isinstance(strategy, StickySessionMigrateStrategy)
    assert strategy._migrate_stats is migrate_stats  # noqa: SLF001
    assert strategy._success_stats is success_stats  # noqa: SLF001
    assert strategy._success_threshold == 0.85  # noqa: SLF001
    assert RouterSettings().sticky_session_migrate_success_threshold == 0.9
