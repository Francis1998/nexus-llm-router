"""Regression tests for cost-optimal routing with restricted catalogs."""

from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    ModelCandidate,
    RouterRequest,
    TaskSignals,
)
from router.strategies import CostOptimalStrategy


def _general_only_catalog() -> dict[str, ModelCandidate]:
    """Build a catalog whose models only support the general domain."""
    return {
        "gpt-4o-mini": ModelCandidate(
            model="gpt-4o-mini",
            provider="openai",
            quality_score=0.82,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
            supports_domains={DomainTag.GENERAL},
        ),
        "kimi-k2": ModelCandidate(
            model="kimi-k2",
            provider="moonshot",
            quality_score=0.90,
            input_cost_per_1k=0.0005,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
        ),
    }


def test_cost_optimal_falls_back_when_no_model_supports_domain() -> None:
    """Cost-optimal routing should not crash when no candidate supports the domain.

    A custom catalog that omits a domain previously raised ``KeyError`` because
    the quality-floor fallback hardcoded ``claude-3-5-sonnet``. It should
    instead fall back to the highest-quality model in the catalog.
    """
    catalog = _general_only_catalog()
    strategy = CostOptimalStrategy(catalog, quality_floor=0.72)
    request = RouterRequest(
        request_id="req-medical",
        messages=[ChatMessage(content="Assess patient symptoms.")],
    )
    signals = TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.MEDICAL,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )

    decision = strategy.choose(request, signals)

    assert decision.chosen_model == "kimi-k2"
    assert "quality floor forced" in decision.rationale


def test_cost_optimal_falls_back_when_quality_floor_unreachable() -> None:
    """An unreachable quality floor should pick the best model in a custom catalog."""
    catalog = _general_only_catalog()
    strategy = CostOptimalStrategy(catalog, quality_floor=1.0)
    request = RouterRequest(
        request_id="req-general",
        messages=[ChatMessage(content="Summarize this.")],
    )
    signals = TaskSignals(
        complexity_score=0.5,
        domain_tag=DomainTag.GENERAL,
        latency_requirement=LatencyRequirement.BATCH,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )

    decision = strategy.choose(request, signals)

    assert decision.chosen_model == "kimi-k2"
    assert "quality floor forced" in decision.rationale
