"""Coverage for routing-strategy and classifier decision paths.

These exercise decision branches that the existing suite does not cover:
the rule-based priority matrix, the classifier complexity tiers, the
cost-optimal quality-floor fallback, and complexity-score monotonicity.
"""

import pytest

from classifier.complexity import LogisticComplexityClassifier
from classifier.features import extract_prompt_features
from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    MOONSHOT_BALANCED_MODEL,
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
    ABRoutingStrategy,
    ClassifierStrategy,
    CostOptimalStrategy,
    RuleBasedStrategy,
)


def _request(content: str = "Hello", requested_model: str | None = None) -> RouterRequest:
    """Build a minimal router request for strategy unit tests."""
    return RouterRequest(
        request_id="req-test",
        messages=[ChatMessage(content=content)],
        requested_model=requested_model,
    )


def _signals(
    complexity_score: float,
    domain_tag: DomainTag,
    latency_requirement: LatencyRequirement = LatencyRequirement.REALTIME,
) -> TaskSignals:
    """Build task signals with explicit complexity, domain, and latency."""
    return TaskSignals(
        complexity_score=complexity_score,
        domain_tag=domain_tag,
        latency_requirement=latency_requirement,
        token_budget=4096,
        prompt_tokens_estimate=8,
    )


def test_rule_based_routes_legal_to_claude() -> None:
    """Legal prompts should route to Claude Sonnet under the rule-based matrix."""
    strategy = RuleBasedStrategy(default_model_catalog())
    decision = strategy.choose(_request(), _signals(0.5, DomainTag.LEGAL))
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "legal" in decision.rationale


def test_rule_based_routes_complex_code_to_gpt5() -> None:
    """Complex code prompts should prefer the OpenAI frontier model."""
    strategy = RuleBasedStrategy(default_model_catalog())
    decision = strategy.choose(_request(), _signals(0.7, DomainTag.CODE))
    assert decision.chosen_model == OPENAI_FRONTIER_MODEL


def test_rule_based_routes_simple_realtime_to_flash() -> None:
    """Simple realtime prompts should prefer the low-latency flash model."""
    strategy = RuleBasedStrategy(default_model_catalog())
    decision = strategy.choose(
        _request(), _signals(0.2, DomainTag.GENERAL, LatencyRequirement.REALTIME)
    )
    assert decision.chosen_model == GEMINI_FLASH_MODEL


def test_rule_based_honors_compatible_requested_model() -> None:
    """An explicit, in-catalog model request should be honored."""
    strategy = RuleBasedStrategy(default_model_catalog())
    decision = strategy.choose(
        _request(requested_model=MOONSHOT_BALANCED_MODEL),
        _signals(0.5, DomainTag.GENERAL),
    )
    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL
    assert decision.routing_strategy == RoutingStrategyName.RULE_BASED


def test_rule_based_defaults_to_balanced_low_cost_model() -> None:
    """A general prompt with no overrides should route to the balanced default."""
    strategy = RuleBasedStrategy(default_model_catalog())
    decision = strategy.choose(_request(), _signals(0.5, DomainTag.GENERAL))
    assert decision.chosen_model == OPENAI_BALANCED_MODEL


def test_classifier_routes_high_complexity_to_sonnet() -> None:
    """High classifier complexity should route to the top-quality model."""
    strategy = ClassifierStrategy(default_model_catalog())
    decision = strategy.choose(_request(), _signals(0.85, DomainTag.GENERAL))
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL


def test_classifier_routes_simple_prompt_to_kimi() -> None:
    """Low-complexity prompts should route to the cost-sensitive model."""
    strategy = ClassifierStrategy(default_model_catalog())
    decision = strategy.choose(_request(), _signals(0.2, DomainTag.GENERAL))
    assert decision.chosen_model == MOONSHOT_BALANCED_MODEL


def test_cost_optimal_forces_fallback_when_floor_unreachable() -> None:
    """An unreachable quality floor should force the highest-quality fallback."""
    strategy = CostOptimalStrategy(default_model_catalog(), quality_floor=1.0)
    decision = strategy.choose(_request(), _signals(0.5, DomainTag.GENERAL))
    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert "quality floor forced" in decision.rationale


def test_ab_strategy_rejects_unknown_experiment_models() -> None:
    """A/B arms outside the catalog should fail fast at construction."""
    with pytest.raises(ValueError, match="not in model catalog"):
        ABRoutingStrategy(
            default_model_catalog(),
            model_a="nonexistent-model",
            model_b=OPENAI_BALANCED_MODEL,
            model_a_weight=0.5,
        )


def test_ab_strategy_rejects_out_of_range_weight() -> None:
    """A/B weights outside [0.0, 1.0] should fail fast at construction."""
    with pytest.raises(ValueError, match="model_a_weight must be within"):
        ABRoutingStrategy(
            default_model_catalog(),
            model_a=OPENAI_BALANCED_MODEL,
            model_b=ANTHROPIC_SAFETY_MODEL,
            model_a_weight=1.5,
        )


def test_ab_strategy_routes_to_in_catalog_arms() -> None:
    """A/B routing should resolve to a configured, in-catalog arm."""
    strategy = ABRoutingStrategy(
        default_model_catalog(),
        model_a=OPENAI_BALANCED_MODEL,
        model_b=ANTHROPIC_SAFETY_MODEL,
        model_a_weight=0.5,
    )
    decision = strategy.choose(_request(), _signals(0.5, DomainTag.GENERAL))
    assert decision.chosen_model in {OPENAI_BALANCED_MODEL, ANTHROPIC_SAFETY_MODEL}


def test_complexity_score_increases_with_instruction_hits() -> None:
    """More instruction keywords should not decrease the complexity score."""
    classifier = LogisticComplexityClassifier()
    simple = classifier.predict_score(extract_prompt_features("Say hello."))
    complex_prompt = classifier.predict_score(
        extract_prompt_features("Analyze, debug, optimize, and compare these approaches.")
    )
    assert complex_prompt > simple
