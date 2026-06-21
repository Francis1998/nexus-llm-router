"""Coverage for routing-strategy and classifier decision paths.

These exercise decision branches that the existing suite does not cover:
the rule-based priority matrix, the classifier complexity tiers, the
cost-optimal quality-floor fallback, and complexity-score monotonicity.
"""

from classifier.complexity import LogisticComplexityClassifier
from classifier.features import extract_prompt_features
from router.config import default_model_catalog
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import (
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
    """Legal prompts should route to Claude under the rule-based matrix."""
    strategy = RuleBasedStrategy(default_model_catalog())
    decision = strategy.choose(_request(), _signals(0.5, DomainTag.LEGAL))
    assert decision.chosen_model == "claude-3-5-sonnet"
    assert "legal" in decision.rationale


def test_rule_based_routes_complex_code_to_gpt4o() -> None:
    """Complex code prompts should prefer GPT-4o."""
    strategy = RuleBasedStrategy(default_model_catalog())
    decision = strategy.choose(_request(), _signals(0.7, DomainTag.CODE))
    assert decision.chosen_model == "gpt-4o"


def test_rule_based_routes_simple_realtime_to_flash() -> None:
    """Simple realtime prompts should prefer the low-latency flash model."""
    strategy = RuleBasedStrategy(default_model_catalog())
    decision = strategy.choose(
        _request(), _signals(0.2, DomainTag.GENERAL, LatencyRequirement.REALTIME)
    )
    assert decision.chosen_model == "gemini-1.5-flash"


def test_rule_based_honors_compatible_requested_model() -> None:
    """An explicit, in-catalog model request should be honored."""
    strategy = RuleBasedStrategy(default_model_catalog())
    decision = strategy.choose(
        _request(requested_model="kimi-k2"),
        _signals(0.5, DomainTag.GENERAL),
    )
    assert decision.chosen_model == "kimi-k2"
    assert decision.routing_strategy == RoutingStrategyName.RULE_BASED


def test_rule_based_defaults_to_balanced_low_cost_model() -> None:
    """A general prompt with no overrides should route to the balanced default."""
    strategy = RuleBasedStrategy(default_model_catalog())
    decision = strategy.choose(_request(), _signals(0.5, DomainTag.GENERAL))
    assert decision.chosen_model == "gpt-4o-mini"


def test_classifier_routes_high_complexity_to_sonnet() -> None:
    """High classifier complexity should route to the top-quality model."""
    strategy = ClassifierStrategy(default_model_catalog())
    decision = strategy.choose(_request(), _signals(0.85, DomainTag.GENERAL))
    assert decision.chosen_model == "claude-3-5-sonnet"


def test_classifier_routes_simple_prompt_to_kimi() -> None:
    """Low-complexity prompts should route to the cost-sensitive model."""
    strategy = ClassifierStrategy(default_model_catalog())
    decision = strategy.choose(_request(), _signals(0.2, DomainTag.GENERAL))
    assert decision.chosen_model == "kimi-k2"


def test_cost_optimal_forces_fallback_when_floor_unreachable() -> None:
    """An unreachable quality floor should force the highest-quality fallback."""
    strategy = CostOptimalStrategy(default_model_catalog(), quality_floor=1.0)
    decision = strategy.choose(_request(), _signals(0.5, DomainTag.GENERAL))
    assert decision.chosen_model == "claude-3-5-sonnet"
    assert "quality floor forced" in decision.rationale


def test_complexity_score_increases_with_instruction_hits() -> None:
    """More instruction keywords should not decrease the complexity score."""
    classifier = LogisticComplexityClassifier()
    simple = classifier.predict_score(extract_prompt_features("Say hello."))
    complex_prompt = classifier.predict_score(
        extract_prompt_features("Analyze, debug, optimize, and compare these approaches.")
    )
    assert complex_prompt > simple
