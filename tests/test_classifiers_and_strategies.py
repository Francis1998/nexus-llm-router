"""Tests for classifiers and routing strategies."""

from router.analyzer import RequestAnalyzer
from router.config import default_model_catalog
from router.schemas import ChatMessage, DomainTag, RouterRequest, RoutingStrategyName
from router.strategies import ABRoutingStrategy, LatencyAwareStrategy, LatencyStats


def test_analyzer_detects_code_domain() -> None:
    """Analyzer should classify code prompts as code domain."""
    analyzer = RequestAnalyzer()
    signals = analyzer.analyze(
        RouterRequest(
            request_id="req-analyze",
            messages=[ChatMessage(content="```python\nasync def run() -> None:\n    pass\n```")],
        ),
    )
    assert signals.domain_tag == DomainTag.CODE
    assert 0.0 <= signals.complexity_score <= 1.0


def test_ab_strategy_is_deterministic_for_request_id() -> None:
    """A/B routing should produce stable decisions for the same request id."""
    strategy = ABRoutingStrategy(default_model_catalog(), "gpt-4o-mini", "claude-3-5-haiku", 0.5)
    request = RouterRequest(
        request_id="stable-request",
        messages=[ChatMessage(content="Hello")],
        strategy=RoutingStrategyName.AB_TEST,
    )
    signals = RequestAnalyzer().analyze(request)
    first_decision = strategy.choose(request, signals)
    second_decision = strategy.choose(request, signals)
    assert first_decision.chosen_model == second_decision.chosen_model


def test_latency_aware_penalizes_slow_provider() -> None:
    """Latency-aware strategy should avoid providers with poor p95 latency."""
    latency_stats = LatencyStats()
    for _ in range(10):
        latency_stats.observe("openai", 2500.0)
        latency_stats.observe("google", 25.0)
    strategy = LatencyAwareStrategy(default_model_catalog(), latency_stats)
    request = RouterRequest(
        request_id="latency-request",
        messages=[ChatMessage(content="Write a short status update.")],
        strategy=RoutingStrategyName.LATENCY_AWARE,
    )
    signals = RequestAnalyzer().analyze(request)
    decision = strategy.choose(request, signals)
    assert decision.provider != "openai"
