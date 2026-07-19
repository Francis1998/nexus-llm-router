"""Tests for the token-budget routing strategy."""

from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    OPENAI_BALANCED_MODEL,
)
from router.schemas import (
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    ModelCandidate,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import TokenBudgetStrategy


def _request(token_budget: int = 4096, max_tokens: int = 512) -> RouterRequest:
    """Build a minimal router request with explicit token limits."""
    return RouterRequest(
        request_id="req-token-budget",
        messages=[ChatMessage(content="Hello")],
        token_budget=token_budget,
        max_tokens=max_tokens,
    )


def _signals(
    domain_tag: DomainTag = DomainTag.GENERAL,
    prompt_tokens_estimate: int = 64,
    token_budget: int = 4096,
) -> TaskSignals:
    """Build task signals for token-budget tests."""
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain_tag,
        latency_requirement=LatencyRequirement.REALTIME,
        token_budget=token_budget,
        prompt_tokens_estimate=prompt_tokens_estimate,
    )


def test_token_budget_picks_highest_quality_when_all_fit() -> None:
    """A generous budget admits every model, so the best quality wins."""
    strategy = TokenBudgetStrategy(default_model_catalog())

    decision = strategy.choose(_request(token_budget=200_000), _signals())

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert decision.routing_strategy is RoutingStrategyName.TOKEN_BUDGET
    assert "token-budget" in decision.rationale


def test_token_budget_excludes_models_below_needed_capacity() -> None:
    """Models whose effective capacity is below tokens_needed are excluded.

    With a 200k request budget and ~150k tokens needed, 128k-context models
    (``min(128k, 200k) = 128k``) drop out while 200k+/1M-context models remain,
    so the highest-quality remaining SKU (Claude Sonnet 4.6) wins.
    """
    strategy = TokenBudgetStrategy(default_model_catalog())
    request = _request(token_budget=200_000, max_tokens=1_000)
    signals = _signals(prompt_tokens_estimate=149_000, token_budget=200_000)

    decision = strategy.choose(request, signals)

    assert decision.chosen_model == ANTHROPIC_SAFETY_MODEL
    assert default_model_catalog()[decision.chosen_model].context_window >= 150_000
    # Balanced OpenAI (128k) must not win when it cannot hold the estimate.
    assert decision.chosen_model != OPENAI_BALANCED_MODEL


def test_token_budget_falls_back_to_largest_context_when_none_fit() -> None:
    """When nothing fits the budget, route to the largest-context eligible model."""
    strategy = TokenBudgetStrategy(default_model_catalog())
    # tokens_needed far exceeds every catalog window and the request budget.
    request = _request(token_budget=1024, max_tokens=900)
    signals = _signals(prompt_tokens_estimate=50_000, token_budget=1024)

    decision = strategy.choose(request, signals)

    assert decision.chosen_model in {GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL}
    assert "no model fitting" in decision.rationale


def test_token_budget_respects_domain_eligibility() -> None:
    """Only domain-eligible models may be chosen for a specialized domain."""
    strategy = TokenBudgetStrategy(default_model_catalog())

    decision = strategy.choose(_request(), _signals(DomainTag.MEDICAL))

    assert DomainTag.MEDICAL in default_model_catalog()[decision.chosen_model].supports_domains


def test_token_budget_prefers_quality_among_fitting_candidates() -> None:
    """Among models that fit, higher quality wins over a larger window."""
    catalog = {
        OPENAI_BALANCED_MODEL: ModelCandidate(
            model=OPENAI_BALANCED_MODEL,
            provider="openai",
            quality_score=0.84,
            input_cost_per_1k=0.0002,
            output_cost_per_1k=0.0008,
            supports_domains={DomainTag.GENERAL},
            context_window=8_000,
        ),
        "wide-but-weaker": ModelCandidate(
            model="wide-but-weaker",
            provider="google",
            quality_score=0.70,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.GENERAL},
            context_window=1_000_000,
        ),
    }
    strategy = TokenBudgetStrategy(catalog)
    request = _request(token_budget=8_000, max_tokens=256)
    signals = _signals(prompt_tokens_estimate=100, token_budget=8_000)

    decision = strategy.choose(request, signals)

    assert decision.chosen_model == OPENAI_BALANCED_MODEL
