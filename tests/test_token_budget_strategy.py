"""Tests for the token-budget routing strategy."""

from pathlib import Path

from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from router.analyzer import RequestAnalyzer
from router.config import RouterSettings, default_model_catalog
from router.engine import NexusRouter
from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
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

    With a 200k request budget and ~150k tokens needed, a 128k-context model
    (``min(128k, 200k) = 128k``) drops out even though it has the highest
    quality, so the best remaining 1M-context SKU wins.
    """
    catalog = default_model_catalog()
    catalog[ANTHROPIC_SAFETY_MODEL] = catalog[ANTHROPIC_SAFETY_MODEL].model_copy(
        update={"context_window": 128_000}
    )
    strategy = TokenBudgetStrategy(catalog)
    request = _request(token_budget=200_000, max_tokens=1_000)
    signals = _signals(prompt_tokens_estimate=149_000, token_budget=200_000)

    decision = strategy.choose(request, signals)

    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert catalog[decision.chosen_model].context_window >= 150_000


def test_token_budget_falls_back_to_largest_context_when_none_fit() -> None:
    """When nothing fits the budget, route to the eligible model accepting the largest prompt."""
    strategy = TokenBudgetStrategy(default_model_catalog())
    # tokens_needed far exceeds every catalog window and the request budget.
    request = _request(token_budget=1024, max_tokens=900)
    signals = _signals(prompt_tokens_estimate=50_000, token_budget=1024)

    decision = strategy.choose(request, signals)

    # GPT-6 caps input at 922,000 tokens, so the 1,048,576-token Gemini models
    # accept the largest prompts.
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


def test_token_budget_fallback_chain_only_contains_fitting_models() -> None:
    """Fallbacks must also accept the prompt; non-fitting models only waste attempts."""
    strategy = TokenBudgetStrategy(default_model_catalog())
    request = _request(token_budget=1_050_000, max_tokens=1_000)
    signals = _signals(prompt_tokens_estimate=1_020_000, token_budget=1_050_000)

    decision = strategy.choose(request, signals)

    # Opus 5.5 (1M window) and GPT-6 (922K max input) cannot accept the prompt.
    assert decision.chosen_model == GEMINI_PRO_MODEL
    assert decision.fallback_chain == [GEMINI_FLASH_MODEL, MOONSHOT_BALANCED_MODEL]


def test_token_budget_no_fit_fallback_reserves_completion_tokens() -> None:
    """The fail-soft pick must still hold the prompt plus the requested completion."""
    strategy = TokenBudgetStrategy(default_model_catalog())
    request = _request(token_budget=1_048_576, max_tokens=128_000)
    signals = _signals(prompt_tokens_estimate=921_000, token_budget=1_048_576)

    decision = strategy.choose(request, signals)

    # 921,000 + 128,000 = 1,049,000 tokens overflow Gemini's 1,048,576-token window
    # but fit GPT-6 Astra (1,050,000-token window, 922,000 max input).
    assert decision.chosen_model == OPENAI_FRONTIER_MODEL
    assert "no model fitting" in decision.rationale


async def test_token_budget_primary_failure_falls_back_to_a_fitting_model(
    tmp_path: Path,
) -> None:
    """When the primary fails, the engine retries only models that accept the prompt."""

    class HugePromptAnalyzer(RequestAnalyzer):
        def analyze(self, request: RouterRequest) -> TaskSignals:
            return _signals(prompt_tokens_estimate=1_020_000, token_budget=request.token_budget)

    router = NexusRouter(
        settings=RouterSettings(audit_log_path=str(tmp_path / "audit.jsonl")),
        adapter_registry=AdapterRegistry(
            {
                "openai": MockProviderAdapter("openai"),
                "anthropic": MockProviderAdapter("anthropic"),
                "google": MockProviderAdapter("google", fail=True),
                "moonshot": MockProviderAdapter("moonshot"),
            },
        ),
        analyzer=HugePromptAnalyzer(),
    )

    response = await router.complete(
        RouterRequest(
            request_id="req-huge-prompt",
            messages=[ChatMessage(content="Summarize the attached corpus.")],
            strategy=RoutingStrategyName.TOKEN_BUDGET,
            token_budget=1_050_000,
            max_tokens=1_000,
        ),
    )

    # Gemini 4 Argon and 3.8 Flash fail; Kimi K3.1 Pro is the remaining model that fits.
    assert response.model_used == MOONSHOT_BALANCED_MODEL
