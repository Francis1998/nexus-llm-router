"""Coverage for routing-strategy and classifier decision paths.

These exercise decision branches that the existing suite does not cover:
the rule-based priority matrix, the classifier complexity tiers, the
cost-optimal quality-floor fallback, and complexity-score monotonicity.
"""

import pytest

from classifier.complexity import DomainClassifier, LogisticComplexityClassifier
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


def test_instruction_hits_ignore_substring_false_positives() -> None:
    """Words that merely contain a keyword must not count as instructions."""
    features = extract_prompt_features(
        "Please improve and approve the paragraph about the province."
    )
    assert features.instruction_hits == 0


def test_instruction_hits_count_keyword_inflections() -> None:
    """Inflected instruction verbs anchored at a word start should count."""
    features = extract_prompt_features("It optimizes throughput and debugged the parser.")
    assert features.instruction_hits == 2


def test_instruction_hits_ignore_words_that_start_with_a_keyword() -> None:
    """A longer, unrelated word that *begins* with a verb must not count.

    The previous permissive ``\\w*`` suffix matched any word starting with an
    instruction verb, so ``proverb`` (from ``prove``) and ``designated`` /
    ``designation`` (from ``design``) were miscounted as instructions, inflating
    the complexity score and misrouting the request. Only the verb and its real
    inflections should count.
    """
    features = extract_prompt_features("Explain the proverb and the designated clause.")
    assert features.instruction_hits == 0


def test_instruction_hits_count_silent_e_ing_inflections() -> None:
    """``-ing`` forms of silent-``e`` verbs must count.

    The previous pattern kept the trailing ``e`` of ``analyze``/``optimize``/
    ``prove``/``compare`` and so could not match the e-dropped ``-ing`` forms,
    silently under-counting genuine instructions.
    """
    features = extract_prompt_features("Start by analyzing, optimizing, and comparing the results.")
    assert features.instruction_hits == 3


def test_code_hits_ignore_substring_false_positives() -> None:
    """Words that merely contain a code keyword must not count as code hits.

    The code pattern matched bare keyword substrings, so plain-English words
    such as ``masterclass`` or ``subclass`` (which contain ``class``) were
    counted as code, inflating the complexity score and biasing the domain
    classifier toward code. Only keywords at a word boundary should count.
    """
    features = extract_prompt_features(
        "Give a masterclass on public speaking and discuss the subclass of birds."
    )
    assert features.code_hits == 0


def test_code_hits_count_real_code_keywords() -> None:
    """Genuine code keywords at a word boundary must still be detected."""
    features = extract_prompt_features("def run(): pass  # a class helper and import os")
    assert features.code_hits >= 3


def test_code_hits_detect_keyword_followed_by_non_space() -> None:
    """Code keywords followed by a newline, colon, or paren must be detected.

    The pattern previously required a trailing literal space, so a keyword
    immediately followed by a newline (an idiomatic SQL ``SELECT\\n``), a colon
    (``class Foo:``), or a paren was missed entirely and the prompt fell through
    to the ``general`` domain. A ``\\b`` boundary detects the token regardless of
    the following character.
    """
    sql_features = extract_prompt_features("SELECT\n  name\nFROM customers")
    assert sql_features.code_hits >= 1
    assert DomainClassifier().classify(sql_features) is DomainTag.CODE

    class_features = extract_prompt_features("class Foo:\n    pass")
    assert class_features.code_hits >= 1


def test_medical_hits_count_common_plurals() -> None:
    """Plural medical keywords must count toward medical_hits.

    The medical pattern previously listed only singular forms (``patient``,
    ``symptom``, ``treatment``, ``diagnosis``), so prompts that used everyday
    plurals such as ``patients`` / ``symptoms`` / ``treatments`` scored zero
    medical hits and fell through to the general domain.
    """
    features = extract_prompt_features("Multiple patients share symptoms after treatments.")
    assert features.medical_hits >= 1


def test_legal_hits_count_common_plurals() -> None:
    """Plural legal keywords must count toward legal_hits.

    The legal pattern previously listed only singular forms (``contract``,
    ``clause``, ``statute``), so prompts referring to ``contracts`` /
    ``statutes`` scored zero legal hits and missed legal-domain routing.
    """
    features = extract_prompt_features("Review the contracts and statutes carefully.")
    assert features.legal_hits >= 1
