"""feedback-aware routing: Thompson sampling over the default policy's candidates."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping

import pytest

from router.config import default_model_catalog
from router.feedback import FeedbackScope, FeedbackStore, ModelFeedback
from router.schemas import (
    AuditRecord,
    ChatMessage,
    DomainTag,
    LatencyRequirement,
    RouterRequest,
    RoutingStrategyName,
    TaskSignals,
)
from router.strategies import FeedbackAwareStrategy, RuleBasedStrategy

GENERAL_CANDIDATES = {"gpt-6-luna", "claude-opus-5-5", "gpt-6-astra"}


class Health:
    """Circuit view with a set of open providers."""

    def __init__(self, open_providers: frozenset[str] = frozenset()) -> None:
        self.open_providers = open_providers

    def is_available(self, provider: str) -> bool:
        return provider not in self.open_providers


class StaticFeedback:
    """Feedback source returning fixed evidence and recording what was asked."""

    def __init__(self, evidence: Mapping[str, ModelFeedback]) -> None:
        self.evidence = dict(evidence)
        self.queries: list[tuple[FeedbackScope, str, str]] = []

    def model_feedback(
        self, scope: FeedbackScope, *, metric: str, domain: str, now: float | None = None
    ) -> Mapping[str, ModelFeedback]:
        self.queries.append((scope, metric, domain))
        return self.evidence


class BrokenFeedback:
    """Feedback source whose database is unavailable."""

    def model_feedback(
        self, scope: FeedbackScope, *, metric: str, domain: str, now: float | None = None
    ) -> Mapping[str, ModelFeedback]:
        raise sqlite3.OperationalError("database is locked")


class Guards:
    """Pre-dispatch checks refusing unpriced models and costs above a ceiling."""

    def __init__(self, ceiling_usd: float = 1.0, unpriced: frozenset[str] = frozenset()) -> None:
        self.ceiling_usd = ceiling_usd
        self.unpriced = unpriced
        self.subjects: set[str] = set()

    def allows_dispatch(self, model: str, subject: str, estimated_cost_usd: float) -> bool:
        self.subjects.add(subject)
        return model not in self.unpriced and estimated_cost_usd < self.ceiling_usd


def _arm(positives: int, negatives: int) -> ModelFeedback:
    return ModelFeedback(n=positives + negatives, positives=positives, negatives=negatives)


def _request(request_id: str = "req-1", tenant_id: str = "acme") -> RouterRequest:
    return RouterRequest(
        request_id=request_id,
        messages=[ChatMessage(content="Plan a phased rollout for our new billing service.")],
        api_key_id="key_acme",
        tenant_id=tenant_id,
        user_id="user-7",
    )


def _signals(
    domain: DomainTag = DomainTag.GENERAL,
    latency: LatencyRequirement = LatencyRequirement.REALTIME,
) -> TaskSignals:
    return TaskSignals(
        complexity_score=0.5,
        domain_tag=domain,
        latency_requirement=latency,
        token_budget=4096,
        prompt_tokens_estimate=64,
    )


def _strategy(
    feedback: object,
    *,
    min_samples: int = 5,
    health: Health | None = None,
    guards: Guards | None = None,
    catalog: Mapping[str, object] | None = None,
) -> FeedbackAwareStrategy:
    return FeedbackAwareStrategy(
        catalog or default_model_catalog(),  # type: ignore[arg-type]
        health or Health(),
        feedback,  # type: ignore[arg-type]
        guards,
        min_samples=min_samples,
    )


PREFERS_OPUS = {
    "claude-opus-5-5": _arm(40, 0),
    "gpt-6-luna": _arm(0, 40),
    "gpt-6-astra": _arm(20, 20),
}


def test_cold_start_returns_the_rule_based_decision() -> None:
    """Below the minimum sample count the decision is rule-based's, labelled as a cold start."""
    feedback = StaticFeedback({"claude-opus-5-5": _arm(3, 0), "gemini-3.8-flash": _arm(9, 0)})
    strategy = _strategy(feedback, min_samples=20)

    decision = strategy.choose(_request(), _signals())
    default = RuleBasedStrategy(default_model_catalog()).choose(_request(), _signals())

    assert decision.routing_strategy is RoutingStrategyName.FEEDBACK_AWARE
    assert (decision.chosen_model, decision.fallback_chain) == (
        default.chosen_model,
        default.fallback_chain,
    )
    assert decision.rationale == (
        "cold start: 3/20 quality feedback samples for general prompts; "
        f"rule-based: {default.rationale}"
    )
    assert feedback.queries == [(FeedbackScope("key_acme", "acme"), "quality", "general")]


def test_without_a_feedback_source_every_request_is_a_cold_start() -> None:
    """No store wired in (for example in unit tests of other strategies) means rule-based."""
    decision = _strategy(None).choose(_request(), _signals())

    assert decision.chosen_model == "gpt-6-luna"
    assert decision.rationale.startswith("cold start: 0/5 quality feedback samples")


def test_same_request_id_replays_the_same_decision() -> None:
    """The RNG is seeded from the request id: same id, same draw; other ids still explore."""
    evidence = {
        "claude-opus-5-5": _arm(3, 2),
        "gpt-6-luna": _arm(2, 3),
        "gpt-6-astra": _arm(3, 3),
    }

    first = _strategy(StaticFeedback(evidence)).choose(_request("req-replay"), _signals())
    again = _strategy(StaticFeedback(evidence)).choose(_request("req-replay"), _signals())
    chosen = {
        _strategy(StaticFeedback(evidence)).choose(_request(f"req-{n}"), _signals()).chosen_model
        for n in range(40)
    }

    assert first == again
    assert first.chosen_model in GENERAL_CANDIDATES
    assert chosen == GENERAL_CANDIDATES


def test_feedback_steers_routing_and_reports_posteriors() -> None:
    """Strong feedback wins every draw; the rationale names posteriors and sample counts."""
    strategy = _strategy(StaticFeedback(PREFERS_OPUS))

    decisions = [strategy.choose(_request(f"req-{n}"), _signals()) for n in range(25)]

    assert {decision.chosen_model for decision in decisions} == {"claude-opus-5-5"}
    decision = decisions[0]
    assert decision.rationale.startswith("feedback-aware: Thompson sample chose claude-opus-5-5 (")
    assert "posterior mean 0.98, n=40) over gpt-6-astra (draw" in decision.rationale
    assert "posterior mean 0.50, n=40); 120 quality samples for general prompts" in (
        decision.rationale
    )
    # Fallbacks: remaining candidates by posterior mean, then the rest of the default attempts.
    assert decision.fallback_chain == ["gpt-6-astra", "gpt-6-luna", "gemini-4-argon"]
    default = RuleBasedStrategy(default_model_catalog()).choose(_request(), _signals())
    assert {decision.chosen_model, *decision.fallback_chain} == {
        default.chosen_model,
        *default.fallback_chain,
    }


def test_open_circuits_are_never_chosen() -> None:
    """A provider with an open circuit is skipped however good its feedback is."""
    strategy = _strategy(StaticFeedback(PREFERS_OPUS), health=Health(frozenset({"anthropic"})))

    chosen = {strategy.choose(_request(f"req-{n}"), _signals()).chosen_model for n in range(25)}

    assert chosen == {"gpt-6-astra"}


def test_candidates_the_engine_would_refuse_are_skipped() -> None:
    """Over-budget or unpriced models are never drawn, however good their feedback."""
    over_budget = Guards(ceiling_usd=0.001)
    evidence = {**PREFERS_OPUS, "gpt-6-luna": _arm(3, 3)}
    affordable = _strategy(StaticFeedback(evidence), guards=over_budget)
    priced = _strategy(
        StaticFeedback(PREFERS_OPUS), guards=Guards(unpriced=frozenset({"claude-opus-5-5"}))
    )

    decision = affordable.choose(_request(), _signals())
    chosen = {priced.choose(_request(f"req-{n}"), _signals()).chosen_model for n in range(25)}

    assert decision.chosen_model == "gpt-6-luna"
    assert "with no alternative candidate; 6 quality samples" in decision.rationale
    assert over_budget.subjects == {"user-7"}
    assert chosen == {"gpt-6-astra"}


def test_feedback_never_widens_the_default_candidates() -> None:
    """Models outside the default attempts, or removed from the catalog, are never chosen."""
    outside = {
        "kimi-k3.1-pro": _arm(50, 0),
        "claude-haiku-5-5": _arm(50, 0),
        "gemini-3.8-flash": _arm(50, 0),
        "gpt-6-luna": _arm(0, 10),
    }
    allowed = {
        name: candidate
        for name, candidate in default_model_catalog().items()
        if name != "claude-opus-5-5"
    }

    widened = {
        _strategy(StaticFeedback(outside)).choose(_request(f"req-{n}"), _signals()).chosen_model
        for n in range(25)
    }
    restricted = {
        _strategy(StaticFeedback(PREFERS_OPUS), catalog=allowed)
        .choose(_request(f"req-{n}"), _signals())
        .chosen_model
        for n in range(25)
    }

    assert widened <= GENERAL_CANDIDATES
    assert "claude-opus-5-5" not in restricted
    assert restricted <= set(allowed)


def test_fallbacks_that_do_not_serve_the_domain_are_not_promoted() -> None:
    """Medical batch prompts: only claude-opus-5-5 and gemini-4-argon serve the domain."""
    evidence = {"gpt-6-astra": _arm(60, 0), "gemini-4-argon": _arm(40, 0)}
    evidence["claude-opus-5-5"] = _arm(0, 40)
    strategy = _strategy(StaticFeedback(evidence))
    medical_batch = _signals(DomainTag.MEDICAL, LatencyRequirement.BATCH)
    medical_realtime = _signals(DomainTag.MEDICAL, LatencyRequirement.REALTIME)

    batch = {strategy.choose(_request(f"req-{n}"), medical_batch).chosen_model for n in range(25)}
    realtime = strategy.choose(_request(), medical_realtime)

    assert batch == {"gemini-4-argon"}
    # Realtime excludes gemini-4-argon, leaving the default pick alone (40 samples).
    assert realtime.chosen_model == "claude-opus-5-5"
    assert "with no alternative candidate" in realtime.rationale


def test_feedback_is_read_for_the_callers_scope_only() -> None:
    """Another tenant's feedback is invisible: the same key with tenant globex is cold."""
    store = FeedbackStore()
    for n in range(6):
        store.register_request(
            AuditRecord(
                request_id=f"acme-{n}",
                chosen_model="claude-opus-5-5",
                provider="anthropic",
                routing_strategy=RoutingStrategyName.RULE_BASED,
                rationale="seed",
                latency_ms=1.0,
                token_cost=0.0,
                input_tokens=1,
                output_tokens=1,
                state="RESPONDED",
                api_key_id="key_acme",
                tenant_id="acme",
                domain=DomainTag.GENERAL,
            )
        )
        store.record(
            FeedbackScope("key_acme", "acme"), request_id=f"acme-{n}", metric="quality", value=True
        )
    strategy = _strategy(store)

    acme = strategy.choose(_request(), _signals())
    globex = strategy.choose(_request(tenant_id="globex"), _signals())

    assert acme.rationale.startswith("feedback-aware: Thompson sample chose")
    assert globex.rationale.startswith("cold start: 0/5 quality feedback samples")


def test_feedback_store_errors_fall_back_to_rule_based() -> None:
    """A broken feedback database never fails routing."""
    decision = _strategy(BrokenFeedback()).choose(_request(), _signals())

    assert decision.chosen_model == "gpt-6-luna"
    assert decision.rationale.startswith("feedback unavailable (database is locked); rule-based:")


def test_min_samples_must_be_positive() -> None:
    """A zero threshold would let a single vote reroute traffic."""
    with pytest.raises(ValueError, match="min_samples"):
        _strategy(None, min_samples=0)
