"""FeedbackStore: request index, scoped idempotent upserts, and summary statistics."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from router.feedback import (
    FeedbackKindConflictError,
    FeedbackScope,
    FeedbackStore,
    mean_interval,
    wilson_interval,
)
from router.schemas import AuditRecord, DomainTag, RoutingStrategyName

ACME = FeedbackScope(api_key_id="key_acme", tenant_id="acme")
NOW = 1_800_000_000.0
PROVIDERS = {
    "claude-opus-5-5": "anthropic",
    "gemini-3.8-flash": "google",
    "gpt-6-astra": "openai",
    "gpt-6-luna": "openai",
}


def _register(
    store: FeedbackStore,
    request_id: str,
    *,
    model: str = "claude-opus-5-5",
    scope: FeedbackScope = ACME,
    domain: DomainTag = DomainTag.GENERAL,
    recorded_at: float = NOW,
) -> None:
    """Index a completed request the way the router does after an audit append."""
    store.register_request(
        AuditRecord(
            request_id=request_id,
            chosen_model=model,
            provider=PROVIDERS[model],
            routing_strategy=RoutingStrategyName.RULE_BASED,
            rationale="test",
            latency_ms=12.0,
            token_cost=0.0001,
            input_tokens=10,
            output_tokens=16,
            state="RESPONDED",
            api_key_id=scope.api_key_id,
            tenant_id=scope.tenant_id,
            domain=domain,
        ),
        recorded_at=recorded_at,
    )


@pytest.mark.parametrize(
    ("positives", "n", "expected"),
    [
        (8, 10, (0.4902, 0.9433)),
        (0, 10, (0.0, 0.2775)),
        (10, 10, (0.7225, 1.0)),
        (5, 6, (0.4365, 0.9699)),
    ],
)
def test_wilson_interval_matches_reference_values(
    positives: int, n: int, expected: tuple[float, float]
) -> None:
    """95% Wilson bounds match published values and stay inside [0, 1]."""
    low, high = wilson_interval(positives, n)

    assert (round(low, 4), round(high, 4)) == expected


def test_intervals_are_uninformative_without_enough_data() -> None:
    """No samples (Wilson) or fewer than two values (normal) give the whole [0, 1] range."""
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert mean_interval(0.7, 0.49, 1) == (0.0, 1.0)


def test_mean_interval_uses_sample_standard_deviation() -> None:
    """Values 0.9, 0.7, 0.8: mean 0.8, s = 0.1, half-width 1.96 * 0.1 / sqrt(3)."""
    low, high = mean_interval(2.4, 0.81 + 0.49 + 0.64, 3)

    assert low == pytest.approx(0.8 - 0.113159, abs=1e-6)
    assert high == pytest.approx(0.8 + 0.113159, abs=1e-6)


def test_record_snapshots_routing_fields_and_upserts_per_metric_and_source() -> None:
    """A second write for the same (request, metric, source) replaces the value in place."""
    store = FeedbackStore()
    _register(store, "req-1", model="gemini-3.8-flash", domain=DomainTag.CODE)

    first = store.record(ACME, request_id="req-1", metric="quality", value=False, now=NOW + 5)
    second = store.record(
        ACME, request_id="req-1", metric="quality", value=True, comment="fixed", now=NOW + 9
    )
    automated = store.record(
        ACME, request_id="req-1", metric="quality", value=False, source="automated", now=NOW + 9
    )

    assert first is not None and second is not None and automated is not None
    assert first.created and not second.created and automated.created
    assert second.record.feedback_id == first.record.feedback_id
    assert second.record.value is True
    assert second.record.comment == "fixed"
    assert (second.record.created_at, second.record.updated_at) == (NOW + 5, NOW + 9)
    assert (second.record.model, second.record.provider, second.record.domain) == (
        "gemini-3.8-flash",
        "google",
        "code",
    )
    assert second.record.strategy == "rule-based"
    assert automated.record.feedback_id != first.record.feedback_id
    assert store.summary(ACME, metric="quality").n == 2


def test_record_returns_none_for_unknown_foreign_or_expired_requests() -> None:
    """Unknown ids, another key, another tenant, and requests past the window look alike."""
    store = FeedbackStore(window_seconds=3600.0)
    _register(store, "req-acme")
    _register(store, "req-old", recorded_at=NOW - 7200.0)

    attempts = [
        (ACME, "req-missing"),
        (FeedbackScope(api_key_id="key_other", tenant_id="acme"), "req-acme"),
        (FeedbackScope(api_key_id="key_acme", tenant_id="globex"), "req-acme"),
        (FeedbackScope(api_key_id="key_acme"), "req-acme"),
        (ACME, "req-old"),
    ]

    for scope, request_id in attempts:
        written = store.record(scope, request_id=request_id, metric="quality", value=True, now=NOW)
        assert written is None, (scope, request_id)
    assert store.record(ACME, request_id="req-acme", metric="quality", value=True, now=NOW)


def test_request_ids_keep_their_first_owner() -> None:
    """Re-registering an id for another caller cannot take over its feedback."""
    store = FeedbackStore()
    _register(store, "req-1")
    _register(store, "req-1", scope=FeedbackScope(api_key_id="key_attacker"))

    attacker = FeedbackScope(api_key_id="key_attacker")
    hijack = store.record(attacker, request_id="req-1", metric="quality", value=True, now=NOW)
    assert hijack is None
    assert store.record(ACME, request_id="req-1", metric="quality", value=True, now=NOW)


def test_metric_keeps_one_value_kind_per_scope() -> None:
    """The first value fixes a metric's kind for the caller, even for the same request."""
    store = FeedbackStore()
    _register(store, "req-1")
    _register(store, "req-2")
    _register(store, "req-3", scope=FeedbackScope(api_key_id="key_other"))
    store.record(ACME, request_id="req-1", metric="quality", value=True, now=NOW)

    for request_id in ("req-1", "req-2"):
        with pytest.raises(FeedbackKindConflictError, match="send true or false"):
            store.record(ACME, request_id=request_id, metric="quality", value=1, now=NOW)
    other = FeedbackScope(api_key_id="key_other")
    assert store.record(other, request_id="req-3", metric="quality", value=0.5, now=NOW)
    assert store.record(ACME, request_id="req-2", metric="relevance", value=0.5, now=NOW)


def test_boolean_summary_groups_with_wilson_bounds_and_sorts_by_lower_bound() -> None:
    """Groups report n, positives, and Wilson bounds; a 1-for-1 group ranks below 4 of 4."""
    store = FeedbackStore()
    ratings = {"claude-opus-5-5": [True] * 4, "gpt-6-luna": [True, False, False, False]}
    ratings["gemini-3.8-flash"] = [True]
    for model, values in ratings.items():
        for index, value in enumerate(values):
            request_id = f"{model}-{index}"
            _register(store, request_id, model=model)
            store.record(ACME, request_id=request_id, metric="quality", value=value, now=NOW)

    summary = store.summary(ACME, metric="quality", group_by="model")

    assert summary.kind == "boolean"
    assert summary.n == 9
    assert [group.key for group in summary.groups] == [
        "claude-opus-5-5",
        "gemini-3.8-flash",
        "gpt-6-luna",
    ]
    opus, flash, luna = summary.groups
    assert (opus.n, opus.positives, opus.mean) == (4, 4, 1.0)
    assert (round(opus.ci_low, 4), opus.ci_high) == (0.5101, 1.0)
    assert (round(flash.ci_low, 4), flash.ci_high) == (0.2065, 1.0)
    assert (luna.positives, round(luna.ci_low, 4), round(luna.ci_high, 4)) == (1, 0.0456, 0.6994)
    by_provider = store.summary(ACME, metric="quality", group_by="provider")
    assert {group.key: group.n for group in by_provider.groups} == {
        "anthropic": 4,
        "google": 1,
        "openai": 4,
    }


def test_numeric_summary_reports_mean_interval() -> None:
    """Numeric metrics report the mean with a normal-approximation interval."""
    store = FeedbackStore()
    for index, value in enumerate([0.9, 0.7, 0.8]):
        _register(store, f"req-{index}")
        store.record(ACME, request_id=f"req-{index}", metric="relevance", value=value, now=NOW)

    summary = store.summary(ACME, metric="relevance", group_by="domain")

    assert summary.kind == "numeric"
    (group,) = summary.groups
    assert (group.key, group.n, group.positives) == ("general", 3, None)
    assert group.mean == pytest.approx(0.8)
    assert (group.ci_low, group.ci_high) == pytest.approx((0.686841, 0.913159), abs=1e-6)


def test_summary_is_empty_for_other_scopes_metrics_and_old_feedback() -> None:
    """Nothing matches: no groups, no kind, n = 0."""
    store = FeedbackStore()
    _register(store, "req-1")
    store.record(ACME, request_id="req-1", metric="quality", value=True, now=NOW)

    empty = [
        store.summary(FeedbackScope(api_key_id="key_acme", tenant_id="globex"), metric="quality"),
        store.summary(ACME, metric="relevance"),
        store.summary(ACME, metric="quality", since=NOW + 1),
    ]

    for summary in empty:
        assert (summary.kind, summary.n, summary.groups) == (None, 0, [])
    assert store.summary(ACME, metric="quality", since=NOW).n == 1


def test_feedback_persists_across_store_instances(tmp_path: Path) -> None:
    """A second store on the same file sees requests and feedback written by the first."""
    path = tmp_path / "feedback.sqlite3"
    writer = FeedbackStore(path)
    _register(writer, "req-1")
    written = writer.record(ACME, request_id="req-1", metric="quality", value=True, now=NOW)

    reader = FeedbackStore(path)
    updated = reader.record(ACME, request_id="req-1", metric="quality", value=False, now=NOW + 1)

    assert written is not None and updated is not None
    assert updated.record.feedback_id == written.record.feedback_id
    assert not updated.created
    evidence = FeedbackStore(path).model_feedback(
        ACME, metric="quality", domain="general", now=NOW + 1
    )
    assert evidence["claude-opus-5-5"].n == 1
    assert evidence["claude-opus-5-5"].positives == 0.0


def test_model_feedback_counts_fractional_evidence_within_the_window() -> None:
    """Numeric values add fractional successes; feedback older than the window is ignored."""
    store = FeedbackStore(window_seconds=3600.0)
    for index, value in enumerate([1.0, 0.5]):
        _register(store, f"req-{index}", model="gpt-6-astra")
        store.record(ACME, request_id=f"req-{index}", metric="score", value=value, now=NOW)

    evidence = store.model_feedback(ACME, metric="score", domain="general", now=NOW + 60)
    expired = store.model_feedback(ACME, metric="score", domain="general", now=NOW + 7200)

    astra = evidence["gpt-6-astra"]
    assert (astra.n, astra.positives, astra.negatives) == (2, 1.5, 0.5)
    assert astra.posterior_mean == pytest.approx(2.5 / 4.0)
    assert expired == {}
    assert store.model_feedback(ACME, metric="score", domain="code", now=NOW) == {}


def test_routing_totals_follow_updates_across_hours() -> None:
    """Replacing a value moves it to the hour it was rewritten in; it is never counted twice."""
    month = FeedbackStore()
    three_hours = FeedbackStore(window_seconds=3 * 3600.0)
    for store in (month, three_hours):
        for request_id in ("req-1", "req-2"):
            _register(store, request_id, recorded_at=NOW)
            store.record(ACME, request_id=request_id, metric="quality", value=True, now=NOW)
        store.record(ACME, request_id="req-1", metric="quality", value=False, now=NOW + 7200)

    both = month.model_feedback(ACME, metric="quality", domain="general", now=NOW + 7200)
    # Four hours later only the rewritten value (hour +2) is inside the 3-hour window.
    recent = three_hours.model_feedback(ACME, metric="quality", domain="general", now=NOW + 14400)

    assert (both["claude-opus-5-5"].n, both["claude-opus-5-5"].positives) == (2, 1.0)
    assert (recent["claude-opus-5-5"].n, recent["claude-opus-5-5"].positives) == (1, 0.0)
    assert month.summary(ACME, metric="quality").groups[0].positives == 1


def test_summary_reads_do_not_block_request_registration(tmp_path: Path) -> None:
    """An open read transaction (a long summary) must not stall the per-request index write."""
    path = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(path)
    _register(store, "req-1")
    reader = sqlite3.connect(path, timeout=0.1)
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT COUNT(*) FROM feedback_requests").fetchone() == (1,)

        _register(store, "req-2")
    finally:
        reader.rollback()
        reader.close()

    assert store.record(ACME, request_id="req-2", metric="quality", value=True, now=NOW)
