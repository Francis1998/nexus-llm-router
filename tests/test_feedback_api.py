"""POST /v1/feedback and GET /v1/feedback/summary over the real app with mock providers."""

from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from prometheus_client import REGISTRY, generate_latest

import api.main as main
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from observability.metrics import BoundedLabelValues
from router.config import RouterSettings, default_model_catalog
from router.engine import NexusRouter
from router.feedback import FeedbackScope
from router.schemas import (
    AuditRecord,
    ChatMessage,
    DomainTag,
    ProviderResponse,
    RoutingStrategyName,
)
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.key_fingerprint import fingerprint_api_key
from safety.tenant_rate_limiter import TenantRateLimiter

PROVIDERS = ("openai", "anthropic", "google", "moonshot")
ACME = {"Authorization": "Bearer sk-acme", "X-Tenant-Id": "acme"}
# General-domain prompt that rule-based answers with an explicitly requested model.
DESIGN_PROMPT = (
    "Design a multi-region failover strategy for a {system}: compare active-active, "
    "active-passive, and pilot-light architectures, analyze cost, latency, and consistency "
    "trade-offs, and recommend one with a phased rollout plan."
)


class CountingAdapter(MockProviderAdapter):
    """Mock provider that counts calls; ``unpriced`` models are missing from its prices."""

    def __init__(self, provider_name: str, unpriced: frozenset[str] = frozenset()) -> None:
        super().__init__(provider_name)
        self.unpriced = unpriced
        self.calls = 0

    def has_price(self, model: str) -> bool:
        return model not in self.unpriced

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        self.calls += 1
        return await super().complete(model, messages, max_tokens)


@dataclass
class Gateway:
    """Test client plus the router and adapters behind it."""

    client: TestClient
    router: NexusRouter
    adapters: dict[str, CountingAdapter]

    def chat(
        self,
        prompt: str = "Summarize this customer call in three bullet points.",
        *,
        headers: dict[str, str] | None = None,
        **body: object,
    ) -> dict[str, Any]:
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}], **body},
            headers=ACME if headers is None else headers,
        )
        assert response.status_code == 200, response.text
        payload: dict[str, Any] = response.json()
        return payload

    def feedback(self, headers: dict[str, str] | None = None, **body: object) -> Response:
        return self.client.post(
            "/v1/feedback", json=body, headers=ACME if headers is None else headers
        )

    def summary(self, headers: dict[str, str] | None = None, **params: str) -> Response:
        return self.client.get(
            "/v1/feedback/summary", params=params, headers=ACME if headers is None else headers
        )


GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GatewayFactory:
    """Build the app over mock providers with private state files."""

    def _build(*, unpriced: frozenset[str] = frozenset(), **overrides: object) -> Gateway:
        settings = RouterSettings(
            _env_file=None,
            audit_log_path=str(tmp_path / "audit-log.jsonl"),
            spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
            idempotency_path=str(tmp_path / "idempotency.sqlite3"),
            feedback_path=str(tmp_path / "feedback.sqlite3"),
            **{"response_cache_enabled": False, **overrides},
        )
        cache = ResponseCache(ttl_seconds=60.0, enabled=settings.response_cache_enabled)
        adapters = {name: CountingAdapter(name, unpriced) for name in PROVIDERS}
        registry = AdapterRegistry(dict(adapters))
        # Unpriced models also carry $0 catalog prices, so nothing can price them.
        catalog = {
            name: candidate.model_copy(update={"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0})
            if name in unpriced
            else candidate
            for name, candidate in default_model_catalog().items()
        }
        router = NexusRouter(settings, registry, model_catalog=catalog)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=1000, refill_per_second=1.0),
        )
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: cache)
        return Gateway(client=TestClient(main.app), router=router, adapters=adapters)

    return _build


def _not_found(request_id: str) -> dict[str, Any]:
    return {
        "error": {
            "message": f"No request found with id '{request_id}' for this API key and tenant.",
            "type": "invalid_request_error",
            "param": "request_id",
            "code": "request_not_found",
        }
    }


def test_feedback_round_trip_stores_the_routing_snapshot(gateway: GatewayFactory) -> None:
    """Rate a chat completion, get the stored record back, and see it in the summary."""
    app = gateway()
    routed = app.chat()["router"]

    response = app.feedback(request_id=routed["request_id"], value=True, comment="Spot on.")

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("fb_")
    assert body["created_at"] == body["updated_at"]
    assert {key: body[key] for key in body if key not in ("id", "created_at", "updated_at")} == {
        "object": "feedback",
        "request_id": routed["request_id"],
        "metric": "quality",
        "value": True,
        "source": "human",
        "comment": "Spot on.",
        "model": routed["model_used"],
        "provider": app.router.model_catalog[routed["model_used"]].provider,
        "strategy": "rule-based",
        "domain": "general",
    }
    summary = app.summary(group_by="model").json()
    assert summary == {
        "object": "feedback.summary",
        "metric": "quality",
        "group_by": "model",
        "kind": "boolean",
        "interval": "wilson",
        "confidence": 0.95,
        "since": None,
        "n": 1,
        "groups": [
            {
                "key": routed["model_used"],
                "n": 1,
                "positives": 1,
                "mean": 1.0,
                "ci_low": pytest.approx(0.2065, abs=1e-4),
                "ci_high": 1.0,
            }
        ],
    }


def test_request_ids_from_responses_and_messages_accept_feedback(gateway: GatewayFactory) -> None:
    """The ``nexus.request_id`` of /v1/responses and /v1/messages works with either key header."""
    app = gateway()
    response = app.client.post("/v1/responses", json={"input": "Draft a note."}, headers=ACME)
    message = app.client.post(
        "/v1/messages",
        json={"model": "auto", "max_tokens": 64, "messages": [{"role": "user", "content": "Hi"}]},
        headers={"x-api-key": "sk-acme", "X-Tenant-Id": "acme"},
    )

    for routed in (response.json()["nexus"], message.json()["nexus"]):
        rated = app.feedback(request_id=routed["request_id"], value=0.5, metric="helpfulness")
        assert rated.status_code == 200, rated.text
    via_x_api_key = app.feedback(
        headers={"x-api-key": "sk-acme", "X-Tenant-Id": "acme"},
        request_id=message.json()["nexus"]["request_id"],
        value=1.0,
        metric="helpfulness",
    )
    assert via_x_api_key.status_code == 200
    assert app.summary(metric="helpfulness").json()["n"] == 2


def test_unknown_request_id_is_404(gateway: GatewayFactory) -> None:
    """An id nexus never issued gets an OpenAI-style 404."""
    response = gateway().feedback(request_id="0" * 32, value=True)

    assert response.status_code == 404
    assert response.json() == _not_found("0" * 32)


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer sk-globex", "X-Tenant-Id": "acme"},
        {"Authorization": "Bearer sk-acme", "X-Tenant-Id": "globex"},
        {"Authorization": "Bearer sk-acme"},
        {},
    ],
    ids=["other-key", "other-tenant", "no-tenant", "anonymous"],
)
def test_feedback_from_another_caller_is_an_indistinguishable_404(
    gateway: GatewayFactory, headers: dict[str, str]
) -> None:
    """Another key or tenant gets exactly the unknown-id 404, and its summary stays empty."""
    app = gateway()
    request_id = app.chat()["router"]["request_id"]

    response = app.feedback(headers=headers, request_id=request_id, value=False)

    assert response.status_code == 404
    assert response.json() == _not_found(request_id)
    assert app.summary(headers=headers).json()["groups"] == []
    assert app.feedback(request_id=request_id, value=True).status_code == 200


def test_repeated_feedback_replaces_the_value(gateway: GatewayFactory) -> None:
    """One value per (request_id, metric, source): retries update instead of double-counting."""
    app = gateway()
    request_id = app.chat()["router"]["request_id"]

    first = app.feedback(request_id=request_id, value=True).json()
    second = app.feedback(request_id=request_id, value=False, comment="Changed my mind").json()
    judge = app.feedback(request_id=request_id, value=False, source="automated").json()

    assert second["id"] == first["id"] != judge["id"]
    assert (second["value"], second["comment"]) == (False, "Changed my mind")
    summary = app.summary().json()
    assert (summary["n"], summary["groups"][0]["positives"]) == (2, 0)


@pytest.mark.parametrize(
    ("body", "param", "code"),
    [
        ({"value": True}, "request_id", "missing_required_parameter"),
        ({"request_id": "r", "value": 1.5}, "value", "invalid_value"),
        ({"request_id": "r", "value": "yes"}, "value", "invalid_value"),
        ({"request_id": "r", "value": None}, "value", "invalid_value"),
        ({"request_id": "r", "value": True, "metric": "Thumbs Up"}, "metric", "invalid_value"),
        ({"request_id": "r", "value": True, "metric": "m" * 65}, "metric", "invalid_value"),
        ({"request_id": "r", "value": True, "source": "robot"}, "source", "invalid_value"),
        ({"request_id": "r", "value": True, "comment": "x" * 1001}, "comment", "invalid_value"),
        ({"request_id": "r", "value": True, "score": 5}, "score", "unknown_parameter"),
    ],
    ids=[
        "missing-request-id",
        "value-above-one",
        "value-string",
        "value-null",
        "metric-pattern",
        "metric-too-long",
        "unknown-source",
        "comment-too-long",
        "unknown-field",
    ],
)
def test_invalid_feedback_is_a_422_with_an_openai_envelope(
    gateway: GatewayFactory, body: dict[str, Any], param: str, code: str
) -> None:
    """Validation failures name the parameter in the OpenAI error envelope."""
    response = gateway().feedback(**body)

    assert response.status_code == 422
    error = response.json()["error"]
    assert (error["type"], error["param"], error["code"]) == ("invalid_request_error", param, code)


@pytest.mark.parametrize(
    ("params", "param"),
    [
        ({"group_by": "tenant"}, "group_by"),
        ({"metric": "Q!"}, "metric"),
        ({"since": "x"}, "since"),
        ({"since": "0001-01-01T00:00:00+14:00"}, "since"),
    ],
    ids=["group-by", "metric", "since", "since-out-of-range"],
)
def test_invalid_summary_query_is_a_422(
    gateway: GatewayFactory, params: dict[str, str], param: str
) -> None:
    """Unsupported grouping, metric names, or timestamps are rejected."""
    response = gateway().summary(**params)

    assert response.status_code == 422
    assert response.json()["error"]["param"] == param


def test_cache_hits_stay_with_the_caller_that_can_rate_them(gateway: GatewayFactory) -> None:
    """A cached answer replays its request id only to the key and tenant that own it."""
    app = gateway(response_cache_enabled=True)
    globex = {"Authorization": "Bearer sk-acme", "X-Tenant-Id": "globex"}

    first = app.chat()["router"]["request_id"]
    replay = app.chat()["router"]["request_id"]
    other_tenant = app.chat(headers=globex)["router"]["request_id"]

    assert replay == first
    assert other_tenant != first
    assert sum(adapter.calls for adapter in app.adapters.values()) == 2
    assert app.feedback(headers=globex, request_id=other_tenant, value=True).status_code == 200


def test_numeric_value_for_a_boolean_metric_is_a_409(gateway: GatewayFactory) -> None:
    """A metric keeps one value kind per caller: ``1`` is not ``true``."""
    app = gateway()
    first, second = app.chat()["router"]["request_id"], app.chat()["router"]["request_id"]
    app.feedback(request_id=first, value=True)

    response = app.feedback(request_id=second, value=1)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "feedback_kind_conflict"


def test_summary_since_filter_and_numeric_metrics(gateway: GatewayFactory) -> None:
    """Numeric metrics report the mean; ``since`` in the future matches nothing."""
    app = gateway()
    for value in (0.9, 0.7, 0.8):
        request_id = app.chat()["router"]["request_id"]
        app.feedback(request_id=request_id, metric="relevance", value=value, source="automated")

    numeric = app.summary(metric="relevance", group_by="strategy").json()
    future = app.summary(metric="relevance", since="2099-01-01T00:00:00").json()

    assert (numeric["kind"], numeric["interval"], numeric["n"]) == ("numeric", "normal", 3)
    group = numeric["groups"][0]
    assert (group["key"], group["positives"]) == ("rule-based", None)
    assert group["mean"] == pytest.approx(0.8)
    assert (future["since"], future["n"], future["groups"]) == ("2099-01-01T00:00:00Z", 0, [])


def test_feedback_is_counted_in_prometheus_but_comments_and_tenants_are_not_exported(
    gateway: GatewayFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    """The counter is labelled by metric and model only; logs and metrics omit the comment."""
    app = gateway()
    routed = app.chat()["router"]
    labels = {"metric": "quality", "model": routed["model_used"]}
    before = REGISTRY.get_sample_value("router_feedback_total", labels) or 0.0
    capsys.readouterr()

    app.feedback(request_id=routed["request_id"], value=True, comment="private: card 4242")

    logged = capsys.readouterr().out
    assert REGISTRY.get_sample_value("router_feedback_total", labels) == before + 1
    assert '"event": "feedback_recorded"' in logged
    assert routed["request_id"] in logged
    exported = generate_latest().decode()
    for secret in ("card 4242", "private", "acme"):
        assert secret not in logged
        assert secret not in exported
    samples = [line for line in exported.splitlines() if line.startswith("router_feedback_total{")]
    label_names = {re.findall(r'(\w+)="', line)[0] for line in samples} | {
        re.findall(r'(\w+)="', line)[1] for line in samples
    }
    assert label_names == {"metric", "model"}


def test_metric_label_values_are_bounded() -> None:
    """Client-chosen metric names beyond the cap share one Prometheus label."""
    labels = BoundedLabelValues(limit=2)

    assert [labels.label(name) for name in ("a", "b", "c", "a")] == ["a", "b", "other", "a"]


def test_feedback_aware_routing_learns_from_posted_feedback(
    gateway: GatewayFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold start defers to rule-based; after enough feedback the bandit routes by it."""
    app = gateway(feedback_min_samples=6)
    request_ids = count(1)
    # Sequential request ids make the seeded Thompson draws reproducible.
    monkeypatch.setattr(main, "uuid4", lambda: uuid.UUID(int=next(request_ids)))
    headers = {**ACME, "X-Router-Strategy": "feedback-aware"}
    cold = app.chat(DESIGN_PROMPT.format(system="payments platform"), headers=headers)["router"]
    ratings = {"gpt-6-luna": False, "claude-opus-5-5": True, "gpt-6-astra": False}
    for index, (model, liked) in enumerate(ratings.items()):
        for attempt in range(4):
            system = f"system {index}-{attempt}"
            routed = app.chat(DESIGN_PROMPT.format(system=system), model=model)["router"]
            assert routed["model_used"] == model
            assert app.feedback(request_id=routed["request_id"], value=liked).status_code == 200

    warm = [
        app.chat(DESIGN_PROMPT.format(system=f"ticket {n}"), headers=headers)["router"]
        for n in range(5)
    ]

    assert cold["routing_strategy"] == "feedback-aware"
    assert cold["model_used"] == "gpt-6-luna"
    assert cold["rationale"].startswith("cold start: 0/6 quality feedback samples")
    assert {routed["model_used"] for routed in warm} == {"claude-opus-5-5"}
    assert "posterior mean 0.83, n=4" in warm[0]["rationale"]
    audit = [
        AuditRecord.model_validate_json(line)
        for line in (tmp_path / "audit-log.jsonl").read_text().splitlines()
    ]
    assert {(record.api_key_id, record.tenant_id, record.domain) for record in audit} == {
        (fingerprint_api_key(ACME["Authorization"]), "acme", DomainTag.GENERAL)
    }
    assert audit[-1].routing_strategy is RoutingStrategyName.FEEDBACK_AWARE


def test_feedback_aware_keeps_the_budget_402_and_unpriced_500(gateway: GatewayFactory) -> None:
    """Refusals before dispatch map to the same statuses as for every other strategy."""
    headers = {**ACME, "X-Router-Strategy": "feedback-aware"}
    body = {"input": DESIGN_PROMPT.format(system="payments platform"), "user": "alice"}
    over_budget = gateway(budget_cap_usd=0.0)
    budget = over_budget.client.post("/v1/responses", json=body, headers=headers)
    unpriced = gateway(unpriced=frozenset(default_model_catalog()))
    price = unpriced.client.post("/v1/responses", json=body, headers=headers)

    assert budget.status_code == 402
    assert budget.json()["error"]["code"] == "budget_exceeded"
    assert price.status_code == 500
    assert price.json()["error"]["code"] == "model_price_unknown"
    assert sum(adapter.calls for adapter in [*over_budget.adapters.values()]) == 0
    assert sum(adapter.calls for adapter in [*unpriced.adapters.values()]) == 0


def test_feedback_aware_never_picks_a_model_the_engine_cannot_price(
    gateway: GatewayFactory,
) -> None:
    """Strong feedback for a model that lost its price does not send traffic to it."""
    app = gateway(unpriced=frozenset({"claude-opus-5-5"}), feedback_min_samples=5)
    store = app.router.feedback_store
    scope = FeedbackScope(fingerprint_api_key(ACME["Authorization"]), "acme")
    seeded = [("claude-opus-5-5", True)] * 10 + [("gpt-6-luna", False), ("gpt-6-astra", True)] * 3
    for n, (model, liked) in enumerate(seeded):
        store.register_request(
            AuditRecord(
                request_id=f"seed-{n}",
                chosen_model=model,
                provider=app.router.model_catalog[model].provider,
                routing_strategy=RoutingStrategyName.RULE_BASED,
                rationale="seed",
                latency_ms=1.0,
                token_cost=0.001,
                input_tokens=1,
                output_tokens=1,
                state="RESPONDED",
                api_key_id=scope.api_key_id,
                tenant_id=scope.tenant_id,
                domain=DomainTag.GENERAL,
            )
        )
        store.record(scope, request_id=f"seed-{n}", metric="quality", value=liked)
    headers = {**ACME, "X-Router-Strategy": "feedback-aware"}

    routed = [
        app.chat(DESIGN_PROMPT.format(system=f"system {n}"), headers=headers)["router"]
        for n in range(5)
    ]

    assert all(route["model_used"] != "claude-opus-5-5" for route in routed)
    assert all("fallback attempt" not in route["rationale"] for route in routed)
    # Only the 6 values of the priced candidates count toward the minimum.
    assert all("; 6 quality samples for general prompts" in route["rationale"] for route in routed)


def test_feedback_index_failure_never_fails_or_retries_the_request(
    gateway: GatewayFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer is returned once; only feedback for that request becomes unavailable."""
    app = gateway()

    def locked(record: AuditRecord, *, recorded_at: float | None = None) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(app.router.feedback_store, "register_request", locked)

    routed = app.chat()["router"]

    assert sum(adapter.calls for adapter in app.adapters.values()) == 1
    assert app.feedback(request_id=routed["request_id"], value=True).status_code == 404


def test_feedback_survives_a_gateway_restart(gateway: GatewayFactory) -> None:
    """Requests and feedback live in SQLite, so a new router on the same file still knows them."""
    app = gateway()
    request_id = app.chat()["router"]["request_id"]
    app.feedback(request_id=request_id, value=True)

    restarted = gateway()

    assert restarted.router is not app.router
    assert restarted.summary().json()["n"] == 1
    updated = restarted.feedback(request_id=request_id, value=False).json()
    assert (updated["request_id"], updated["value"]) == (request_id, False)
    assert restarted.summary().json()["groups"][0]["positives"] == 0
