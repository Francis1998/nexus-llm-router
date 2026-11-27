"""Virtual keys end to end: the key admin API and enforcement on every model endpoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as main
from adapters.base import BaseProviderAdapter
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings
from router.engine import NexusRouter
from router.schemas import ChatMessage, ProviderResponse
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimiter
from safety.virtual_keys import KeyRateLimiter, VirtualKeyStore, budget_window

MASTER_KEY = "sk-master-test-0123456789abcdef"
MASTER = {"Authorization": f"Bearer {MASTER_KEY}"}
PROMPT = "Summarize the incident report for the on-call channel."
CHAT = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 64}
RESPONSES = {"input": PROMPT, "max_output_tokens": 64}
MESSAGES = {"model": "auto", "max_tokens": 64, "messages": [{"role": "user", "content": PROMPT}]}


class CountingAdapter(MockProviderAdapter):
    """Mock adapter that records every model it is asked to serve."""

    def __init__(self, provider_name: str, calls: list[str], *, fail: bool = False) -> None:
        super().__init__(provider_name, fail=fail)
        self._calls = calls

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        self._calls.append(model)
        return await super().complete(model, messages, max_tokens)


@dataclass
class Gateway:
    """A test client over mock providers plus the state its requests touch."""

    client: TestClient
    settings: RouterSettings
    store: VirtualKeyStore
    ledger: SpendLedger
    calls: list[str]

    def create_key(self, **fields: object) -> dict[str, Any]:
        response = self.client.post("/v1/keys", json=fields, headers=MASTER)
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        return body


def _bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Gateway]:
    """Build a nexus over counting mock providers; ``virtual_keys`` mode by default."""

    def build(*, failing: tuple[str, ...] = (), **overrides: object) -> Gateway:
        fields: dict[str, Any] = {
            "auth_mode": "virtual_keys",
            "master_key": MASTER_KEY,
            "audit_log_path": str(tmp_path / "audit-log.jsonl"),
            "spend_ledger_path": str(tmp_path / "spend-ledger.sqlite3"),
            "idempotency_path": str(tmp_path / "idempotency.sqlite3"),
            "feedback_path": str(tmp_path / "feedback.sqlite3"),
            "virtual_keys_path": str(tmp_path / "virtual-keys.sqlite3"),
        }
        settings = RouterSettings(**(fields | overrides))
        calls: list[str] = []
        adapters: dict[str, BaseProviderAdapter] = {
            name: CountingAdapter(name, calls, fail=name in failing)
            for name in ("openai", "anthropic", "google", "moonshot")
        }
        registry = AdapterRegistry(adapters)
        router = NexusRouter(settings, registry)
        store = VirtualKeyStore(settings.virtual_keys_path)
        ledger = SpendLedger(settings.spend_ledger_path)
        limiter = KeyRateLimiter()
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=60, refill_per_second=1.0),
        )
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: ResponseCache(enabled=False))
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        monkeypatch.setattr(main, "get_spend_ledger", lambda: ledger)
        monkeypatch.setattr(main, "get_virtual_key_store", lambda: store)
        monkeypatch.setattr(main, "get_key_rate_limiter", lambda: limiter)
        return Gateway(TestClient(main.app), settings, store, ledger, calls)

    return build


def _error(response: httpx.Response) -> tuple[str | None, str | None]:
    """Return ``(type, code)`` from an OpenAI or Anthropic error envelope."""
    body = response.json()
    if body.get("type") == "error":
        return body["error"]["type"], None
    return body["error"]["type"], body["error"]["code"]


# --- Admin API -------------------------------------------------------------------


def test_create_returns_the_secret_once_and_listings_only_show_its_prefix(
    gateway: Callable[..., Gateway],
) -> None:
    """The secret appears in the create response only; reads show a non-secret prefix."""
    gw = gateway()
    created = gw.create_key(alias="team-a", models=["gpt-6.2-sol"], max_budget_usd=5)
    secret = created["key"]
    assert secret.startswith("sk-nexus-")
    assert created["key_prefix"] == secret[: len("sk-nexus-") + 4]
    assert created["status"] == "active"
    assert created["models"] == ["gpt-6.2-sol"]
    assert created["spend"]["remaining_usd"] == 5

    reads = [
        gw.client.get(f"/v1/keys/{created['id']}", headers=MASTER),
        gw.client.get("/v1/keys", headers=MASTER),
        gw.client.get("/v1/key/info", headers=_bearer(secret)),
    ]
    for response in reads:
        assert response.status_code == 200
        assert secret not in response.text
        assert '"key":' not in response.text
    assert secret not in Path(gw.settings.virtual_keys_path).read_bytes().decode("latin-1")


@pytest.mark.parametrize(
    ("headers", "status", "code"),
    [
        pytest.param({}, 401, "invalid_api_key", id="missing"),
        pytest.param(_bearer("sk-nexus-not-issued"), 401, "invalid_api_key", id="unknown"),
        pytest.param(_bearer(MASTER_KEY + "x"), 401, "invalid_api_key", id="wrong-master"),
        pytest.param({"x-api-key": MASTER_KEY}, 200, None, id="master-x-api-key"),
    ],
)
def test_admin_api_requires_the_master_key(
    gateway: Callable[..., Gateway], headers: dict[str, str], status: int, code: str | None
) -> None:
    """Missing or invalid credentials get 401 ``invalid_api_key``."""
    response = gateway().client.get("/v1/keys", headers=headers)
    assert response.status_code == status
    if code is not None:
        assert _error(response) == ("authentication_error", code)


def test_admin_api_refuses_virtual_keys_with_403(gateway: Callable[..., Gateway]) -> None:
    """A valid virtual key is authenticated but may not manage keys."""
    gw = gateway()
    secret = gw.create_key()["key"]
    for method, path in [("GET", "/v1/keys"), ("POST", "/v1/keys"), ("DELETE", "/v1/keys/x")]:
        response = gw.client.request(method, path, json={}, headers=_bearer(secret))
        assert response.status_code == 403
        assert _error(response) == ("permission_error", "permission_denied")


def test_admin_api_checks_credentials_before_the_body(gateway: Callable[..., Gateway]) -> None:
    """An unauthenticated caller learns nothing from body validation errors."""
    response = gateway().client.post("/v1/keys", json={"models": "not-a-list"})
    assert response.status_code == 401


def test_admin_api_is_disabled_without_a_master_key(gateway: Callable[..., Gateway]) -> None:
    """In ``open`` mode without ``NEXUS_MASTER_KEY`` the admin API answers 403."""
    gw = gateway(auth_mode="open", master_key=None)
    response = gw.client.get("/v1/keys", headers=MASTER)
    assert response.status_code == 403
    assert "NEXUS_MASTER_KEY" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("body", "param", "code"),
    [
        pytest.param({"budget": 1}, "budget", "unknown_parameter", id="unknown-field"),
        pytest.param({"max_budget_usd": -1}, "max_budget_usd", "invalid_value", id="negative"),
        pytest.param({"budget_duration": "1y"}, "budget_duration", "invalid_value", id="window"),
        pytest.param({"rpm_limit": 0}, "rpm_limit", "invalid_value", id="rpm"),
        pytest.param({"models": ["gpt-unknown"]}, "models", "invalid_value", id="unknown-model"),
        pytest.param({"models": "gpt-6.2-sol"}, "models", "invalid_value", id="models-type"),
        pytest.param({"expires_at": "2001-01-01T00:00:00Z"}, "expires_at", "invalid_value"),
        pytest.param({"alias": " "}, "alias", "invalid_value", id="blank-alias"),
        pytest.param({"metadata": {"x": "y" * 5000}}, "metadata", "invalid_value", id="meta"),
    ],
)
def test_create_validation_errors_are_openai_400s(
    gateway: Callable[..., Gateway], body: dict[str, Any], param: str, code: str
) -> None:
    """Invalid fields get an OpenAI-envelope 400 naming the parameter."""
    response = gateway().client.post("/v1/keys", json=body, headers=MASTER)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert (error["type"], error["param"], error["code"]) == ("invalid_request_error", param, code)


def test_list_filters_and_paginates_including_revoked_keys(
    gateway: Callable[..., Gateway],
) -> None:
    """Listings filter by alias / blocked / revoked and page with limit and offset."""
    gw = gateway()
    first = gw.create_key(alias="search")
    second = gw.create_key(alias="search", blocked=True)
    third = gw.create_key(alias="billing")
    assert gw.client.delete(f"/v1/keys/{third['id']}", headers=MASTER).status_code == 200

    def ids(query: str) -> list[str]:
        response = gw.client.get(f"/v1/keys?{query}", headers=MASTER)
        assert response.status_code == 200, response.text
        return [key["id"] for key in response.json()["data"]]

    assert ids("") == [third["id"], second["id"], first["id"]]
    assert ids("alias=search") == [second["id"], first["id"]]
    assert ids("blocked=true") == [second["id"]]
    assert ids("revoked=true") == [third["id"]]
    assert ids("revoked=false&blocked=false") == [first["id"]]
    assert ids("expired=true") == []
    page = gw.client.get("/v1/keys?limit=2&offset=1", headers=MASTER).json()
    assert [key["id"] for key in page["data"]] == [second["id"], first["id"]]
    assert (page["total"], page["has_more"]) == (3, False)
    assert gw.client.get("/v1/keys?limit=1", headers=MASTER).json()["has_more"] is True
    assert gw.client.get("/v1/keys?limit=101", headers=MASTER).status_code == 400
    listed = gw.client.get("/v1/keys?revoked=true", headers=MASTER).json()["data"][0]
    assert listed["status"] == "revoked"
    assert listed["revoked_at"] is not None


def test_update_changes_and_clears_limits(gateway: Callable[..., Gateway]) -> None:
    """PATCH applies present fields, ``null`` clears a limit, omitted fields stay."""
    gw = gateway()
    key = gw.create_key(alias="a", models=["gpt-6.2-sol"], max_budget_usd=1, rpm_limit=5)
    response = gw.client.patch(
        f"/v1/keys/{key['id']}",
        json={"models": None, "max_budget_usd": 2.5, "budget_duration": "1d", "blocked": True},
        headers=MASTER,
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["models"] == []
    assert (updated["max_budget_usd"], updated["budget_duration"]) == (2.5, "1d")
    assert (updated["rpm_limit"], updated["alias"]) == (5, "a")
    assert (updated["blocked"], updated["status"]) == (True, "blocked")
    assert updated["spend"]["resets_at"] is not None

    missing = gw.client.patch("/v1/keys/vk_missing", json={"alias": "b"}, headers=MASTER)
    assert missing.status_code == 404
    assert _error(missing) == ("invalid_request_error", "key_not_found")
    null_blocked = gw.client.patch(f"/v1/keys/{key['id']}", json={"blocked": None}, headers=MASTER)
    assert null_blocked.status_code == 400


def test_revoked_keys_cannot_be_updated_or_rotated(gateway: Callable[..., Gateway]) -> None:
    """Revocation is final: PATCH and rotate answer 409, revoke stays idempotent."""
    gw = gateway()
    key_id = gw.create_key()["id"]
    assert gw.client.post(f"/v1/keys/{key_id}/revoke", headers=MASTER).status_code == 200
    assert gw.client.delete(f"/v1/keys/{key_id}", headers=MASTER).status_code == 200
    for response in [
        gw.client.patch(f"/v1/keys/{key_id}", json={"blocked": False}, headers=MASTER),
        gw.client.post(f"/v1/keys/{key_id}/rotate", headers=MASTER),
    ]:
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "key_revoked"


def test_rotate_replaces_the_secret_and_keeps_the_key(gateway: Callable[..., Gateway]) -> None:
    """After rotation the old secret is a 401 and the new one is the same key id."""
    gw = gateway()
    created = gw.create_key(alias="rotating")
    rotated = gw.client.post(f"/v1/keys/{created['id']}/rotate", headers=MASTER).json()
    assert rotated["id"] == created["id"]
    assert rotated["key"] != created["key"]
    assert gw.client.get("/v1/key/info", headers=_bearer(created["key"])).status_code == 401
    info = gw.client.get("/v1/key/info", headers=_bearer(rotated["key"]))
    assert info.json()["id"] == created["id"]


def test_key_info_reports_window_spend_and_reset(gateway: Callable[..., Gateway]) -> None:
    """``/v1/key/info`` shows the caller's limits, window spend, remaining budget, and reset."""
    gw = gateway()
    key = gw.create_key(models=["gpt-6.2-sol"], max_budget_usd=1, budget_duration="1d")
    assert gw.client.post("/v1/chat/completions", json=CHAT, headers=_bearer(key["key"])).is_success
    info = gw.client.get("/v1/key/info", headers={"x-api-key": key["key"]}).json()
    window = budget_window("1d", float(info["updated_at"]))
    assert window is not None
    spend = info["spend"]
    assert spend["request_count"] == 1
    assert spend["spend_usd"] > 0
    assert spend["remaining_usd"] == pytest.approx(1 - spend["spend_usd"])
    assert spend["resets_at"] == int(window[1])
    master = gw.client.get("/v1/key/info", headers=MASTER)
    assert master.status_code == 400
    assert gw.client.get("/v1/key/info").status_code == 401


# --- Enforcement on every endpoint -------------------------------------------------

SPEND_EVENT = {
    "request_id": "manual-1",
    "tenant": "acme",
    "provider": "openai",
    "model": "gpt-6.2-sol",
    "cost_usd": 0.5,
}
COUNT = {"model": "auto", "messages": [{"role": "user", "content": PROMPT}]}
# (method, path, body) for every endpoint virtual-keys mode guards.
ENDPOINTS: dict[str, tuple[str, str, dict[str, Any] | None]] = {
    "chat": ("POST", "/v1/chat/completions", CHAT),
    "chat-stream": ("POST", "/v1/chat/completions", CHAT | {"stream": True}),
    "responses": ("POST", "/v1/responses", RESPONSES),
    "responses-stream": ("POST", "/v1/responses", RESPONSES | {"stream": True}),
    "messages": ("POST", "/v1/messages", MESSAGES),
    "messages-stream": ("POST", "/v1/messages", MESSAGES | {"stream": True}),
    "count-tokens": ("POST", "/v1/messages/count_tokens", COUNT),
    "feedback": ("POST", "/v1/feedback", {"request_id": "req-1", "value": True}),
    "feedback-summary": ("GET", "/v1/feedback/summary", None),
    "spend": ("GET", "/v1/spend", None),
    "spend-record": ("POST", "/v1/spend", SPEND_EVENT),
    "models": ("GET", "/v1/models", None),
    "model": ("GET", "/v1/models/gpt-6.2-sol", None),
    "key-info": ("GET", "/v1/key/info", None),
}


def _call(gw: Gateway, endpoint: str, headers: dict[str, str]) -> httpx.Response:
    method, path, body = ENDPOINTS[endpoint]
    return gw.client.request(method, path, json=body, headers=headers)


def _credential(gw: Gateway, state: str) -> dict[str, str]:
    """Return headers carrying a key in ``state``."""
    if state == "missing":
        return {}
    if state == "unknown":
        return _bearer("sk-nexus-" + "A" * 43)
    key = gw.create_key(blocked=state == "blocked")
    if state == "revoked":
        assert gw.client.delete(f"/v1/keys/{key['id']}", headers=MASTER).status_code == 200
    if state == "expired":
        expire = gw.client.patch(f"/v1/keys/{key['id']}", json={"expires_at": 1}, headers=MASTER)
        assert expire.json()["status"] == "expired"
    return _bearer(key["key"])


@pytest.mark.parametrize("endpoint", list(ENDPOINTS))
@pytest.mark.parametrize(
    ("state", "code"),
    [
        ("missing", "invalid_api_key"),
        ("unknown", "invalid_api_key"),
        ("revoked", "api_key_revoked"),
        ("expired", "api_key_expired"),
        ("blocked", "api_key_blocked"),
    ],
)
def test_invalid_keys_get_401_before_any_provider_call(
    gateway: Callable[..., Gateway], endpoint: str, state: str, code: str
) -> None:
    """Unknown, revoked, expired, and blocked keys never reach a provider."""
    gw = gateway()
    response = _call(gw, endpoint, _credential(gw, state))
    assert response.status_code == 401, response.text
    body = response.json()
    if endpoint.startswith("chat"):
        assert "API key" in body["detail"]
    elif endpoint.startswith(("messages", "count")):
        assert body == {"type": "error", "error": body["error"]}
        assert body["error"]["type"] == "authentication_error"
    else:
        assert _error(response) == ("authentication_error", code)
    assert gw.calls == []
    assert gw.ledger.summary().request_count == 0


@pytest.mark.parametrize(
    ("endpoint", "body"),
    [
        ("chat", CHAT | {"model": "claude-opus-5-5"}),
        ("responses", RESPONSES | {"model": "claude-opus-5-5"}),
        ("messages", MESSAGES | {"model": "claude-opus-5-5"}),
        ("count-tokens", COUNT | {"model": "claude-opus-5-5"}),
    ],
)
def test_an_explicit_model_outside_the_allowlist_is_403(
    gateway: Callable[..., Gateway], endpoint: str, body: dict[str, Any]
) -> None:
    """Naming a catalog model the key may not use is a 403 ``model_not_allowed``."""
    gw = gateway()
    secret = gw.create_key(models=["gpt-6.2-sol"])["key"]
    _, path, _ = ENDPOINTS[endpoint]
    response = gw.client.post(path, json=body, headers={"x-api-key": secret})
    assert response.status_code == 403, response.text
    if endpoint == "chat":
        assert "may not use model 'claude-opus-5-5'" in response.json()["detail"]
    elif endpoint == "responses":
        assert _error(response) == ("permission_error", "model_not_allowed")
    else:
        assert response.json()["error"]["type"] == "permission_error"
    assert gw.calls == []


def test_routing_hints_and_rule_choices_stay_inside_the_allowlist(
    gateway: Callable[..., Gateway],
) -> None:
    """``auto`` and strategy picks outside the allowlist are routed to an allowed model."""
    gw = gateway()
    headers = _bearer(gw.create_key(models=["gpt-6.2-sol"])["key"])
    medical = {"messages": [{"role": "user", "content": "Patient dosage and diagnosis triage."}]}
    for body in [CHAT | {"model": "auto"}, medical, CHAT | {"model": "gpt-6.2-sol"}]:
        response = gw.client.post("/v1/chat/completions", json=body, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["model"] == "gpt-6.2-sol"
    rationale = gw.client.post("/v1/chat/completions", json=medical, headers=headers).json()
    assert "is not allowed for this API key" in rationale["router"]["rationale"]
    assert set(gw.calls) == {"gpt-6.2-sol"}


def test_fallbacks_never_reach_a_disallowed_model(gateway: Callable[..., Gateway]) -> None:
    """When the allowed provider fails, the request fails instead of leaving the allowlist."""
    gw = gateway(failing=("openai",))
    restricted = _bearer(gw.create_key(models=["gpt-6.2-sol", "gpt-6-luna"])["key"])
    response = gw.client.post("/v1/responses", json=RESPONSES, headers=restricted)
    assert response.status_code == 503, response.text
    assert set(gw.calls) <= {"gpt-6.2-sol", "gpt-6-luna"}

    gw.calls.clear()
    unrestricted = _bearer(gw.create_key()["key"])
    fallback = gw.client.post("/v1/responses", json=RESPONSES, headers=unrestricted)
    assert fallback.status_code == 200
    assert fallback.json()["model"] not in {"gpt-6.2-sol", "gpt-6-luna"}


def test_a_key_whose_models_left_the_catalog_gets_no_eligible_model(
    gateway: Callable[..., Gateway],
) -> None:
    """Allowlists naming only models the catalog no longer has are a 403."""
    gw = gateway()
    secret, _ = gw.store.create(allowed_models=["retired-model"])
    response = gw.client.post("/v1/responses", json=RESPONSES, headers=_bearer(secret))
    assert response.status_code == 403
    assert _error(response) == ("permission_error", "no_eligible_model")
    assert gw.calls == []


# --- Budgets, RPM limits, listings, and attribution ----------------------------------


@pytest.mark.parametrize(
    ("endpoint", "shape"),
    [
        ("chat", "detail"),
        ("responses", ("budget_exceeded", "budget_exceeded")),
        ("responses-stream", ("budget_exceeded", "budget_exceeded")),
        ("messages", ("billing_error", None)),
    ],
)
def test_an_exhausted_window_budget_is_402(
    gateway: Callable[..., Gateway], endpoint: str, shape: object
) -> None:
    """Spend settled in the current window up to the budget refuses the next request."""
    gw = gateway()
    key = gw.create_key(max_budget_usd=0.01, budget_duration="1h")
    gw.ledger.record(
        request_id="earlier", tenant="t", provider="openai", model="gpt-6.2-sol",
        cost_usd=0.01, api_key_id=key["id"],
    )
    response = _call(gw, endpoint, {"x-api-key": key["key"]})
    assert response.status_code == 402, response.text
    if shape == "detail":
        assert "API key budget exhausted" in response.json()["detail"]
    else:
        assert _error(response) == shape
    assert gw.calls == []


def test_the_forecast_refuses_a_request_the_remaining_budget_cannot_cover(
    gateway: Callable[..., Gateway],
) -> None:
    """Pre-dispatch, every candidate whose estimated cost exceeds the remainder is skipped."""
    gw = gateway()
    key = gw.create_key(max_budget_usd=0.0001)
    response = gw.client.post("/v1/responses", json=RESPONSES, headers=_bearer(key["key"]))
    assert response.status_code == 402
    assert _error(response) == ("budget_exceeded", "budget_exceeded")
    assert "API key budget exceeded" in response.json()["error"]["message"]
    assert gw.calls == []


def test_spend_settles_from_the_real_cost_and_resets_with_the_window(
    gateway: Callable[..., Gateway],
) -> None:
    """The ledger holds the provider's real cost per key; last window's spend does not count."""
    gw = gateway()
    key = gw.create_key(max_budget_usd=0.01, budget_duration="1h")
    window = budget_window("1h", float(key["created_at"]))
    assert window is not None
    gw.ledger.record(
        request_id="last-hour", tenant="t", provider="openai", model="gpt-6.2-sol",
        cost_usd=0.01, recorded_at=window[0] - 1, api_key_id=key["id"],
    )
    response = gw.client.post("/v1/chat/completions", json=CHAT, headers=_bearer(key["key"]))
    assert response.status_code == 200, response.text
    cost = response.json()["router"]["cost_usd"]
    assert cost > 0
    spend = gw.client.get(f"/v1/keys/{key['id']}", headers=MASTER).json()["spend"]
    assert spend["spend_usd"] == pytest.approx(cost)
    assert spend["total_spend_usd"] == pytest.approx(cost + 0.01)
    assert spend["window_start"] == int(window[0])
    assert spend["remaining_usd"] == pytest.approx(0.01 - cost)


def test_rpm_limit_returns_429_and_only_metered_calls_count(
    gateway: Callable[..., Gateway],
) -> None:
    """``rpm_limit`` caps calls that can spend; listing models or key info is free."""
    gw = gateway()
    headers = _bearer(gw.create_key(rpm_limit=2)["key"])
    for endpoint in ["models", "key-info", "count-tokens", "spend"]:
        assert _call(gw, endpoint, headers).status_code == 200
    assert _call(gw, "chat", headers).status_code == 200
    assert _call(gw, "responses", headers).status_code == 200
    limited = _call(gw, "responses", headers)
    assert limited.status_code == 429
    assert _error(limited) == ("rate_limit_error", "rate_limit_exceeded")
    assert _call(gw, "chat", headers).status_code == 429
    assert len(gw.calls) == 2


def test_models_lists_only_the_keys_allowed_models(gateway: Callable[..., Gateway]) -> None:
    """``/v1/models`` is filtered per key; other catalog models are 404 for that key."""
    gw = gateway()
    headers = _bearer(gw.create_key(models=["claude-opus-5-5", "gpt-6.2-sol"])["key"])
    listed = gw.client.get("/v1/models", headers=headers).json()["data"]
    catalog = gw.client.get("/v1/models", headers=MASTER).json()["data"]
    assert [model["id"] for model in listed] == [
        model["id"] for model in catalog if model["id"] in {"claude-opus-5-5", "gpt-6.2-sol"}
    ]
    assert len(catalog) > 2
    hidden = gw.client.get("/v1/models/gemini-4-argon", headers=headers)
    assert hidden.status_code == 404
    assert _error(hidden) == ("invalid_request_error", "model_not_found")
    assert gw.client.get("/v1/models/gemini-4-argon", headers=MASTER).status_code == 200


def test_spend_endpoint_is_scoped_to_the_calling_key(gateway: Callable[..., Gateway]) -> None:
    """A virtual key sees its own spend; the master key sees all and alone may record."""
    gw = gateway()
    first, second = _bearer(gw.create_key()["key"]), _bearer(gw.create_key()["key"])
    assert gw.client.post("/v1/chat/completions", json=CHAT, headers=first).is_success
    assert gw.client.post("/v1/responses", json=RESPONSES, headers=second).is_success
    assert gw.client.get("/v1/spend", headers=first).json()["request_count"] == 1
    assert gw.client.get("/v1/spend", headers=MASTER).json()["request_count"] == 2
    denied = gw.client.post("/v1/spend", json=SPEND_EVENT, headers=first)
    assert denied.status_code == 403
    assert _error(denied) == ("permission_error", "permission_denied")
    assert gw.client.post("/v1/spend", json=SPEND_EVENT, headers=MASTER).status_code == 200


def test_ledger_and_audit_attribute_requests_to_the_key_id_never_the_secret(
    gateway: Callable[..., Gateway], capsys: pytest.CaptureFixture[str]
) -> None:
    """Spend rows and audit records carry ``vk_...``; the secret is in no file or log."""
    gw = gateway()
    key = gw.create_key(tenant="acme")
    response = gw.client.post("/v1/messages", json=MESSAGES, headers={"x-api-key": key["key"]})
    assert response.status_code == 200, response.text
    assert gw.ledger.key_spend(key["id"])[1] == 1
    audit = json.loads(Path(gw.settings.audit_log_path).read_text().splitlines()[-1])
    assert (audit["api_key_id"], audit["tenant_id"]) == (key["id"], "acme")
    for path in [gw.settings.spend_ledger_path, gw.settings.audit_log_path]:
        assert key["key"] not in Path(path).read_bytes().decode("latin-1")
    assert key["key"] not in capsys.readouterr().out


def test_tenant_bound_keys_reject_another_tenant(gateway: Callable[..., Gateway]) -> None:
    """``X-Tenant-Id`` naming another tenant than the key's is a 403 ``tenant_mismatch``.

    The refusal does not use up the key's only request of the minute.
    """
    gw = gateway()
    headers = _bearer(gw.create_key(tenant="acme", rpm_limit=1)["key"])
    mismatch = gw.client.post(
        "/v1/responses", json=RESPONSES, headers=headers | {"X-Tenant-Id": "globex"}
    )
    assert mismatch.status_code == 403
    assert _error(mismatch) == ("permission_error", "tenant_mismatch")
    same = headers | {"X-Tenant-Id": "acme"}
    assert gw.client.post("/v1/responses", json=RESPONSES, headers=same).status_code == 200


def test_idempotency_cache_and_feedback_stay_per_key(
    gateway: Callable[..., Gateway], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replays, cache hits, and feedback rights never cross from one key to another."""
    gw = gateway()
    cache = ResponseCache(enabled=True)
    monkeypatch.setattr(main, "get_response_cache", lambda: cache)
    first, second = _bearer(gw.create_key()["key"]), _bearer(gw.create_key()["key"])
    replay = {"Idempotency-Key": "retry-1"}
    original = gw.client.post("/v1/responses", json=RESPONSES, headers=first | replay).json()
    assert gw.client.post("/v1/responses", json=RESPONSES, headers=first | replay).json() == (
        original
    )
    other = gw.client.post("/v1/responses", json=RESPONSES, headers=second | replay).json()
    assert other["id"] != original["id"]
    request_id = original["nexus"]["request_id"]
    assert other["nexus"]["request_id"] != request_id
    rate = {"request_id": request_id, "value": True}
    assert gw.client.post("/v1/feedback", json=rate, headers=second).status_code == 404
    assert gw.client.post("/v1/feedback", json=rate, headers=first).status_code == 200


def test_open_mode_is_unchanged(gateway: Callable[..., Gateway]) -> None:
    """Without ``NEXUS_AUTH_MODE=virtual_keys`` no key is needed and nothing is filtered."""
    gw = gateway(auth_mode="open")
    secret = gw.create_key(models=["gpt-6.2-sol"], max_budget_usd=0)["key"]
    for headers in [{}, _bearer("any-client-key"), _bearer(secret)]:
        assert gw.client.post("/v1/responses", json=RESPONSES, headers=headers).status_code == 200
    models = gw.client.get("/v1/models", headers=_bearer(secret)).json()["data"]
    assert len(models) > 1
    assert gw.client.get("/v1/spend").status_code == 200
    assert gw.client.post("/v1/spend", json=SPEND_EVENT).status_code == 200


def test_virtual_keys_mode_requires_a_master_key() -> None:
    """Settings refuse ``virtual_keys`` without ``NEXUS_MASTER_KEY`` (and short master keys)."""
    with pytest.raises(ValueError, match="NEXUS_MASTER_KEY is required"):
        RouterSettings(auth_mode="virtual_keys")
    with pytest.raises(ValueError, match="at least 16"):
        RouterSettings(auth_mode="virtual_keys", master_key="short")


def test_x_api_key_and_authorization_must_agree(gateway: Callable[..., Gateway]) -> None:
    """Two different credentials on one request are a 401, never billed to either key."""
    gw = gateway()
    first, second = gw.create_key()["key"], gw.create_key()["key"]
    headers = _bearer(first) | {"x-api-key": second}
    response = gw.client.post("/v1/responses", json=RESPONSES, headers=headers)
    assert response.status_code == 401
    both = _bearer(first) | {"x-api-key": first}
    assert gw.client.post("/v1/responses", json=RESPONSES, headers=both).status_code == 200
