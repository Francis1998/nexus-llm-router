"""Regression tests: a failed spend-ledger or audit-log write must not re-dispatch a request.

After a provider answers, the router records the completion's spend in the SQLite
spend ledger and appends its audit record to the JSONL audit log. Both writes ran
inside the block that treats any exception as a provider failure, so a write error
(SQLite ``database is locked`` while another connection holds the ledger's write
lock, a full disk, a read-only volume):

* dispatched the already-answered request to every fallback model, each one billed,
  and recorded a provider error against every provider that had answered;
* then returned a bare HTTP 500 (``StateTransitionError``) instead of the answer;
* ended a fully delivered stream with an error that blamed the provider
  (``provider google failed after streaming output began: database is locked``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from prometheus_client import REGISTRY

import api.main as main
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.audit import AuditLog
from router.config import RouterSettings
from router.engine import NexusRouter
from router.schemas import AuditRecord, ChatMessage, ProviderResponse
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.spend_ledger import SpendLedger, SpendRecord
from safety.tenant_rate_limiter import TenantRateLimiter

PROVIDERS = ("openai", "anthropic", "google", "moonshot")
PROMPT = "Summarize this customer call in three bullet points."
HEADERS = {"Authorization": "Bearer sk-accounting", "x-api-key": "sk-accounting"}
# One non-streaming request per API surface; all three share the router.
REQUESTS: dict[str, dict[str, Any]] = {
    "/v1/chat/completions": {"messages": [{"role": "user", "content": PROMPT}]},
    "/v1/responses": {"input": PROMPT},
    "/v1/messages": {
        "model": "claude-opus-5-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": PROMPT}],
    },
}
# Last frame of a successful stream on each surface (SSE event name, or chat's data).
STREAM_ENDINGS = {
    "/v1/chat/completions": "[DONE]",
    "/v1/responses": "response.completed",
    "/v1/messages": "message_stop",
}
# Chat error frame, Responses failure event, and Messages error event.
FAILURE_FRAMES = ('{"error"', "response.failed", "error")


class CountingAdapter(MockProviderAdapter):
    """Mock provider that counts the requests it answers, streamed or not."""

    def __init__(self, provider_name: str) -> None:
        super().__init__(provider_name)
        self.calls = 0

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        self.calls += 1
        return await super().complete(model, messages, max_tokens)


@dataclass
class Gateway:
    """Test client plus the router, adapters, and ledger behind it."""

    client: TestClient
    router: NexusRouter
    adapters: dict[str, CountingAdapter]
    ledger_path: Path

    @property
    def provider_calls(self) -> int:
        return sum(adapter.calls for adapter in self.adapters.values())

    def post(self, path: str, **body: object) -> Response:
        return self.client.post(path, json={**REQUESTS[path], **body}, headers=HEADERS)


GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GatewayFactory:
    """Build the app over counting mock providers with private state files."""

    def _build(**overrides: object) -> Gateway:
        paths = {
            "audit_log_path": str(tmp_path / "audit-log.jsonl"),
            "spend_ledger_path": str(tmp_path / "spend-ledger.sqlite3"),
            "idempotency_path": str(tmp_path / "idempotency.sqlite3"),
            "feedback_path": str(tmp_path / "feedback.sqlite3"),
        }
        settings = RouterSettings(
            _env_file=None, response_cache_enabled=False, **{**paths, **overrides}
        )
        adapters = {name: CountingAdapter(name) for name in PROVIDERS}
        registry = AdapterRegistry(dict(adapters))
        router = NexusRouter(settings, registry)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=1000, refill_per_second=1.0),
        )
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: ResponseCache(enabled=False))
        client = TestClient(main.app, raise_server_exceptions=False)
        return Gateway(client, router, adapters, Path(settings.spend_ledger_path))

    return _build


@pytest.fixture
def locked_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every spend-ledger write fail as SQLite does when another connection holds the lock."""

    def locked(self: SpendLedger, **fields: object) -> SpendRecord:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(SpendLedger, "record", locked)


def _frames(body: str) -> list[str]:
    """Return each SSE frame's ``event`` name, or its ``data`` when it has no event name."""
    frames: list[str] = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        frames.append(fields.get("event", fields.get("data", "")))
    return frames


def _provider_errors() -> float:
    """Return the sum of every ``provider_error_rate`` sample in the default registry."""
    return sum(
        sample.value
        for metric in REGISTRY.collect()
        if metric.name == "provider_error_rate"
        for sample in metric.samples
        if sample.name == "provider_error_rate_total"
    )


@pytest.mark.parametrize("path", list(REQUESTS))
@pytest.mark.usefixtures("locked_ledger")
def test_locked_spend_ledger_returns_the_answer_from_one_provider_call(
    gateway: GatewayFactory, path: str
) -> None:
    """The provider already answered: return that answer, never re-dispatch or blame it."""
    app = gateway()
    errors_before = _provider_errors()

    response = app.post(path)

    # (HTTP status, provider calls, provider errors recorded)
    assert (response.status_code, app.provider_calls, _provider_errors() - errors_before) == (
        200,
        1,
        0,
    )


def test_unwritable_audit_log_returns_the_answer_and_keeps_its_spend(
    gateway: GatewayFactory, tmp_path: Path
) -> None:
    """A full or read-only audit volume loses the audit line, not the answer or its spend."""
    audit_path = tmp_path / "audit-volume"
    audit_path.mkdir()  # Appending to a directory raises a real OSError.
    app = gateway(audit_log_path=str(audit_path))
    errors_before = _provider_errors()

    response = app.post("/v1/chat/completions")

    assert (response.status_code, app.provider_calls, _provider_errors() - errors_before) == (
        200,
        1,
        0,
    )
    ledger = SpendLedger(app.ledger_path).summary()
    assert ledger.request_count == 1
    assert ledger.total_cost_usd == pytest.approx(response.json()["router"]["cost_usd"])


@pytest.mark.parametrize("path", list(STREAM_ENDINGS))
@pytest.mark.usefixtures("locked_ledger")
def test_stream_ends_normally_when_the_spend_ledger_write_fails(
    gateway: GatewayFactory, path: str
) -> None:
    """A fully delivered stream ends with its normal terminal event, not a provider error."""
    app = gateway()
    errors_before = _provider_errors()

    response = app.post(path, stream=True)

    assert response.status_code == 200, response.text
    frames = _frames(response.text)
    failures = [frame for frame in frames if frame.startswith(FAILURE_FRAMES)]
    assert (failures, _provider_errors() - errors_before) == ([], 0)
    assert frames[-1] == STREAM_ENDINGS[path]
    assert app.provider_calls == 1


def test_unexpected_accounting_error_is_never_retried_on_a_fallback(
    gateway: GatewayFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever fails after the provider answered, the request is dispatched exactly once."""

    def rejecting_sink(self: AuditLog, record: AuditRecord) -> None:
        raise RuntimeError("audit sink rejected the record")

    monkeypatch.setattr(AuditLog, "append", rejecting_sink)
    app = gateway()
    errors_before = _provider_errors()

    response = app.post("/v1/chat/completions")

    assert (response.status_code, app.provider_calls, _provider_errors() - errors_before) == (
        500,
        1,
        0,
    )
