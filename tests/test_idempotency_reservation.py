"""Idempotency-Key reservations: concurrent duplicates never dispatch twice."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

import api.main as main
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from cache.response_cache import ResponseCache
from router.config import RouterSettings
from router.engine import NexusRouter
from router.schemas import ChatMessage, ProviderResponse
from safety.gateway_guards import GatewayGuardService
from safety.idempotency import IdempotencyStore
from safety.tenant_rate_limiter import TenantRateLimiter

PROVIDERS = ("openai", "anthropic", "google", "moonshot")


@dataclass
class DispatchLog:
    """Counts provider calls across every mock adapter."""

    calls: list[str] = field(default_factory=list)


class SlowAdapter(MockProviderAdapter):
    """Mock provider that takes a moment to answer, so requests overlap."""

    def __init__(self, provider_name: str, log: DispatchLog, *, fail: bool = False) -> None:
        super().__init__(provider_name, fail=fail)
        self._log = log

    async def complete(
        self, model: str, messages: list[ChatMessage], max_tokens: int
    ) -> ProviderResponse:
        self._log.calls.append(model)
        await asyncio.sleep(0.2)
        return await super().complete(model, messages, max_tokens)


@dataclass
class Gateway:
    """ASGI client plus the dispatch log of its router."""

    log: DispatchLog

    async def post(self, path: str, body: dict[str, Any], key: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://nexus") as client:
            return await client.post(path, json=body, headers={"Idempotency-Key": key})


GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GatewayFactory:
    """Build an app whose providers are slow mock adapters."""

    def _build(*, fail: bool = False) -> Gateway:
        log = DispatchLog()
        settings = RouterSettings(
            audit_log_path=str(tmp_path / "audit-log.jsonl"),
            spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
            idempotency_path=str(tmp_path / "idempotency.sqlite3"),
            response_cache_enabled=False,
        )
        registry = AdapterRegistry({name: SlowAdapter(name, log, fail=fail) for name in PROVIDERS})
        router = NexusRouter(settings, registry)
        guards = GatewayGuardService(
            idempotency_store=IdempotencyStore(settings.idempotency_path),
            tenant_rate_limiter=TenantRateLimiter(capacity=60, refill_per_second=1.0),
        )
        cache = ResponseCache(ttl_seconds=60.0)
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
        monkeypatch.setattr(main, "get_router", lambda: router)
        monkeypatch.setattr(main, "get_response_cache", lambda: cache)
        monkeypatch.setattr(main, "get_gateway_guards", lambda: guards)
        return Gateway(log=log)

    return _build


CHAT = ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Draft a note."}]})
RESPONSES = ("/v1/responses", {"input": "Draft a note."})


@pytest.mark.parametrize(("path", "body"), [CHAT, RESPONSES], ids=["chat", "responses"])
async def test_concurrent_duplicate_is_rejected_not_dispatched(
    gateway: GatewayFactory, path: str, body: dict[str, Any]
) -> None:
    """A retry that overlaps the original gets 409 instead of a second provider call."""
    client = gateway()

    first, second = await asyncio.gather(
        client.post(path, body, "order-42"), client.post(path, body, "order-42")
    )

    assert sorted([first.status_code, second.status_code]) == [200, 409]
    assert len(client.log.calls) == 1
    conflict = first if first.status_code == 409 else second
    assert "Idempotency-Key" in conflict.text


@pytest.mark.parametrize(("path", "body"), [CHAT, RESPONSES], ids=["chat", "responses"])
async def test_retry_after_completion_replays(
    gateway: GatewayFactory, path: str, body: dict[str, Any]
) -> None:
    """Once the original finished, the same key replays the stored response."""
    client = gateway()

    original = await client.post(path, body, "order-43")
    retry = await client.post(path, body, "order-43")

    assert original.status_code == retry.status_code == 200
    assert retry.json() == original.json()
    assert len(client.log.calls) == 1


async def test_failed_request_releases_the_key(gateway: GatewayFactory) -> None:
    """A routing failure frees the key so the client can retry with it."""
    client = gateway(fail=True)
    path, body = CHAT

    failed = await client.post(path, body, "order-44")
    retried = await client.post(path, body, "order-44")

    assert failed.status_code == 503
    assert retried.status_code == 503
