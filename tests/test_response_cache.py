"""Tests for exact-match response caching."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from api.main import app, get_response_cache, get_router, get_settings
from cache.response_cache import ResponseCache
from router.config import RouterSettings
from router.engine import NexusRouter
from router.schemas import (
    ChatMessage,
    RouterRequest,
    RouterResponse,
    RoutingStrategyName,
)


def _sample_response(*, content: str = "cached-answer") -> RouterResponse:
    return RouterResponse(
        content=content,
        model_used="gpt-5.5",
        routing_strategy=RoutingStrategyName.RULE_BASED,
        latency_ms=12.0,
        input_tokens=4,
        output_tokens=8,
        cost_usd=0.001,
        rationale="unit-test",
        request_id="req-1",
    )


def test_cache_hit_returns_value_and_miss_stores() -> None:
    """A miss stores the value; the next get is a hit."""
    cache = ResponseCache(ttl_seconds=60.0)
    key = ResponseCache.make_key(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
        tenant="acme",
    )
    assert cache.get(key) is None
    assert cache.misses == 1
    cache.set(key, _sample_response())
    hit = cache.get(key)
    assert hit is not None
    assert hit.content == "cached-answer"
    assert cache.hits == 1


def test_cache_ttl_expiry_forces_miss() -> None:
    """Expired entries are treated as misses."""
    cache = ResponseCache(ttl_seconds=0.05)
    key = ResponseCache.make_key(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "ttl"}],
        temperature=1.0,
        tenant="t1",
    )
    cache.set(key, _sample_response(content="stale"))
    time.sleep(0.06)
    assert cache.get(key) is None
    assert cache.misses == 1


def test_cache_tenant_isolation() -> None:
    """Identical prompts under different tenants do not share entries."""
    cache = ResponseCache(ttl_seconds=60.0)
    messages = [{"role": "user", "content": "shared prompt"}]
    key_a = ResponseCache.make_key("gemini-3.5-flash", messages, 0.0, tenant="tenant-a")
    key_b = ResponseCache.make_key("gemini-3.5-flash", messages, 0.0, tenant="tenant-b")
    assert key_a != key_b
    cache.set(key_a, _sample_response(content="a-only"))
    assert cache.get(key_a) is not None
    assert cache.get(key_b) is None


def test_api_cache_hit_skips_provider_dispatch() -> None:
    """Second identical API request should be served from cache without routing."""
    import api.main as main

    # Restore getters in case earlier API tests rebound module attributes.
    main.get_settings = get_settings
    main.get_router = get_router
    main.get_response_cache = get_response_cache
    get_settings.cache_clear()
    get_router.cache_clear()
    get_response_cache.cache_clear()
    cache = get_response_cache()
    cache.clear()

    mock_complete = AsyncMock(return_value=_sample_response(content="from-provider"))
    with patch.object(get_router(), "complete", mock_complete):
        client = TestClient(app)
        body = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "cache me"}],
            "temperature": 0.0,
            "user": "tenant-cache",
        }
        first = client.post("/v1/chat/completions", json=body)
        second = client.post("/v1/chat/completions", json=body)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["choices"][0]["message"]["content"] == "from-provider"
    assert second.json()["choices"][0]["message"]["content"] == "from-provider"
    assert mock_complete.await_count == 1


def test_make_key_includes_temperature_and_message_objects() -> None:
    """Keys differ by temperature and accept ChatMessage objects."""
    messages = [ChatMessage(role="user", content="hi")]
    warm = ResponseCache.make_key("kimi-k2", messages, 0.2, tenant=None)
    cold = ResponseCache.make_key("kimi-k2", messages, 0.8, tenant=None)
    assert warm != cold


def _install_cached_mock_gateway(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[TestClient, list[RouterRequest]]:
    """Wire mock providers plus a fresh enabled cache; record dispatched requests."""
    import api.main as main

    settings = RouterSettings(
        audit_log_path=str(tmp_path / "audit-log.jsonl"),
        spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
        idempotency_path=str(tmp_path / "idempotency.sqlite3"),
    )
    registry = AdapterRegistry(
        {name: MockProviderAdapter(name) for name in ("openai", "anthropic", "google", "moonshot")}
    )
    router = NexusRouter(settings, registry)
    dispatched: list[RouterRequest] = []
    original_complete = router.complete

    async def _spy(request: RouterRequest) -> RouterResponse:
        dispatched.append(request)
        return await original_complete(request)

    monkeypatch.setattr(router, "complete", _spy)
    cache = ResponseCache(ttl_seconds=60.0)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_adapter_registry", lambda: registry)
    monkeypatch.setattr(main, "get_router", lambda: router)
    monkeypatch.setattr(main, "get_response_cache", lambda: cache)
    return TestClient(main.app), dispatched


def test_api_cache_does_not_replay_truncated_answer_for_larger_max_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A max_tokens=1 completion must not be served for a max_tokens=512 request."""
    client, dispatched = _install_cached_mock_gateway(monkeypatch, tmp_path)
    body = {"messages": [{"role": "user", "content": "explain caching"}], "user": "t-max"}

    short = client.post("/v1/chat/completions", json={**body, "max_tokens": 1})
    long = client.post("/v1/chat/completions", json={**body, "max_tokens": 512})
    long_again = client.post("/v1/chat/completions", json={**body, "max_tokens": 512})

    assert short.json()["usage"]["completion_tokens"] == 1
    assert long.json()["usage"]["completion_tokens"] == 16
    assert [request.max_tokens for request in dispatched] == [1, 512]
    assert long_again.json()["router"] == long.json()["router"]


def test_api_cache_is_scoped_to_the_routing_strategy_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A response routed by one X-Router-Strategy must not be replayed for another."""
    client, dispatched = _install_cached_mock_gateway(monkeypatch, tmp_path)
    body = {"messages": [{"role": "user", "content": "route me"}], "user": "t-strategy"}

    cheap = client.post(
        "/v1/chat/completions", json=body, headers={"X-Router-Strategy": "cost-optimal"}
    )
    ruled = client.post(
        "/v1/chat/completions", json=body, headers={"X-Router-Strategy": "rule-based"}
    )
    default = client.post("/v1/chat/completions", json=body)

    assert cheap.json()["router"]["routing_strategy"] == "cost-optimal"
    assert ruled.json()["router"]["routing_strategy"] == "rule-based"
    # No header resolves to the default strategy (rule-based), so it is a cache hit.
    assert default.json()["router"] == ruled.json()["router"]
    assert [request.strategy for request in dispatched] == [
        RoutingStrategyName.COST_OPTIMAL,
        RoutingStrategyName.RULE_BASED,
    ]


def test_make_key_includes_max_tokens_and_strategy_with_compatible_defaults() -> None:
    """New keyword args change the key; omitting them keeps the old call shape working."""
    messages = [{"role": "user", "content": "same prompt"}]
    base = ResponseCache.make_key("gpt-5.5", messages, 0.0, "acme")

    assert base == ResponseCache.make_key("gpt-5.5", messages, 0.0, tenant="acme")
    assert ResponseCache.make_key("gpt-5.5", messages, 0.0, "acme", max_tokens=1) != (
        ResponseCache.make_key("gpt-5.5", messages, 0.0, "acme", max_tokens=512)
    )
    assert ResponseCache.make_key("gpt-5.5", messages, 0.0, "acme", strategy="cost-optimal") != (
        ResponseCache.make_key("gpt-5.5", messages, 0.0, "acme", strategy="rule-based")
    )
