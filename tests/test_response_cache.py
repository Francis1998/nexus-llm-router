"""Tests for exact-match response caching."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.main import app, get_response_cache, get_router, get_settings
from cache.response_cache import ResponseCache
from router.schemas import (
    ChatMessage,
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
