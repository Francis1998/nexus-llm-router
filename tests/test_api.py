"""Tests for FastAPI request handling."""

from fastapi.testclient import TestClient

from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from api.main import app, get_adapter_registry, get_router, get_settings
from router.config import load_settings
from router.engine import NexusRouter


def _install_mock_router() -> None:
    get_settings.cache_clear()
    get_adapter_registry.cache_clear()
    get_router.cache_clear()
    settings = load_settings()
    registry = AdapterRegistry(
        {
            "openai": MockProviderAdapter("openai"),
            "anthropic": MockProviderAdapter("anthropic"),
            "google": MockProviderAdapter("google"),
            "moonshot": MockProviderAdapter("moonshot"),
        }
    )
    router = NexusRouter(settings, registry)
    import api.main as main

    main.get_settings = lambda: settings  # type: ignore[assignment]
    main.get_adapter_registry = lambda: registry  # type: ignore[assignment]
    main.get_router = lambda: router  # type: ignore[assignment]


def test_chat_completion_non_streaming_returns_json() -> None:
    """API should return a normal JSON completion when stream is false."""
    _install_mock_router()
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert "routed response" in body["choices"][0]["message"]["content"]
