"""Tests for OpenAI-compatible SSE streaming on /v1/chat/completions."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from api.main import app, get_adapter_registry, get_router, get_settings
from router.config import load_settings
from router.engine import NexusRouter


def _install_mock_router() -> NexusRouter:
    """Wire mock providers into the FastAPI app singletons."""
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

    app.dependency_overrides.clear()
    # The app uses lru_cache getters rather than Depends, so patch callables.
    import api.main as main

    main.get_settings = lambda: settings  # type: ignore[assignment]
    main.get_adapter_registry = lambda: registry  # type: ignore[assignment]
    main.get_router = lambda: router  # type: ignore[assignment]
    return router


def _parse_sse_frames(body: str) -> list[dict[str, object] | str]:
    frames: list[dict[str, object] | str] = []
    for block in body.split("\n\n"):
        line = block.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            frames.append("[DONE]")
        else:
            frames.append(json.loads(payload))
    return frames


def test_stream_true_returns_sse_chunks_and_done() -> None:
    """stream=true should return text/event-stream frames ending with [DONE]."""
    _install_mock_router()
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello stream"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse_frames(response.text)
    assert frames, "expected SSE data frames"
    assert frames[-1] == "[DONE]"
    content_chunks = [
        frame["choices"][0]["delta"].get("content", "")  # type: ignore[index]
        for frame in frames[:-1]
        if isinstance(frame, dict) and "choices" in frame
    ]
    assembled = "".join(str(chunk) for chunk in content_chunks if chunk)
    assert "routed response" in assembled
    assert any(
        isinstance(frame, dict) and frame["choices"][0].get("finish_reason") == "stop"  # type: ignore[index]
        for frame in frames[:-1]
    )


def test_stream_false_still_returns_json_completion() -> None:
    """Non-streaming requests remain JSON chat.completion responses."""
    _install_mock_router()
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello json"}], "stream": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert "routed response" in body["choices"][0]["message"]["content"]
