"""Tests for OpenAI-compatible SSE streaming on /v1/chat/completions."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adapters.base import BaseProviderAdapter, ProviderError
from adapters.mock import MockProviderAdapter
from adapters.registry import AdapterRegistry
from api.main import app, get_adapter_registry, get_router, get_settings
from router.config import RouterSettings, load_settings
from router.engine import NexusRouter, RoutingFailedError
from router.model_ids import GEMINI_FLASH_MODEL
from router.schemas import ChatMessage, RouterRequest


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


def test_stream_fails_instead_of_splicing_a_fallback_after_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider dropping mid-stream must not get a fallback answer appended."""

    class _PartialStreamAdapter(MockProviderAdapter):
        async def stream(
            self, model: str, messages: list[ChatMessage], max_tokens: int
        ) -> AsyncGenerator[str, None]:
            yield f"[{model}] partial "
            raise ProviderError("connection reset mid-stream")

    import api.main as main

    settings = RouterSettings(
        audit_log_path=str(tmp_path / "audit-log.jsonl"),
        spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
    )
    adapters: dict[str, BaseProviderAdapter] = {
        name: MockProviderAdapter(name) for name in ("openai", "anthropic", "moonshot")
    }
    adapters["google"] = _PartialStreamAdapter("google")
    router = NexusRouter(settings, AdapterRegistry(adapters))
    monkeypatch.setattr(main, "get_router", lambda: router)

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    frames = _parse_sse_frames(response.text)
    content = "".join(
        str(frame["choices"][0]["delta"].get("content", ""))  # type: ignore[index]
        for frame in frames
        if isinstance(frame, dict) and "choices" in frame
    )
    assert content == f"[{GEMINI_FLASH_MODEL}] partial "
    assert frames[-2] == {
        "error": {
            "message": "provider google failed after streaming output began: "
            "connection reset mid-stream",
            "type": "routing_failed",
        }
    }
    assert frames[-1] == "[DONE]"


async def test_stream_post_output_failure_after_responded_state_raises_routing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure while finalizing a streamed answer surfaces as RoutingFailedError."""
    settings = RouterSettings(
        audit_log_path=str(tmp_path / "audit-log.jsonl"),
        spend_ledger_path=str(tmp_path / "spend-ledger.sqlite3"),
    )
    registry = AdapterRegistry(
        {name: MockProviderAdapter(name) for name in ("openai", "anthropic", "google", "moonshot")}
    )
    router = NexusRouter(settings, registry)

    def _broken_audit_log(record: object) -> None:
        raise OSError("audit log volume is read-only")

    monkeypatch.setattr(router._audit_log, "append", _broken_audit_log)
    request = RouterRequest(request_id="stream-audit", messages=[ChatMessage(content="hi")])

    chunks: list[str] = []
    with pytest.raises(RoutingFailedError, match="after streaming output began"):
        async for _, chunk in router.complete_stream(request):
            chunks.append(chunk)
    assert "".join(chunks) == f"[{GEMINI_FLASH_MODEL}] routed response"
