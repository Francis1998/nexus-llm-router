"""Tests for Anthropic adapter system-message handling."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.anthropic import AnthropicAdapter, _build_anthropic_payload
from router.schemas import ChatMessage


def test_build_payload_extracts_system_to_top_level() -> None:
    """System messages must map to Anthropic's top-level system field."""
    payload = _build_anthropic_payload(
        "claude-3-5-sonnet",
        [
            ChatMessage(role="system", content="You are a careful medical assistant."),
            ChatMessage(role="user", content="Summarize this case."),
        ],
        max_tokens=128,
    )
    assert payload["system"] == "You are a careful medical assistant."
    assert payload["messages"] == [{"role": "user", "content": "Summarize this case."}]


def test_build_payload_joins_multiple_system_messages() -> None:
    """Multiple system messages are joined for Anthropic's single system field."""
    payload = _build_anthropic_payload(
        "claude-3-5-sonnet",
        [
            ChatMessage(role="system", content="Policy A."),
            ChatMessage(role="system", content="Policy B."),
            ChatMessage(role="user", content="Question."),
        ],
        max_tokens=64,
    )
    assert payload["system"] == "Policy A.\n\nPolicy B."
    assert len(payload["messages"]) == 1


@pytest.mark.asyncio()
async def test_complete_posts_system_field_to_anthropic() -> None:
    """complete() must not drop system instructions from the HTTP payload."""
    adapter = AnthropicAdapter(api_key="test-key", timeout_seconds=5.0)
    captured: dict[str, Any] = {}

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        del args
        captured.update(kwargs)
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "content": [{"text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        return response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("adapters.anthropic.httpx.AsyncClient", return_value=mock_client):
        await adapter.complete(
            "claude-3-5-sonnet",
            [
                ChatMessage(role="system", content="Safety policy."),
                ChatMessage(role="user", content="Hello."),
            ],
            max_tokens=32,
        )

    payload = captured["json"]
    assert payload["system"] == "Safety policy."
    assert all(message["role"] != "system" for message in payload["messages"])
