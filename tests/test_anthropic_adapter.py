"""Tests for Anthropic adapter system-message handling."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.anthropic import (
    AnthropicAdapter,
    _build_anthropic_payload,
    _extract_anthropic_text,
)
from router.model_ids import ANTHROPIC_SAFETY_MODEL
from router.schemas import ChatMessage


def test_extract_text_joins_blocks_after_leading_thinking_block() -> None:
    """Text must survive a leading non-text block and span multiple text blocks.

    Anthropic returns ``content`` as ordered typed blocks. Reading only
    ``content[0].text`` yielded an empty answer when a ``thinking`` (or other
    non-text) block came first, and dropped later segments when the answer was
    split across several ``text`` blocks. Every ``text`` block must be joined in
    order.
    """
    blocks = [
        {"type": "thinking", "thinking": "internal reasoning"},
        {"type": "text", "text": "Hello "},
        {"type": "tool_use", "id": "t1", "input": {}},
        {"type": "text", "text": "world"},
    ]
    assert _extract_anthropic_text(blocks) == "Hello world"


def test_build_payload_extracts_system_to_top_level() -> None:
    """System messages must map to Anthropic's top-level system field."""
    payload = _build_anthropic_payload(
        ANTHROPIC_SAFETY_MODEL,
        [
            ChatMessage(role="system", content="You are a careful medical assistant."),
            ChatMessage(role="user", content="Summarize this case."),
        ],
        max_tokens=128,
    )
    assert payload["system"] == "You are a careful medical assistant."
    assert payload["messages"] == [{"role": "user", "content": "Summarize this case."}]


def test_build_payload_maps_tool_role_to_user() -> None:
    """Tool turns must map to Anthropic's ``user`` role, not a raw ``tool`` role.

    The Anthropic Messages API rejects any turn whose role is not ``user`` or
    ``assistant`` (tool results are carried on a ``user`` turn). Forwarding a
    ``{"role": "tool"}`` entry verbatim caused the whole request to fail with a
    400, so tool turns must be remapped to ``user``.
    """
    payload = _build_anthropic_payload(
        ANTHROPIC_SAFETY_MODEL,
        [
            ChatMessage(role="user", content="What is the weather?"),
            ChatMessage(role="assistant", content="Calling a tool."),
            ChatMessage(role="tool", content='{"temp_c": 21}'),
        ],
        max_tokens=128,
    )

    roles = [message["role"] for message in payload["messages"]]
    assert "tool" not in roles
    assert roles == ["user", "assistant", "user"]
    assert payload["messages"][-1]["content"] == '{"temp_c": 21}'


def test_build_payload_joins_multiple_system_messages() -> None:
    """Multiple system messages are joined for Anthropic's single system field."""
    payload = _build_anthropic_payload(
        ANTHROPIC_SAFETY_MODEL,
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
            ANTHROPIC_SAFETY_MODEL,
            [
                ChatMessage(role="system", content="Safety policy."),
                ChatMessage(role="user", content="Hello."),
            ],
            max_tokens=32,
        )

    payload = captured["json"]
    assert payload["system"] == "Safety policy."
    assert all(message["role"] != "system" for message in payload["messages"])
