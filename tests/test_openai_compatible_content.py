"""Tests for OpenAI-compatible adapters parsing structured message content."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.moonshot import MoonshotAdapter
from adapters.openai import OpenAIAdapter
from router.model_ids import MOONSHOT_BALANCED_MODEL, OPENAI_FRONTIER_MODEL
from router.schemas import ChatMessage


def _mock_client_returning(body: dict[str, object]) -> MagicMock:
    """Build a mocked httpx.AsyncClient that returns a fixed JSON body.

    Args:
        body: JSON body the mocked POST should return.

    Returns:
        Configured mock client usable as an async context manager.
    """

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        del args, kwargs
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = body
        return response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post
    return mock_client


_STRUCTURED_BODY: dict[str, object] = {
    "choices": [
        {
            "message": {
                "content": [
                    {"type": "text", "text": "Hello, "},
                    {"type": "text", "text": "world."},
                ]
            }
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}


@pytest.mark.asyncio()
async def test_openai_adapter_joins_structured_content_parts() -> None:
    """OpenAI content delivered as a parts list must be joined, not dropped.

    Some OpenAI-compatible gateways (LiteLLM, vLLM, OpenRouter) return
    ``message.content`` as a list of ``{"type": "text", "text": ...}`` parts.
    The adapter previously accepted only a string and silently produced an empty
    completion for the list shape. Text parts must be concatenated in order.
    """
    adapter = OpenAIAdapter(api_key="test-key", timeout_seconds=5.0)

    with patch(
        "adapters.openai.httpx.AsyncClient", return_value=_mock_client_returning(_STRUCTURED_BODY)
    ):
        result = await adapter.complete(
            OPENAI_FRONTIER_MODEL,
            [ChatMessage(role="user", content="Hi.")],
            max_tokens=32,
        )

    assert result.content == "Hello, world."


_REFUSAL_BODY: dict[str, object] = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "refusal": "I cannot help with that request.",
            }
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 8},
}


@pytest.mark.asyncio()
async def test_openai_adapter_surfaces_refusal_when_content_is_null() -> None:
    """A refusal must be surfaced, not dropped, when content is null.

    When a model declines a request, OpenAI sets ``message.content`` to ``null``
    and carries the explanation in a sibling ``refusal`` string. The adapter
    previously read only ``content`` and returned an empty completion, silently
    discarding the model's stated reason for refusing. The refusal text must be
    surfaced as the completion instead.
    """
    adapter = OpenAIAdapter(api_key="test-key", timeout_seconds=5.0)

    with patch(
        "adapters.openai.httpx.AsyncClient", return_value=_mock_client_returning(_REFUSAL_BODY)
    ):
        result = await adapter.complete(
            OPENAI_FRONTIER_MODEL,
            [ChatMessage(role="user", content="Do something disallowed.")],
            max_tokens=32,
        )

    assert result.content == "I cannot help with that request."


@pytest.mark.asyncio()
async def test_openai_adapter_uses_max_completion_tokens_for_gpt5() -> None:
    """GPT-5.x completions must send ``max_completion_tokens``, not ``max_tokens``.

    OpenAI's GPT-5 chat-completions API rejects the legacy ``max_tokens`` field
    (returning 400) and requires ``max_completion_tokens`` instead. Sending the
    wrong key silently broke every GPT-5.5 primary route. Legacy GPT-4.x / mini
    SKUs must keep ``max_tokens``.
    """
    adapter = OpenAIAdapter(api_key="test-key", timeout_seconds=5.0)
    captured: dict[str, object] = {}

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        del args
        captured["json"] = kwargs.get("json")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("adapters.openai.httpx.AsyncClient", return_value=mock_client):
        await adapter.complete(
            OPENAI_FRONTIER_MODEL,
            [ChatMessage(role="user", content="Hi.")],
            max_tokens=64,
        )

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload.get("max_completion_tokens") == 64
    assert "max_tokens" not in payload


@pytest.mark.asyncio()
async def test_openai_adapter_keeps_max_tokens_for_non_gpt5() -> None:
    """Non-GPT-5 OpenAI models must continue to send ``max_tokens``."""
    adapter = OpenAIAdapter(api_key="test-key", timeout_seconds=5.0)
    captured: dict[str, object] = {}

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        del args
        captured["json"] = kwargs.get("json")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("adapters.openai.httpx.AsyncClient", return_value=mock_client):
        await adapter.complete(
            "gpt-4.1-mini",
            [ChatMessage(role="user", content="Hi.")],
            max_tokens=32,
        )

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload.get("max_tokens") == 32
    assert "max_completion_tokens" not in payload


@pytest.mark.asyncio()
async def test_moonshot_adapter_joins_structured_content_parts() -> None:
    """Moonshot (OpenAI-compatible) must also join structured content parts."""
    adapter = MoonshotAdapter(
        api_key="test-key", base_url="https://api.moonshot.ai/v1", timeout_seconds=5.0
    )

    with patch(
        "adapters.moonshot.httpx.AsyncClient",
        return_value=_mock_client_returning(_STRUCTURED_BODY),
    ):
        result = await adapter.complete(
            MOONSHOT_BALANCED_MODEL,
            [ChatMessage(role="user", content="Hi.")],
            max_tokens=32,
        )

    assert result.content == "Hello, world."
