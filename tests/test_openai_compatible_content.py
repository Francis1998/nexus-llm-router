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
