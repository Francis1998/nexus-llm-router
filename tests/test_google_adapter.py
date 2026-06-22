"""Tests for Google Gemini adapter system-message handling."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.google import GoogleGeminiAdapter
from router.model_ids import GEMINI_FLASH_MODEL
from router.schemas import ChatMessage


@pytest.mark.asyncio()
async def test_complete_posts_system_instruction_to_gemini() -> None:
    """complete() must map system messages to Gemini's systemInstruction field."""
    adapter = GoogleGeminiAdapter(api_key="test-key", timeout_seconds=5.0)
    captured: dict[str, Any] = {}

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        del args
        captured.update(kwargs)
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        return response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("adapters.google.httpx.AsyncClient", return_value=mock_client):
        await adapter.complete(
            GEMINI_FLASH_MODEL,
            [
                ChatMessage(role="system", content="Safety policy."),
                ChatMessage(role="user", content="Hello."),
            ],
            max_tokens=32,
        )

    payload = captured["json"]
    assert payload["systemInstruction"] == {"parts": [{"text": "Safety policy."}]}
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "Hello."}]}]
