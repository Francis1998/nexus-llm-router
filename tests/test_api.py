"""Tests for FastAPI request handling."""

from fastapi.testclient import TestClient

from api.main import app


def test_chat_completion_rejects_streaming_until_sse_is_supported() -> None:
    """API should not silently treat streaming requests as non-streaming."""
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 400
    assert (
        response.json()["detail"] == "streaming responses are not supported by the API endpoint yet"
    )
