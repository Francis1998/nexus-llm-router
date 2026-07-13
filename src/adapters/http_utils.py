"""Helpers for parsing provider HTTP responses."""

from typing import cast

import httpx

JsonObject = dict[str, object]


def json_object(response: httpx.Response) -> JsonObject:
    """Parse a JSON response body as an object.

    Args:
        response: HTTP response.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the body is not a JSON object.
    """
    payload = cast(object, response.json())
    if not isinstance(payload, dict):
        raise ValueError("provider returned non-object JSON payload")
    return cast(JsonObject, payload)


def nested_str(payload: JsonObject, path: list[str], default: str = "") -> str:
    """Read a nested string field.

    Args:
        payload: Source JSON object.
        path: Nested object path.
        default: Value returned when the path is missing.

    Returns:
        String field value.
    """
    value: object = payload
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value if isinstance(value, str) else default


def message_text(message: object) -> str:
    """Extract assistant text from an OpenAI-compatible message object.

    The base Chat Completions contract returns ``content`` as a string, but
    several OpenAI-compatible gateways (LiteLLM, vLLM, OpenRouter) return it as a
    list of ``{"type": "text", "text": ...}`` parts. Both shapes are supported.

    When the model declines a request, OpenAI sets ``content`` to ``null`` and
    carries the explanation in a sibling ``refusal`` string instead. Ignoring it
    would surface an empty completion and silently drop the model's stated reason
    for refusing, so the ``refusal`` text is returned when no content is present.

    Args:
        message: The ``choices[i].message`` object.

    Returns:
        Concatenated assistant text (or the refusal text when content is
        absent), or an empty string when neither is present.
    """
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        segments: list[str] = []
        for part in content:
            if isinstance(part, str) and part:
                segments.append(part)
            elif isinstance(part, dict):
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value:
                    segments.append(text_value)
        if segments:
            return "".join(segments)
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal:
        return refusal
    return content if isinstance(content, str) else ""


def nested_int(payload: JsonObject, path: list[str], default: int = 0) -> int:
    """Read a nested integer field.

    Args:
        payload: Source JSON object.
        path: Nested object path.
        default: Value returned when the path is missing.

    Returns:
        Integer field value.
    """
    value: object = payload
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    # Some OpenAI-compatible gateways serialize usage counts as strings
    # (for example ``"12"``); a whitespace-trimmed non-negative integer string
    # is a valid count and must not be dropped to the default.
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return default
