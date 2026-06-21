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
    return value if isinstance(value, int) else default
