"""Non-reversible API key fingerprints for gateway identity.

The raw ``Authorization`` header is a credential and must never be used as an
identifier: identifiers flow into HTTP error details, the durable spend ledger
(``GET /v1/spend``), the idempotency store, cache namespaces, and logs.
``fingerprint_api_key`` derives a stable, non-reversible id instead so per-key
isolation (rate limits, budgets, idempotency and cache tenants) keeps working
without ever echoing or persisting the secret.
"""

from __future__ import annotations

import hashlib

ANONYMOUS_KEY_ID = "anonymous"
_BEARER_SCHEME = "bearer"
_FINGERPRINT_HEX_CHARS = 16


class ConflictingCredentialsError(ValueError):
    """Raised when ``x-api-key`` and ``Authorization`` carry different keys."""


def fingerprint_api_key(authorization: str | None) -> str:
    """Return a stable, non-reversible identifier for an ``Authorization`` header.

    An optional ``Bearer`` scheme (case-insensitive) is stripped so ``Bearer sk-x``
    and a bare ``sk-x`` map to the same id. Other schemes are hashed verbatim.

    Args:
        authorization: Raw ``Authorization`` header value, or ``None`` when absent.

    Returns:
        ``"anonymous"`` when the header is missing or carries no credential,
        otherwise ``"key_"`` plus the first 16 hex characters of the credential's
        SHA-256 digest.
    """
    if authorization is None:
        return ANONYMOUS_KEY_ID
    credential = authorization.strip()
    scheme, _, remainder = credential.partition(" ")
    if scheme.lower() == _BEARER_SCHEME:
        credential = remainder.strip()
    if not credential:
        return ANONYMOUS_KEY_ID
    digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()
    return f"key_{digest[:_FINGERPRINT_HEX_CHARS]}"


def fingerprint_request_credentials(authorization: str | None, x_api_key: str | None) -> str:
    """Fingerprint the API key sent as ``Authorization`` and/or ``x-api-key``.

    Anthropic SDKs and curl examples send ``x-api-key: <key>``; newer SDKs and
    OpenAI-style clients send ``Authorization: Bearer <key>``. Both forms of the
    same key map to the same fingerprint, so rate limits, budgets, and cache or
    idempotency scopes do not depend on which header a client used.

    Args:
        authorization: Raw ``Authorization`` header value, or ``None``.
        x_api_key: Raw ``x-api-key`` header value, or ``None``.

    Returns:
        The shared ``key_`` fingerprint, or ``"anonymous"`` when neither header
        carries a credential.

    Raises:
        ConflictingCredentialsError: When both headers carry different keys.
            The message never includes either key.
    """
    bearer_id = fingerprint_api_key(authorization)
    api_key_id = fingerprint_api_key(x_api_key)
    if api_key_id == ANONYMOUS_KEY_ID:
        return bearer_id
    if bearer_id not in (ANONYMOUS_KEY_ID, api_key_id):
        raise ConflictingCredentialsError(
            "The x-api-key and Authorization headers carry different API keys; "
            "send the key in one of them (or the same key in both)."
        )
    return api_key_id
