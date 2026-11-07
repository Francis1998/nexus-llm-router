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
