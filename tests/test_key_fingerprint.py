"""Unit tests for non-reversible API key fingerprints."""

from __future__ import annotations

import hashlib

import pytest

from safety.key_fingerprint import ANONYMOUS_KEY_ID, fingerprint_api_key


def test_fingerprint_is_stable_prefix_insensitive_and_hides_the_secret() -> None:
    """Same credential → same id regardless of the optional Bearer scheme."""
    api_key = "sk-live-unit-0123456789abcdef"
    expected = "key_" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]

    variants = [api_key, f"Bearer {api_key}", f"bearer {api_key}", f"  BEARER   {api_key}  "]

    assert {fingerprint_api_key(value) for value in variants} == {expected}
    assert api_key not in expected
    assert fingerprint_api_key("Bearer sk-live-other") != expected
    assert fingerprint_api_key("Basic dXNlcjpwYXNz").startswith("key_")


@pytest.mark.parametrize("header", [None, "", "   ", "Bearer", "Bearer   ", "bearer"])
def test_fingerprint_missing_or_blank_credential_is_anonymous(header: str | None) -> None:
    """Missing headers and empty bearer tokens share the anonymous identity."""
    assert fingerprint_api_key(header) == ANONYMOUS_KEY_ID
