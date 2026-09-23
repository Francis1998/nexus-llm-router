"""Tests for ResponseCacheTtlAdvisor."""

from __future__ import annotations

import pytest

from safety.response_cache_ttl import ResponseCacheTtlAdvisor


def test_ephemeral_short() -> None:
    advice = ResponseCacheTtlAdvisor().advise(request_class="ephemeral")
    assert advice.band == "short"
    assert advice.ttl_seconds == 30


def test_stable_medium() -> None:
    advice = ResponseCacheTtlAdvisor().advise(request_class="stable")
    assert advice.band == "medium"


def test_static_long() -> None:
    advice = ResponseCacheTtlAdvisor().advise(request_class="static")
    assert advice.band == "long"
    assert advice.ttl_seconds == 86_400


def test_invalid() -> None:
    with pytest.raises(ValueError, match="request_class"):
        ResponseCacheTtlAdvisor().advise(request_class="nope")
