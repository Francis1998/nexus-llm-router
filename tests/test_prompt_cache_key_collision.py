"""Unit tests for PromptCacheKeyCollisionAdvisor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from safety.prompt_cache_key_collision import PromptCacheKeyCollisionAdvisor


def test_within_band() -> None:
    """Band within under soft limit."""

    advice = PromptCacheKeyCollisionAdvisor().advise(request_id="r1", collision_rate=0.01)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Band soft between soft and hard."""

    advice = PromptCacheKeyCollisionAdvisor().advise(request_id="r1", collision_rate=0.06)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """Band breach above hard limit."""

    advice = PromptCacheKeyCollisionAdvisor().advise(request_id="r1", collision_rate=0.11)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Empty request_id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        PromptCacheKeyCollisionAdvisor().advise(request_id=" ", collision_rate=0.01)


def test_no_network_calls() -> None:
    """Advisor never performs HTTP calls."""

    with (
        patch("httpx.Client", MagicMock()) as client_cls,
        patch("httpx.AsyncClient", MagicMock()) as async_cls,
        patch("httpx.get", MagicMock()) as get_fn,
        patch("httpx.post", MagicMock()) as post_fn,
    ):
        PromptCacheKeyCollisionAdvisor().advise(request_id="r1", collision_rate=0.06)
        client_cls.assert_not_called()
        async_cls.assert_not_called()
        get_fn.assert_not_called()
        post_fn.assert_not_called()
