"""Unit tests for PromptCacheKeyCollisionAdvisor."""

from __future__ import annotations

import pytest

from safety.prompt_cache_key_collision import PromptCacheKeyCollisionAdvisor


def test_within() -> None:
    """Low metric is within."""

    advice = PromptCacheKeyCollisionAdvisor().advise(request_id="r1", collision_rate=0.01)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = PromptCacheKeyCollisionAdvisor().advise(request_id="r1", collision_rate=0.06)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = PromptCacheKeyCollisionAdvisor().advise(request_id="r1", collision_rate=0.11)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="collision_rate"):
        PromptCacheKeyCollisionAdvisor().advise(request_id="r1", collision_rate=-0.1)
