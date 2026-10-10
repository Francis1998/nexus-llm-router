"""Unit tests for PromptCacheHitRateFloorAdvisor."""

from __future__ import annotations

import pytest

from safety.prompt_cache_hit_rate_floor import PromptCacheHitRateFloorAdvisor


def test_within() -> None:
    """Band within (healthy hit rate)."""

    advice = PromptCacheHitRateFloorAdvisor().advise(request_id="r1", hit_rate=0.85)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = PromptCacheHitRateFloorAdvisor().advise(request_id="r1", hit_rate=0.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = PromptCacheHitRateFloorAdvisor().advise(request_id="r1", hit_rate=0.1)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="hit_rate"):
        PromptCacheHitRateFloorAdvisor().advise(request_id="r1", hit_rate=-0.1)
