"""Tests for EngineWarmupColdStartBandAdvisor."""

from __future__ import annotations

import pytest

from safety.engine_warmup_cold_start_band import EngineWarmupColdStartBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = EngineWarmupColdStartBandAdvisor().advise(
        request_id="r1", cold_start_ratio=0.1, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = EngineWarmupColdStartBandAdvisor().advise(
        request_id="r1", cold_start_ratio=0.3, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = EngineWarmupColdStartBandAdvisor().advise(
        request_id="r1", cold_start_ratio=0.9, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        EngineWarmupColdStartBandAdvisor().advise(request_id=" ", cold_start_ratio=0.1)


def test_bad_limits() -> None:
    """Invalid limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        EngineWarmupColdStartBandAdvisor().advise(
            request_id="r1", cold_start_ratio=0.1, soft_limit=0.5, hard_limit=0.2
        )
