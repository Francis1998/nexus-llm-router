"""Unit tests for GuidedJsonSchemaThrashBandAdvisor."""

from __future__ import annotations

import pytest

from safety.guided_json_schema_thrash_band import (
    GuidedJsonSchemaThrashBandAdvisor,
)


def test_within() -> None:
    """Below soft -> within."""

    advice = GuidedJsonSchemaThrashBandAdvisor().advise(request_id="r1", thrash_ratio=0.1000)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard -> soft."""

    advice = GuidedJsonSchemaThrashBandAdvisor().advise(request_id="r1", thrash_ratio=0.3500)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard -> breach."""

    advice = GuidedJsonSchemaThrashBandAdvisor().advise(request_id="r1", thrash_ratio=0.7000)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        GuidedJsonSchemaThrashBandAdvisor().advise(request_id=" ", thrash_ratio=0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        GuidedJsonSchemaThrashBandAdvisor().advise(
            request_id="r1",
            thrash_ratio=0.1,
            soft_limit=0.5,
            hard_limit=0.2,
        )
