"""Tests for KvMirrorLagBandAdvisor."""

from __future__ import annotations

import pytest

from safety.kv_mirror_lag_band import KvMirrorLagBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = KvMirrorLagBandAdvisor().advise(request_id="r1", lag_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = KvMirrorLagBandAdvisor().advise(request_id="r1", lag_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = KvMirrorLagBandAdvisor().advise(request_id="r1", lag_ratio=0.9)
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        KvMirrorLagBandAdvisor().advise(request_id=" ", lag_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="lag_ratio"):
        KvMirrorLagBandAdvisor().advise(request_id="r1", lag_ratio=-0.1)
