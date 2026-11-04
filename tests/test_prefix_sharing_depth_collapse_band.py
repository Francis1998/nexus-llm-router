"""Tests for PrefixSharingDepthCollapseBandAdvisor."""

from __future__ import annotations

import pytest

from safety.prefix_sharing_depth_collapse_band import PrefixSharingDepthCollapseBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = PrefixSharingDepthCollapseBandAdvisor().advise(
        request_id="r1", collapse_ratio=0.1, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = PrefixSharingDepthCollapseBandAdvisor().advise(
        request_id="r1", collapse_ratio=0.3, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = PrefixSharingDepthCollapseBandAdvisor().advise(
        request_id="r1", collapse_ratio=0.9, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        PrefixSharingDepthCollapseBandAdvisor().advise(request_id=" ", collapse_ratio=0.1)


def test_bad_limits() -> None:
    """Invalid limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        PrefixSharingDepthCollapseBandAdvisor().advise(
            request_id="r1", collapse_ratio=0.1, soft_limit=0.5, hard_limit=0.2
        )
