"""Tests for FlashAttnWorkspaceThrashBandAdvisor."""

from __future__ import annotations

import pytest

from safety.flash_attn_workspace_thrash_band import FlashAttnWorkspaceThrashBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = FlashAttnWorkspaceThrashBandAdvisor().advise(
        request_id="r1", thrash_ratio=0.1, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = FlashAttnWorkspaceThrashBandAdvisor().advise(
        request_id="r1", thrash_ratio=0.3, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = FlashAttnWorkspaceThrashBandAdvisor().advise(
        request_id="r1", thrash_ratio=0.6, soft_limit=0.2, hard_limit=0.5
    )
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        FlashAttnWorkspaceThrashBandAdvisor().advise(request_id=" ", thrash_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="thrash_ratio"):
        FlashAttnWorkspaceThrashBandAdvisor().advise(request_id="r1", thrash_ratio=-0.1)
