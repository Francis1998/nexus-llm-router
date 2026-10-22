"""Unit tests for GuidedDecodeFsmBlowupAdvisor."""

from __future__ import annotations

import pytest

from safety.guided_decode_fsm_blowup import GuidedDecodeFsmBlowupAdvisor


def test_within() -> None:
    """Below soft -> within."""

    advice = GuidedDecodeFsmBlowupAdvisor().advise(request_id="r1", fsm_blowup_ratio=0.05)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard -> soft."""

    advice = GuidedDecodeFsmBlowupAdvisor().advise(request_id="r1", fsm_blowup_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard -> breach."""

    advice = GuidedDecodeFsmBlowupAdvisor().advise(request_id="r1", fsm_blowup_ratio=0.8)
    assert advice.band == "breach"


def test_invalid_request_id() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        GuidedDecodeFsmBlowupAdvisor().advise(request_id=" ", fsm_blowup_ratio=0.1)


def test_invalid_limits() -> None:
    """Bad limits raise."""

    with pytest.raises(ValueError, match="hard_limit"):
        GuidedDecodeFsmBlowupAdvisor().advise(
            request_id="r1",
            fsm_blowup_ratio=0.1,
            soft_limit=0.5,
            hard_limit=0.2,
        )
