"""Unit tests for WebGroundingInjectionGateAdvisor."""

from __future__ import annotations

import pytest

from safety.web_grounding_injection_gate import WebGroundingInjectionGateAdvisor


def test_within() -> None:
    """Band within."""

    advice = WebGroundingInjectionGateAdvisor().advise(request_id="r1", injection_score=0.125)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = WebGroundingInjectionGateAdvisor().advise(request_id="r1", injection_score=0.425)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = WebGroundingInjectionGateAdvisor().advise(request_id="r1", injection_score=1.6)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="injection_score"):
        WebGroundingInjectionGateAdvisor().advise(request_id="r1", injection_score=-0.1)
