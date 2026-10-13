"""Tests for PromptInjectionCanaryFloorAdvisor."""

from __future__ import annotations

import pytest

from safety.prompt_injection_canary_floor import PromptInjectionCanaryFloorAdvisor


def test_within() -> None:
    """Band within."""

    advice = PromptInjectionCanaryFloorAdvisor().advise(request_id="r1", canary_hit_rate=0.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.02 + 0.1) / 2
    advice = PromptInjectionCanaryFloorAdvisor().advise(request_id="r1", canary_hit_rate=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = PromptInjectionCanaryFloorAdvisor().advise(request_id="r1", canary_hit_rate=0.2000)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError, match="request_id"):
        PromptInjectionCanaryFloorAdvisor().advise(request_id=" ", canary_hit_rate=0.0)
