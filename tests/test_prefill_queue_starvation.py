"""Tests for PrefillQueueStarvationAdvisor."""

from __future__ import annotations

import pytest

from safety.prefill_queue_starvation import PrefillQueueStarvationAdvisor


def test_within() -> None:
    """Band within."""

    advice = PrefillQueueStarvationAdvisor().advise(request_id="r1", starvation_rate=0.1000)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.2 + 0.5) / 2
    advice = PrefillQueueStarvationAdvisor().advise(request_id="r1", starvation_rate=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = PrefillQueueStarvationAdvisor().advise(request_id="r1", starvation_rate=1.0000)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError):
        PrefillQueueStarvationAdvisor().advise(request_id="r1", starvation_rate=-0.1)
