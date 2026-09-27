"""Unit tests for QueueWaitSloAdvisor."""

from __future__ import annotations

import pytest

from safety.queue_wait_slo import QueueWaitSloAdvisor


def test_fresh_band() -> None:
    """Under 70% of SLO is fresh."""

    advice = QueueWaitSloAdvisor().advise(request_id="r1", wait_s=1.0, slo_s=5.0)
    assert advice.band == "fresh"


def test_aging_band() -> None:
    """70%+ of SLO is aging."""

    advice = QueueWaitSloAdvisor().advise(request_id="r1", wait_s=4.0, slo_s=5.0)
    assert advice.band == "aging"


def test_late_band() -> None:
    """At/over SLO is late."""

    advice = QueueWaitSloAdvisor().advise(request_id="r1", wait_s=5.0, slo_s=5.0)
    assert advice.band == "late"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        QueueWaitSloAdvisor().advise(request_id=" ", wait_s=0.0, slo_s=1.0)
