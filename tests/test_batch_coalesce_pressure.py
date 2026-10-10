"""Unit tests for BatchCoalescePressureAdvisor."""

from __future__ import annotations

import pytest

from safety.batch_coalesce_pressure import BatchCoalescePressureAdvisor


def test_within() -> None:
    """Band within."""

    advice = BatchCoalescePressureAdvisor().advise(request_id="r1", coalesce_pressure=0.15)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = BatchCoalescePressureAdvisor().advise(request_id="r1", coalesce_pressure=0.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = BatchCoalescePressureAdvisor().advise(request_id="r1", coalesce_pressure=1.7)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="coalesce_pressure"):
        BatchCoalescePressureAdvisor().advise(request_id="r1", coalesce_pressure=-0.1)
