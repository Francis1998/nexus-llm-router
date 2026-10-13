"""Tests for BatchPaddingWasteAdvisor."""

from __future__ import annotations

import pytest

from safety.batch_padding_waste import BatchPaddingWasteAdvisor


def test_within() -> None:
    """Band within."""

    advice = BatchPaddingWasteAdvisor().advise(request_id="r1", padding_waste_ratio=0.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.25 + 0.55) / 2
    advice = BatchPaddingWasteAdvisor().advise(request_id="r1", padding_waste_ratio=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = BatchPaddingWasteAdvisor().advise(request_id="r1", padding_waste_ratio=1.1000)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError, match="request_id"):
        BatchPaddingWasteAdvisor().advise(request_id=" ", padding_waste_ratio=0.0)
