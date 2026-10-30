"""Tests for DisaggKvTransferStallBandAdvisor."""

from __future__ import annotations

import pytest

from safety.disagg_kv_transfer_stall_band import DisaggKvTransferStallBandAdvisor


def test_within() -> None:
    """Within soft budget."""

    advice = DisaggKvTransferStallBandAdvisor().advise(request_id="r1", stall_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Soft band."""

    advice = DisaggKvTransferStallBandAdvisor().advise(request_id="r1", stall_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """Breach band."""

    advice = DisaggKvTransferStallBandAdvisor().advise(request_id="r1", stall_ratio=0.9)
    assert advice.band == "breach"


def test_invalid_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        DisaggKvTransferStallBandAdvisor().advise(request_id=" ", stall_ratio=0.1)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="stall_ratio"):
        DisaggKvTransferStallBandAdvisor().advise(request_id="r1", stall_ratio=-0.1)
