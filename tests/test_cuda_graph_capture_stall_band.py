"""Tests for CudaGraphCaptureStallBandAdvisor."""

from __future__ import annotations

import pytest

from safety.cuda_graph_capture_stall_band import CudaGraphCaptureStallBandAdvisor


def test_within() -> None:
    """Below soft is within."""

    advice = CudaGraphCaptureStallBandAdvisor().advise(request_id="r1", stall_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Between soft and hard is soft."""

    advice = CudaGraphCaptureStallBandAdvisor().advise(request_id="r1", stall_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """At/above hard is breach."""

    advice = CudaGraphCaptureStallBandAdvisor().advise(request_id="r1", stall_ratio=0.9)
    assert advice.band == "breach"


def test_empty_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        CudaGraphCaptureStallBandAdvisor().advise(request_id=" ", stall_ratio=0.1)


def test_negative() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="stall_ratio"):
        CudaGraphCaptureStallBandAdvisor().advise(request_id="r1", stall_ratio=-0.1)
