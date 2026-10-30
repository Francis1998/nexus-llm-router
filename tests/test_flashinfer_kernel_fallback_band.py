"""Tests for FlashinferKernelFallbackBandAdvisor."""

from __future__ import annotations

import pytest

from safety.flashinfer_kernel_fallback_band import FlashinferKernelFallbackBandAdvisor


def test_within() -> None:
    """Within soft budget."""

    advice = FlashinferKernelFallbackBandAdvisor().advise(request_id="r1", fallback_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Soft band."""

    advice = FlashinferKernelFallbackBandAdvisor().advise(request_id="r1", fallback_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """Breach band."""

    advice = FlashinferKernelFallbackBandAdvisor().advise(request_id="r1", fallback_ratio=0.9)
    assert advice.band == "breach"


def test_invalid_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        FlashinferKernelFallbackBandAdvisor().advise(request_id=" ", fallback_ratio=0.1)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="fallback_ratio"):
        FlashinferKernelFallbackBandAdvisor().advise(request_id="r1", fallback_ratio=-0.1)
