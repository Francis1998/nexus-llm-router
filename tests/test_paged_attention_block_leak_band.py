"""Tests for PagedAttentionBlockLeakBandAdvisor."""

from __future__ import annotations

import pytest

from safety.paged_attention_block_leak_band import PagedAttentionBlockLeakBandAdvisor


def test_within() -> None:
    """Within soft budget."""

    advice = PagedAttentionBlockLeakBandAdvisor().advise(request_id="r1", leak_ratio=0.1)
    assert advice.band == "within"


def test_soft() -> None:
    """Soft band."""

    advice = PagedAttentionBlockLeakBandAdvisor().advise(request_id="r1", leak_ratio=0.3)
    assert advice.band == "soft"


def test_breach() -> None:
    """Breach band."""

    advice = PagedAttentionBlockLeakBandAdvisor().advise(request_id="r1", leak_ratio=0.9)
    assert advice.band == "breach"


def test_invalid_request() -> None:
    """Empty request_id raises."""

    with pytest.raises(ValueError, match="request_id"):
        PagedAttentionBlockLeakBandAdvisor().advise(request_id=" ", leak_ratio=0.1)


def test_invalid_metric() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="leak_ratio"):
        PagedAttentionBlockLeakBandAdvisor().advise(request_id="r1", leak_ratio=-0.1)
