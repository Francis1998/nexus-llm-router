"""Tests for KvBlockStealAdvisor."""

from __future__ import annotations

import pytest

from safety.kv_block_steal import KvBlockStealAdvisor


def test_within() -> None:
    """Band within."""

    advice = KvBlockStealAdvisor().advise(request_id="r1", steal_rate=0.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    mid = (0.2 + 0.5) / 2
    advice = KvBlockStealAdvisor().advise(request_id="r1", steal_rate=mid)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = KvBlockStealAdvisor().advise(request_id="r1", steal_rate=1.0000)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid input raises."""

    with pytest.raises(ValueError, match="request_id"):
        KvBlockStealAdvisor().advise(request_id=" ", steal_rate=0.0)
