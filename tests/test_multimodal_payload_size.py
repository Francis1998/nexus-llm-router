"""Unit tests for MultimodalPayloadSizeGate."""

from __future__ import annotations

import pytest

from safety.multimodal_payload_size import MultimodalPayloadSizeGate


def test_allow_under_cap() -> None:
    """Payload under cap is allow."""

    decision = MultimodalPayloadSizeGate().check(payload_bytes=1_000_000, max_bytes=5_000_000)
    assert decision.allowed is True
    assert decision.band == "allow"


def test_deny_over_cap() -> None:
    """Payload over cap is deny."""

    decision = MultimodalPayloadSizeGate().check(payload_bytes=6_000_000, max_bytes=5_000_000)
    assert decision.allowed is False
    assert decision.band == "deny"


def test_allow_at_exact_cap() -> None:
    """Exact cap is allow."""

    decision = MultimodalPayloadSizeGate().check(payload_bytes=100, max_bytes=100)
    assert decision.allowed is True


def test_invalid_max_raises() -> None:
    """Non-positive max_bytes raises ValueError."""

    with pytest.raises(ValueError, match="max_bytes"):
        MultimodalPayloadSizeGate().check(payload_bytes=0, max_bytes=0)
