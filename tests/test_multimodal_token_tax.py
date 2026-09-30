"""Unit tests for MultimodalTokenTaxAdvisor."""

from __future__ import annotations

import pytest

from safety.multimodal_token_tax import MultimodalTokenTaxAdvisor


def test_within_band() -> None:
    """Low tax is within."""

    advice = MultimodalTokenTaxAdvisor().advise(request_id="r1", vision_tokens=100, text_tokens=200)
    assert advice.band == "within"


def test_soft_band() -> None:
    """Mid tax is soft."""

    advice = MultimodalTokenTaxAdvisor().advise(request_id="r1", vision_tokens=500, text_tokens=100)
    assert advice.band == "soft"


def test_breach_band() -> None:
    """High tax is breach."""

    advice = MultimodalTokenTaxAdvisor().advise(
        request_id="r1", vision_tokens=2000, text_tokens=100
    )
    assert advice.band == "breach"


def test_empty_request_raises() -> None:
    """Empty request id raises ValueError."""

    with pytest.raises(ValueError, match="request_id"):
        MultimodalTokenTaxAdvisor().advise(request_id=" ", vision_tokens=1, text_tokens=1)
