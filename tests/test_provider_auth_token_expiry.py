"""Unit tests for ProviderAuthTokenExpiryAdvisor."""

from __future__ import annotations

import pytest

from safety.provider_auth_token_expiry import ProviderAuthTokenExpiryAdvisor


def test_within() -> None:
    """Good metric is within."""

    advice = ProviderAuthTokenExpiryAdvisor().advise(request_id="r1", minutes_to_expiry=120.0)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = ProviderAuthTokenExpiryAdvisor().advise(request_id="r1", minutes_to_expiry=30.0)
    assert advice.band == "soft"


def test_breach() -> None:
    """Bad metric is breach."""

    advice = ProviderAuthTokenExpiryAdvisor().advise(request_id="r1", minutes_to_expiry=5.0)
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="minutes_to_expiry"):
        ProviderAuthTokenExpiryAdvisor().advise(request_id="r1", minutes_to_expiry=-0.1)
