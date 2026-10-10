"""Unit tests for ModalityMixBudgetAdvisor."""

from __future__ import annotations

import pytest

from safety.modality_mix_budget import ModalityMixBudgetAdvisor


def test_within() -> None:
    """Band within."""

    advice = ModalityMixBudgetAdvisor().advise(request_id="r1", mix_ratio=0.15)
    assert advice.band == "within"


def test_soft() -> None:
    """Band soft."""

    advice = ModalityMixBudgetAdvisor().advise(request_id="r1", mix_ratio=0.5)
    assert advice.band == "soft"


def test_breach() -> None:
    """Band breach."""

    advice = ModalityMixBudgetAdvisor().advise(request_id="r1", mix_ratio=1.7)
    assert advice.band == "breach"


def test_invalid_raises() -> None:
    """Invalid metric raises ValueError."""

    with pytest.raises(ValueError, match="mix_ratio"):
        ModalityMixBudgetAdvisor().advise(request_id="r1", mix_ratio=-0.1)
