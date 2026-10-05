"""Unit tests for LogitBiasMagnitudeBudgetGuard."""

from __future__ import annotations

import pytest

from safety.logit_bias_magnitude_budget import LogitBiasMagnitudeBudgetGuard


def test_within() -> None:
    """Low metric is within."""

    advice = LogitBiasMagnitudeBudgetGuard().advise(request_id="r1", l1_magnitude=3.2)
    assert advice.band == "within"


def test_soft() -> None:
    """Mid metric is soft."""

    advice = LogitBiasMagnitudeBudgetGuard().advise(request_id="r1", l1_magnitude=16.0)
    assert advice.band == "soft"


def test_breach() -> None:
    """High metric is breach."""

    advice = LogitBiasMagnitudeBudgetGuard().advise(
        request_id="r1", l1_magnitude=28.799999999999997
    )
    assert advice.band == "breach"


def test_invalid() -> None:
    """Negative metric raises."""

    with pytest.raises(ValueError, match="l1_magnitude"):
        LogitBiasMagnitudeBudgetGuard().advise(request_id="r1", l1_magnitude=-0.1)
