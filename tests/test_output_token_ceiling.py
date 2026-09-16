"""Tests for OutputTokenCeilingGuard ok/near/over advisory bands."""

from __future__ import annotations

import pytest

from safety.output_token_ceiling import OutputTokenCeilingAdvice, OutputTokenCeilingGuard


def test_ok_band_when_well_under_ceiling() -> None:
    """Low utilization yields ok with positive headroom advisory."""
    guard = OutputTokenCeilingGuard(near_ratio=0.85)
    advice = guard.advise(requested_max_tokens=512, ceiling=4096, model="gpt-5.5")
    assert isinstance(advice, OutputTokenCeilingAdvice)
    assert advice.band == "ok"
    assert advice.utilization == pytest.approx(512 / 4096)
    assert advice.headroom_tokens == 4096 - 512
    assert "ok" in advice.advisory.lower()
    assert advice.model == "gpt-5.5"


def test_near_band_at_near_ratio() -> None:
    """Utilization at/above near_ratio but under ceiling is near."""
    guard = OutputTokenCeilingGuard(near_ratio=0.85)
    advice = guard.advise(
        requested_max_tokens=3600,
        ceiling=4096,
        model="claude-sonnet-4.6",
    )
    assert advice.band == "near"
    assert advice.utilization >= 0.85
    assert advice.headroom_tokens == 496
    assert "near" in advice.advisory.lower()


def test_over_band_when_requested_exceeds_ceiling() -> None:
    """requested_max_tokens > ceiling yields over (never kills itself)."""
    guard = OutputTokenCeilingGuard()
    advice = guard.advise(
        requested_max_tokens=5000,
        ceiling=4096,
        model="gemini-3.1-pro-preview",
    )
    assert advice.band == "over"
    assert advice.headroom_tokens == 4096 - 5000
    assert "over" in advice.advisory.lower()
    assert advice.model == "gemini-3.1-pro-preview"


def test_frontier_models_share_ceiling_thresholds() -> None:
    """Same thresholds apply across GPT-5.5 / Sonnet / Gemini / Kimi."""
    guard = OutputTokenCeilingGuard(near_ratio=0.9)
    for model in (
        "gpt-5.5",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "kimi-k2",
    ):
        advice = guard.advise(requested_max_tokens=900, ceiling=1000, model=model)
        assert advice.band == "near"
        assert advice.model == model


def test_rejects_invalid_inputs() -> None:
    """Constructor and advise validate ratios and non-negative token counts."""
    with pytest.raises(ValueError, match="near_ratio"):
        OutputTokenCeilingGuard(near_ratio=0.0)
    with pytest.raises(ValueError, match="near_ratio"):
        OutputTokenCeilingGuard(near_ratio=1.0)
    guard = OutputTokenCeilingGuard()
    with pytest.raises(ValueError, match="requested_max_tokens"):
        guard.advise(requested_max_tokens=-1, ceiling=1000)
    with pytest.raises(ValueError, match="ceiling"):
        guard.advise(requested_max_tokens=10, ceiling=0)
    # Exact ceiling is still ok (not over) — over is strictly greater.
    at_ceiling = guard.advise(requested_max_tokens=1000, ceiling=1000)
    assert at_ceiling.band == "near"
    assert at_ceiling.headroom_tokens == 0
