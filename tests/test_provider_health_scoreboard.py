"""Tests for ProviderHealthScoreboard rolling health bands."""

from __future__ import annotations

import pytest

from safety.provider_health import ProviderHealthScoreAdvice, ProviderHealthScoreboard


def test_healthy_degraded_and_unhealthy_bands() -> None:
    """Success rate maps to healthy / degraded / unhealthy advisory bands."""
    board = ProviderHealthScoreboard(
        window_size=20,
        healthy_min_success_rate=0.90,
        degraded_min_success_rate=0.70,
        min_samples=5,
    )
    for _ in range(10):
        board.record_success("openai")
    healthy = board.snapshot("openai")
    assert isinstance(healthy, ProviderHealthScoreAdvice)
    assert healthy.band == "healthy"
    assert healthy.health_score == pytest.approx(1.0)

    for _ in range(3):
        board.record_failure("anthropic")
    for _ in range(7):
        board.record_success("anthropic")
    degraded = board.snapshot("anthropic")
    assert degraded is not None
    assert degraded.band == "degraded"
    assert degraded.success_rate == pytest.approx(0.7)

    for _ in range(8):
        board.record_failure("gemini")
    for _ in range(2):
        board.record_success("gemini")
    unhealthy = board.snapshot("gemini")
    assert unhealthy is not None
    assert unhealthy.band == "unhealthy"
    assert "unhealthy" in unhealthy.advisory


def test_cold_start_stays_healthy_until_min_samples() -> None:
    """Fewer than min_samples keeps the band healthy even with failures."""
    board = ProviderHealthScoreboard(min_samples=5, degraded_min_success_rate=0.5)
    for _ in range(4):
        board.record_failure("moonshot")
    snap = board.snapshot("moonshot")
    assert snap is not None
    assert snap.band == "healthy"
    assert snap.samples == 4
    board.record_failure("moonshot")
    snap = board.snapshot("moonshot")
    assert snap is not None
    assert snap.band == "unhealthy"


def test_window_evicts_oldest_outcomes_and_scoreboard_sorts() -> None:
    """Rolling window evicts oldest samples; scoreboard sorts by health."""
    board = ProviderHealthScoreboard(
        window_size=3,
        healthy_min_success_rate=0.99,
        degraded_min_success_rate=0.5,
        min_samples=1,
    )
    for _ in range(3):
        board.record_failure("openai")
    assert board.snapshot("openai") is not None
    assert board.snapshot("openai").band == "unhealthy"  # type: ignore[union-attr]
    for _ in range(3):
        board.record_success("openai")
    snap = board.snapshot("openai")
    assert snap is not None
    assert snap.samples == 3
    assert snap.band == "healthy"
    board.record_outcome("anthropic", success=False)
    ranked = board.scoreboard()
    assert [item.provider for item in ranked] == ["openai", "anthropic"]
    assert board.unhealthy()[0].provider == "anthropic"
    assert "openai" in board.providers()


def test_frontier_providers_gpt_claude_gemini_kimi() -> None:
    """Frontier provider ids for GPT-5.5 / Sonnet 4.6 / Gemini 3.x / Kimi K2."""
    board = ProviderHealthScoreboard(min_samples=1)
    board.record_success("openai-gpt-5.5")
    board.record_success("anthropic-claude-sonnet-4-6")
    board.record_failure("google-gemini-3")
    board.record_success("moonshot-kimi-k2")
    names = board.providers()
    assert names == [
        "openai-gpt-5.5",
        "anthropic-claude-sonnet-4-6",
        "google-gemini-3",
        "moonshot-kimi-k2",
    ]
    assert board.snapshot("google-gemini-3") is not None
    assert board.snapshot("google-gemini-3").band == "unhealthy"  # type: ignore[union-attr]


def test_rejects_invalid_inputs() -> None:
    """Construction and record paths validate thresholds and provider ids."""
    with pytest.raises(ValueError, match="window_size"):
        ProviderHealthScoreboard(window_size=0)
    with pytest.raises(ValueError, match="healthy_min_success_rate"):
        ProviderHealthScoreboard(healthy_min_success_rate=0.0)
    with pytest.raises(ValueError, match="degraded_min_success_rate"):
        ProviderHealthScoreboard(degraded_min_success_rate=1.0)
    with pytest.raises(ValueError, match="healthy_min_success_rate must be >"):
        ProviderHealthScoreboard(
            healthy_min_success_rate=0.8,
            degraded_min_success_rate=0.9,
        )
    with pytest.raises(ValueError, match="min_samples"):
        ProviderHealthScoreboard(min_samples=0)
    board = ProviderHealthScoreboard()
    with pytest.raises(ValueError, match="provider"):
        board.record_success("")
    with pytest.raises(ValueError, match="provider"):
        board.snapshot("")
    assert board.snapshot("missing") is None
    board.clear()
    assert board.providers() == []
