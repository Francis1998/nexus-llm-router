"""Tests for FirstTokenLatencySloAdvisor TTFT / first-token SLO bands."""

from __future__ import annotations

import pytest

from safety.first_token_slo import FirstTokenLatencySloAdvisor, FirstTokenSloAdvice


def test_advise_within_warn_and_breach_bands() -> None:
    """TTFT maps to within / warn / breach relative to the SLO and warn ratio."""
    advisor = FirstTokenLatencySloAdvisor(slo_ttft_ms=500.0, warn_ratio=0.8)
    within = advisor.advise(model="gpt-5.5", ttft_ms=200.0)
    assert isinstance(within, FirstTokenSloAdvice)
    assert within.band == "within"
    assert within.headroom_ms == pytest.approx(300.0)
    warn = advisor.advise(model="claude-sonnet-4-6", ttft_ms=420.0)
    assert warn.band == "warn"
    breach = advisor.advise(model="gemini-3.5-flash", ttft_ms=501.0)
    assert breach.band == "breach"
    assert breach.headroom_ms < 0


def test_per_model_slo_override() -> None:
    """Per-model TTFT SLO overrides the default threshold."""
    advisor = FirstTokenLatencySloAdvisor(
        slo_ttft_ms=300.0,
        warn_ratio=0.9,
        per_model_slo_ttft_ms={"kimi-k2": 800.0},
    )
    snap = advisor.advise(model="kimi-k2", ttft_ms=500.0)
    assert snap.band == "within"
    assert snap.slo_ttft_ms == 800.0
    assert advisor.slo_for("gpt-5.5") == 300.0
    assert advisor.slo_for("kimi-k2") == 800.0


def test_record_tracks_rolling_ttft_and_breaches() -> None:
    """record() keeps a rolling window; breaches() lists models over TTFT SLO."""
    advisor = FirstTokenLatencySloAdvisor(
        slo_ttft_ms=100.0,
        warn_ratio=0.8,
        window_size=5,
    )
    advisor.record("gpt-5.5", 40.0)
    for ttft in (150.0, 160.0, 170.0):
        advisor.record("claude-sonnet-4-6", ttft)
    breached = advisor.breaches()
    assert [item.model for item in breached] == ["claude-sonnet-4-6"]
    assert breached[0].band == "breach"
    assert "gpt-5.5" in advisor.models()


def test_window_evicts_oldest_ttft_samples() -> None:
    """Only the newest window_size TTFT samples drive the rolling snapshot."""
    advisor = FirstTokenLatencySloAdvisor(slo_ttft_ms=1_000.0, window_size=3)
    for _ in range(3):
        advisor.record("gemini-3.5-flash", 5_000.0)
    snap = advisor.snapshot("gemini-3.5-flash")
    assert snap is not None
    assert snap.band == "breach"
    for ttft in (10.0, 12.0, 14.0):
        advisor.record("gemini-3.5-flash", ttft)
    snap = advisor.snapshot("gemini-3.5-flash")
    assert snap is not None
    assert snap.samples == 3
    assert snap.band == "within"


def test_rejects_invalid_inputs() -> None:
    """Construction and advise/record validate thresholds and inputs."""
    with pytest.raises(ValueError, match="slo_ttft_ms"):
        FirstTokenLatencySloAdvisor(slo_ttft_ms=0)
    with pytest.raises(ValueError, match="warn_ratio"):
        FirstTokenLatencySloAdvisor(warn_ratio=0)
    with pytest.raises(ValueError, match="warn_ratio"):
        FirstTokenLatencySloAdvisor(warn_ratio=1.0)
    with pytest.raises(ValueError, match="window_size"):
        FirstTokenLatencySloAdvisor(window_size=0)
    with pytest.raises(ValueError, match="per-model"):
        FirstTokenLatencySloAdvisor(per_model_slo_ttft_ms={"": 10.0})
    advisor = FirstTokenLatencySloAdvisor()
    with pytest.raises(ValueError, match="model"):
        advisor.advise(model="", ttft_ms=10.0)
    with pytest.raises(ValueError, match="ttft_ms"):
        advisor.advise(model="gpt-5.5", ttft_ms=-1.0)
    with pytest.raises(ValueError, match="model"):
        advisor.slo_for("")
    assert advisor.snapshot("missing") is None
    advisor.clear()
    assert advisor.models() == []
