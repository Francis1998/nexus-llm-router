"""Tests for ProviderFallbackChainPlanner ordered provider fallback chains."""

from __future__ import annotations

import pytest

from safety.fallback_chain import FallbackChainPlan, ProviderFallbackChainPlanner


def test_plan_orders_by_preference_and_skips_unavailable() -> None:
    """Preference order wins; unavailable providers are skipped with rationale."""
    planner = ProviderFallbackChainPlanner(
        preference_order=("openai", "anthropic", "google", "moonshot"),
    )
    plan = planner.plan(
        candidates=("google", "openai", "moonshot", "anthropic"),
        unavailable={"anthropic"},
    )
    assert isinstance(plan, FallbackChainPlan)
    assert plan.primary == "openai"
    assert plan.chain == ["google", "moonshot"]
    assert plan.skipped_unavailable == ["anthropic"]
    assert "preference" in plan.rationale.lower() or "openai" in plan.rationale


def test_explicit_primary_anchors_chain() -> None:
    """Caller-supplied primary stays first when available."""
    planner = ProviderFallbackChainPlanner(
        preference_order=("openai", "anthropic", "google", "moonshot"),
    )
    plan = planner.plan(
        candidates=("openai", "anthropic", "google", "moonshot"),
        primary="moonshot",
    )
    assert plan.primary == "moonshot"
    assert plan.chain == ["openai", "anthropic", "google"]
    assert "moonshot" not in plan.chain


def test_unknown_candidates_preserve_relative_order_after_preferred() -> None:
    """Providers absent from preference order keep input order after known ones."""
    planner = ProviderFallbackChainPlanner(preference_order=("openai", "anthropic"))
    plan = planner.plan(
        candidates=("custom-a", "openai", "custom-b", "anthropic"),
    )
    assert plan.primary == "openai"
    assert plan.chain == ["anthropic", "custom-a", "custom-b"]


def test_frontier_provider_labels_for_catalog_models() -> None:
    """Planner accepts frontier provider ids used by GPT-5.5 / Sonnet / Gemini / Kimi."""
    planner = ProviderFallbackChainPlanner(
        preference_order=("openai", "anthropic", "google", "moonshot"),
    )
    plan = planner.plan(
        candidates=("moonshot", "google", "anthropic", "openai"),
        unavailable=set(),
    )
    assert plan.ordered() == ["openai", "anthropic", "google", "moonshot"]
    assert plan.primary == "openai"


def test_rejects_empty_candidates_and_invalid_primary() -> None:
    """Empty candidates / unknown primary / empty preference entries raise."""
    with pytest.raises(ValueError, match="preference"):
        ProviderFallbackChainPlanner(preference_order=("", "openai"))
    planner = ProviderFallbackChainPlanner(preference_order=("openai", "anthropic"))
    with pytest.raises(ValueError, match="candidates"):
        planner.plan(candidates=())
    with pytest.raises(ValueError, match="primary"):
        planner.plan(candidates=("openai",), primary="missing")
    with pytest.raises(ValueError, match="primary"):
        planner.plan(candidates=("openai",), primary="")
    # All unavailable → error
    with pytest.raises(ValueError, match="available"):
        planner.plan(candidates=("openai", "anthropic"), unavailable={"openai", "anthropic"})
