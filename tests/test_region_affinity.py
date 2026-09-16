"""Tests for ProviderRegionAffinityAdvisor ranked region advice."""

from __future__ import annotations

import pytest

from safety.region_affinity import (
    ProviderRegionAffinityAdvisor,
    RegionAffinityAdvice,
    RegionAffinityRank,
)


def test_ranks_primary_region_match_first() -> None:
    """Provider covering the first preferred region ranks highest."""
    advisor = ProviderRegionAffinityAdvisor()
    advice = advisor.advise(
        preferred_regions=["us-east", "eu-west"],
        provider_regions={
            "openai": ["us-east", "us-west"],
            "anthropic": ["eu-west"],
            "moonshot": ["ap-southeast"],
        },
    )
    assert isinstance(advice, RegionAffinityAdvice)
    assert advice.rankings[0].provider == "openai"
    assert advice.rankings[0].rank == 1
    assert advice.rankings[0].affinity_score >= 1.0
    assert "us-east" in advice.rankings[0].matched_regions
    assert advice.rankings[-1].provider == "moonshot"
    assert advice.rankings[-1].affinity_score == 0.0


def test_secondary_overlap_outranks_none() -> None:
    """Secondary preferred-region overlap scores above zero-overlap."""
    advisor = ProviderRegionAffinityAdvisor()
    advice = advisor.advise(
        preferred_regions=["us-east", "eu-west"],
        provider_regions={
            "anthropic": ["eu-west", "eu-central"],
            "gemini": ["asia-northeast"],
        },
    )
    assert advice.rankings[0].provider == "anthropic"
    assert advice.rankings[0].affinity_score > 0.0
    assert advice.rankings[0].matched_regions == ("eu-west",)
    assert advice.rankings[1].provider == "gemini"
    assert advice.rankings[1].affinity_score == 0.0


def test_tie_breaks_by_provider_id() -> None:
    """Equal affinity scores sort by provider id ascending."""
    advisor = ProviderRegionAffinityAdvisor()
    advice = advisor.advise(
        preferred_regions=["us-east"],
        provider_regions={
            "zeta": ["us-east"],
            "alpha": ["us-east"],
        },
    )
    assert [r.provider for r in advice.rankings] == ["alpha", "zeta"]
    assert all(isinstance(r, RegionAffinityRank) for r in advice.rankings)


def test_frontier_models_share_region_ranking() -> None:
    """Same ranking applies for GPT-5.5 / Sonnet / Gemini / Kimi gateways."""
    advisor = ProviderRegionAffinityAdvisor()
    provider_regions = {
        "openai": ["us-east"],
        "anthropic": ["eu-west", "us-east"],
        "google": ["us-central", "eu-west"],
        "moonshot": ["ap-southeast"],
    }
    for _model in (
        "gpt-5.5",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "kimi-k2",
    ):
        advice = advisor.advise(
            preferred_regions=["us-east", "eu-west"],
            provider_regions=provider_regions,
        )
        assert advice.rankings[0].provider == "anthropic"
        assert advice.rankings[0].affinity_score > advice.rankings[1].affinity_score
        assert "preferred=" in advice.advisory


def test_rejects_invalid_inputs() -> None:
    """advise validates non-empty preferred regions and provider maps."""
    advisor = ProviderRegionAffinityAdvisor()
    with pytest.raises(ValueError, match="preferred_regions"):
        advisor.advise(preferred_regions=[], provider_regions={"openai": ["us-east"]})
    with pytest.raises(ValueError, match="provider_regions"):
        advisor.advise(preferred_regions=["us-east"], provider_regions={})
    with pytest.raises(ValueError, match="non-empty"):
        advisor.advise(preferred_regions=[""], provider_regions={"openai": ["us-east"]})
    with pytest.raises(ValueError, match="non-empty"):
        advisor.advise(preferred_regions=["us-east"], provider_regions={"openai": []})
    # Case-insensitive region matching
    advice = advisor.advise(
        preferred_regions=["US-East"],
        provider_regions={"openai": ["us-east"]},
    )
    assert advice.rankings[0].affinity_score >= 1.0
