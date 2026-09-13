"""Tests for StickyProviderAffinity session → provider continuity map."""

from __future__ import annotations

import pytest

from safety.sticky_provider import StickyProviderAffinity, StickyProviderBinding


def test_bind_then_get_returns_provider() -> None:
    """bind(session_key, provider) pins the provider for later get()."""
    affinity = StickyProviderAffinity()
    binding = affinity.bind("sess-openai", "openai")
    assert isinstance(binding, StickyProviderBinding)
    assert binding.session_key == "sess-openai"
    assert binding.provider == "openai"
    assert affinity.get("sess-openai") == "openai"


def test_rebind_overwrites_previous_provider() -> None:
    """Re-binding the same session replaces the sticky provider."""
    affinity = StickyProviderAffinity()
    affinity.bind("chat-1", "openai")
    affinity.bind("chat-1", "anthropic")
    assert affinity.get("chat-1") == "anthropic"


def test_clear_removes_binding() -> None:
    """clear() drops the sticky mapping and returns whether it existed."""
    affinity = StickyProviderAffinity()
    affinity.bind("sess-a", "google")
    assert affinity.clear("sess-a") is True
    assert affinity.get("sess-a") is None
    assert affinity.clear("sess-a") is False


def test_independent_sessions_across_frontier_providers() -> None:
    """Distinct session keys keep independent sticky providers."""
    affinity = StickyProviderAffinity()
    pairs = (
        ("s-gpt", "openai"),
        ("s-claude", "anthropic"),
        ("s-gemini", "google"),
        ("s-kimi", "moonshot"),
    )
    for session_key, provider in pairs:
        affinity.bind(session_key, provider)
    for session_key, provider in pairs:
        assert affinity.get(session_key) == provider
    assert sorted(affinity.sessions()) == sorted(s for s, _ in pairs)


def test_rejects_empty_session_or_provider() -> None:
    """Empty session_key / provider raise ValueError."""
    affinity = StickyProviderAffinity()
    with pytest.raises(ValueError, match="session_key"):
        affinity.bind("", "openai")
    with pytest.raises(ValueError, match="provider"):
        affinity.bind("sess", "")
    with pytest.raises(ValueError, match="session_key"):
        affinity.get("")
    with pytest.raises(ValueError, match="session_key"):
        affinity.clear("")
