"""Tests for character-trigram semantic fuzzy caching."""

from __future__ import annotations

import time

from cache.semantic_fuzzy_cache import SemanticFuzzyCache


def test_exact_text_is_fuzzy_hit() -> None:
    """Identical normalized messages hit at similarity 1.0."""
    cache = SemanticFuzzyCache(ttl_seconds=60.0, threshold=0.92)
    messages = [{"role": "user", "content": "Summarize the refund policy"}]
    cache.set(messages, {"content": "cached"}, model="gpt-5.5", tenant="acme")
    hit = cache.get(messages, model="gpt-5.5", tenant="acme")
    assert hit is not None
    assert hit.value == {"content": "cached"}
    assert hit.similarity == 1.0
    assert cache.hits == 1


def test_near_duplicate_hits_above_threshold() -> None:
    """A small paraphrase still exceeds the default Jaccard threshold."""
    cache = SemanticFuzzyCache(ttl_seconds=60.0, threshold=0.85)
    original = [{"role": "user", "content": "Explain quantum entanglement briefly"}]
    near = [{"role": "user", "content": "Explain quantum entanglement briefly."}]
    cache.set(original, "answer-a", model="claude-sonnet-4-6", tenant="t1")
    hit = cache.get(near, model="claude-sonnet-4-6", tenant="t1")
    assert hit is not None
    assert hit.value == "answer-a"
    assert hit.similarity >= 0.85


def test_unrelated_prompt_misses() -> None:
    """Unrelated prompts stay below the similarity threshold."""
    cache = SemanticFuzzyCache(ttl_seconds=60.0, threshold=0.92)
    cache.set(
        [{"role": "user", "content": "Write a haiku about autumn leaves"}],
        "poem",
        model="gemini-3.5-flash",
        tenant="t1",
    )
    miss = cache.get(
        [{"role": "user", "content": "Debug this Python TypeError stacktrace"}],
        model="gemini-3.5-flash",
        tenant="t1",
    )
    assert miss is None
    assert cache.misses == 1


def test_tenant_and_model_namespace_isolation() -> None:
    """Hits do not cross tenant or model namespaces."""
    cache = SemanticFuzzyCache(ttl_seconds=60.0, threshold=0.5)
    messages = [{"role": "user", "content": "shared fuzzy prompt"}]
    cache.set(messages, "a-only", model="gpt-5.5", tenant="tenant-a")
    assert cache.get(messages, model="gpt-5.5", tenant="tenant-a") is not None
    assert cache.get(messages, model="gpt-5.5", tenant="tenant-b") is None
    assert cache.get(messages, model="kimi-k2", tenant="tenant-a") is None


def test_ttl_expiry_forces_miss() -> None:
    """Expired fuzzy entries are purged and treated as misses."""
    cache = SemanticFuzzyCache(ttl_seconds=0.05, threshold=0.5)
    messages = [{"role": "user", "content": "ttl fuzzy"}]
    cache.set(messages, "stale", model="gpt-5.5", tenant="t")
    time.sleep(0.06)
    assert cache.get(messages, model="gpt-5.5", tenant="t") is None


def test_disabled_cache_never_hits() -> None:
    """When disabled, set is a no-op and get always misses."""
    cache = SemanticFuzzyCache(enabled=False)
    messages = [{"role": "user", "content": "ignored"}]
    assert cache.set(messages, "x") == ""
    assert cache.get(messages) is None
    assert cache.misses == 1


def test_clear_resets_entries_and_counters() -> None:
    """clear drops stored entries and hit/miss counters."""
    cache = SemanticFuzzyCache(ttl_seconds=60.0)
    messages = [{"role": "user", "content": "clear me"}]
    cache.set(messages, "v", model="gpt-5.5")
    assert cache.get(messages, model="gpt-5.5") is not None
    cache.clear()
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0
