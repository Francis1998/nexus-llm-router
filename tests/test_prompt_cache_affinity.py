"""Tests for PromptCacheAffinityRouter sticky prefix → model table."""

from __future__ import annotations

import hashlib
import threading

import pytest

from router.prompt_cache_affinity import PromptCacheAffinityRouter, fingerprint


def test_fingerprint_hashes_first_n_chars() -> None:
    """fingerprint SHA-256s prompt[:chars] and is stable."""
    prompt = "A" * 400 + "TAIL"
    expected = hashlib.sha256(("A" * 256).encode("utf-8")).hexdigest()
    assert fingerprint(prompt) == expected
    assert fingerprint(prompt, chars=10) == hashlib.sha256(b"A" * 10).hexdigest()
    short = "hi"
    assert fingerprint(short, chars=256) == hashlib.sha256(b"hi").hexdigest()


def test_fingerprint_rejects_invalid_chars() -> None:
    """chars must be >= 1."""
    with pytest.raises(ValueError, match="chars"):
        fingerprint("prompt", chars=0)


def test_remember_choose_returns_sticky_when_in_candidates() -> None:
    """choose returns remembered model when still eligible."""
    router = PromptCacheAffinityRouter()
    fp = fingerprint("system: you are a coding agent\n" + "x" * 300)
    router.remember(fp, "gpt-5.5")
    assert router.choose(fp, ["claude-sonnet-4-6", "gpt-5.5", "kimi-k2"]) == "gpt-5.5"


def test_choose_returns_none_when_sticky_not_in_candidates() -> None:
    """If sticky model left the pool, choose returns None."""
    router = PromptCacheAffinityRouter()
    fp = fingerprint("shared prefix for gemini cache")
    router.remember(fp, "gemini-3.5-flash")
    assert router.choose(fp, ["gpt-5.5", "claude-sonnet-4-6"]) is None


def test_choose_unknown_fingerprint_returns_none() -> None:
    """Unknown fingerprints do not invent affinity."""
    router = PromptCacheAffinityRouter()
    assert router.choose("deadbeef", ["gpt-5.5"]) is None


def test_remember_overwrites_previous_model() -> None:
    """Later remember() updates the sticky model for a fingerprint."""
    router = PromptCacheAffinityRouter()
    fp = fingerprint("tools schema blob")
    router.remember(fp, "gpt-5.5")
    router.remember(fp, "claude-sonnet-4-6")
    assert router.choose(fp, ["gpt-5.5", "claude-sonnet-4-6"]) == "claude-sonnet-4-6"


def test_snapshot_is_isolated_copy() -> None:
    """snapshot returns a copy safe from later mutations."""
    router = PromptCacheAffinityRouter()
    fp = fingerprint("kimi sticky prefix")
    router.remember(fp, "kimi-k2")
    snap = router.snapshot()
    assert snap == {fp: "kimi-k2"}
    router.remember(fp, "gpt-5.5")
    router.remember(fingerprint("other"), "gemini-3.5-flash")
    assert snap == {fp: "kimi-k2"}
    assert len(router) == 2


def test_forget_and_clear() -> None:
    """forget removes one entry; clear empties the table."""
    router = PromptCacheAffinityRouter()
    fp_a = fingerprint("prefix-a")
    fp_b = fingerprint("prefix-b")
    router.remember(fp_a, "gpt-5.5")
    router.remember(fp_b, "claude-sonnet-4-6")
    assert router.forget(fp_a) is True
    assert router.forget(fp_a) is False
    assert router.choose(fp_a, ["gpt-5.5"]) is None
    assert router.choose(fp_b, ["claude-sonnet-4-6"]) == "claude-sonnet-4-6"
    router.clear()
    assert len(router) == 0
    assert router.snapshot() == {}


def test_rejects_empty_fingerprint_and_model() -> None:
    """remember/choose/forget validate non-empty fingerprint and model_id."""
    router = PromptCacheAffinityRouter()
    with pytest.raises(ValueError, match="prefix_fingerprint"):
        router.remember("", "gpt-5.5")
    with pytest.raises(ValueError, match="model_id"):
        router.remember("abc", "")
    with pytest.raises(ValueError, match="prefix_fingerprint"):
        router.choose("", ["gpt-5.5"])
    with pytest.raises(ValueError, match="prefix_fingerprint"):
        router.forget("")


def test_thread_safe_remember_choose_snapshot() -> None:
    """Concurrent remember/choose/snapshot do not raise or corrupt state."""
    router = PromptCacheAffinityRouter()
    errors: list[BaseException] = []

    def worker(start: int) -> None:
        try:
            for index in range(start, start + 40):
                fp = fingerprint(f"prefix-{index}", chars=64)
                router.remember(fp, f"model-{index % 4}")
                _ = router.choose(fp, [f"model-{index % 4}", "gpt-5.5"])
                _ = router.snapshot()
        except BaseException as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i * 40,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(router) == 160
