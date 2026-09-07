"""Tests for SQLite-backed virtual API keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from safety.virtual_keys import VirtualKeyError, VirtualKeyStore


def test_create_and_authenticate_round_trip() -> None:
    """create returns a raw key that authenticates to the stored record."""
    store = VirtualKeyStore(":memory:")
    raw_key, record = store.create(
        tenant="acme",
        max_budget_usd=10.0,
        allowed_models=["gpt-5.5", "claude-sonnet-4-6"],
    )
    assert raw_key
    assert record.tenant == "acme"
    assert record.max_budget_usd == 10.0
    assert record.allowed_models == frozenset({"gpt-5.5", "claude-sonnet-4-6"})
    assert record.key_hash != raw_key
    authenticated = store.authenticate(raw_key)
    assert authenticated.key_id == record.key_id
    assert authenticated.key_hash == record.key_hash


def test_authenticate_unknown_and_disabled(tmp_path: Path) -> None:
    """Unknown keys raise; disabled keys also raise."""
    db_path = tmp_path / "keys.sqlite3"
    store = VirtualKeyStore(db_path)
    with pytest.raises(VirtualKeyError, match="unknown"):
        store.authenticate("not-a-real-key")

    raw_key, record = store.create(tenant="globex", max_budget_usd=1.0)
    # Disable via direct SQL to exercise authenticate's enabled check.
    import sqlite3

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE virtual_keys SET enabled = 0 WHERE key_id = ?",
            (record.key_id,),
        )
        connection.commit()
    with pytest.raises(VirtualKeyError, match="disabled"):
        store.authenticate(raw_key)


def test_empty_allowlist_permits_all_models() -> None:
    """Empty allowed_models means every model is allowed."""
    store = VirtualKeyStore(":memory:")
    _, key = store.create(tenant="open", max_budget_usd=5.0, allowed_models=None)
    assert key.allowed_models == frozenset()
    store.assert_model_allowed(key, "gpt-5.5")
    store.assert_model_allowed(key, "kimi-k2")


def test_assert_model_allowed_rejects_outside_allowlist() -> None:
    """Non-empty allowlist rejects models outside the set."""
    store = VirtualKeyStore(":memory:")
    _, key = store.create(
        tenant="restricted",
        max_budget_usd=5.0,
        allowed_models=["gemini-3.5-flash"],
    )
    store.assert_model_allowed(key, "gemini-3.5-flash")
    with pytest.raises(VirtualKeyError, match="not allowed"):
        store.assert_model_allowed(key, "gpt-5.5")


def test_assert_budget_enforces_ceiling() -> None:
    """spent + estimated cost must stay within max_budget_usd."""
    store = VirtualKeyStore(":memory:")
    _, key = store.create(tenant="budget", max_budget_usd=1.0)
    store.assert_budget(key, spent_usd=0.4, estimated_cost_usd=0.5)
    with pytest.raises(VirtualKeyError, match="budget exceeded"):
        store.assert_budget(key, spent_usd=0.6, estimated_cost_usd=0.5)


def test_persists_across_instances(tmp_path: Path) -> None:
    """A new store on the same DB path authenticates previously issued keys."""
    db_path = tmp_path / "durable-keys.sqlite3"
    first = VirtualKeyStore(db_path)
    raw_key, record = first.create(
        tenant="durable",
        max_budget_usd=2.5,
        allowed_models=["gpt-5.5"],
    )
    second = VirtualKeyStore(db_path)
    authenticated = second.authenticate(raw_key)
    assert authenticated.key_id == record.key_id
    assert authenticated.tenant == "durable"
    assert authenticated.allowed_models == frozenset({"gpt-5.5"})
