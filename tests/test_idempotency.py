"""Tests for SQLite-backed request idempotency key store."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from safety.idempotency import IdempotencyRecord, IdempotencyStore


def test_put_returns_true_then_false_on_duplicate() -> None:
    """First put wins; second put with the same key/tenant returns False."""
    store = IdempotencyStore(":memory:", ttl_seconds=60.0)
    body = '{"id":"chatcmpl-1","model":"gpt-5.5"}'
    assert store.put("key-1", "acme", body, 200) is True
    assert store.put("key-1", "acme", '{"id":"other"}', 200) is False
    record = store.get("key-1", "acme")
    assert record is not None
    assert record.response_json == body
    assert record.status_code == 200
    assert isinstance(record, IdempotencyRecord)


def test_keys_are_scoped_by_tenant() -> None:
    """Same idempotency key under different tenants stores independently."""
    store = IdempotencyStore(":memory:")
    assert store.put("shared-key", "acme", '{"tenant":"acme"}', 200) is True
    assert store.put("shared-key", "globex", '{"tenant":"globex"}', 201) is True
    acme = store.get("shared-key", "acme")
    globex = store.get("shared-key", "globex")
    assert acme is not None and globex is not None
    assert acme.response_json == '{"tenant":"acme"}'
    assert globex.status_code == 201


def test_get_miss_returns_none() -> None:
    """Unknown keys return None."""
    store = IdempotencyStore(":memory:")
    assert store.get("missing", "acme") is None


def test_ttl_eviction_allows_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expired records are evicted and the key can be stored again."""
    store = IdempotencyStore(":memory:", ttl_seconds=10.0)
    clock = {"now": 1_000.0}

    def fake_time() -> float:
        return clock["now"]

    monkeypatch.setattr(time, "time", fake_time)
    assert store.put("ttl-key", "acme", '{"ok":true}', 200) is True
    clock["now"] = 1_005.0
    assert store.get("ttl-key", "acme") is not None
    clock["now"] = 1_011.0
    assert store.get("ttl-key", "acme") is None
    assert store.put("ttl-key", "acme", '{"ok":"again"}', 200) is True
    record = store.get("ttl-key", "acme")
    assert record is not None
    assert record.response_json == '{"ok":"again"}'


def test_evict_expired_returns_deleted_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """evict_expired removes stale rows and reports the count."""
    store = IdempotencyStore(":memory:", ttl_seconds=5.0)
    clock = {"now": 100.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    assert store.put("a", "t", "{}", 200) is True
    assert store.put("b", "t", "{}", 200) is True
    clock["now"] = 110.0
    assert store.evict_expired() == 2
    assert store.get("a", "t") is None


def test_persists_across_instances(tmp_path: Path) -> None:
    """A new store on the same DB path replays previously stored responses."""
    db_path = tmp_path / "idempotency.sqlite3"
    first = IdempotencyStore(db_path, ttl_seconds=3600.0)
    assert first.put("durable", "acme", '{"model":"claude-sonnet-4-6"}', 200) is True
    second = IdempotencyStore(db_path, ttl_seconds=3600.0)
    record = second.get("durable", "acme")
    assert record is not None
    assert record.tenant == "acme"
    assert "claude-sonnet-4-6" in record.response_json


def test_rejects_empty_key_or_tenant() -> None:
    """Empty key/tenant values raise ValueError."""
    store = IdempotencyStore(":memory:")
    with pytest.raises(ValueError, match="key"):
        store.put("", "acme", "{}", 200)
    with pytest.raises(ValueError, match="tenant"):
        store.put("k", "", "{}", 200)
