"""IdempotencyStore reservations: atomic claim, conflict, completion, release, lease."""

from __future__ import annotations

from pathlib import Path

import pytest

from safety.idempotency import IdempotencyKeyInUseError, IdempotencyStore


def test_store_reservation_lifecycle(tmp_path: Path) -> None:
    """Reserve, conflict while pending, replay after completion, reclaim after release."""
    store = IdempotencyStore(tmp_path / "idempotency.sqlite3")

    assert store.reserve("k1", "tenant", lease_seconds=60.0) is None
    with pytest.raises(IdempotencyKeyInUseError):
        store.reserve("k1", "tenant", lease_seconds=60.0)
    assert store.get("k1", "tenant") is None
    assert store.reserve("k1", "other-tenant", lease_seconds=60.0) is None

    assert store.put("k1", "tenant", '{"id": "done"}', 200) is True
    replay = store.reserve("k1", "tenant", lease_seconds=60.0)
    assert replay is not None and replay.response_json == '{"id": "done"}'

    store.release("k1", "other-tenant")
    assert store.reserve("k1", "other-tenant", lease_seconds=60.0) is None


def test_expired_reservation_can_be_reclaimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reservation left by a crashed request expires after its lease."""
    store = IdempotencyStore(tmp_path / "idempotency.sqlite3")
    now = [1_000.0]
    monkeypatch.setattr("safety.idempotency.time.time", lambda: now[0])

    assert store.reserve("k2", "tenant", lease_seconds=30.0) is None
    now[0] += 31.0

    assert store.reserve("k2", "tenant", lease_seconds=30.0) is None
