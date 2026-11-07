"""IdempotencyStore reservations: atomic claim, ownership, completion, release, lease."""

from __future__ import annotations

import multiprocessing
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Barrier
from pathlib import Path

import pytest

from safety.idempotency import (
    IdempotencyKeyInUseError,
    IdempotencyRecord,
    IdempotencyReservation,
    IdempotencyStore,
)


def _claim(store: IdempotencyStore, key: str, tenant: str = "tenant") -> IdempotencyReservation:
    claim = store.reserve(key, tenant, lease_seconds=30.0)
    assert isinstance(claim, IdempotencyReservation)
    return claim


def test_store_reservation_lifecycle(tmp_path: Path) -> None:
    """Reserve, conflict while pending, replay after completion, reclaim after release."""
    store = IdempotencyStore(tmp_path / "idempotency.sqlite3")

    reservation = _claim(store, "k1")
    with pytest.raises(IdempotencyKeyInUseError):
        store.reserve("k1", "tenant", lease_seconds=30.0)
    assert store.get("k1", "tenant") is None
    assert store.put("k1", "tenant", '{"id": "other"}', 200) is False
    other_tenant = _claim(store, "k1", "other-tenant")

    assert store.complete(reservation, '{"id": "done"}', 200) is True
    replay = store.reserve("k1", "tenant", lease_seconds=30.0)
    assert isinstance(replay, IdempotencyRecord)
    assert replay.response_json == '{"id": "done"}'

    store.release(other_tenant)
    _claim(store, "k1", "other-tenant")


def test_expired_reservation_can_be_reclaimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reservation left by a crashed request expires after its lease."""
    store = IdempotencyStore(tmp_path / "idempotency.sqlite3")
    now = [1_000.0]
    monkeypatch.setattr("safety.idempotency.time.time", lambda: now[0])

    _claim(store, "k2")
    now[0] += 31.0

    _claim(store, "k2")


def test_stale_owner_cannot_release_or_overwrite_newer_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that outlived its lease cannot free or replace the new owner's claim."""
    store = IdempotencyStore(tmp_path / "idempotency.sqlite3")
    now = [1_000.0]
    monkeypatch.setattr("safety.idempotency.time.time", lambda: now[0])
    stale = _claim(store, "k3")
    now[0] += 31.0
    current = _claim(store, "k3")

    store.release(stale)
    with pytest.raises(IdempotencyKeyInUseError):
        store.reserve("k3", "tenant", lease_seconds=30.0)
    assert store.complete(stale, '{"owner": "stale"}', 200) is False
    assert store.complete(current, '{"owner": "current"}', 200) is True

    replay = store.reserve("k3", "tenant", lease_seconds=30.0)
    assert isinstance(replay, IdempotencyRecord)
    assert replay.response_json == '{"owner": "current"}'


def test_expired_owner_still_stores_result_when_key_is_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late completion is kept when no newer request reclaimed the key."""
    store = IdempotencyStore(tmp_path / "idempotency.sqlite3")
    now = [1_000.0]
    monkeypatch.setattr("safety.idempotency.time.time", lambda: now[0])
    late = _claim(store, "k4")
    now[0] += 31.0

    assert store.complete(late, '{"owner": "late"}', 200) is True
    replay = store.reserve("k4", "tenant", lease_seconds=30.0)
    assert isinstance(replay, IdempotencyRecord)
    assert replay.response_json == '{"owner": "late"}'


def _race_for_key(db_path: str, start: Barrier, attempted: Barrier, results: Queue[str]) -> None:
    """Worker process: reserve the shared key with its own store and connection."""
    store = IdempotencyStore(db_path)
    start.wait()
    try:
        claim = store.reserve("shared", "tenant", lease_seconds=30.0)
    except IdempotencyKeyInUseError:
        results.put("in_use")
        attempted.wait()
        return
    results.put("reserved")
    attempted.wait()
    assert isinstance(claim, IdempotencyReservation)
    store.complete(claim, '{"winner": true}', 200)


def test_reservation_is_atomic_across_processes(tmp_path: Path) -> None:
    """Independent workers sharing one SQLite file: exactly one wins the key."""
    context = multiprocessing.get_context("spawn")
    workers = 4
    # Barrier timeouts and daemon workers keep a failing worker from hanging CI.
    start = context.Barrier(workers, timeout=30)
    attempted = context.Barrier(workers, timeout=30)
    results = context.Queue()
    db_path = str(tmp_path / "idempotency.sqlite3")
    IdempotencyStore(db_path)
    processes = [
        context.Process(
            target=_race_for_key, args=(db_path, start, attempted, results), daemon=True
        )
        for _ in range(workers)
    ]
    try:
        for process in processes:
            process.start()
        outcomes = sorted(results.get(timeout=30) for _ in range(workers))
        for process in processes:
            process.join(timeout=30)
    finally:
        for process in processes:
            if process.pid is None:
                continue
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    assert outcomes == ["in_use"] * (workers - 1) + ["reserved"]
    assert [process.exitcode for process in processes] == [0] * workers
    replay = IdempotencyStore(db_path).reserve("shared", "tenant", lease_seconds=30.0)
    assert isinstance(replay, IdempotencyRecord)
    assert replay.response_json == '{"winner": true}'
