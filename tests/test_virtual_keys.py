"""Tests for SQLite-backed virtual API keys."""

from __future__ import annotations

import hashlib
import hmac
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from safety.spend_ledger import SpendLedger
from safety.tenant_rate_limiter import TenantRateLimitExceededError
from safety.virtual_keys import (
    KEY_PREFIX,
    InvalidVirtualKeyError,
    KeyRateLimiter,
    VirtualKeyError,
    VirtualKeyNotFoundError,
    VirtualKeyRevokedError,
    VirtualKeyStore,
    budget_window,
    parse_budget_duration,
)


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


class _Clock:
    """Settable Unix-time source for expiry and window tests."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# 2026-11-26T12:00:00Z, a Thursday.
THURSDAY_NOON = datetime(2026, 11, 26, 12, tzinfo=UTC).timestamp()


def test_secrets_are_hmac_hashed_and_shown_only_by_prefix(tmp_path: Path) -> None:
    """Rows hold HMAC-SHA-256 under the pepper and a display prefix, never the secret."""
    db_path = tmp_path / "keys.sqlite3"
    store = VirtualKeyStore(db_path, pepper="pepper-0123456789abcdef")
    raw_key, record = store.create(alias="team-a")
    assert raw_key.startswith(KEY_PREFIX)
    assert len(raw_key) == len(KEY_PREFIX) + 43
    assert record.key_prefix == raw_key[: len(KEY_PREFIX) + 4]
    expected = hmac.new(b"pepper-0123456789abcdef", raw_key.encode(), hashlib.sha256)
    assert record.key_hash == expected.hexdigest()
    assert record.key_hash != hashlib.sha256(raw_key.encode()).hexdigest()
    assert raw_key not in db_path.read_bytes().decode("latin-1")
    assert all(raw_key not in repr(key) for key in store.list_keys()[0])
    other_pepper = VirtualKeyStore(db_path, pepper="another-pepper-0123456789")
    with pytest.raises(InvalidVirtualKeyError) as unknown:
        other_pepper.authenticate(raw_key)
    assert unknown.value.reason == "unknown"


def test_keys_from_the_original_schema_still_authenticate(tmp_path: Path) -> None:
    """Stores written before the upgrade (plain SHA-256 rows) migrate in place."""
    import sqlite3

    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE virtual_keys (key_id TEXT PRIMARY KEY, key_hash TEXT NOT NULL UNIQUE, "
            "tenant TEXT NOT NULL, max_budget_usd REAL NOT NULL, allowed_models TEXT NOT NULL, "
            "enabled INTEGER NOT NULL DEFAULT 1)"
        )
        connection.execute(
            "INSERT INTO virtual_keys VALUES (?, ?, ?, ?, ?, 1)",
            ("legacy-id", hashlib.sha256(b"legacy-secret").hexdigest(), "acme", 3.0, "gpt-6.2-sol"),
        )
    key = VirtualKeyStore(db_path, pepper="pepper-0123456789abcdef").authenticate("legacy-secret")
    assert (key.key_id, key.tenant, key.max_budget_usd) == ("legacy-id", "acme", 3.0)
    assert key.allowed_models == frozenset({"gpt-6.2-sol"})
    assert key.status(THURSDAY_NOON) == "active"


def test_expiry_blocking_revocation_and_rotation() -> None:
    """Each state is its own 401 reason; revocation is final, rotation keeps the key id."""
    clock = _Clock(THURSDAY_NOON)
    store = VirtualKeyStore(":memory:", clock=clock)
    raw_key, record = store.create(expires_at=THURSDAY_NOON + 60)
    assert store.authenticate(raw_key).key_id == record.key_id
    clock.now += 60
    with pytest.raises(InvalidVirtualKeyError) as expired:
        store.authenticate(raw_key)
    assert (expired.value.reason, str(expired.value)) == (
        "expired",
        "virtual key expired at 2026-11-26T12:01:00Z",
    )

    store.update(record.key_id, {"expires_at": None, "enabled": False})
    with pytest.raises(InvalidVirtualKeyError) as blocked:
        store.authenticate(raw_key)
    assert blocked.value.reason == "blocked"
    store.update(record.key_id, {"enabled": True})
    new_key, rotated = store.rotate(record.key_id)
    assert rotated.key_id == record.key_id
    assert store.authenticate(new_key).key_id == record.key_id
    with pytest.raises(InvalidVirtualKeyError):
        store.authenticate(raw_key)

    revoked = store.revoke(record.key_id)
    assert revoked.status(clock.now) == "revoked"
    assert store.revoke(record.key_id) == revoked
    with pytest.raises(InvalidVirtualKeyError) as error:
        store.authenticate(new_key)
    assert error.value.reason == "revoked"
    with pytest.raises(VirtualKeyRevokedError):
        store.update(record.key_id, {"alias": "again"})
    with pytest.raises(VirtualKeyRevokedError):
        store.rotate(record.key_id)
    assert store.list_keys(revoked=True)[1] == 1


def test_update_validates_fields_and_none_clears_limits() -> None:
    """Unknown or invalid fields raise ``ValueError``; ``None`` removes a limit."""
    store = VirtualKeyStore(":memory:")
    _, record = store.create(max_budget_usd=1.0, budget_duration="1d", rpm_limit=10)
    with pytest.raises(ValueError, match="cannot update"):
        store.update(record.key_id, {"key_hash": "x"})
    for changes in [
        {"rpm_limit": 0},
        {"budget_duration": "0d"},
        {"max_budget_usd": float("nan")},
        {"allowed_models": "gpt-6.2-sol"},
        {"metadata": {"bad": object()}},
    ]:
        with pytest.raises(ValueError):
            store.update(record.key_id, changes)
    cleared = store.update(record.key_id, {"max_budget_usd": None, "rpm_limit": None})
    assert (cleared.has_budget, cleared.rpm_limit, cleared.budget_duration) == (False, None, "1d")
    with pytest.raises(VirtualKeyNotFoundError):
        store.update("vk_missing", {"alias": "x"})


@pytest.mark.parametrize(
    ("duration", "start", "end"),
    [
        ("1h", "2026-11-26T12:00:00", "2026-11-26T13:00:00"),
        ("6h", "2026-11-26T12:00:00", "2026-11-26T18:00:00"),
        ("1d", "2026-11-26T00:00:00", "2026-11-27T00:00:00"),
        ("1w", "2026-11-23T00:00:00", "2026-11-30T00:00:00"),
        ("1mo", "2026-11-01T00:00:00", "2026-12-01T00:00:00"),
        ("3mo", "2026-10-01T00:00:00", "2027-01-01T00:00:00"),
        ("30m", "2026-11-26T12:00:00", "2026-11-26T12:30:00"),
    ],
)
def test_budget_windows_are_fixed_utc_windows(duration: str, start: str, end: str) -> None:
    """Windows align to UTC hours, midnights, Mondays, and calendar months."""
    window = budget_window(duration, THURSDAY_NOON)
    bounds = [datetime.fromisoformat(value).replace(tzinfo=UTC) for value in (start, end)]
    assert window == (bounds[0].timestamp(), bounds[1].timestamp())


def test_budget_windows_are_half_open_and_lifetime_budgets_have_none() -> None:
    """The window end belongs to the next window; no duration means a lifetime budget."""
    window = budget_window("1d", THURSDAY_NOON)
    assert window is not None
    assert budget_window("1d", window[1]) == (window[1], window[1] + 86_400)
    assert budget_window("1d", window[1] - 0.001) == window
    assert budget_window(None, THURSDAY_NOON) is None
    for invalid in ["", "1y", "0d", "10000d", "1.5h", "d"]:
        with pytest.raises(ValueError, match="budget_duration"):
            parse_budget_duration(invalid)


def test_list_keys_filters_and_pages() -> None:
    """``list_keys`` returns one page newest first plus the total matching count."""
    clock = _Clock(THURSDAY_NOON)
    store = VirtualKeyStore(":memory:", clock=clock)
    ids = []
    for index in range(5):
        clock.now += 1
        ids.append(store.create(alias="team" if index % 2 else "ops")[1].key_id)
    keys, total = store.list_keys(alias="team", limit=1)
    assert ([key.key_id for key in keys], total) == ([ids[3]], 2)
    keys, total = store.list_keys(limit=2, offset=3)
    assert ([key.key_id for key in keys], total) == ([ids[1], ids[0]], 5)


def test_key_rate_limiter_allows_the_limit_per_minute() -> None:
    """A key may send ``rpm_limit`` requests at once; a changed limit starts a new bucket."""
    limiter = KeyRateLimiter()
    for _ in range(3):
        limiter.assert_allowed("vk_a", 3)
    with pytest.raises(TenantRateLimitExceededError):
        limiter.assert_allowed("vk_a", 3)
    limiter.assert_allowed("vk_b", 3)
    limiter.assert_allowed("vk_a", 4)


def test_concurrent_settlement_loses_no_spend(tmp_path: Path) -> None:
    """Spend written by many threads through two ledger handles sums exactly per key."""
    ledger_path = tmp_path / "spend-ledger.sqlite3"
    ledgers = [SpendLedger(ledger_path), SpendLedger(ledger_path)]

    def settle(worker: int) -> None:
        for index in range(25):
            ledgers[worker % 2].record(
                request_id=f"req-{worker}-{index}",
                tenant="t",
                provider="openai",
                model="gpt-6.2-sol",
                cost_usd=0.001,
                api_key_id="vk_shared" if index % 5 else "vk_other",
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(settle, range(8)))
    spent, count = SpendLedger(ledger_path).key_spend("vk_shared")
    assert count == 160
    assert spent == pytest.approx(0.160)
    assert SpendLedger(ledger_path).key_spend("vk_other") == (pytest.approx(0.040), 40)
    assert SpendLedger(ledger_path).key_spend("vk_shared", since=time.time() + 60) == (0.0, 0)
