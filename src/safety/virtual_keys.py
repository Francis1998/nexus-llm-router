"""Virtual API keys: hashed gateway keys with budgets, model allowlists, limits, and expiry.

``VirtualKeyStore`` keeps one SQLite row per issued key. The secret
(``sk-nexus-`` plus 43 random URL-safe characters, 256 bits) is returned once
by :meth:`VirtualKeyStore.create` / :meth:`VirtualKeyStore.rotate` and never
stored: rows hold its HMAC-SHA-256 under a server-side pepper and a short,
non-secret display prefix. Spend is not stored here; the gateway sums the
durable spend ledger per key id over the key's budget window
(:func:`budget_window`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

KEY_PREFIX = "sk-nexus-"
# Random characters shown after ``sk-nexus-`` in listings (24 of the key's 256 bits).
DISPLAY_RANDOM_CHARS = 4
MAX_ALLOWED_MODELS = 256
MAX_MODEL_ID_CHARS = 128
MAX_LABEL_CHARS = 64
MAX_TENANT_CHARS = 128
MAX_METADATA_BYTES = 4096
MAX_RPM_LIMIT = 1_000_000
# HMAC key used when no pepper is configured: still a one-way hash of a 256-bit
# secret, but anyone holding the database can then check candidate keys offline.
DEFAULT_PEPPER = b"nexus-virtual-keys/v1"
_HMAC_SCHEME = "hmac-sha256"
# Rows written before hashing used a pepper (plain SHA-256); they still authenticate.
_LEGACY_SCHEME = "sha256"
_DURATION_PATTERN = re.compile(r"^([1-9][0-9]{0,3})(mo|m|h|d|w)$")
_UNIT_SECONDS = {"m": 60, "h": 3_600, "d": 86_400, "w": 604_800}
# 1970-01-05 was the first Monday after the Unix epoch: weekly windows start on Mondays.
_FIRST_MONDAY_OFFSET = 4 * 86_400
_UPDATABLE_FIELDS = frozenset(
    {
        "alias",
        "tenant",
        "allowed_models",
        "max_budget_usd",
        "budget_duration",
        "rpm_limit",
        "expires_at",
        "metadata",
        "enabled",
    }
)

KeyStatus = Literal["active", "blocked", "expired", "revoked"]
InvalidKeyReason = Literal["unknown", "revoked", "expired", "blocked"]


@dataclass(frozen=True, slots=True)
class VirtualKey:
    """Durable virtual-key record (hash only; the raw key is never stored).

    ``max_budget_usd`` is ``math.inf`` for keys without a budget, and an empty
    ``allowed_models`` permits every model.
    """

    key_id: str
    key_hash: str
    tenant: str
    max_budget_usd: float
    allowed_models: frozenset[str]
    enabled: bool = True
    alias: str | None = None
    key_prefix: str = ""
    budget_duration: str | None = None
    rpm_limit: int | None = None
    expires_at: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    revoked_at: float | None = None

    @property
    def has_budget(self) -> bool:
        """Return whether the key has a finite USD budget."""
        return math.isfinite(self.max_budget_usd)

    def status(self, now: float) -> KeyStatus:
        """Return the key's state at ``now`` (revoked > expired > blocked > active).

        Args:
            now: Unix timestamp.

        Returns:
            ``revoked``, ``expired``, ``blocked``, or ``active``.
        """
        if self.revoked_at is not None:
            return "revoked"
        if self.expires_at is not None and self.expires_at <= now:
            return "expired"
        if not self.enabled:
            return "blocked"
        return "active"


class VirtualKeyError(RuntimeError):
    """Raised when a virtual key is missing, disabled, or out of policy."""


class InvalidVirtualKeyError(VirtualKeyError):
    """Raised when a presented key is unknown, revoked, expired, or blocked."""

    def __init__(self, reason: InvalidKeyReason, message: str) -> None:
        """Initialize the error.

        Args:
            reason: Why the key cannot be used.
            message: Human-readable message (never contains the key).
        """
        super().__init__(message)
        self.reason: InvalidKeyReason = reason


class VirtualKeyNotFoundError(VirtualKeyError):
    """Raised when an administrative operation names an unknown key id."""


class VirtualKeyRevokedError(VirtualKeyError):
    """Raised when updating or rotating a key that has been revoked."""


class ModelNotAllowedError(VirtualKeyError):
    """Raised when a model is outside a key's allowlist."""


class VirtualKeyBudgetError(VirtualKeyError):
    """Raised when spent plus estimated cost would exceed a key's budget."""


def parse_budget_duration(text: str) -> str:
    """Validate a budget window such as ``30m``, ``1h``, ``1d``, ``1w``, or ``1mo``.

    Args:
        text: ``<count><unit>`` with a count of 1-9999 and a unit of ``m``
            (minutes), ``h``, ``d``, ``w`` (weeks), or ``mo`` (calendar months).

    Returns:
        The normalized duration (surrounding whitespace removed).

    Raises:
        ValueError: When the duration is malformed.
    """
    normalized = text.strip()
    if _DURATION_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "budget_duration must look like 30m, 1h, 1d, 1w, or 1mo "
            "(a count of 1-9999 and a unit of m, h, d, w, or mo)"
        )
    return normalized


def budget_window(duration: str | None, now: float) -> tuple[float, float] | None:
    """Return the UTC budget window ``[start, end)`` containing ``now``.

    Windows are fixed, not rolling: ``Nm``/``Nh``/``Nd`` windows are aligned to
    the Unix epoch (so ``1h`` starts on the hour and ``1d`` at 00:00 UTC),
    ``Nw`` windows start on Mondays at 00:00 UTC, and ``Nmo`` windows on the
    first day of a month at 00:00 UTC (``3mo`` gives calendar quarters).

    Args:
        duration: Budget duration, or ``None`` for a lifetime budget.
        now: Unix timestamp.

    Returns:
        ``(start, end)`` Unix timestamps, or ``None`` for a lifetime budget.
    """
    if duration is None:
        return None
    match = _DURATION_PATTERN.fullmatch(parse_budget_duration(duration))
    if match is None:  # pragma: no cover - parse_budget_duration already validated it
        raise ValueError(f"invalid budget_duration: {duration!r}")
    count, unit = int(match.group(1)), match.group(2)
    if unit == "mo":
        moment = datetime.fromtimestamp(now, UTC)
        month_index = (moment.year - 1970) * 12 + moment.month - 1
        first = month_index - month_index % count
        return _month_start(first), _month_start(first + count)
    length = count * _UNIT_SECONDS[unit]
    offset = _FIRST_MONDAY_OFFSET if unit == "w" else 0
    start = math.floor((now - offset) / length) * length + offset
    return float(start), float(start + length)


def _month_start(month_index: int) -> float:
    """Return the Unix timestamp of the first day of month ``month_index`` (0 = 1970-01)."""
    year, month = divmod(month_index, 12)
    return datetime(1970 + year, month + 1, 1, tzinfo=UTC).timestamp()


def _new_secret() -> str:
    """Return a fresh raw key: ``sk-nexus-`` plus 256 random bits, URL-safe base64."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def _display_prefix(raw_key: str) -> str:
    """Return the non-secret prefix listings show for a raw key."""
    if raw_key.startswith(KEY_PREFIX):
        return raw_key[: len(KEY_PREFIX) + DISPLAY_RANDOM_CHARS]
    return raw_key[:DISPLAY_RANDOM_CHARS]


def _check_label(value: object, name: str, max_chars: int) -> str | None:
    """Validate an optional free-text label such as ``alias`` or ``tenant``."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    stripped = value.strip()
    if len(stripped) > max_chars or not stripped.isprintable():
        raise ValueError(f"{name} must be at most {max_chars} printable characters")
    return stripped


def _check_models(value: object) -> frozenset[str]:
    """Validate a model allowlist (``None`` or empty permits every model)."""
    if value is None:
        return frozenset()
    if isinstance(value, str) or not isinstance(value, Sequence | frozenset | set):
        raise ValueError("allowed_models must be a list of model ids")
    models: set[str] = set()
    for model in value:
        if (
            not isinstance(model, str)
            or not model
            or len(model) > MAX_MODEL_ID_CHARS
            or "," in model
            or not model.isprintable()
            or model != model.strip()
        ):
            raise ValueError(
                f"each allowed model must be a model id of 1-{MAX_MODEL_ID_CHARS} printable "
                "characters without commas or surrounding spaces"
            )
        models.add(model)
    if len(models) > MAX_ALLOWED_MODELS:
        raise ValueError(f"allowed_models accepts at most {MAX_ALLOWED_MODELS} models")
    return frozenset(models)


def _check_budget(value: object) -> float:
    """Validate a USD budget; ``None`` means unlimited (``math.inf``)."""
    if value is None:
        return math.inf
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("max_budget_usd must be a number")
    budget = float(value)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("max_budget_usd must be >= 0")
    return budget


def _check_duration(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("budget_duration must be a string such as 1d")
    return parse_budget_duration(value)


def _check_rpm(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_RPM_LIMIT:
        raise ValueError(f"rpm_limit must be an integer between 1 and {MAX_RPM_LIMIT}")
    return value


def _check_expiry(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return moment.timestamp()
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expires_at must be a datetime or a Unix timestamp")
    if not math.isfinite(float(value)):
        raise ValueError("expires_at must be finite")
    return float(value)


def _check_metadata(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError("metadata must be a JSON object")
    try:
        encoded = json.dumps(dict(value), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be a JSON object") from error
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError(f"metadata must be at most {MAX_METADATA_BYTES} bytes of JSON")
    decoded: dict[str, object] = json.loads(encoded)
    return decoded


def _check_enabled(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("enabled must be true or false")
    return value


_FIELD_CHECKS: dict[str, Callable[[object], object]] = {
    "alias": lambda value: _check_label(value, "alias", MAX_LABEL_CHARS),
    "tenant": lambda value: _check_label(value, "tenant", MAX_TENANT_CHARS) or "",
    "allowed_models": _check_models,
    "max_budget_usd": _check_budget,
    "budget_duration": _check_duration,
    "rpm_limit": _check_rpm,
    "expires_at": _check_expiry,
    "metadata": _check_metadata,
    "enabled": _check_enabled,
}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS virtual_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    tenant TEXT NOT NULL,
    max_budget_usd REAL NOT NULL,
    allowed_models TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1
)
"""
# Columns added after the first schema; ``ALTER TABLE`` upgrades older stores in place.
_ADDED_COLUMNS = (
    ("hash_scheme", f"TEXT NOT NULL DEFAULT '{_LEGACY_SCHEME}'"),
    ("key_prefix", "TEXT NOT NULL DEFAULT ''"),
    ("alias", "TEXT"),
    ("budget_duration", "TEXT"),
    ("rpm_limit", "INTEGER"),
    ("expires_at", "REAL"),
    ("metadata", "TEXT NOT NULL DEFAULT '{}'"),
    ("created_at", "REAL NOT NULL DEFAULT 0"),
    ("updated_at", "REAL NOT NULL DEFAULT 0"),
    ("revoked_at", "REAL"),
)
# Updatable field -> column; only these fixed names are ever formatted into SQL.
_FIELD_COLUMNS = {
    "alias": "alias",
    "tenant": "tenant",
    "allowed_models": "allowed_models",
    "max_budget_usd": "max_budget_usd",
    "budget_duration": "budget_duration",
    "rpm_limit": "rpm_limit",
    "expires_at": "expires_at",
    "metadata": "metadata",
    "enabled": "enabled",
}


class VirtualKeyStore:
    """SQLite-backed virtual key store for multi-tenant gateways.

    Issues hashed keys with an optional USD budget and budget window, a model
    allowlist (empty permits every model), a requests-per-minute limit, an
    expiry, and an alias, tenant, and metadata. Keys can be blocked,
    unblocked, rotated, and revoked; revoked keys stay in the store for audit.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        pepper: str | bytes | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the store and ensure the schema exists.

        Args:
            db_path: Filesystem path for SQLite, or ``":memory:"`` for tests.
            pepper: Server-side secret keying the HMAC-SHA-256 of every key;
                changing it invalidates every issued key. ``None`` uses a
                public constant.
            clock: Unix-time source (injectable for tests).
        """
        self._memory = str(db_path) == ":memory:"
        self._db_path: Path | str = ":memory:" if self._memory else Path(db_path)
        if not self._memory:
            path = Path(self._db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = path
        if isinstance(pepper, str):
            pepper = pepper.encode("utf-8")
        self._pepper = pepper or DEFAULT_PEPPER
        self._clock = clock
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        if self._memory:
            self._connection = self._open()
        self._initialize()

    @property
    def db_path(self) -> Path | str:
        """Return the SQLite database path (or ``":memory:"``)."""
        return self._db_path

    def _open(self) -> sqlite3.Connection:
        # Autocommit mode: writes run in explicit ``BEGIN IMMEDIATE`` transactions.
        connection = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _session(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield a connection; ``write`` wraps the block in one immediate transaction."""
        with self._lock:
            connection = self._connection or self._open()
            try:
                if write:
                    connection.execute("BEGIN IMMEDIATE")
                try:
                    yield connection
                except BaseException:
                    if write:
                        connection.execute("ROLLBACK")
                    raise
                if write:
                    connection.execute("COMMIT")
            finally:
                if self._connection is None:
                    connection.close()

    def _initialize(self) -> None:
        with self._session(write=True) as connection:
            connection.execute(_CREATE_TABLE)
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(virtual_keys)").fetchall()
            }
            for name, declaration in _ADDED_COLUMNS:
                if name not in columns:
                    connection.execute(f"ALTER TABLE virtual_keys ADD COLUMN {name} {declaration}")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_virtual_keys_hash ON virtual_keys(key_hash)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_virtual_keys_created ON virtual_keys(created_at)"
            )

    def _hash_key(self, raw_key: str, scheme: str = _HMAC_SCHEME) -> str:
        """Return the stored digest of ``raw_key`` under ``scheme``."""
        data = raw_key.encode("utf-8")
        if scheme == _LEGACY_SCHEME:
            return hashlib.sha256(data).hexdigest()
        return hmac.new(self._pepper, data, hashlib.sha256).hexdigest()

    @staticmethod
    def _serialize_models(models: frozenset[str]) -> str:
        return ",".join(sorted(models))

    @staticmethod
    def _deserialize_models(blob: str) -> frozenset[str]:
        if not blob.strip():
            return frozenset()
        return frozenset(part.strip() for part in blob.split(",") if part.strip())

    @staticmethod
    def _row_to_key(row: sqlite3.Row) -> VirtualKey:
        metadata = json.loads(str(row["metadata"] or "{}"))
        return VirtualKey(
            key_id=str(row["key_id"]),
            key_hash=str(row["key_hash"]),
            tenant=str(row["tenant"]),
            max_budget_usd=float(row["max_budget_usd"]),
            allowed_models=VirtualKeyStore._deserialize_models(str(row["allowed_models"])),
            enabled=bool(row["enabled"]),
            alias=None if row["alias"] is None else str(row["alias"]),
            key_prefix=str(row["key_prefix"]),
            budget_duration=None if row["budget_duration"] is None else str(row["budget_duration"]),
            rpm_limit=None if row["rpm_limit"] is None else int(row["rpm_limit"]),
            expires_at=None if row["expires_at"] is None else float(row["expires_at"]),
            metadata=metadata if isinstance(metadata, dict) else {},
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            revoked_at=None if row["revoked_at"] is None else float(row["revoked_at"]),
        )

    @staticmethod
    def _column_value(name: str, value: object) -> object:
        """Return the SQLite value stored for a checked field value."""
        if name == "allowed_models" and isinstance(value, frozenset):
            return VirtualKeyStore._serialize_models(value)
        if name == "metadata":
            return json.dumps(value, sort_keys=True)
        if name == "enabled":
            return int(bool(value))
        return value

    def create(
        self,
        *,
        tenant: str | None = None,
        max_budget_usd: float | None = None,
        allowed_models: Sequence[str] | None = None,
        alias: str | None = None,
        budget_duration: str | None = None,
        rpm_limit: int | None = None,
        expires_at: datetime | float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[str, VirtualKey]:
        """Create a virtual key and return ``(raw_key, record)``.

        Args:
            tenant: Optional tenant the key's requests are attributed to.
            max_budget_usd: USD budget per ``budget_duration`` window (or for the
                key's lifetime); ``None`` means unlimited.
            allowed_models: Optional model allowlist; empty/None allows all.
            alias: Optional human-readable label.
            budget_duration: Optional budget window such as ``1d`` or ``1mo``.
            rpm_limit: Optional requests-per-minute limit.
            expires_at: Optional expiry (datetime, naive means UTC, or Unix time).
            metadata: Optional JSON object stored with the key.

        Returns:
            The one-time raw key string and the durable record (hash only).

        Raises:
            ValueError: When a field is invalid.
        """
        values = {
            "tenant": _FIELD_CHECKS["tenant"](tenant),
            "max_budget_usd": _check_budget(max_budget_usd),
            "allowed_models": _check_models(allowed_models),
            "alias": _check_label(alias, "alias", MAX_LABEL_CHARS),
            "budget_duration": _check_duration(budget_duration),
            "rpm_limit": _check_rpm(rpm_limit),
            "expires_at": _check_expiry(expires_at),
            "metadata": _check_metadata(metadata),
        }
        raw_key = _new_secret()
        now = self._clock()
        key_id = "vk_" + secrets.token_hex(12)
        with self._session(write=True) as connection:
            connection.execute(
                """
                INSERT INTO virtual_keys (
                    key_id, key_hash, tenant, max_budget_usd, allowed_models, enabled,
                    hash_scheme, key_prefix, alias, budget_duration, rpm_limit, expires_at,
                    metadata, created_at, updated_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    key_id,
                    self._hash_key(raw_key),
                    values["tenant"],
                    values["max_budget_usd"],
                    self._column_value("allowed_models", values["allowed_models"]),
                    _HMAC_SCHEME,
                    _display_prefix(raw_key),
                    values["alias"],
                    values["budget_duration"],
                    values["rpm_limit"],
                    values["expires_at"],
                    self._column_value("metadata", values["metadata"]),
                    now,
                    now,
                ),
            )
            record = self._fetch(connection, key_id)
        return raw_key, record

    @staticmethod
    def _fetch(connection: sqlite3.Connection, key_id: str) -> VirtualKey:
        query = "SELECT * FROM virtual_keys WHERE key_id = ?"
        row = connection.execute(query, (key_id,)).fetchone()
        if row is None:
            raise VirtualKeyNotFoundError(f"no virtual key with id {key_id!r}")
        return VirtualKeyStore._row_to_key(row)

    def get(self, key_id: str) -> VirtualKey:
        """Return the key with ``key_id`` (including revoked keys).

        Args:
            key_id: Key identifier such as ``vk_...``.

        Returns:
            The stored record.

        Raises:
            VirtualKeyNotFoundError: When no key has this id.
        """
        with self._session() as connection:
            return self._fetch(connection, key_id)

    def list_keys(
        self,
        *,
        alias: str | None = None,
        tenant: str | None = None,
        blocked: bool | None = None,
        expired: bool | None = None,
        revoked: bool | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[VirtualKey], int]:
        """Return one page of keys, newest first, and the number of matching keys.

        Args:
            alias: Only keys with this alias.
            tenant: Only keys bound to this tenant.
            blocked: Only blocked (``True``) or unblocked (``False``) keys.
            expired: Only keys past (``True``) or before (``False``) their expiry.
            revoked: Only revoked (``True``) or unrevoked (``False``) keys.
            limit: Page size (at least 1).
            offset: Number of matching keys to skip.

        Returns:
            ``(keys, total)``.
        """
        clauses: list[str] = []
        params: list[object] = []
        if alias is not None:
            clauses.append("alias = ?")
            params.append(alias)
        if tenant is not None:
            clauses.append("tenant = ?")
            params.append(tenant)
        if blocked is not None:
            clauses.append("enabled = ?")
            params.append(0 if blocked else 1)
        if revoked is not None:
            clauses.append("revoked_at IS NOT NULL" if revoked else "revoked_at IS NULL")
        if expired is not None:
            clauses.append(
                "(expires_at IS NOT NULL AND expires_at <= ?)"
                if expired
                else "(expires_at IS NULL OR expires_at > ?)"
            )
            params.append(self._clock())
        # Only the fixed clauses above are formatted in; every value is bound.
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        count_sql = "SELECT COUNT(*) FROM virtual_keys" + where  # noqa: S608  # nosec B608
        page_sql = (
            "SELECT * FROM virtual_keys"  # noqa: S608  # nosec B608
            + where
            + " ORDER BY created_at DESC, key_id LIMIT ? OFFSET ?"
        )
        with self._session() as connection:
            total = int(connection.execute(count_sql, params).fetchone()[0])
            rows = connection.execute(page_sql, [*params, max(1, limit), max(0, offset)])
            keys = [self._row_to_key(row) for row in rows.fetchall()]
        return keys, total

    def update(self, key_id: str, changes: Mapping[str, object]) -> VirtualKey:
        """Apply a partial update and return the updated key.

        ``None`` clears a limit: no budget, no budget window, no RPM limit, no
        expiry, no alias, no tenant, or (for ``allowed_models``) every model.

        Args:
            key_id: Key identifier.
            changes: Field values by name (``alias``, ``tenant``, ``allowed_models``,
                ``max_budget_usd``, ``budget_duration``, ``rpm_limit``,
                ``expires_at``, ``metadata``, ``enabled``).

        Returns:
            The updated record.

        Raises:
            ValueError: When a field is unknown or invalid.
            VirtualKeyNotFoundError: When no key has this id.
            VirtualKeyRevokedError: When the key has been revoked.
        """
        unknown = sorted(set(changes) - _UPDATABLE_FIELDS)
        if unknown:
            raise ValueError(f"cannot update {', '.join(unknown)}")
        checked = {name: _FIELD_CHECKS[name](value) for name, value in changes.items()}
        with self._session(write=True) as connection:
            current = self._fetch(connection, key_id)
            if current.revoked_at is not None:
                raise VirtualKeyRevokedError(f"virtual key {key_id} has been revoked")
            assignments = [f"{_FIELD_COLUMNS[name]} = ?" for name in checked]
            values = [self._column_value(name, value) for name, value in checked.items()]
            connection.execute(
                "UPDATE virtual_keys SET "  # noqa: S608  # nosec B608
                + ", ".join([*assignments, "updated_at = ?"])
                + " WHERE key_id = ?",
                [*values, self._clock(), key_id],
            )
            return self._fetch(connection, key_id)

    def revoke(self, key_id: str) -> VirtualKey:
        """Revoke a key for good; it stays listed (``status: revoked``) for audit.

        Revoking an already revoked key returns it unchanged.

        Args:
            key_id: Key identifier.

        Returns:
            The revoked record.

        Raises:
            VirtualKeyNotFoundError: When no key has this id.
        """
        with self._session(write=True) as connection:
            current = self._fetch(connection, key_id)
            if current.revoked_at is not None:
                return current
            now = self._clock()
            connection.execute(
                "UPDATE virtual_keys SET revoked_at = ?, updated_at = ? WHERE key_id = ?",
                (now, now, key_id),
            )
            return self._fetch(connection, key_id)

    def rotate(self, key_id: str) -> tuple[str, VirtualKey]:
        """Replace a key's secret; the old secret stops working immediately.

        The key id, limits, and spend history are kept.

        Args:
            key_id: Key identifier.

        Returns:
            The new one-time raw key and the updated record.

        Raises:
            VirtualKeyNotFoundError: When no key has this id.
            VirtualKeyRevokedError: When the key has been revoked.
        """
        raw_key = _new_secret()
        with self._session(write=True) as connection:
            current = self._fetch(connection, key_id)
            if current.revoked_at is not None:
                raise VirtualKeyRevokedError(f"virtual key {key_id} has been revoked")
            connection.execute(
                """
                UPDATE virtual_keys
                SET key_hash = ?, hash_scheme = ?, key_prefix = ?, updated_at = ?
                WHERE key_id = ?
                """,
                (
                    self._hash_key(raw_key),
                    _HMAC_SCHEME,
                    _display_prefix(raw_key),
                    self._clock(),
                    key_id,
                ),
            )
            return raw_key, self._fetch(connection, key_id)

    def authenticate(self, raw_key: str) -> VirtualKey:
        """Resolve a raw key to its durable record.

        The lookup is by keyed hash, and the stored digest is compared in
        constant time; error messages never contain the key.

        Args:
            raw_key: Bearer / virtual key presented by the client.

        Returns:
            The matching active ``VirtualKey``.

        Raises:
            InvalidVirtualKeyError: When the key is unknown, revoked, expired, or
                disabled (blocked).
        """
        if not raw_key:
            raise InvalidVirtualKeyError("unknown", "unknown virtual key")
        expected = {
            scheme: self._hash_key(raw_key, scheme) for scheme in (_HMAC_SCHEME, _LEGACY_SCHEME)
        }
        with self._session() as connection:
            rows = connection.execute(
                "SELECT * FROM virtual_keys WHERE key_hash IN (?, ?)",
                (expected[_HMAC_SCHEME], expected[_LEGACY_SCHEME]),
            ).fetchall()
        match = next(
            (
                row
                for row in rows
                if str(row["hash_scheme"]) in expected
                and hmac.compare_digest(str(row["key_hash"]), expected[str(row["hash_scheme"])])
            ),
            None,
        )
        if match is None:
            raise InvalidVirtualKeyError("unknown", "unknown virtual key")
        key = self._row_to_key(match)
        now = self._clock()
        status = key.status(now)
        if status == "revoked":
            raise InvalidVirtualKeyError("revoked", "virtual key has been revoked")
        if status == "expired":
            expired_at = datetime.fromtimestamp(key.expires_at or now, UTC)
            raise InvalidVirtualKeyError(
                "expired", f"virtual key expired at {expired_at.isoformat().replace('+00:00', 'Z')}"
            )
        if status == "blocked":
            raise InvalidVirtualKeyError("blocked", "virtual key is disabled (blocked)")
        return key

    def assert_model_allowed(self, key: VirtualKey, model: str) -> None:
        """Raise when ``model`` is outside the key's allowlist.

        An empty allowlist permits every model.

        Args:
            key: Authenticated virtual key.
            model: Requested model id.

        Raises:
            ModelNotAllowedError: When the model is not allowed.
        """
        if key.allowed_models and model not in key.allowed_models:
            raise ModelNotAllowedError(f"model {model!r} is not allowed for this virtual key")

    def assert_budget(
        self,
        key: VirtualKey,
        spent_usd: float,
        estimated_cost_usd: float,
    ) -> None:
        """Raise when spent + estimated cost would exceed the key budget.

        Args:
            key: Authenticated virtual key.
            spent_usd: Spend already recorded for this key in its budget window.
            estimated_cost_usd: Projected cost of the next request.

        Raises:
            VirtualKeyBudgetError: When the budget would be exceeded.
        """
        projected = float(spent_usd) + float(estimated_cost_usd)
        if projected > key.max_budget_usd:
            raise VirtualKeyBudgetError(
                f"virtual key budget exceeded: {projected:.6f} > {key.max_budget_usd:.6f}"
            )
