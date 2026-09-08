"""Durable request idempotency key store (SQLite-backed)."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """Stored response for a previously processed idempotency key."""

    key: str
    tenant: str
    response_json: str
    status_code: int
    created_at: float
    expires_at: float


class IdempotencyStore:
    """SQLite-backed idempotency key store for chat-completion gateways.

    Closes the gap versus LiteLLM / Portkey durable idempotency keys: clients
    can safely retry ``POST /v1/chat/completions`` with the same
    ``Idempotency-Key`` and receive the original response instead of
    double-billing GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        ttl_seconds: float = 86_400.0,
    ) -> None:
        """Initialize the store and ensure the schema exists.

        Args:
            db_path: Filesystem path for SQLite, or ``":memory:"`` for tests.
            ttl_seconds: Default lifetime for stored responses (seconds).
        """
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._ttl_seconds = float(ttl_seconds)
        self._memory = str(db_path) == ":memory:"
        self._db_path: Path | str = ":memory:" if self._memory else Path(db_path)
        if not self._memory:
            path = Path(self._db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = path
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        if self._memory:
            self._connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        self._initialize()

    @property
    def db_path(self) -> Path | str:
        """Return the SQLite database path (or ``":memory:"``)."""
        return self._db_path

    @property
    def ttl_seconds(self) -> float:
        """Return the configured response TTL in seconds."""
        return self._ttl_seconds

    def _connect(self) -> sqlite3.Connection:
        if self._memory:
            assert self._connection is not None
            return self._connection
        connection = sqlite3.connect(self._db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    key TEXT NOT NULL,
                    tenant TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (key, tenant)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency_keys(expires_at)"
            )
            connection.commit()
            if not self._memory:
                connection.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> IdempotencyRecord:
        return IdempotencyRecord(
            key=str(row["key"]),
            tenant=str(row["tenant"]),
            response_json=str(row["response_json"]),
            status_code=int(row["status_code"]),
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
        )

    def _evict_expired_locked(self, connection: sqlite3.Connection, now: float) -> None:
        connection.execute(
            "DELETE FROM idempotency_keys WHERE expires_at <= ?",
            (now,),
        )

    def put(self, key: str, tenant: str, response_json: str, status_code: int) -> bool:
        """Store a response for ``(key, tenant)`` if the key is new.

        Expired rows are evicted before insert. Returns ``False`` when an
        unexpired record already exists for the same key and tenant.

        Args:
            key: Client-supplied idempotency key.
            tenant: Tenant / customer identifier.
            response_json: Serialized response body to replay on retry.
            status_code: HTTP status code to replay on retry.

        Returns:
            ``True`` when the key was stored; ``False`` if it already exists.
        """
        if not key:
            raise ValueError("key must be non-empty")
        if not tenant:
            raise ValueError("tenant must be non-empty")
        now = time.time()
        expires_at = now + self._ttl_seconds
        with self._lock:
            connection = self._connect()
            self._evict_expired_locked(connection, now)
            existing = connection.execute(
                "SELECT 1 FROM idempotency_keys WHERE key = ? AND tenant = ?",
                (key, tenant),
            ).fetchone()
            if existing is not None:
                if not self._memory:
                    connection.close()
                return False
            connection.execute(
                """
                INSERT INTO idempotency_keys (
                    key, tenant, response_json, status_code, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, tenant, response_json, int(status_code), now, expires_at),
            )
            connection.commit()
            if not self._memory:
                connection.close()
        return True

    def get(self, key: str, tenant: str) -> IdempotencyRecord | None:
        """Return the stored record for ``(key, tenant)``, or ``None``.

        Expired rows are evicted and treated as misses.

        Args:
            key: Client-supplied idempotency key.
            tenant: Tenant / customer identifier.

        Returns:
            The matching ``IdempotencyRecord``, or ``None`` if missing/expired.
        """
        now = time.time()
        with self._lock:
            connection = self._connect()
            self._evict_expired_locked(connection, now)
            connection.commit()
            row = connection.execute(
                "SELECT * FROM idempotency_keys WHERE key = ? AND tenant = ?",
                (key, tenant),
            ).fetchone()
            if not self._memory:
                connection.close()
        if row is None:
            return None
        return self._row_to_record(row)

    def evict_expired(self) -> int:
        """Delete expired rows and return how many were removed."""
        now = time.time()
        with self._lock:
            connection = self._connect()
            cursor = connection.execute(
                "DELETE FROM idempotency_keys WHERE expires_at <= ?",
                (now,),
            )
            deleted = int(cursor.rowcount)
            connection.commit()
            if not self._memory:
                connection.close()
        return deleted
