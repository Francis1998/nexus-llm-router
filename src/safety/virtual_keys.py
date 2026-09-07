"""Virtual API keys with tenant budget and model allowlists (SQLite-backed)."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VirtualKey:
    """Durable virtual-key record (hash only; raw key is never stored)."""

    key_id: str
    key_hash: str
    tenant: str
    max_budget_usd: float
    allowed_models: frozenset[str]
    enabled: bool = True


class VirtualKeyError(RuntimeError):
    """Raised when a virtual key is missing, disabled, or out of policy."""


class VirtualKeyStore:
    """SQLite-backed virtual key store for multi-tenant gateways.

    Closes the gap versus LiteLLM virtual keys: issue hashed keys with a
    per-tenant USD budget and optional model allowlist. Empty
    ``allowed_models`` means all models are permitted.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        """Initialize the store and ensure the schema exists.

        Args:
            db_path: Filesystem path for SQLite, or ``":memory:"`` for tests.
        """
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
                CREATE TABLE IF NOT EXISTS virtual_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL UNIQUE,
                    tenant TEXT NOT NULL,
                    max_budget_usd REAL NOT NULL,
                    allowed_models TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_virtual_keys_hash ON virtual_keys(key_hash)"
            )
            connection.commit()
            if not self._memory:
                connection.close()

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

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
        return VirtualKey(
            key_id=str(row["key_id"]),
            key_hash=str(row["key_hash"]),
            tenant=str(row["tenant"]),
            max_budget_usd=float(row["max_budget_usd"]),
            allowed_models=VirtualKeyStore._deserialize_models(str(row["allowed_models"])),
            enabled=bool(row["enabled"]),
        )

    def create(
        self,
        *,
        tenant: str,
        max_budget_usd: float,
        allowed_models: Sequence[str] | None = None,
    ) -> tuple[str, VirtualKey]:
        """Create a virtual key and return ``(raw_key, record)``.

        Args:
            tenant: Tenant / customer identifier.
            max_budget_usd: Hard USD budget for this key.
            allowed_models: Optional model allowlist; empty/None allows all.

        Returns:
            The one-time raw key string and the durable record (hash only).
        """
        if max_budget_usd < 0:
            raise ValueError("max_budget_usd must be >= 0")
        models = frozenset(allowed_models or ())
        raw_key = secrets.token_urlsafe(32)
        record = VirtualKey(
            key_id=str(uuid.uuid4()),
            key_hash=self._hash_key(raw_key),
            tenant=tenant,
            max_budget_usd=float(max_budget_usd),
            allowed_models=models,
            enabled=True,
        )
        with self._lock:
            connection = self._connect()
            connection.execute(
                """
                INSERT INTO virtual_keys (
                    key_id, key_hash, tenant, max_budget_usd, allowed_models, enabled
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.key_id,
                    record.key_hash,
                    record.tenant,
                    record.max_budget_usd,
                    self._serialize_models(record.allowed_models),
                    1,
                ),
            )
            connection.commit()
            if not self._memory:
                connection.close()
        return raw_key, record

    def authenticate(self, raw_key: str) -> VirtualKey:
        """Resolve a raw key to its durable record.

        Args:
            raw_key: Bearer / virtual key presented by the client.

        Returns:
            The matching enabled ``VirtualKey``.

        Raises:
            VirtualKeyError: When the key is unknown or disabled.
        """
        key_hash = self._hash_key(raw_key)
        with self._lock:
            connection = self._connect()
            row = connection.execute(
                "SELECT * FROM virtual_keys WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()
            if not self._memory:
                connection.close()
        if row is None:
            raise VirtualKeyError("unknown virtual key")
        key = self._row_to_key(row)
        if not key.enabled:
            raise VirtualKeyError("virtual key is disabled")
        return key

    def assert_model_allowed(self, key: VirtualKey, model: str) -> None:
        """Raise when ``model`` is outside the key's allowlist.

        An empty allowlist permits every model.

        Args:
            key: Authenticated virtual key.
            model: Requested model id.

        Raises:
            VirtualKeyError: When the model is not allowed.
        """
        if key.allowed_models and model not in key.allowed_models:
            raise VirtualKeyError(f"model {model!r} is not allowed for this virtual key")

    def assert_budget(
        self,
        key: VirtualKey,
        spent_usd: float,
        estimated_cost_usd: float,
    ) -> None:
        """Raise when spent + estimated cost would exceed the key budget.

        Args:
            key: Authenticated virtual key.
            spent_usd: Already-recorded spend for this key/tenant.
            estimated_cost_usd: Projected cost of the next request.

        Raises:
            VirtualKeyError: When the budget would be exceeded.
        """
        projected = float(spent_usd) + float(estimated_cost_usd)
        if projected > key.max_budget_usd:
            raise VirtualKeyError(
                f"virtual key budget exceeded: {projected:.6f} > {key.max_budget_usd:.6f}"
            )
