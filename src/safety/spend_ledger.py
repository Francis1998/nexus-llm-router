"""Durable spend ledger for routed completions (SQLite-backed)."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SpendRecord:
    """One durable spend event."""

    request_id: str
    tenant: str
    provider: str
    model: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    recorded_at: float
    api_key_id: str | None = None


@dataclass(frozen=True, slots=True)
class SpendSummary:
    """Aggregated spend view across ledger rows."""

    total_cost_usd: float
    request_count: int
    by_tenant: dict[str, float]
    by_provider: dict[str, float]
    by_model: dict[str, float]


class SpendLedger:
    """Append-only SQLite spend ledger with cross-instance durability.

    Multiple ``SpendLedger`` instances pointing at the same ``db_path`` share
    durable state, closing the gap versus LiteLLM virtual-key ``/spend`` APIs.
    """

    def __init__(self, db_path: str | Path = "migrations/spend-ledger.sqlite3") -> None:
        """Initialize the ledger and ensure the schema exists.

        Args:
            db_path: Filesystem path for the SQLite database.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Records whose write failed with ``retain_on_failure``, oldest first.
        self._unsettled: list[SpendRecord] = []
        self._initialize()

    @property
    def db_path(self) -> Path:
        """Return the SQLite database path."""
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS spend_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    tenant TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    cost_usd REAL NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    recorded_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spend_tenant ON spend_events(tenant)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spend_provider ON spend_events(provider)"
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(spend_events)")
            }
            if "api_key_id" not in columns:
                # Ledgers created before per-key attribution: their rows keep NULL.
                try:
                    connection.execute("ALTER TABLE spend_events ADD COLUMN api_key_id TEXT")
                except sqlite3.OperationalError as error:
                    # Another process added the column first.
                    if "duplicate column" not in str(error):
                        raise
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spend_api_key "
                "ON spend_events(api_key_id, recorded_at)"
            )
            connection.commit()

    def record(
        self,
        *,
        request_id: str,
        tenant: str,
        provider: str,
        model: str,
        cost_usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        recorded_at: float | None = None,
        api_key_id: str | None = None,
        retain_on_failure: bool = False,
    ) -> SpendRecord:
        """Persist a spend event and return the stored record.

        Args:
            request_id: Router request identifier.
            tenant: Tenant / user / API-key subject.
            provider: Provider name.
            model: Model identifier.
            cost_usd: Cost in USD.
            input_tokens: Prompt tokens.
            output_tokens: Completion tokens.
            recorded_at: Optional unix timestamp; defaults to now.
            api_key_id: Gateway API key the request was made with (a virtual
                key id or a key fingerprint, never the secret).
            retain_on_failure: Keep the record in memory when the write fails,
                so :meth:`has_unsettled` reports it and :meth:`settle` retries it.

        Returns:
            The durable spend record.

        Raises:
            sqlite3.Error: When the write fails.
            OSError: When the database file cannot be written.
        """
        stamp = time.time() if recorded_at is None else float(recorded_at)
        record = SpendRecord(
            request_id=request_id,
            tenant=tenant or "anonymous",
            provider=provider,
            model=model,
            cost_usd=float(cost_usd),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            recorded_at=stamp,
            api_key_id=api_key_id,
        )
        try:
            self._insert(record)
        except (sqlite3.Error, OSError):
            if retain_on_failure:
                with self._lock:
                    self._unsettled.append(record)
            raise
        return record

    def has_unsettled(self, api_key_id: str) -> bool:
        """Return whether a retained record of ``api_key_id`` is not in the ledger yet.

        Args:
            api_key_id: Gateway API key id.

        Returns:
            ``True`` while a failed write of the key is waiting for :meth:`settle`.
        """
        with self._lock:
            return any(record.api_key_id == api_key_id for record in self._unsettled)

    def settle(self) -> None:
        """Retry writing every retained record; records that fail again stay retained."""
        with self._lock:
            pending, self._unsettled = self._unsettled, []
        failed: list[SpendRecord] = []
        for record in pending:
            try:
                self._insert(record)
            except (sqlite3.Error, OSError):
                failed.append(record)
        if failed:
            with self._lock:
                self._unsettled = failed + self._unsettled

    def _insert(self, record: SpendRecord) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO spend_events (
                    request_id, tenant, provider, model, cost_usd,
                    input_tokens, output_tokens, recorded_at, api_key_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    record.tenant,
                    record.provider,
                    record.model,
                    record.cost_usd,
                    record.input_tokens,
                    record.output_tokens,
                    record.recorded_at,
                    record.api_key_id,
                ),
            )
            connection.commit()

    def summary(
        self,
        *,
        tenant: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        since: float | None = None,
        until: float | None = None,
        api_key_id: str | None = None,
    ) -> SpendSummary:
        """Aggregate spend with optional filters.

        Args:
            tenant: Optional tenant filter.
            provider: Optional provider filter.
            model: Optional model filter.
            since: Optional unix timestamp lower bound (inclusive:
                ``recorded_at >= since``).
            until: Optional unix timestamp upper bound (exclusive:
                ``recorded_at < until``).
            api_key_id: Optional gateway API key filter.

        Returns:
            Aggregated spend summary.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if tenant:
            clauses.append("tenant = ?")
            params.append(tenant)
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if model:
            clauses.append("model = ?")
            params.append(model)
        if since is not None:
            clauses.append("recorded_at >= ?")
            params.append(float(since))
        if until is not None:
            clauses.append("recorded_at < ?")
            params.append(float(until))
        if api_key_id:
            clauses.append("api_key_id = ?")
            params.append(api_key_id)
        # WHERE fragments are fixed allowlisted clauses only; values are bound
        # via ``params`` (no user-controlled SQL identifiers).
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._lock, self._connect() as connection:
            # nosec B608 - allowlisted WHERE fragments; values bound in params
            total_sql = (
                "SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM spend_events"  # noqa: S608
                + ((" " + where) if where else "")  # nosec B608
            )
            total_row = connection.execute(total_sql, params).fetchone()
            tenant_sql = (
                "SELECT tenant, SUM(cost_usd) AS total FROM spend_events"  # noqa: S608
                + ((" " + where) if where else "")  # nosec B608
                + " GROUP BY tenant ORDER BY total DESC"
            )
            by_tenant = {
                row["tenant"]: float(row["total"]) for row in connection.execute(tenant_sql, params)
            }
            provider_sql = (
                "SELECT provider, SUM(cost_usd) AS total FROM spend_events"  # noqa: S608
                + ((" " + where) if where else "")  # nosec B608
                + " GROUP BY provider ORDER BY total DESC"
            )
            by_provider = {
                row["provider"]: float(row["total"])
                for row in connection.execute(provider_sql, params)
            }
            model_sql = (
                "SELECT model, SUM(cost_usd) AS total FROM spend_events"  # noqa: S608
                + ((" " + where) if where else "")  # nosec B608
                + " GROUP BY model ORDER BY total DESC"
            )
            by_model = {
                row["model"]: float(row["total"]) for row in connection.execute(model_sql, params)
            }

        return SpendSummary(
            total_cost_usd=float(total_row[0]),
            request_count=int(total_row[1]),
            by_tenant=by_tenant,
            by_provider=by_provider,
            by_model=by_model,
        )

    def key_spend(self, api_key_id: str, *, since: float | None = None) -> tuple[float, int]:
        """Return ``(total USD, request count)`` recorded for one API key.

        Args:
            api_key_id: Gateway API key id (for example a virtual key id).
            since: Optional unix timestamp lower bound (inclusive).

        Returns:
            Spend and number of ledger rows for the key since ``since``.
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM spend_events "
                "WHERE api_key_id = ? AND recorded_at >= ?",
                (api_key_id, float("-inf") if since is None else float(since)),
            ).fetchone()
        return float(row[0]), int(row[1])
