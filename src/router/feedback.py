"""Durable request feedback and the statistics behind feedback-aware routing.

Every completed request is registered (request id, caller scope, and the routing
snapshot of its audit record) so feedback can later be attached by request id.
Feedback rows copy the model, provider, strategy, and domain of that snapshot at
write time and are upserted per ``(request_id, metric, source)``. Requests and
feedback belong to the caller scope that made the request: the API key
fingerprint plus the optional ``X-Tenant-Id``.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from router.schemas import AuditRecord

FeedbackSource = Literal["human", "automated"]
FeedbackKind = Literal["boolean", "numeric"]
FeedbackGroupBy = Literal["model", "strategy", "domain", "provider"]

METRIC_PATTERN = r"^[a-z0-9_.-]{1,64}$"
DEFAULT_FEEDBACK_WINDOW_SECONDS = 30 * 86_400.0
# Two-sided 95% quantile of the standard normal distribution.
Z_95 = 1.959963984540054
# Fixed column names that ``group_by`` may select; never user-controlled SQL.
_GROUP_COLUMNS: dict[str, str] = {
    "model": "model",
    "strategy": "strategy",
    "domain": "domain",
    "provider": "provider",
}


class FeedbackKindConflictError(ValueError):
    """Raised when a metric already holds feedback of the other value kind in a scope."""

    def __init__(self, metric: str, existing_kind: FeedbackKind) -> None:
        """Initialize the error.

        Args:
            metric: Metric name of the rejected write.
            existing_kind: Value kind the metric already holds.
        """
        expected = "true or false" if existing_kind == "boolean" else "a number in [0, 1]"
        super().__init__(
            f"metric '{metric}' already holds {existing_kind} feedback for this caller; "
            f"send {expected}, or use another metric name"
        )
        self.metric = metric
        self.existing_kind = existing_kind


@dataclass(frozen=True, slots=True)
class FeedbackScope:
    """Caller that owns requests and their feedback.

    Attributes:
        api_key_id: Non-reversible API key fingerprint (``key_...`` or ``anonymous``).
        tenant_id: Optional ``X-Tenant-Id`` of the caller.
    """

    api_key_id: str
    tenant_id: str | None = None

    @property
    def tenant_key(self) -> str:
        """Return the stored tenant column value (empty when no tenant was sent)."""
        return self.tenant_id or ""


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    """One stored feedback value with the routing snapshot of its request."""

    feedback_id: str
    request_id: str
    metric: str
    source: FeedbackSource
    value: bool | float
    comment: str | None
    model: str
    provider: str
    strategy: str
    domain: str
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class FeedbackWrite:
    """Result of an upsert: the stored record and whether it was newly created."""

    record: FeedbackRecord
    created: bool


@dataclass(frozen=True, slots=True)
class ModelFeedback:
    """Beta-Bernoulli evidence for one model.

    Boolean feedback counts one success or failure per value; a numeric value
    ``v`` in ``[0, 1]`` counts as ``v`` successes and ``1 - v`` failures.
    """

    n: int
    positives: float
    negatives: float

    @property
    def posterior_mean(self) -> float:
        """Return the mean of ``Beta(1 + positives, 1 + negatives)``."""
        return (1.0 + self.positives) / (2.0 + self.positives + self.negatives)


@dataclass(frozen=True, slots=True)
class FeedbackGroup:
    """Aggregate of one ``group_by`` value.

    ``positives`` is set for boolean metrics only. The interval is a Wilson
    score interval for boolean metrics and a normal approximation of the mean
    for numeric ones.
    """

    key: str
    n: int
    mean: float
    ci_low: float
    ci_high: float
    positives: int | None = None


@dataclass(frozen=True, slots=True)
class FeedbackSummary:
    """Per-group feedback aggregates for one metric of one caller scope."""

    metric: str
    group_by: FeedbackGroupBy
    kind: FeedbackKind | None
    n: int
    groups: list[FeedbackGroup]


def wilson_interval(positives: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    Args:
        positives: Number of successes.
        n: Number of trials.
        z: Standard-normal quantile (``Z_95`` for a 95% interval).

    Returns:
        ``(low, high)`` within ``[0, 1]``; ``(0, 1)`` when ``n`` is 0.
    """
    if n <= 0:
        return 0.0, 1.0
    proportion = positives / n
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = (proportion + z_squared / (2 * n)) / denominator
    spread = proportion * (1.0 - proportion) / n + z_squared / (4 * n * n)
    half_width = z * math.sqrt(max(0.0, spread)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def mean_interval(
    total: float, total_squares: float, n: int, z: float = Z_95
) -> tuple[float, float]:
    """Return a normal-approximation interval for the mean of values in ``[0, 1]``.

    Args:
        total: Sum of the values.
        total_squares: Sum of the squared values.
        n: Number of values.
        z: Standard-normal quantile (``Z_95`` for a 95% interval).

    Returns:
        ``mean ± z * s / sqrt(n)`` with the sample standard deviation ``s``,
        clipped to ``[0, 1]``; ``(0, 1)`` below two values, where ``s`` is undefined.
    """
    if n < 2:
        return 0.0, 1.0
    mean = total / n
    variance = max(0.0, (total_squares - n * mean * mean) / (n - 1))
    half_width = z * math.sqrt(variance / n)
    return max(0.0, mean - half_width), min(1.0, mean + half_width)


class FeedbackStore:
    """SQLite store for the request index and request feedback.

    Multiple instances pointing at the same ``db_path`` share durable state,
    like :class:`safety.spend_ledger.SpendLedger`. File databases use SQLite's
    write-ahead log, so summary reads never block the per-request index writes;
    each call opens its own connection, and SQLite serializes writers across
    threads and processes. ``":memory:"`` keeps one locked connection for tests.

    Routing reads per-model totals kept in hourly buckets (``feedback_totals``),
    updated in the same transaction as each upsert, so a ``feedback-aware``
    decision costs a few rows per model however much feedback a caller sent.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        window_seconds: float = DEFAULT_FEEDBACK_WINDOW_SECONDS,
    ) -> None:
        """Initialize the store and ensure the schema exists.

        Args:
            db_path: Filesystem path for SQLite, or ``":memory:"`` for tests.
            window_seconds: How long a request accepts feedback, and how far
                back :meth:`model_feedback` looks.

        Raises:
            ValueError: If ``window_seconds`` is not positive.
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._window_seconds = float(window_seconds)
        self._memory = str(db_path) == ":memory:"
        self._db_path: Path | str = ":memory:" if self._memory else Path(db_path)
        if not self._memory:
            path = Path(self._db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = path
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = self._open() if self._memory else None
        self._initialize()

    @property
    def db_path(self) -> Path | str:
        """Return the SQLite database path (or ``":memory:"``)."""
        return self._db_path

    @property
    def window_seconds(self) -> float:
        """Return the feedback window in seconds."""
        return self._window_seconds

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection: the shared in-memory one (locked) or a fresh file one."""
        if self._connection is not None:
            with self._lock:
                yield self._connection
            return
        connection = self._open()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._session() as connection:
            if not self._memory:
                connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_requests (
                    request_id TEXT PRIMARY KEY,
                    api_key_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_requests_recorded
                    ON feedback_requests(recorded_at);
                CREATE TABLE IF NOT EXISTS feedback (
                    request_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    source TEXT NOT NULL,
                    feedback_id TEXT NOT NULL UNIQUE,
                    api_key_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    value REAL NOT NULL,
                    value_kind TEXT NOT NULL,
                    comment TEXT,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (request_id, metric, source)
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_scope
                    ON feedback(api_key_id, tenant_id, metric, updated_at);
                CREATE TABLE IF NOT EXISTS feedback_totals (
                    api_key_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    model TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    n INTEGER NOT NULL,
                    total REAL NOT NULL,
                    PRIMARY KEY (api_key_id, tenant_id, metric, domain, model, hour)
                );
                """
            )

    def register_request(self, record: AuditRecord, *, recorded_at: float | None = None) -> None:
        """Make a completed request addressable by feedback.

        Index rows older than the feedback window are pruned on the way. The
        first registration of a request id wins, so a request can never change
        owner.

        Args:
            record: Audit record of the completed request.
            recorded_at: Optional unix timestamp; defaults to now.
        """
        stamp = time.time() if recorded_at is None else float(recorded_at)
        with self._session() as connection:
            connection.execute(
                "DELETE FROM feedback_requests WHERE recorded_at < ?",
                (stamp - self._window_seconds,),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO feedback_requests (
                    request_id, api_key_id, tenant_id, model, provider, strategy,
                    domain, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    record.api_key_id or "anonymous",
                    record.tenant_id or "",
                    record.chosen_model,
                    record.provider,
                    record.routing_strategy.value,
                    record.domain.value if record.domain is not None else "",
                    stamp,
                ),
            )
            connection.commit()

    def record(
        self,
        scope: FeedbackScope,
        *,
        request_id: str,
        metric: str,
        value: bool | float,
        source: FeedbackSource = "human",
        comment: str | None = None,
        now: float | None = None,
    ) -> FeedbackWrite | None:
        """Insert or replace the caller's feedback for ``(request_id, metric, source)``.

        Args:
            scope: Caller writing the feedback.
            request_id: Request the feedback is about.
            metric: Metric name.
            value: ``True``/``False``, or a number in ``[0, 1]``.
            source: ``human`` or ``automated``.
            comment: Optional free text, stored but never logged.
            now: Optional unix timestamp; defaults to now.

        Returns:
            The stored record, or ``None`` when the request is unknown, outside
            the feedback window, or belongs to another caller scope.

        Raises:
            ValueError: If a numeric ``value`` is outside ``[0, 1]``.
            FeedbackKindConflictError: If ``metric`` already holds the other
                value kind for this caller scope.
        """
        kind: FeedbackKind = "boolean" if isinstance(value, bool) else "numeric"
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("feedback value must be a boolean or a number in [0, 1]")
        write = _Write(
            scope=scope,
            request_id=request_id,
            metric=metric,
            source=source,
            feedback_id=f"fb_{uuid4().hex}",
            value=numeric,
            kind=kind,
            comment=comment,
            stamp=time.time() if now is None else float(now),
        )
        with self._session() as connection:
            # Take the write lock first so the ownership and kind checks hold
            # until commit, even across processes sharing the file.
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._upsert(connection, write)
            except BaseException:
                connection.rollback()
                raise
            connection.commit()
        if row is None:
            return None
        return FeedbackWrite(
            record=self._row_to_record(row), created=row["feedback_id"] == write.feedback_id
        )

    def _upsert(self, connection: sqlite3.Connection, write: _Write) -> sqlite3.Row | None:
        """Check ownership and value kind, upsert, update totals, return the stored row."""
        scope = write.scope
        target = connection.execute(
            """
            SELECT model, provider, strategy, domain FROM feedback_requests
            WHERE request_id = ? AND api_key_id = ? AND tenant_id = ? AND recorded_at >= ?
            """,
            (
                write.request_id,
                scope.api_key_id,
                scope.tenant_key,
                write.stamp - self._window_seconds,
            ),
        ).fetchone()
        if target is None:
            return None
        # A metric holds one kind per caller, so any existing row tells which.
        existing_kind = connection.execute(
            "SELECT value_kind FROM feedback WHERE api_key_id = ? AND tenant_id = ? "
            "AND metric = ? LIMIT 1",
            (scope.api_key_id, scope.tenant_key, write.metric),
        ).fetchone()
        if existing_kind is not None and existing_kind["value_kind"] != write.kind:
            raise FeedbackKindConflictError(write.metric, existing_kind["value_kind"])
        key = (write.request_id, write.metric, write.source)
        previous = connection.execute(
            "SELECT value, updated_at FROM feedback "
            "WHERE request_id = ? AND metric = ? AND source = ?",
            key,
        ).fetchone()
        connection.execute(
            """
            INSERT INTO feedback (
                request_id, metric, source, feedback_id, api_key_id, tenant_id, value,
                value_kind, comment, model, provider, strategy, domain, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (request_id, metric, source) DO UPDATE SET
                value = excluded.value,
                value_kind = excluded.value_kind,
                comment = excluded.comment,
                updated_at = excluded.updated_at
            """,
            (
                *key,
                write.feedback_id,
                scope.api_key_id,
                scope.tenant_key,
                write.value,
                write.kind,
                write.comment,
                target["model"],
                target["provider"],
                target["strategy"],
                target["domain"],
                write.stamp,
                write.stamp,
            ),
        )
        totals_key = (scope.api_key_id, scope.tenant_key, write.metric, target["domain"])
        if previous is not None:
            self._add_to_totals(
                connection,
                (*totals_key, target["model"], _hour(previous["updated_at"])),
                count=-1,
                total=-float(previous["value"]),
            )
        self._add_to_totals(
            connection,
            (*totals_key, target["model"], _hour(write.stamp)),
            count=1,
            total=write.value,
        )
        row: sqlite3.Row | None = connection.execute(
            "SELECT * FROM feedback WHERE request_id = ? AND metric = ? AND source = ?", key
        ).fetchone()
        return row

    @staticmethod
    def _add_to_totals(
        connection: sqlite3.Connection,
        bucket: tuple[str, str, str, str, str, int],
        *,
        count: int,
        total: float,
    ) -> None:
        """Add ``count`` values summing to ``total`` to one hourly bucket."""
        connection.execute(
            """
            INSERT INTO feedback_totals (
                api_key_id, tenant_id, metric, domain, model, hour, n, total
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (api_key_id, tenant_id, metric, domain, model, hour) DO UPDATE SET
                n = n + excluded.n,
                total = total + excluded.total
            """,
            (*bucket, count, total),
        )
        connection.execute(
            "DELETE FROM feedback_totals WHERE api_key_id = ? AND tenant_id = ? AND metric = ? "
            "AND domain = ? AND model = ? AND hour = ? AND n <= 0",
            bucket,
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FeedbackRecord:
        value = float(row["value"])
        source: FeedbackSource = "automated" if row["source"] == "automated" else "human"
        return FeedbackRecord(
            feedback_id=str(row["feedback_id"]),
            request_id=str(row["request_id"]),
            metric=str(row["metric"]),
            source=source,
            value=value >= 0.5 if row["value_kind"] == "boolean" else value,
            comment=row["comment"],
            model=str(row["model"]),
            provider=str(row["provider"]),
            strategy=str(row["strategy"]),
            domain=str(row["domain"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def summary(
        self,
        scope: FeedbackScope,
        *,
        metric: str,
        group_by: FeedbackGroupBy = "model",
        since: float | None = None,
    ) -> FeedbackSummary:
        """Aggregate the caller's feedback for one metric per ``group_by`` value.

        Groups are ordered by their lower confidence bound (then sample size),
        so a 1-for-1 group does not outrank 45 positives out of 50.

        Args:
            scope: Caller whose feedback is aggregated.
            metric: Metric name.
            group_by: Snapshot column to group by.
            since: Optional unix timestamp; only feedback last written at or
                after it is counted.

        Returns:
            Summary with one entry per group; no groups when nothing matches.

        Raises:
            ValueError: If ``group_by`` is not a supported column.
        """
        column = _GROUP_COLUMNS.get(group_by)
        if column is None:
            raise ValueError(f"unsupported group_by: {group_by}")
        where = "api_key_id = ? AND tenant_id = ? AND metric = ?"
        params: list[object] = [scope.api_key_id, scope.tenant_key, metric]
        if since is not None:
            where += " AND updated_at >= ?"
            params.append(float(since))
        # ``column`` comes from the fixed allowlist above and ``where`` from fixed
        # fragments; every value is bound, so this is not an injection vector.
        query = (
            f"SELECT {column} AS key, COUNT(*) AS n, SUM(value) AS total, "  # noqa: S608  # nosec B608
            "SUM(value * value) AS total_squares, "
            "SUM(value_kind = 'numeric') AS numeric_rows "
            f"FROM feedback WHERE {where} GROUP BY {column}"
        )
        with self._session() as connection:
            rows = connection.execute(query, params).fetchall()
        if not rows:
            return FeedbackSummary(metric=metric, group_by=group_by, kind=None, n=0, groups=[])
        kind: FeedbackKind = (
            "numeric" if any(int(row["numeric_rows"]) for row in rows) else "boolean"
        )
        groups = [self._group(row, kind) for row in rows]
        groups.sort(key=lambda group: (-group.ci_low, -group.n, group.key))
        return FeedbackSummary(
            metric=metric,
            group_by=group_by,
            kind=kind,
            n=sum(group.n for group in groups),
            groups=groups,
        )

    @staticmethod
    def _group(row: sqlite3.Row, kind: FeedbackKind) -> FeedbackGroup:
        n = int(row["n"])
        total = float(row["total"])
        key = str(row["key"]) or "unknown"
        if kind == "boolean":
            positives = round(total)
            low, high = wilson_interval(positives, n)
            return FeedbackGroup(
                key=key, n=n, mean=positives / n, ci_low=low, ci_high=high, positives=positives
            )
        low, high = mean_interval(total, float(row["total_squares"]), n)
        return FeedbackGroup(key=key, n=n, mean=total / n, ci_low=low, ci_high=high)

    def model_feedback(
        self,
        scope: FeedbackScope,
        *,
        metric: str,
        domain: str,
        now: float | None = None,
    ) -> dict[str, ModelFeedback]:
        """Return per-model evidence for one caller, metric, and domain in the window.

        The window is applied in whole hours: feedback written in the hour that
        contains ``now - window_seconds`` still counts.

        Args:
            scope: Caller whose feedback is read.
            metric: Metric name.
            domain: Request domain tag value (for example ``general``).
            now: Optional unix timestamp; defaults to now.

        Returns:
            Evidence keyed by model, for models with at least one value.
        """
        stamp = time.time() if now is None else float(now)
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT model, SUM(n) AS n, SUM(total) AS positives FROM feedback_totals
                WHERE api_key_id = ? AND tenant_id = ? AND metric = ? AND domain = ?
                  AND hour >= ?
                GROUP BY model
                HAVING SUM(n) > 0
                """,
                (
                    scope.api_key_id,
                    scope.tenant_key,
                    metric,
                    domain,
                    _hour(stamp - self._window_seconds),
                ),
            ).fetchall()
        evidence: dict[str, ModelFeedback] = {}
        for row in rows:
            n = int(row["n"])
            # Clamp float drift from repeated replace-and-subtract updates.
            positives = min(float(n), max(0.0, float(row["positives"])))
            evidence[str(row["model"])] = ModelFeedback(
                n=n, positives=positives, negatives=n - positives
            )
        return evidence


@dataclass(frozen=True, slots=True)
class _Write:
    """One validated feedback write."""

    scope: FeedbackScope
    request_id: str
    metric: str
    source: FeedbackSource
    feedback_id: str
    value: float
    kind: FeedbackKind
    comment: str | None
    stamp: float


def _hour(stamp: float) -> int:
    """Return the hourly totals bucket of a unix timestamp."""
    return math.floor(stamp / 3600.0)
