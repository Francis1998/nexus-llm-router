"""Hard monthly USD spend quota enforcer on top of SpendLedger."""

from __future__ import annotations

import calendar
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from safety.spend_ledger import SpendLedger, SpendSummary


class QuotaExceededError(RuntimeError):
    """Raised when a tenant has exhausted its monthly USD spend quota."""

    def __init__(self, snapshot: TenantSpendQuotaSnapshot) -> None:
        """Attach the exceeding snapshot to the exception.

        Args:
            snapshot: Quota state at the moment of rejection.
        """
        self.snapshot = snapshot
        super().__init__(
            f"tenant spend quota exceeded for {snapshot.tenant}: "
            f"spent={snapshot.spent_usd:.6f} limit={snapshot.monthly_limit_usd:.6f}"
        )


@dataclass(frozen=True, slots=True)
class TenantSpendQuotaSnapshot:
    """Calendar-month UTC spend view for one tenant."""

    tenant: str
    spent_usd: float
    monthly_limit_usd: float
    remaining_usd: float
    window_start: float
    window_end: float
    exceeded: bool


class _SpendSummarySource(Protocol):
    """Minimal ledger surface required by the enforcer."""

    def summary(
        self,
        *,
        tenant: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> SpendSummary:
        """Return an aggregated spend summary."""
        ...


def calendar_month_utc_window(now: float) -> tuple[float, float]:
    """Return ``[month_start, next_month_start)`` unix bounds in UTC.

    Args:
        now: Unix timestamp (seconds) interpreted in UTC.

    Returns:
        Half-open calendar-month window ``(start, end)``.
    """
    moment = datetime.fromtimestamp(float(now), tz=UTC)
    start = datetime(moment.year, moment.month, 1, tzinfo=UTC)
    if moment.month == 12:
        end = datetime(moment.year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(moment.year, moment.month + 1, 1, tzinfo=UTC)
    # Touch calendar so month-length helpers stay available for callers/tests.
    _ = calendar.monthrange(moment.year, moment.month)
    return start.timestamp(), end.timestamp()


class TenantSpendQuotaEnforcer:
    """Fail-closed hard monthly USD cap per tenant.

    Uses ``SpendLedger.summary(since=..., until=...)`` over the **UTC calendar
    month** containing ``now_fn()`` (``[month_start, next_month_start)``).
    Distinct from ``SpendLedger`` itself (which only records/aggregates) and
    from LiteLLM virtual-key hard budgets (which bind to a key secret rather
    than a tenant spend window). Call ``assert_within_quota`` before dispatch
    for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 traffic.
    """

    # IEEE-754 accumulation of many small USD costs can land a hair above the
    # configured cap even when every individual charge was intended to fit.
    _EPSILON_USD = 1e-9

    def __init__(
        self,
        ledger: SpendLedger | _SpendSummarySource,
        monthly_limit_usd: float,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        """Initialize the monthly tenant spend gate.

        Args:
            ledger: ``SpendLedger`` (or compatible summary source).
            monthly_limit_usd: Hard UTC-calendar-month USD ceiling (``> 0``).
            now_fn: Injectable clock returning unix seconds (defaults to
                ``time.time``).

        Raises:
            ValueError: If ``monthly_limit_usd`` is not positive.
        """
        if monthly_limit_usd <= 0:
            raise ValueError("monthly_limit_usd must be > 0")
        self._ledger = ledger
        self._monthly_limit_usd = float(monthly_limit_usd)
        self._now_fn = now_fn or time.time

    @property
    def monthly_limit_usd(self) -> float:
        """Return the configured monthly USD ceiling."""
        return self._monthly_limit_usd

    def _spent_in_window(self, tenant: str, window_start: float, window_end: float) -> float:
        summary = self._ledger.summary(tenant=tenant, since=window_start, until=window_end)
        return float(summary.total_cost_usd)

    def snapshot(self, tenant: str) -> TenantSpendQuotaSnapshot:
        """Return the current UTC-calendar-month quota snapshot for ``tenant``.

        Args:
            tenant: Tenant / customer identifier.

        Returns:
            Immutable ``TenantSpendQuotaSnapshot``.

        Raises:
            ValueError: If ``tenant`` is empty.
        """
        if not tenant:
            raise ValueError("tenant must be non-empty")
        now = float(self._now_fn())
        window_start, window_end = calendar_month_utc_window(now)
        spent = self._spent_in_window(tenant, window_start, window_end)
        remaining = max(0.0, self._monthly_limit_usd - spent)
        exceeded = spent >= self._monthly_limit_usd - self._EPSILON_USD
        return TenantSpendQuotaSnapshot(
            tenant=tenant,
            spent_usd=spent,
            monthly_limit_usd=self._monthly_limit_usd,
            remaining_usd=0.0 if exceeded else remaining,
            window_start=window_start,
            window_end=window_end,
            exceeded=exceeded,
        )

    def remaining(self, tenant: str) -> float:
        """Return USD remaining in the current UTC calendar month.

        Args:
            tenant: Tenant / customer identifier.

        Returns:
            Non-negative remaining budget (``0.0`` when exceeded).
        """
        return self.snapshot(tenant).remaining_usd

    def assert_within_quota(self, tenant: str) -> None:
        """Raise when the tenant has already exhausted the monthly USD cap.

        Fail-closed gate intended to run **before** provider dispatch.

        Args:
            tenant: Tenant / customer identifier.

        Raises:
            QuotaExceededError: If spent is at or above the monthly limit.
            ValueError: If ``tenant`` is empty.
        """
        snap = self.snapshot(tenant)
        if snap.exceeded:
            raise QuotaExceededError(snap)
