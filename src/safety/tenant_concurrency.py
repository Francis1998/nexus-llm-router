"""Per-tenant in-flight concurrency slot guard (advisory / hard)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

SlotBand = Literal["ok", "near", "full"]


@dataclass(frozen=True, slots=True)
class TenantConcurrencySnapshot:
    """In-flight concurrency view for one tenant."""

    tenant_id: str
    in_flight: int
    max_slots: int
    band: SlotBand
    remaining: int
    advisory: str
    blocked: bool


class TenantConcurrencyExceededError(RuntimeError):
    """Raised by ``acquire`` when hard gate blocks a full tenant."""

    def __init__(self, snapshot: TenantConcurrencySnapshot) -> None:
        self.snapshot = snapshot
        super().__init__(
            f"tenant concurrency full for {snapshot.tenant_id}: "
            f"in_flight={snapshot.in_flight} max_slots={snapshot.max_slots}"
        )


class TenantConcurrencySlotGuard:
    """Track per-tenant in-flight request slots with ok/near/full bands.

    Closes the Portkey / LiteLLM / OpenRouter *per-tenant concurrency* gap for
    offline Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
    Kimi K2. Distinct from ``TenantRateLimiter`` (request rate) and
    ``TenantSpendQuotaGuard`` (spend). Optional ``hard_gate`` blocks acquire
    when full.
    """

    def __init__(
        self,
        *,
        max_slots: int = 8,
        near_ratio: float = 0.75,
        hard_gate: bool = False,
    ) -> None:
        if max_slots < 1:
            raise ValueError("max_slots must be >= 1")
        if not 0.0 < near_ratio < 1.0:
            raise ValueError("near_ratio must be in (0, 1)")
        self._max = int(max_slots)
        self._near = float(near_ratio)
        self._hard_gate = bool(hard_gate)
        self._inflight: dict[str, int] = {}
        self._lock = threading.Lock()

    def _snap(self, tenant_id: str, n: int) -> TenantConcurrencySnapshot:
        ratio = n / float(self._max)
        if n >= self._max:
            band: SlotBand = "full"
            blocked = True
            advisory = f"full: in_flight={n} >= max_slots={self._max} for tenant_id={tenant_id}"
        elif ratio >= self._near:
            band = "near"
            blocked = False
            advisory = (
                f"near: in_flight={n}/{self._max} "
                f"(near_ratio={self._near}) for tenant_id={tenant_id}"
            )
        else:
            band = "ok"
            blocked = False
            advisory = f"ok: in_flight={n}/{self._max} for tenant_id={tenant_id}"
        return TenantConcurrencySnapshot(
            tenant_id=tenant_id,
            in_flight=n,
            max_slots=self._max,
            band=band,
            remaining=max(0, self._max - n),
            advisory=advisory,
            blocked=blocked,
        )

    def acquire(self, tenant_id: str) -> TenantConcurrencySnapshot:
        """Increment in-flight slots for ``tenant_id``."""
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        with self._lock:
            n = self._inflight.get(tenant_id, 0)
            if self._hard_gate and n >= self._max:
                snap = self._snap(tenant_id, n)
                raise TenantConcurrencyExceededError(snap)
            n += 1
            self._inflight[tenant_id] = n
            return self._snap(tenant_id, n)

    def release(self, tenant_id: str) -> TenantConcurrencySnapshot | None:
        """Decrement in-flight slots; return snapshot or None if unknown."""
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        with self._lock:
            if tenant_id not in self._inflight:
                return None
            n = max(0, self._inflight[tenant_id] - 1)
            if n == 0:
                del self._inflight[tenant_id]
                return self._snap(tenant_id, 0)
            self._inflight[tenant_id] = n
            return self._snap(tenant_id, n)

    def snapshot(self, tenant_id: str) -> TenantConcurrencySnapshot | None:
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        with self._lock:
            if tenant_id not in self._inflight:
                return None
            return self._snap(tenant_id, self._inflight[tenant_id])
