"""Prometheus metrics for routing lifecycle."""

import threading

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

router_requests_total = Counter(
    "router_requests_total",
    "Total router requests.",
    ["strategy", "state"],
)
router_cost_usd_total = Counter(
    "router_cost_usd_total",
    "Total routed completion cost in USD.",
    ["provider", "model"],
)
router_latency_seconds = Histogram(
    "router_latency_seconds",
    "End-to-end router latency in seconds.",
    ["strategy", "provider"],
)
provider_error_rate = Counter(
    "provider_error_rate",
    "Provider errors by provider and model.",
    ["provider", "model"],
)
response_cache_hits_total = Counter(
    "response_cache_hits_total",
    "Exact-match response cache hits.",
)
response_cache_misses_total = Counter(
    "response_cache_misses_total",
    "Exact-match response cache misses.",
)
router_feedback_total = Counter(
    "router_feedback_total",
    "Request feedback writes (new and updated values) by metric and served model.",
    ["metric", "model"],
)


class BoundedLabelValues:
    """Admit at most ``limit`` distinct label values; later ones share one label.

    Client-chosen strings such as feedback metric names would otherwise create
    an unbounded number of Prometheus series.
    """

    def __init__(self, limit: int, overflow: str = "other") -> None:
        """Initialize the label set.

        Args:
            limit: Distinct values that keep their own label.
            overflow: Label used for every value beyond ``limit``.
        """
        self._limit = limit
        self._overflow = overflow
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def label(self, value: str) -> str:
        """Return ``value`` while it fits the limit, otherwise the overflow label.

        Args:
            value: Candidate label value.

        Returns:
            The label value to record.
        """
        with self._lock:
            if value in self._seen:
                return value
            if len(self._seen) < self._limit:
                self._seen.add(value)
                return value
        return self._overflow


def metrics_response() -> Response:
    """Return Prometheus metrics response.

    Returns:
        Starlette response containing Prometheus exposition data.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
