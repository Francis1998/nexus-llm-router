"""Prometheus metrics for routing lifecycle."""

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


def metrics_response() -> Response:
    """Return Prometheus metrics response.

    Returns:
        Starlette response containing Prometheus exposition data.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
