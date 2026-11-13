"""Gateway errors with OpenAI- and Anthropic-compatible error envelopes."""

from __future__ import annotations

from collections.abc import Sequence

from router.engine import ProviderRejectedRequestError, RoutingFailedError, UnpricedModelError
from router.schemas import ChatMessage
from safety.budget import BudgetExceededError
from safety.rate_limiter import RateLimitExceededError

ROUTING_ERRORS = (RateLimitExceededError, BudgetExceededError, RoutingFailedError)

# Anthropic error ``type`` for each HTTP status nexus can return. Nexus keeps its
# usual statuses on every endpoint (503 rather than Anthropic's 529 when every
# provider failed), so only the envelope changes on the Anthropic paths.
_ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    503: "overloaded_error",
    504: "timeout_error",
    529: "overloaded_error",
}


def anthropic_error_type(status_code: int) -> str:
    """Return the Anthropic error ``type`` for an HTTP status.

    Args:
        status_code: HTTP status of the error response.

    Returns:
        The documented Anthropic type; other 4xx map to ``invalid_request_error``
        and other 5xx to ``api_error``, as the Anthropic API does.
    """
    if status_code in _ANTHROPIC_ERROR_TYPES:
        return _ANTHROPIC_ERROR_TYPES[status_code]
    return "api_error" if status_code >= 500 else "invalid_request_error"


def anthropic_error_body(status_code: int, message: str) -> dict[str, str | dict[str, str]]:
    """Return the Anthropic ``{"type": "error", "error": {...}}`` envelope.

    Args:
        status_code: HTTP status of the error response.
        message: Human-readable error message.

    Returns:
        JSON-serializable error body.
    """
    return {
        "type": "error",
        "error": {"type": anthropic_error_type(status_code), "message": message},
    }


class GatewayError(Exception):
    """Request failure carrying an HTTP status and OpenAI-style error fields."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            status_code: HTTP status returned to the client.
            message: Human-readable error message.
            error_type: OpenAI-style error ``type``.
            code: Optional machine-readable error ``code``.
            param: Optional request parameter that caused the error.
        """
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param

    def to_openai_body(self) -> dict[str, dict[str, str | None]]:
        """Return the OpenAI ``{"error": {...}}`` envelope.

        Returns:
            JSON-serializable error body.
        """
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }

    def to_anthropic_body(self) -> dict[str, str | dict[str, str]]:
        """Return the Anthropic ``{"type": "error", "error": {...}}`` envelope.

        The Anthropic ``type`` follows the HTTP status (see
        :func:`anthropic_error_type`); Anthropic errors have no ``param`` or
        ``code``, so the message names the offending field instead.

        Returns:
            JSON-serializable error body.
        """
        return anthropic_error_body(self.status_code, self.message)


def routing_error(
    exception: RateLimitExceededError | BudgetExceededError | RoutingFailedError,
) -> GatewayError:
    """Map a router failure to its HTTP status and error fields.

    Args:
        exception: Failure raised by ``NexusRouter``.

    Returns:
        429 for rate limits, 402 when the budget guardrail refused every
        candidate, the provider's 4xx when it rejected the request itself, 500
        (``model_price_unknown``) when no candidate model has a price, and 503
        when every provider attempt failed.
    """
    if isinstance(exception, RateLimitExceededError):
        return GatewayError(
            429, str(exception), error_type="rate_limit_error", code="rate_limit_exceeded"
        )
    if isinstance(exception, BudgetExceededError):
        return GatewayError(
            402, str(exception), error_type="budget_exceeded", code="budget_exceeded"
        )
    if isinstance(exception, ProviderRejectedRequestError):
        return GatewayError(exception.status_code, str(exception), code="provider_rejected_request")
    if isinstance(exception, UnpricedModelError):
        return GatewayError(
            500, str(exception), error_type="server_error", code="model_price_unknown"
        )
    return GatewayError(503, str(exception), error_type="routing_failed", code="routing_failed")


def conversation_error(messages: Sequence[ChatMessage], param: str) -> GatewayError | None:
    """Return a 400 for a conversation that not every provider can answer.

    Every provider rejects an empty conversation, and Anthropic and Gemini also
    reject one with only system instructions (they send system text apart from
    the turns), so routing it would fail or succeed depending on the provider.

    Args:
        messages: Router messages built from the request.
        param: Request parameter holding the conversation (``messages`` or ``input``).

    Returns:
        The OpenAI-style error to return, or ``None`` when the conversation is routable.
    """
    if not messages:
        return GatewayError(
            400,
            f"Invalid '{param}': empty array. Expected an array with minimum length 1, "
            "but got an empty array instead.",
            code="empty_array",
            param=param,
        )
    if all(message.role == "system" for message in messages):
        return GatewayError(
            400,
            f"Invalid '{param}': expected at least one user or assistant message, "
            "but got only system instructions.",
            code="invalid_value",
            param=param,
        )
    return None
