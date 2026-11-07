"""Gateway errors with OpenAI-compatible error envelopes."""

from __future__ import annotations

from collections.abc import Sequence

from router.engine import ProviderRejectedRequestError, RoutingFailedError
from router.schemas import ChatMessage
from safety.budget import BudgetExceededError
from safety.rate_limiter import RateLimitExceededError

ROUTING_ERRORS = (RateLimitExceededError, BudgetExceededError, RoutingFailedError)


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


def routing_error(
    exception: RateLimitExceededError | BudgetExceededError | RoutingFailedError,
) -> GatewayError:
    """Map a router failure to its HTTP status and error fields.

    Args:
        exception: Failure raised by ``NexusRouter``.

    Returns:
        429 for rate limits, 402 for budget caps, the provider's 4xx when it
        rejected the request itself, 503 when every provider failed.
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
