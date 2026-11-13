"""Request feedback API: ``POST /v1/feedback`` and ``GET /v1/feedback/summary``.

Translates between the HTTP schemas and :class:`router.feedback.FeedbackStore`
and renders failures as OpenAI-style error envelopes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from api.errors import GatewayError
from api.schemas import FeedbackObject, FeedbackSummaryGroup, FeedbackSummaryResponse
from router.feedback import FeedbackKindConflictError, FeedbackRecord, FeedbackSummary

FEEDBACK_PATH = "/v1/feedback"
FEEDBACK_SUMMARY_PATH = "/v1/feedback/summary"
_VALUE_MESSAGE = "expected true, false, or a number between 0 and 1"


def is_feedback_path(path: str) -> bool:
    """Return whether ``path`` is one of the feedback endpoints.

    Args:
        path: Request URL path.

    Returns:
        ``True`` for ``/v1/feedback`` and ``/v1/feedback/summary``.
    """
    return path in (FEEDBACK_PATH, FEEDBACK_SUMMARY_PATH)


def validation_error(errors: Sequence[Any]) -> GatewayError:
    """Convert FastAPI request-validation errors into an OpenAI-style 422.

    Args:
        errors: ``RequestValidationError.errors()`` entries.

    Returns:
        Gateway error naming the first invalid parameter.
    """
    first = errors[0] if errors and isinstance(errors[0], Mapping) else {}
    error_type = first.get("type")
    if error_type == "json_invalid":
        return GatewayError(422, "We could not parse the JSON body of your request.")
    location = [part for part in first.get("loc", ()) if part not in ("body", "query")]
    param = str(location[0]) if location else None
    if param is None:
        detail = str(first.get("msg", "invalid request"))
        return GatewayError(422, f"Invalid request: {detail}.", code="invalid_value")
    if error_type == "missing":
        return GatewayError(
            422,
            f"Missing required parameter: '{param}'.",
            code="missing_required_parameter",
            param=param,
        )
    if error_type == "extra_forbidden":
        return GatewayError(
            422,
            f"Unrecognized request argument supplied: {param}",
            code="unknown_parameter",
            param=param,
        )
    detail = _VALUE_MESSAGE if param == "value" else str(first.get("msg", "invalid value"))
    return GatewayError(
        422, f"Invalid value for '{param}': {detail}.", code="invalid_value", param=param
    )


def request_not_found(request_id: str) -> GatewayError:
    """Return the 404 for an unknown, expired, or foreign request id.

    The message is identical in all three cases so a caller cannot tell
    whether another tenant's request exists.

    Args:
        request_id: Request id from the feedback body.

    Returns:
        OpenAI-style ``request_not_found`` error.
    """
    return GatewayError(
        404,
        f"No request found with id '{request_id}' for this API key and tenant.",
        param="request_id",
        code="request_not_found",
    )


def kind_conflict(error: FeedbackKindConflictError) -> GatewayError:
    """Return the 409 for a value of the other kind than the metric already holds.

    Args:
        error: Conflict raised by the store.

    Returns:
        OpenAI-style ``feedback_kind_conflict`` error.
    """
    return GatewayError(409, str(error), param="value", code="feedback_kind_conflict")


def since_out_of_range() -> GatewayError:
    """Return the 422 for a ``since`` that cannot be converted to UTC.

    Returns:
        OpenAI-style ``invalid_value`` error for ``since``.
    """
    return GatewayError(
        422,
        "Invalid value for 'since': the timestamp is out of range.",
        code="invalid_value",
        param="since",
    )


def feedback_object(record: FeedbackRecord) -> FeedbackObject:
    """Render a stored record as the ``feedback`` object.

    Args:
        record: Stored feedback.

    Returns:
        API object including the routing snapshot and the comment.
    """
    return FeedbackObject(
        id=record.feedback_id,
        request_id=record.request_id,
        metric=record.metric,
        value=record.value,
        source=record.source,
        comment=record.comment,
        model=record.model,
        provider=record.provider,
        strategy=record.strategy,
        domain=record.domain,
        created_at=int(record.created_at),
        updated_at=int(record.updated_at),
    )


def as_utc(since: datetime | None) -> datetime | None:
    """Return ``since`` as an aware UTC datetime; naive values are read as UTC.

    Args:
        since: Parsed ``since`` query parameter.

    Returns:
        UTC datetime, or ``None``.

    Raises:
        OverflowError: If the converted timestamp falls outside ``datetime``'s range.
    """
    if since is None:
        return None
    if since.tzinfo is None:
        return since.replace(tzinfo=UTC)
    return since.astimezone(UTC)


def summary_response(summary: FeedbackSummary, since: datetime | None) -> FeedbackSummaryResponse:
    """Render a store summary as the ``feedback.summary`` object.

    Args:
        summary: Aggregates from the store.
        since: Normalized ``since`` filter echoed back to the caller.

    Returns:
        API summary payload.
    """
    interval: Literal["wilson", "normal"] | None = None
    if summary.kind is not None:
        interval = "wilson" if summary.kind == "boolean" else "normal"
    return FeedbackSummaryResponse(
        metric=summary.metric,
        group_by=summary.group_by,
        kind=summary.kind,
        interval=interval,
        since=since,
        n=summary.n,
        groups=[
            FeedbackSummaryGroup(
                key=group.key,
                n=group.n,
                positives=group.positives,
                mean=group.mean,
                ci_low=group.ci_low,
                ci_high=group.ci_high,
            )
            for group in summary.groups
        ],
    )
