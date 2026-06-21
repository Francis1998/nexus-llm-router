"""State machine primitives for request routing."""

from enum import StrEnum


class RequestState(StrEnum):
    """Lifecycle states for a routed request."""

    RECEIVED = "RECEIVED"
    CLASSIFIED = "CLASSIFIED"
    ROUTED = "ROUTED"
    DISPATCHED = "DISPATCHED"
    RESPONDED = "RESPONDED"
    FALLBACK = "FALLBACK"
    FAILED = "FAILED"


class StateTransitionError(ValueError):
    """Raised when a request attempts an invalid state transition."""


class RoutingStateMachine:
    """Validate Observe-Decide-Act lifecycle transitions."""

    _allowed_transitions: dict[RequestState, set[RequestState]] = {
        RequestState.RECEIVED: {RequestState.CLASSIFIED, RequestState.FAILED},
        RequestState.CLASSIFIED: {RequestState.ROUTED, RequestState.FAILED},
        RequestState.ROUTED: {RequestState.DISPATCHED, RequestState.FAILED},
        RequestState.DISPATCHED: {
            RequestState.RESPONDED,
            RequestState.FALLBACK,
            RequestState.FAILED,
        },
        RequestState.FALLBACK: {
            RequestState.DISPATCHED,
            RequestState.RESPONDED,
            RequestState.FAILED,
        },
        RequestState.RESPONDED: set(),
        RequestState.FAILED: set(),
    }

    def __init__(self) -> None:
        """Initialize the state machine at the received state."""
        self.current_state = RequestState.RECEIVED

    def transition(self, next_state: RequestState) -> None:
        """Move to the next lifecycle state.

        Args:
            next_state: The target lifecycle state.

        Raises:
            StateTransitionError: If the transition is not allowed.
        """
        allowed_states = self._allowed_transitions[self.current_state]
        if next_state not in allowed_states:
            message = f"invalid transition {self.current_state.value} -> {next_state.value}"
            raise StateTransitionError(message)
        self.current_state = next_state
