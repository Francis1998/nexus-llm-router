"""Tests for request routing state-machine transitions."""

from __future__ import annotations

import pytest

from router.state import RequestState, RoutingStateMachine, StateTransitionError


def test_state_machine_accepts_primary_success_path() -> None:
    """The normal routing lifecycle reaches the responded terminal state."""

    state_machine = RoutingStateMachine()

    state_machine.transition(RequestState.CLASSIFIED)
    state_machine.transition(RequestState.ROUTED)
    state_machine.transition(RequestState.DISPATCHED)
    state_machine.transition(RequestState.RESPONDED)

    assert state_machine.current_state == RequestState.RESPONDED


def test_state_machine_accepts_fallback_dispatch_path() -> None:
    """Fallback routing can redispatch before responding."""

    state_machine = RoutingStateMachine()

    state_machine.transition(RequestState.CLASSIFIED)
    state_machine.transition(RequestState.ROUTED)
    state_machine.transition(RequestState.DISPATCHED)
    state_machine.transition(RequestState.FALLBACK)
    state_machine.transition(RequestState.DISPATCHED)
    state_machine.transition(RequestState.RESPONDED)

    assert state_machine.current_state == RequestState.RESPONDED


def test_state_machine_rejects_skipped_lifecycle_steps() -> None:
    """Requests cannot jump directly from received to responded."""

    state_machine = RoutingStateMachine()

    with pytest.raises(StateTransitionError, match="RECEIVED -> RESPONDED"):
        state_machine.transition(RequestState.RESPONDED)

    assert state_machine.current_state == RequestState.RECEIVED


@pytest.mark.parametrize("terminal_state", [RequestState.RESPONDED, RequestState.FAILED])
def test_state_machine_terminal_states_reject_followup_transitions(
    terminal_state: RequestState,
) -> None:
    """Terminal states reject additional routing transitions."""

    state_machine = RoutingStateMachine()
    state_machine.transition(RequestState.FAILED)
    if terminal_state == RequestState.RESPONDED:
        state_machine = RoutingStateMachine()
        state_machine.transition(RequestState.CLASSIFIED)
        state_machine.transition(RequestState.ROUTED)
        state_machine.transition(RequestState.DISPATCHED)
        state_machine.transition(RequestState.RESPONDED)

    with pytest.raises(StateTransitionError):
        state_machine.transition(RequestState.FAILED)

    assert state_machine.current_state == terminal_state
