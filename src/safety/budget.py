"""Budget guardrails for routed requests."""


class BudgetExceededError(RuntimeError):
    """Raised when a user or session spend cap is exceeded."""


class BudgetGuardrail:
    """Track spend and enforce hard budget caps."""

    # IEEE-754 accumulation of many small USD costs can land a hair above the
    # configured cap even when every individual charge was intended to fit.
    # Treat overruns within this epsilon as still within budget.
    _EPSILON_USD = 1e-9

    def __init__(self, cap_usd: float) -> None:
        """Initialize the budget guardrail.

        Args:
            cap_usd: Maximum spend per subject.
        """
        self._cap_usd = cap_usd
        self._spend_by_subject: dict[str, float] = {}

    def assert_can_spend(self, subject: str, estimated_cost_usd: float) -> None:
        """Raise when an estimated spend would exceed the cap.

        Args:
            subject: User or session budget key.
            estimated_cost_usd: Estimated next request cost.

        Raises:
            BudgetExceededError: If the spend cap would be exceeded.
        """
        current_spend = self._spend_by_subject.get(subject, 0.0)
        if current_spend + estimated_cost_usd > self._cap_usd + self._EPSILON_USD:
            raise BudgetExceededError(f"budget cap exceeded for {subject}")

    def record_spend(self, subject: str, cost_usd: float) -> None:
        """Record actual spend for a subject.

        Args:
            subject: User or session budget key.
            cost_usd: Cost to record.
        """
        self._spend_by_subject[subject] = self._spend_by_subject.get(subject, 0.0) + cost_usd

    def spent(self, subject: str) -> float:
        """Return recorded spend for a subject.

        Args:
            subject: User or session budget key.

        Returns:
            Spend in USD.
        """
        return self._spend_by_subject.get(subject, 0.0)
