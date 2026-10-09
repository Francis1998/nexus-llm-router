"""ToolLoopDepthGateAdvisor.

Advises loop_depth vs budgets.
Closes the LiteLLM/vLLM/OpenAI-compatible tool-loop depth gates gap. Distinct from
`ToolArgSizeForesightAdvisor` and `ToolSchemaDriftAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolLoopDepthGateAdvice:
    """ToolLoopDepthGateAdvisor advice."""

    request_id: str
    loop_depth: float
    soft_limit: float
    hard_limit: float
    band: str


class ToolLoopDepthGateAdvisor:
    """Advise loop_depth bands."""

    def advise(
        self,
        *,
        request_id: str,
        loop_depth: float,
        soft_limit: float = 3.0,
        hard_limit: float = 8.0,
    ) -> ToolLoopDepthGateAdvice:
        """Return loop_depth band.

        Args:
            request_id: Non-empty request id.
            loop_depth: Observed ratio/value (``>= 0``).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if loop_depth < 0:
            raise ValueError("loop_depth must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if loop_depth >= hard_limit:
            band = "breach"
        elif loop_depth >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return ToolLoopDepthGateAdvice(
            request_id=rid,
            loop_depth=float(loop_depth),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
