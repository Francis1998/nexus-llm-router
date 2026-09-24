"""Provider exploration epsilon advisor.

Advises exploit / explore / forced_explore bands from an epsilon schedule for
provider selection. Closes the OpenRouter / LiteLLM / Portkey exploration vs
sticky-routing gap. Distinct from ``StickyProviderAffinity`` and
``ProviderCanaryAdvisor``. Works with GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2. Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderExplorationAdvice:
    """Epsilon-greedy exploration advice."""

    epsilon: float
    sample: float
    band: str
    explore: bool


class ProviderExplorationEpsilonAdvisor:
    """Advise explore vs exploit from epsilon and a unit sample."""

    def advise(self, *, epsilon: float, sample: float) -> ProviderExplorationAdvice:
        """Return exploration band.

        Args:
            epsilon: Exploration probability in ``[0, 1]``.
            sample: Unit-interval draw in ``[0, 1]`` (caller-supplied RNG).

        Returns:
            ProviderExplorationAdvice with band ``exploit`` / ``explore`` /
            ``forced_explore``.
        """

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 <= sample <= 1.0:
            raise ValueError("sample must be in [0, 1]")
        if epsilon >= 1.0:
            band = "forced_explore"
            explore = True
        elif sample < epsilon:
            band = "explore"
            explore = True
        else:
            band = "exploit"
            explore = False
        return ProviderExplorationAdvice(
            epsilon=float(epsilon),
            sample=float(sample),
            band=band,
            explore=explore,
        )
