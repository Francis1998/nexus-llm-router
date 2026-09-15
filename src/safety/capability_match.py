"""Match request capability needs to model capability flags."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

CapabilityFit = Literal["fit", "partial", "mismatch"]

KNOWN_CAPABILITIES = frozenset(
    {
        "vision",
        "tools",
        "json_schema",
        "long_context",
    }
)


@dataclass(frozen=True, slots=True)
class CapabilityMatchAdvice:
    """Structured capability fit advisory for one model vs request needs."""

    model: str
    required: frozenset[str]
    supported: frozenset[str]
    missing: frozenset[str]
    matched: frozenset[str]
    verdict: CapabilityFit
    advisory: str


class ModelCapabilityMatcher:
    """Match request needs to model capability flags (fit / partial / mismatch).

    Closes the OpenRouter / LiteLLM / Portkey *capability catalog match* gap
    for offline Nexus gateways serving GPT-5.5 / Claude Sonnet 4.6 /
    Gemini 3.x / Kimi K2: given request needs (vision / tools / json_schema /
    long_context) and a model's capability flags, return whether the model is
    a full fit, a partial fit, or a mismatch.

    Distinct from ``ContextWindowFitAdvisor`` (token-count vs context limit
    bands) and ``OutputJsonSchemaGuard`` (post-check JSON required keys).
    This matcher never rejects traffic — callers use ``verdict`` to prefer
    alternate models or warn operators.
    """

    def __init__(
        self,
        *,
        catalog: Mapping[str, Iterable[str]] | None = None,
        known_capabilities: Iterable[str] | None = None,
    ) -> None:
        """Initialize with an optional model capability catalog.

        Args:
            catalog: Optional ``model_id -> capability flags`` map.
            known_capabilities: Allowed capability names. Defaults to
                ``vision`` / ``tools`` / ``json_schema`` / ``long_context``.

        Raises:
            ValueError: If catalog keys/flags or known capabilities are empty.
        """
        if known_capabilities is None:
            known = KNOWN_CAPABILITIES
        else:
            known = frozenset(known_capabilities)
            if not known or any(not name for name in known):
                raise ValueError("known_capabilities must be non-empty strings")
        self._known = known
        parsed: dict[str, frozenset[str]] = {}
        if catalog is not None:
            for model, caps in catalog.items():
                if not model:
                    raise ValueError("catalog model keys must be non-empty")
                flags = frozenset(caps)
                if any(not flag for flag in flags):
                    raise ValueError("catalog capability flags must be non-empty")
                unknown = flags - known
                if unknown:
                    raise ValueError(f"unknown capability flags for {model}: {sorted(unknown)}")
                parsed[model] = flags
        self._catalog = parsed

    @property
    def known_capabilities(self) -> frozenset[str]:
        """Return the configured known capability flag set."""
        return self._known

    def capabilities_for(self, model: str) -> frozenset[str]:
        """Return catalog capabilities for ``model`` (empty if unknown)."""
        if not model:
            raise ValueError("model must be non-empty")
        return self._catalog.get(model, frozenset())

    def match(
        self,
        *,
        model: str,
        needs: Iterable[str],
        capabilities: Iterable[str] | None = None,
    ) -> CapabilityMatchAdvice:
        """Return fit / partial / mismatch for ``model`` vs ``needs``.

        Args:
            model: Model identifier (non-empty).
            needs: Required capability flags for the request.
            capabilities: Optional per-call capability override. When omitted,
                catalog capabilities for ``model`` are used.

        Returns:
            Immutable ``CapabilityMatchAdvice`` with matched / missing sets
            and a ``fit`` / ``partial`` / ``mismatch`` verdict.

        Raises:
            ValueError: If model/needs/capabilities are invalid.
        """
        if not model:
            raise ValueError("model must be non-empty")
        required = frozenset(needs)
        if any(not flag for flag in required):
            raise ValueError("needs entries must be non-empty")
        unknown_needs = required - self._known
        if unknown_needs:
            raise ValueError(f"unknown needs: {sorted(unknown_needs)}")

        if capabilities is None:
            supported = self._catalog.get(model, frozenset())
        else:
            supported = frozenset(capabilities)
            if any(not flag for flag in supported):
                raise ValueError("capabilities entries must be non-empty")
            unknown_caps = supported - self._known
            if unknown_caps:
                raise ValueError(f"unknown capabilities: {sorted(unknown_caps)}")

        matched = required & supported
        missing = required - supported

        if not required:
            verdict: CapabilityFit = "fit"
            advisory = (
                f"fit: model={model} has no required capabilities (supported={sorted(supported)})"
            )
        elif not missing:
            verdict = "fit"
            advisory = f"fit: model={model} supports all required capabilities {sorted(required)}"
        elif matched:
            verdict = "partial"
            advisory = (
                f"partial: model={model} matches {sorted(matched)} but missing {sorted(missing)}"
            )
        else:
            verdict = "mismatch"
            advisory = (
                f"mismatch: model={model} missing all required capabilities "
                f"{sorted(required)} (supported={sorted(supported)})"
            )

        return CapabilityMatchAdvice(
            model=model,
            required=required,
            supported=supported,
            missing=missing,
            matched=matched,
            verdict=verdict,
            advisory=advisory,
        )

    def models(self) -> list[str]:
        """Return catalog model ids in insertion order."""
        return list(self._catalog.keys())
