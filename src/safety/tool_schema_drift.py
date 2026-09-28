"""Tool schema drift gate.

Compares declared tool-schema hash to observed hash and emits advisory
bands. Distinct from ``OutputSchemaGuard`` (response JSON schema) and
``JsonSchemaRetryAdvisor`` (retry). Closes OpenRouter / LiteLLM / Portkey
tool-schema drift gaps. Works with GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2. Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolSchemaDriftAdvice:
    """Tool schema drift advice."""

    tool_name: str
    declared_hash: str
    observed_hash: str
    band: str


class ToolSchemaDriftGate:
    """Gate requests whose tool schema hash drifted."""

    def advise(
        self,
        *,
        tool_name: str,
        declared_hash: str,
        observed_hash: str,
    ) -> ToolSchemaDriftAdvice:
        """Return band for declared vs observed schema hash.

        Args:
            tool_name: Non-empty tool name.
            declared_hash: Non-empty declared schema hash.
            observed_hash: Non-empty observed schema hash.

        Returns:
            ToolSchemaDriftAdvice with ``stable`` / ``soft_drift`` / ``hard_drift``.
        """

        name = tool_name.strip()
        declared = declared_hash.strip()
        observed = observed_hash.strip()
        if not name:
            raise ValueError("tool_name must be non-empty")
        if not declared:
            raise ValueError("declared_hash must be non-empty")
        if not observed:
            raise ValueError("observed_hash must be non-empty")

        if declared == observed:
            band = "stable"
        elif declared[:8] == observed[:8]:
            band = "soft_drift"
        else:
            band = "hard_drift"
        return ToolSchemaDriftAdvice(
            tool_name=name,
            declared_hash=declared,
            observed_hash=observed,
            band=band,
        )
