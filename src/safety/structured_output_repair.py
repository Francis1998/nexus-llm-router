"""Structured output repair advisor (repair / accept / fail).

Advises whether invalid model JSON should be repaired, accepted, or failed
against a required key set. Closes the LiteLLM / Portkey / OpenRouter
structured-output repair gap. Distinct from ``JsonSchemaRetryGuard`` (retry
loops) and ``OutputSchemaGuard`` (schema hard fail). Works with GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2. Never performs network I/O.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

RepairBand = Literal["accept", "repair", "fail"]


@dataclass(frozen=True, slots=True)
class StructuredOutputRepairAdvice:
    """Repair advice for one structured output payload."""

    request_id: str
    band: RepairBand
    missing_keys: tuple[str, ...]
    parse_ok: bool
    advisory: str


class StructuredOutputRepairAdvisor:
    """Advise accept/repair/fail for structured JSON outputs."""

    def advise(
        self,
        request_id: str,
        *,
        payload: str,
        required_keys: tuple[str, ...] | list[str],
    ) -> StructuredOutputRepairAdvice:
        """Return repair advice for ``payload``.

        Args:
            request_id: Non-empty request id.
            payload: Raw model output string.
            required_keys: Keys that must be present in a JSON object.

        Returns:
            StructuredOutputRepairAdvice with band accept/repair/fail.
        """

        if not request_id:
            raise ValueError("request_id must be non-empty")
        keys = tuple(k for k in required_keys if k)
        if not keys:
            raise ValueError("required_keys must be non-empty")

        try:
            parsed = json.loads(payload)
            parse_ok = True
        except (json.JSONDecodeError, TypeError):
            return StructuredOutputRepairAdvice(
                request_id=request_id,
                band="fail",
                missing_keys=keys,
                parse_ok=False,
                advisory=f"fail: payload is not JSON for request_id={request_id}",
            )

        if not isinstance(parsed, dict):
            return StructuredOutputRepairAdvice(
                request_id=request_id,
                band="fail",
                missing_keys=keys,
                parse_ok=True,
                advisory=f"fail: JSON root is not an object for request_id={request_id}",
            )

        missing = tuple(k for k in keys if k not in parsed)
        if not missing:
            band: RepairBand = "accept"
            advisory = f"accept: all required keys present for request_id={request_id}"
        elif len(missing) <= max(1, len(keys) // 2):
            band = "repair"
            advisory = (
                f"repair: missing {missing} for request_id={request_id}; request a patch completion"
            )
        else:
            band = "fail"
            advisory = f"fail: too many missing keys {missing} for request_id={request_id}"

        return StructuredOutputRepairAdvice(
            request_id=request_id,
            band=band,
            missing_keys=missing,
            parse_ok=parse_ok,
            advisory=advisory,
        )
