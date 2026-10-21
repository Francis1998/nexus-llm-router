"""QuantCalibrationDriftBandAdvisor.

Advises calib_drift_ratio vs budgets.
Closes the vLLM/TGI/TensorRT-LLM AWQ/GPTQ quant-calibration drift monitors gap. Distinct from
`EmbeddingCachePoisonAdvisor` / `TokenizerVocabularyDriftAdvisor`.
Works with frontier multi-LLM stacks.
Never performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuantCalibrationDriftBandAdvice:
    """QuantCalibrationDriftBandAdvisor advice."""

    request_id: str
    calib_drift_ratio: float
    soft_limit: float
    hard_limit: float
    band: str


class QuantCalibrationDriftBandAdvisor:
    """Advise calib_drift_ratio bands."""

    def advise(
        self,
        *,
        request_id: str,
        calib_drift_ratio: float,
        soft_limit: float = 0.2,
        hard_limit: float = 0.5,
    ) -> QuantCalibrationDriftBandAdvice:
        """Return calib_drift_ratio band.

        Args:
            request_id: Non-empty request id.
            calib_drift_ratio: Observed ratio/value (>= 0).
            soft_limit: Soft budget.
            hard_limit: Hard budget.
        """

        rid = request_id.strip()
        if not rid:
            raise ValueError("request_id must be non-empty")
        if calib_drift_ratio < 0:
            raise ValueError("calib_drift_ratio must be >= 0")
        if soft_limit <= 0:
            raise ValueError("soft_limit must be > 0")
        if hard_limit <= soft_limit:
            raise ValueError("hard_limit must be > soft_limit")
        if calib_drift_ratio >= hard_limit:
            band = "breach"
        elif calib_drift_ratio >= soft_limit:
            band = "soft"
        else:
            band = "within"
        return QuantCalibrationDriftBandAdvice(
            request_id=rid,
            calib_drift_ratio=float(calib_drift_ratio),
            soft_limit=float(soft_limit),
            hard_limit=float(hard_limit),
            band=band,
        )
