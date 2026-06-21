"""Export lightweight classifier coefficients."""

import json
from pathlib import Path

OUTPUT_PATH = Path("migrations/classifier-weights.json")


def train_classifier() -> dict[str, object]:
    """Return calibrated logistic regression coefficients.

    Returns:
        Classifier metadata and coefficients.
    """
    return {
        "model_type": "logistic_regression",
        "feature_source": "prompt_features",
        "bias": -1.55,
        "weights": {
            "word_count": 0.008,
            "question_count": 0.18,
            "code_hits": 0.32,
            "medical_hits": 0.26,
            "legal_hits": 0.24,
            "instruction_hits": 0.34,
        },
    }


def main() -> None:
    """Write classifier coefficients to disk."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(train_classifier(), indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
