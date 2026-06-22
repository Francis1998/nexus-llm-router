"""Tests that the built-in catalog and docs use current model SKUs."""

from pathlib import Path

from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_FAST_MODEL,
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    OPENAI_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_catalog_contains_current_routing_models() -> None:
    """Default catalog must include the June 2026 routing SKUs."""
    catalog = default_model_catalog()
    expected_models = {
        OPENAI_FRONTIER_MODEL,
        OPENAI_BALANCED_MODEL,
        ANTHROPIC_SAFETY_MODEL,
        ANTHROPIC_FAST_MODEL,
        GEMINI_PRO_MODEL,
        GEMINI_FLASH_MODEL,
    }
    assert expected_models.issubset(catalog.keys())


def test_readme_documents_current_domain_routing_models() -> None:
    """README domain-routing copy must reference current frontier models."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Claude Sonnet 4.6" in readme
    assert "GPT-5.5" in readme
    assert "GPT-4o" not in readme
