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
from scripts.create_demo_gif import decision_flow_slides, demo_lines, use_case_slides

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


def test_demo_gif_script_uses_current_model_ids() -> None:
    """Generated demo copy must stay aligned with canonical model IDs."""
    rendered_demo_text = "\n".join(
        [
            *(line for line, _color in demo_lines()),
            *(
                line
                for _title, slide_lines in decision_flow_slides()
                for line, _color in slide_lines
            ),
            *(
                line
                for _title, slide_lines in use_case_slides()
                for line, _color in slide_lines
            ),
        ]
    )

    assert OPENAI_FRONTIER_MODEL in rendered_demo_text
    assert OPENAI_BALANCED_MODEL in rendered_demo_text
    assert GEMINI_FLASH_MODEL in rendered_demo_text
    assert ANTHROPIC_SAFETY_MODEL in rendered_demo_text
    assert "gpt-4o" not in rendered_demo_text
