"""Tests that the built-in catalog and docs use current model SKUs."""

from pathlib import Path

from adapters.anthropic import AnthropicAdapter
from adapters.base import BaseProviderAdapter
from adapters.google import GoogleGeminiAdapter
from adapters.moonshot import MoonshotAdapter
from adapters.openai import OpenAIAdapter
from router import model_ids
from router.config import default_model_catalog
from router.model_ids import (
    ANTHROPIC_FAST_MODEL,
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    MOONSHOT_BALANCED_MODEL,
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
        MOONSHOT_BALANCED_MODEL,
    }
    assert expected_models.issubset(catalog.keys())


def test_catalog_includes_all_model_id_constants() -> None:
    """Every canonical model ID constant should be present in the default catalog."""

    catalog = default_model_catalog()
    canonical_model_ids = {
        value
        for name, value in vars(model_ids).items()
        if name.endswith("_MODEL") and isinstance(value, str)
    }

    assert canonical_model_ids.issubset(catalog.keys())


def test_adapter_cost_estimates_match_catalog_priors() -> None:
    """Provider adapter cost estimates should match router catalog priors."""

    adapters_by_provider: dict[str, BaseProviderAdapter] = {
        "openai": OpenAIAdapter(api_key="test-key", timeout_seconds=30.0),
        "anthropic": AnthropicAdapter(api_key="test-key", timeout_seconds=30.0),
        "google": GoogleGeminiAdapter(api_key="test-key", timeout_seconds=30.0),
        "moonshot": MoonshotAdapter(
            api_key="test-key",
            base_url="https://api.moonshot.ai/v1",
            timeout_seconds=30.0,
        ),
    }

    for model, candidate in default_model_catalog().items():
        adapter = adapters_by_provider[candidate.provider]
        assert adapter.estimate_cost(model, 1000, 1000) == candidate.estimate_cost(
            1000,
            1000,
        )


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
            *(line for _title, slide_lines in use_case_slides() for line, _color in slide_lines),
        ]
    )

    assert OPENAI_FRONTIER_MODEL in rendered_demo_text
    assert OPENAI_BALANCED_MODEL in rendered_demo_text
    assert GEMINI_FLASH_MODEL in rendered_demo_text
    assert ANTHROPIC_SAFETY_MODEL in rendered_demo_text
    assert "gpt-4o" not in rendered_demo_text
