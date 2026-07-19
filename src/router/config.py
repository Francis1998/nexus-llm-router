"""Runtime configuration for Nexus LLM Router."""

from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from router.model_ids import (
    ANTHROPIC_FAST_MODEL,
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    MOONSHOT_BALANCED_MODEL,
    OPENAI_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
)
from router.schemas import DomainTag, ModelCandidate, RoutingStrategyName


class ProviderSettings(BaseSettings):
    """Provider API settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    moonshot_api_key: str | None = None
    moonshot_base_url: str = "https://api.moonshot.ai/v1"
    request_timeout_seconds: float = 30.0


class RouterSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NEXUS_", extra="ignore")

    app_name: str = "nexus-llm-router"
    environment: str = "development"
    default_strategy: RoutingStrategyName = RoutingStrategyName.RULE_BASED
    audit_log_path: str = "migrations/audit-log.jsonl"
    provider_settings: ProviderSettings = Field(default_factory=ProviderSettings)
    budget_cap_usd: float = 25.0
    rate_limit_capacity: int = 120
    rate_limit_refill_per_second: float = 2.0
    enable_pii_scrubbing: bool = False
    ab_model_a: str = OPENAI_BALANCED_MODEL
    ab_model_b: str = ANTHROPIC_FAST_MODEL
    ab_model_a_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    quality_floor: Annotated[float, Field(ge=0.0, le=1.0)] = 0.72
    blend_quality_weight: Annotated[float, Field(ge=0.0)] = 0.5
    blend_cost_weight: Annotated[float, Field(ge=0.0)] = 0.3
    blend_latency_weight: Annotated[float, Field(ge=0.0)] = 0.2
    request_cost_ceiling_usd: Annotated[float, Field(ge=0.0)] = 0.05
    canary_stable_model: str = OPENAI_BALANCED_MODEL
    canary_model: str = OPENAI_FRONTIER_MODEL
    canary_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    latency_sla_ms: Annotated[float, Field(ge=0.0)] = 750.0
    epsilon: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1


def default_model_catalog() -> dict[str, ModelCandidate]:
    """Return the built-in model catalog with cost and quality priors.

    Returns:
        Mapping of model names to model candidates.
    """
    return {
        OPENAI_FRONTIER_MODEL: ModelCandidate(
            model=OPENAI_FRONTIER_MODEL,
            provider="openai",
            quality_score=0.97,
            input_cost_per_1k=0.006,
            output_cost_per_1k=0.018,
            supports_domains={DomainTag.CODE, DomainTag.LEGAL, DomainTag.GENERAL},
            context_window=200_000,
            supported_regions={"us", "global"},
        ),
        OPENAI_BALANCED_MODEL: ModelCandidate(
            model=OPENAI_BALANCED_MODEL,
            provider="openai",
            quality_score=0.84,
            input_cost_per_1k=0.0002,
            output_cost_per_1k=0.0008,
            supports_domains={DomainTag.CODE, DomainTag.GENERAL},
            context_window=128_000,
            supported_regions={"us", "global"},
        ),
        ANTHROPIC_SAFETY_MODEL: ModelCandidate(
            model=ANTHROPIC_SAFETY_MODEL,
            provider="anthropic",
            quality_score=0.98,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            supports_domains={
                DomainTag.CODE,
                DomainTag.MEDICAL,
                DomainTag.LEGAL,
                DomainTag.GENERAL,
            },
            context_window=200_000,
            supported_regions={"us", "eu", "global"},
        ),
        ANTHROPIC_FAST_MODEL: ModelCandidate(
            model=ANTHROPIC_FAST_MODEL,
            provider="anthropic",
            quality_score=0.82,
            input_cost_per_1k=0.0008,
            output_cost_per_1k=0.004,
            supports_domains={DomainTag.GENERAL, DomainTag.LEGAL},
            context_window=200_000,
            supported_regions={"us", "eu", "global"},
        ),
        GEMINI_PRO_MODEL: ModelCandidate(
            model=GEMINI_PRO_MODEL,
            provider="google",
            quality_score=0.95,
            input_cost_per_1k=0.0035,
            output_cost_per_1k=0.0105,
            supports_domains={
                DomainTag.CODE,
                DomainTag.MEDICAL,
                DomainTag.LEGAL,
                DomainTag.GENERAL,
            },
            supports_realtime=False,
            context_window=1_000_000,
            supported_regions={"us", "eu", "global"},
        ),
        GEMINI_FLASH_MODEL: ModelCandidate(
            model=GEMINI_FLASH_MODEL,
            provider="google",
            quality_score=0.81,
            input_cost_per_1k=0.0015,
            output_cost_per_1k=0.009,
            supports_domains={DomainTag.CODE, DomainTag.GENERAL},
            context_window=1_000_000,
            supported_regions={"us", "eu", "global"},
        ),
        MOONSHOT_BALANCED_MODEL: ModelCandidate(
            model=MOONSHOT_BALANCED_MODEL,
            provider="moonshot",
            quality_score=0.76,
            input_cost_per_1k=0.0005,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.CODE, DomainTag.GENERAL},
            context_window=128_000,
            supported_regions={"cn", "global"},
        ),
    }


@lru_cache(maxsize=1)
def load_settings() -> RouterSettings:
    """Load and cache application settings.

    Returns:
        Runtime settings instance.
    """
    return RouterSettings()
