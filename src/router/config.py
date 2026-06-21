"""Runtime configuration for Nexus LLM Router."""

from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    ab_model_a: str = "gpt-4o-mini"
    ab_model_b: str = "claude-3-5-haiku"
    ab_model_a_weight: float = 0.5
    quality_floor: Annotated[float, Field(ge=0.0, le=1.0)] = 0.72


def default_model_catalog() -> dict[str, ModelCandidate]:
    """Return the built-in model catalog with cost and quality priors.

    Returns:
        Mapping of model names to model candidates.
    """
    return {
        "gpt-4o": ModelCandidate(
            model="gpt-4o",
            provider="openai",
            quality_score=0.95,
            input_cost_per_1k=0.005,
            output_cost_per_1k=0.015,
            supports_domains={DomainTag.CODE, DomainTag.LEGAL, DomainTag.GENERAL},
        ),
        "gpt-4o-mini": ModelCandidate(
            model="gpt-4o-mini",
            provider="openai",
            quality_score=0.82,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
            supports_domains={DomainTag.CODE, DomainTag.GENERAL},
        ),
        "claude-3-5-sonnet": ModelCandidate(
            model="claude-3-5-sonnet",
            provider="anthropic",
            quality_score=0.96,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            supports_domains={
                DomainTag.CODE,
                DomainTag.MEDICAL,
                DomainTag.LEGAL,
                DomainTag.GENERAL,
            },
        ),
        "claude-3-5-haiku": ModelCandidate(
            model="claude-3-5-haiku",
            provider="anthropic",
            quality_score=0.80,
            input_cost_per_1k=0.0008,
            output_cost_per_1k=0.004,
            supports_domains={DomainTag.GENERAL, DomainTag.LEGAL},
        ),
        "gemini-1.5-pro": ModelCandidate(
            model="gemini-1.5-pro",
            provider="google",
            quality_score=0.93,
            input_cost_per_1k=0.0035,
            output_cost_per_1k=0.0105,
            supports_domains={
                DomainTag.CODE,
                DomainTag.MEDICAL,
                DomainTag.LEGAL,
                DomainTag.GENERAL,
            },
            supports_realtime=False,
        ),
        "gemini-1.5-flash": ModelCandidate(
            model="gemini-1.5-flash",
            provider="google",
            quality_score=0.78,
            input_cost_per_1k=0.00035,
            output_cost_per_1k=0.00105,
            supports_domains={DomainTag.CODE, DomainTag.GENERAL},
        ),
        "kimi-k2": ModelCandidate(
            model="kimi-k2",
            provider="moonshot",
            quality_score=0.76,
            input_cost_per_1k=0.0005,
            output_cost_per_1k=0.002,
            supports_domains={DomainTag.CODE, DomainTag.GENERAL},
        ),
    }


@lru_cache(maxsize=1)
def load_settings() -> RouterSettings:
    """Load and cache application settings.

    Returns:
        Runtime settings instance.
    """
    return RouterSettings()
