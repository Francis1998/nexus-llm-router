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
    provider_family_cost_ceiling_usd: Annotated[float, Field(ge=0.0)] = 0.05
    soft_family_budget_usd: Annotated[float, Field(ge=0.0)] = 5.0
    soft_family_budget_window_seconds: Annotated[float, Field(gt=0.0)] = 3600.0
    canary_stable_model: str = OPENAI_BALANCED_MODEL
    canary_model: str = OPENAI_FRONTIER_MODEL
    canary_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    latency_sla_ms: Annotated[float, Field(ge=0.0)] = 750.0
    epsilon: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    adaptive_exploration_base: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    adaptive_exploration_min: Annotated[float, Field(ge=0.0, le=1.0)] = 0.02
    availability_slo: Annotated[float, Field(ge=0.0, le=1.0)] = 0.99
    health_blend_success_weight: Annotated[float, Field(ge=0.0)] = 0.35
    health_blend_latency_weight: Annotated[float, Field(ge=0.0)] = 0.25
    health_blend_quality_weight: Annotated[float, Field(ge=0.0)] = 0.25
    health_blend_cost_weight: Annotated[float, Field(ge=0.0)] = 0.15
    hcl_health_weight: Annotated[float, Field(ge=0.0)] = 0.4
    hcl_cost_weight: Annotated[float, Field(ge=0.0)] = 0.3
    hcl_latency_weight: Annotated[float, Field(ge=0.0)] = 0.3
    prompt_prefix_cache_min_chars: Annotated[int, Field(ge=1)] = 512
    concurrency_cap: Annotated[int, Field(ge=1)] = 8
    token_bucket_capacity: Annotated[int, Field(ge=1)] = 10
    token_bucket_refill_per_sec: Annotated[float, Field(gt=0.0)] = 1.0
    tier_frontier_rpm: Annotated[int, Field(ge=1)] = 30
    tier_mid_rpm: Annotated[int, Field(ge=1)] = 60
    tier_economy_rpm: Annotated[int, Field(ge=1)] = 120
    failover_priority: list[str] = Field(
        default_factory=lambda: [
            OPENAI_FRONTIER_MODEL,
            ANTHROPIC_SAFETY_MODEL,
            GEMINI_PRO_MODEL,
            MOONSHOT_BALANCED_MODEL,
        ]
    )
    sticky_region_failover_preferences: list[str] = Field(
        default_factory=lambda: ["eu", "us", "cn", "global"]
    )
    latency_slo_ms: Annotated[float, Field(ge=0.0)] = 2000.0
    shadow_traffic_percent: Annotated[float, Field(ge=0.0, le=100.0)] = 5.0
    canary_cost_blend_percent: Annotated[float, Field(ge=0.0, le=100.0)] = 10.0
    token_cost_anomaly_ratio: Annotated[float, Field(gt=0.0)] = 2.0
    latency_hedge_ms: Annotated[float, Field(ge=0.0)] = 500.0
    prompt_length_tier_tokens: Annotated[int, Field(ge=1)] = 8000
    retry_budget_default: Annotated[int, Field(ge=0)] = 3
    cache_hit_sticky_min_chars: Annotated[int, Field(ge=1)] = 64
    embedding_cache_namespace_prefix: str = "embed"
    circuit_half_open_probe_budget: Annotated[int, Field(ge=1)] = 2
    semantic_cache_ttl_seconds: Annotated[float, Field(ge=0.0)] = 300.0
    provider_spend_soft_usd: Annotated[float, Field(ge=0.0)] = 10.0
    carbon_aware_max_intensity: Annotated[float, Field(ge=0.0)] = 400.0
    tenant_concurrency_lease: Annotated[int, Field(ge=1)] = 8
    provider_error_budget_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15
    region_latency_p99_ms: Annotated[float, Field(ge=0.0)] = 3000.0
    sticky_canary_cost_percent: Annotated[float, Field(ge=0.0, le=100.0)] = 10.0
    queue_depth_soft_cap: Annotated[int, Field(ge=1)] = 4
    provider_quota_lookback: Annotated[int, Field(ge=1)] = 100
    adaptive_timeout_hedge_ratio: Annotated[float, Field(ge=1.0)] = 1.5
    token_bucket_tenant_rate: Annotated[float, Field(gt=0.0)] = 5.0
    region_carbon_blend_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    provider_weight_decay_factor: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    provider_weight_recover: Annotated[float, Field(ge=0.0)] = 0.1
    retry_after_default_seconds: Annotated[float, Field(ge=0.0)] = 30.0
    latency_slope_window: Annotated[int, Field(ge=2)] = 10
    latency_slope_threshold_ms: Annotated[float, Field(ge=0.0)] = 25.0

    provider_hourly_cost_ceiling_usd: Annotated[float, Field(ge=0.0)] = 5.0
    token_rpm_ceiling: Annotated[int, Field(ge=1)] = 100_000
    provider_circuit_probe_budget: Annotated[int, Field(ge=1)] = 1
    carbon_latency_latency_weight: Annotated[float, Field(ge=0.0)] = 0.5
    adaptive_concurrency_base_cap: Annotated[int, Field(ge=1)] = 8
    adaptive_concurrency_min_cap: Annotated[int, Field(ge=1)] = 1
    adaptive_concurrency_latency_ms: Annotated[float, Field(gt=0.0)] = 2000.0
    provider_token_fair_share_ceiling: Annotated[int, Field(ge=1)] = 100_000
    region_failover_hysteresis_successes: Annotated[int, Field(ge=1)] = 3
    carbon_latency_carbon_weight: Annotated[float, Field(ge=0.0)] = 0.5
    tenant_budget_cascade_soft: Annotated[float, Field(ge=0.0)] = 10.0
    provider_error_budget_reset_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15
    provider_error_budget_reset_seconds: Annotated[float, Field(gt=0.0)] = 60.0
    sticky_region_warmup_requests: Annotated[int, Field(ge=1)] = 3
    tenant_budget_cascade_hard: Annotated[float, Field(gt=0.0)] = 12.5
    tenant_quota_burst_soft: Annotated[int, Field(ge=1)] = 60
    tenant_quota_burst_hard: Annotated[int, Field(ge=1)] = 75
    tenant_quota_burst_window_seconds: Annotated[float, Field(gt=0.0)] = 60.0
    provider_tail_latency_hedge_ms: Annotated[float, Field(ge=0.0)] = 1500.0
    sticky_session_migrate_success_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9
    provider_cold_start_lookback: Annotated[int, Field(ge=1)] = 100
    provider_cold_start_target: Annotated[int, Field(ge=1)] = 5
    tenant_fair_queue_lookback: Annotated[int, Field(ge=1)] = 100
    sticky_region_drain_regions: list[str] = Field(default_factory=list)
    provider_canary_primary_provider: str = "openai"
    provider_canary_shadow_percent: Annotated[float, Field(ge=0.0, le=100.0)] = 5.0
    sticky_model_pin_ttl_seconds: Annotated[float, Field(gt=0.0)] = 300.0


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
