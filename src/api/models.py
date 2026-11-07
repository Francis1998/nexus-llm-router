"""OpenAI-compatible model discovery (``GET /v1/models``) for the nexus catalog."""

from time import time

from api.schemas import (
    ModelHealth,
    ModelLongPromptPricing,
    ModelObject,
    ModelPricing,
    NexusModelInfo,
)
from router.schemas import ModelCandidate
from router.strategies import known_model_capabilities

# OpenAI's ``created`` is a model release time; nexus has none, so it reports when
# this process loaded its catalog.
CATALOG_LOADED_AT = int(time())


def _per_million(cost_per_1k: float) -> float:
    return round(cost_per_1k * 1000, 6)


def model_object(candidate: ModelCandidate, health: ModelHealth) -> ModelObject:
    """Render a catalog model as an OpenAI ``model`` object with nexus metadata.

    Args:
        candidate: Routable catalog model.
        health: Live health of the model's provider.

    Returns:
        Model object for ``GET /v1/models`` and ``GET /v1/models/{id}``.
    """
    tier = candidate.long_prompt_pricing
    long_prompt = (
        None
        if tier is None
        else ModelLongPromptPricing(
            threshold_tokens=tier.threshold_tokens,
            input_per_1m_tokens=_per_million(tier.input_cost_per_1k),
            output_per_1m_tokens=_per_million(tier.output_cost_per_1k),
        )
    )
    return ModelObject(
        id=candidate.model,
        created=CATALOG_LOADED_AT,
        owned_by=candidate.provider,
        nexus=NexusModelInfo(
            provider=candidate.provider,
            context_window=candidate.context_window,
            max_input_tokens=candidate.max_input_tokens,
            pricing=ModelPricing(
                input_per_1m_tokens=_per_million(candidate.input_cost_per_1k),
                output_per_1m_tokens=_per_million(candidate.output_cost_per_1k),
                long_prompt=long_prompt,
            ),
            capabilities=sorted(known_model_capabilities(candidate.model)),
            domains=sorted(domain.value for domain in candidate.supports_domains),
            regions=sorted(candidate.supported_regions),
            supports_realtime=candidate.supports_realtime,
            quality_score=candidate.quality_score,
            health=health,
        ),
    )
