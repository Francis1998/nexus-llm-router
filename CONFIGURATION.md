# Configuration

Nexus uses `pydantic-settings`. Application settings use the `NEXUS_` prefix. Provider settings use provider-native environment names.

## Core Settings

```dotenv
NEXUS_ENVIRONMENT=development
NEXUS_DEFAULT_STRATEGY=rule-based
NEXUS_AUDIT_LOG_PATH=migrations/audit-log.jsonl
NEXUS_BUDGET_CAP_USD=25.0
NEXUS_RATE_LIMIT_CAPACITY=120
NEXUS_RATE_LIMIT_REFILL_PER_SECOND=2.0
NEXUS_ENABLE_PII_SCRUBBING=false
NEXUS_QUALITY_FLOOR=0.72
```

## Provider Credentials

```dotenv
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
MOONSHOT_API_KEY=
MOONSHOT_BASE_URL=https://api.moonshot.ai/v1
REQUEST_TIMEOUT_SECONDS=30
```

## A/B Routing Settings

The `ab` strategy assigns a stable bucket from the request ID and compares two configured model arms without changing application code.

```dotenv
NEXUS_AB_MODEL_A=gpt-4.1-mini
NEXUS_AB_MODEL_B=claude-haiku-4-5
NEXUS_AB_MODEL_A_WEIGHT=0.5
```

Use `gpt-5.5`, `claude-sonnet-4-6`, `gemini-3.1-pro-preview`, or `kimi-k2` for higher-quality evaluation arms when the experiment budget allows it.

## Weighted-Blend Routing Settings

The `weighted-blend` strategy selects the model that maximizes a tunable
composite of normalized quality, cost, and rolling p95 latency (cost and latency
are min-max inverted, so cheaper and faster candidates score higher). Weights are
normalized to sum to one, so only their ratios matter; all-zero weights fall back
to pure quality.

```dotenv
NEXUS_BLEND_QUALITY_WEIGHT=0.5
NEXUS_BLEND_COST_WEIGHT=0.3
NEXUS_BLEND_LATENCY_WEIGHT=0.2
```

## Per-Request Strategy Selection

Set `X-Router-Strategy` to one of:

- `rule-based`
- `classifier`
- `cost-optimal`
- `latency-aware`
- `reliability-aware`
- `weighted-blend`
- `ab`

If the header is absent, Nexus uses `NEXUS_DEFAULT_STRATEGY`.

## Built-In Model Catalog

The default catalog lives in `src/router/config.py` and includes:

- OpenAI: `gpt-5.5`, `gpt-4.1-mini`
- Anthropic: `claude-sonnet-4-6`, `claude-haiku-4-5`
- Google: `gemini-3.1-pro-preview`, `gemini-3.5-flash`
- Moonshot: `kimi-k2`

Each model has quality, cost, domain, and realtime-support priors. Replace or extend the catalog when onboarding provider-specific SKUs.
