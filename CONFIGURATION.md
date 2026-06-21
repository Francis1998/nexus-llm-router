# Configuration

Nexus uses `pydantic-settings`. Application settings use the `NEXUS_` prefix. Provider settings use provider-native environment names.

## Core Settings

```dotenv
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

## Per-Request Strategy Selection

Set `X-Router-Strategy` to one of:

- `rule-based`
- `classifier`
- `cost-optimal`
- `latency-aware`
- `ab`

If the header is absent, Nexus uses `NEXUS_DEFAULT_STRATEGY`.

## Built-In Model Catalog

The default catalog lives in `src/router/config.py` and includes:

- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Anthropic: `claude-3-5-sonnet`, `claude-3-5-haiku`
- Google: `gemini-1.5-pro`, `gemini-1.5-flash`
- Moonshot: `kimi-k2`

Each model has quality, cost, domain, and realtime-support priors. Replace or extend the catalog when onboarding provider-specific SKUs.
