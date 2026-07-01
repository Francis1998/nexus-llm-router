# nexus-llm-router

![Tests](https://img.shields.io/badge/tests-47%20passing-brightgreen) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![CI](https://github.com/Francis1998/nexus-llm-router/actions/workflows/ci.yml/badge.svg)

> Intelligent multi-LLM routing middleware with task-aware model selection, cost optimization, fallback safety, and a drop-in OpenAI-compatible API.

![Nexus use cases](assets/use-cases.gif)

## Why Nexus

Most teams start with one LLM endpoint. That works until traffic grows, latency starts swinging, finance asks why every request hits the most expensive model, and incident review asks why the app kept calling a degraded provider. Nexus gives the application one stable OpenAI-compatible API while moving model choice, fallback, budget, audit, and routing rationale into infra-owned middleware.

Nexus is designed for AI infrastructure engineers running multi-model production pipelines where quality, latency, and cost must be optimized at the same time.

## Problems It Solves

- Issue: every prompt is sent to the same frontier model.
  Nexus solves this by classifying prompt complexity and routing simple tasks to cheaper low-latency models while reserving premium models for hard prompts.

- Issue: spend grows faster than product usage.
  Nexus solves this with cost-aware routing, model cost estimates, per-user budget guardrails, and Prometheus cost metrics.

- Issue: code, medical, legal, and general prompts need different quality defaults.
  Nexus solves this by extracting a domain tag and applying deterministic policy rules such as medical/legal to Claude Sonnet 4.6 and complex code to GPT-5.5.

- Issue: one provider has an incident and the app fails hard.
  Nexus solves this with per-provider circuit breakers and automatic fallback chains.

- Issue: provider latency spikes during peak traffic.
  Nexus solves this with latency-aware routing that tracks rolling p95 latency and penalizes slow providers.

- Issue: teams want to compare models without rewriting product code.
  Nexus solves this with stable request-id A/B routing selected by the `X-Router-Strategy` header.

- Issue: support and compliance teams ask why a model answered a request.
  Nexus solves this by persisting durable audit records with `request_id`, selected model, strategy, rationale, latency, token usage, and cost.

- Issue: a single API key can overwhelm the router.
  Nexus solves this with a token-bucket rate limiter keyed by API key identifier.

- Issue: session or tenant budgets need hard enforcement.
  Nexus solves this by rejecting requests before dispatch when estimated spend would exceed the configured cap.

- Issue: PII can leak into third-party providers.
  Nexus solves this with optional regex redaction and a Presidio extension path before provider dispatch.

- Issue: teams need OpenAI compatibility without giving up provider choice.
  Nexus solves this by exposing `/v1/chat/completions` while normalizing OpenAI, Anthropic, Gemini, and Moonshot adapters behind one interface.

- Issue: model routing becomes a hidden product decision.
  Nexus solves this by making routing policy explicit, testable, observable, and owned in infra.

## Demo Gallery

Terminal routing demo with JSON rationale logs:

![Nexus terminal demo](assets/demo.gif)

Observe -> Decide -> Act state-machine demo:

![Nexus decision flow](assets/decision-flow.gif)

## Features

- **Router engine** with configurable strategies
- **Adapter pipeline** with full observability
- **Async-first** design using `asyncio` + `httpx`
- **Type-safe** with full `mypy` compliance
- **Production-ready** with Docker, CI/CD, and structured logging

## Quick Start

```bash
git clone https://github.com/Francis1998/nexus-llm-router.git
cd nexus-llm-router
pip install -e ".[dev]"
cp .env.example .env
PYTHONPATH=src uvicorn api.main:app --reload
```

## Quality Gates

```bash
ruff check src/ tests/ scripts/
mypy src/
pytest tests/ -v
```

## Docker Compose

```bash
docker compose up --build
```

Services:

- Router: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Routing Strategies

Select a strategy with `X-Router-Strategy`:

- `rule-based`: domain and complexity priority matrix
- `classifier`: logistic-regression-style complexity and domain features
- `cost-optimal`: minimizes estimated cost subject to quality floor
- `latency-aware`: penalizes providers with poor rolling p95 latency
- `ab`: deterministic request-id buckets across two model arms

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](ARCHITECTURE.md) | System design and component overview |
| [Configuration](CONFIGURATION.md) | All configuration options |
| [Quickstart](QUICKSTART.md) | Local setup and first request |
| [Safety](SAFETY.md) | Guardrails, fallback, and PII controls |
| [Contributing](CONTRIBUTING.md) | Development workflow and PR process |
| [Security](SECURITY.md) | Vulnerability reporting policy |
| [Changelog](CHANGELOG.md) | Version history |

## License

Apache-2.0 © [Francis1998](https://github.com/Francis1998)

*Last updated: 2026-06-26*
