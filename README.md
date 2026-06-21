# Nexus LLM Router

Intelligent multi-LLM router with task-aware routing strategies, cost optimization, and production safety controls. Nexus exposes a drop-in OpenAI-compatible `/v1/chat/completions` API and routes each request through an opinionated `Observe -> Decide -> Act` pipeline.

![Nexus use cases](assets/use-cases.gif)

## Why Nexus

Nexus is designed for AI infrastructure engineers running multi-model production pipelines where quality, latency, and cost all matter. It is inspired by the problem space explored by LiteLLM, RouteLLM, and anyscale/llm-router, but uses an original architecture centered on deterministic routing decisions, durable audit logs, and explicit safety controls.

Most teams start with one LLM endpoint. That works until traffic grows, latency starts swinging, finance asks why every request hits the most expensive model, and incident review asks why the app kept calling a degraded provider. Nexus gives the application one stable OpenAI-compatible API while moving model choice, fallback, budget, audit, and routing rationale into infra-owned middleware.

## Problems It Solves

- Issue: every prompt is sent to the same frontier model.
  Nexus solves this by classifying prompt complexity and routing simple tasks to cheaper low-latency models while reserving premium models for hard prompts.

- Issue: spend grows faster than product usage.
  Nexus solves this with cost-aware routing, model cost estimates, per-user budget guardrails, and Prometheus cost metrics.

- Issue: code, medical, legal, and general prompts need different quality defaults.
  Nexus solves this by extracting a domain tag and applying deterministic policy rules such as medical/legal to Claude and complex code to GPT-4o.

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

Observe → Decide → Act state-machine demo:

![Nexus decision flow](assets/decision-flow.gif)

## Core Features

- `Observe`: extracts complexity score, domain tag, latency requirement, and token budget.
- `Decide`: selects models with rule-based, classifier, cost-optimal, latency-aware, or A/B strategies.
- `Act`: dispatches through provider adapters and automatically walks fallback chains.
- State machine: `RECEIVED -> CLASSIFIED -> ROUTED -> DISPATCHED -> RESPONDED | FALLBACK | FAILED`.
- Provider adapters: OpenAI, Anthropic, Google Gemini, and Moonshot Kimi.
- Safety: circuit breakers, request timeouts, budget caps, PII scrubbing, and token-bucket rate limits.
- Observability: structlog JSON logs, OpenTelemetry traces, Prometheus metrics, and Grafana dashboard.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[dev]"
cp .env.example .env
PYTHONPATH=src uvicorn api.main:app --reload
```

Send a request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Router-Strategy: rule-based" \
  -d '{"messages":[{"role":"user","content":"Debug this Python retry loop."}],"max_tokens":128}'
```

Run the offline routing demo:

```bash
PYTHONPATH=src python scripts/benchmark.py
```

## Quality Gates

```bash
ruff check src/
mypy src/
pytest tests/ -v
```

## Docker Compose

```bash
cp .env.example .env
docker-compose up --build
```

Services:

- Router: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Routing Strategies

Select a strategy with `X-Router-Strategy`:

- `rule-based`: domain and complexity priority matrix.
- `classifier`: logistic-regression-style complexity and domain features.
- `cost-optimal`: minimizes estimated cost subject to quality floor.
- `latency-aware`: penalizes providers with poor rolling p95 latency.
- `ab`: deterministic request-id buckets across two model arms.

## Documentation

- [QUICKSTART.md](QUICKSTART.md)
- [CONFIGURATION.md](CONFIGURATION.md)
- [SAFETY.md](SAFETY.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)

## License

Apache-2.0
