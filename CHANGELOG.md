# Changelog

All notable changes to **nexus-llm-router** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- OpenAI-compatible FastAPI router API.
- Observe -> Decide -> Act routing engine.
- Rule-based, classifier, cost-optimal, latency-aware, and A/B routing strategies.
- OpenAI, Anthropic, Google Gemini, Moonshot, and mock provider adapters.
- Circuit breaker, budget guardrail, PII scrubber, and token-bucket rate limiter.
- Prometheus metrics, OpenTelemetry tracing, structlog JSON logging, and JSONL audit logs.
- Docker Compose stack with router, Prometheus, and Grafana.
- Demo GIF gallery and offline benchmark/demo script.

### Fixed
- Restored README positioning after generated documentation overwrote project-specific content.
- Restored CI to supported Python versions and verified quality gates.
- Restored the real offline routing demo script after a placeholder benchmark overwrite.

## [v0.1.0] — 2026-06-20

### Added
- Initial project scaffold with routing core, safety controls, observability, tests, docs, and deployment assets.
