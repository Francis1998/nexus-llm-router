# Changelog

All notable changes to **nexus-llm-router** are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `weighted-blend` routing strategy: selects the model maximizing a tunable composite of normalized quality, cost, and latency (min-max inverted so cheaper/faster scores higher). Weights are configurable via `NEXUS_BLEND_QUALITY_WEIGHT`, `NEXUS_BLEND_COST_WEIGHT`, and `NEXUS_BLEND_LATENCY_WEIGHT`, are normalized to sum to one, and fall back to pure quality when all are zero.
- `reliability-aware` routing strategy: routes to the highest-quality model whose provider circuit breaker is closed and orders the fallback chain healthy-providers-first, backed by a non-mutating `CircuitBreakerRegistry.is_available()` read.
- OpenAI-compatible FastAPI router API.
- Observe -> Decide -> Act routing engine.
- Rule-based, classifier, cost-optimal, latency-aware, and A/B routing strategies.
- OpenAI, Anthropic, Google Gemini, Moonshot, and mock provider adapters.
- Circuit breaker, budget guardrail, PII scrubber, and token-bucket rate limiter.
- Prometheus metrics, OpenTelemetry tracing, structlog JSON logging, and JSONL audit logs.
- Docker Compose stack with router, Prometheus, and Grafana.
- Demo GIF gallery and offline benchmark/demo script.

### Fixed
- Prompt code-feature detection now word-anchors its language keywords (`def`, `class`, `import`, `function`, `const`, `async`), so plain-English words that merely contain a keyword (e.g. `masterclass`, `subclass`) are no longer counted as code, which previously inflated complexity scores and biased the domain classifier toward code.
- OpenAI and Moonshot adapters now join structured `message.content` part lists (returned by OpenAI-compatible gateways such as LiteLLM, vLLM, and OpenRouter) via a shared `message_text()` helper, instead of silently returning an empty completion for the non-string shape.
- Anthropic adapter now joins all `content` text blocks (skipping non-text blocks such as `thinking`/`tool_use`) instead of reading only `content[0].text`, which returned an empty completion when a non-text block came first and truncated multi-block answers.
- Gemini adapter now concatenates all `content.parts` text segments (skipping non-text parts) instead of reading only the first, preventing silent truncation of multi-part completions.
- Restored README positioning after generated documentation overwrote project-specific content.
- Restored CI to supported Python versions and verified quality gates.
- Restored the real offline routing demo script after a placeholder benchmark overwrite.
- Removed automation artifact `DAILY_IMPROVEMENTS.md` and added repository integrity tests.
- Aligned contributor/security docs, Dockerfile production install, and full-source Bandit scan.

## [v0.1.0] — 2026-06-20

### Added
- Initial project scaffold with routing core, safety controls, observability, tests, docs, and deployment assets.
