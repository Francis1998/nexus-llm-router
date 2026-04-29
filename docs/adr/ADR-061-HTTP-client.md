# ADR-061: Http Client for nexus-llm-router

**Date:** 2026-04-28
**Status:** Accepted
**Context:** Llm Routing

## Context

The `router` module needs a reliable HTTP client solution
that integrates cleanly with our async strategy pipeline.

## Decision

Use **httpx (async)** for HTTP client.

## Considered Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **httpx (async)** (chosen) | Native async, well-maintained | Slightly higher cold-start |
| aiohttp | Mature ecosystem | Sync-first, harder to integrate |
| requests | Zero dependencies | Limited features for production |

## Consequences

- All new strategy components will use `httpx (async)` as the HTTP client layer.
- Existing code will be migrated incrementally.
- Added to `pyproject.toml` as a core dependency.
