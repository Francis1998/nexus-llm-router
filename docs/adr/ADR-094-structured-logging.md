# ADR-094: Structured Logging for nexus-llm-router

**Date:** 2024-08-01
**Status:** Accepted
**Context:** Llm Routing

## Context

The `router` module needs a reliable structured logging solution
that integrates cleanly with our async router pipeline.

## Decision

Use **structlog** for structured logging.

## Considered Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **structlog** (chosen) | Native async, well-maintained | Slightly higher cold-start |
| loguru | Mature ecosystem | Sync-first, harder to integrate |
| stdlib logging | Zero dependencies | Limited features for production |

## Consequences

- All new router components will use `structlog` as the structured logging layer.
- Existing code will be migrated incrementally.
- Added to `pyproject.toml` as a core dependency.
