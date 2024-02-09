# ADR-023: Config Management for nexus-llm-router

**Date:** 2024-02-09
**Status:** Accepted
**Context:** Llm Routing

## Context

The `router` module needs a reliable config management solution
that integrates cleanly with our async latency pipeline.

## Decision

Use **pydantic-settings** for config management.

## Considered Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **pydantic-settings** (chosen) | Native async, well-maintained | Slightly higher cold-start |
| dynaconf | Mature ecosystem | Sync-first, harder to integrate |
| raw os.environ | Zero dependencies | Limited features for production |

## Consequences

- All new latency components will use `pydantic-settings` as the config management layer.
- Existing code will be migrated incrementally.
- Added to `pyproject.toml` as a core dependency.
