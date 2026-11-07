# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x     | Yes       |

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Use GitHub private vulnerability reporting instead:

https://github.com/Francis1998/nexus-llm-router/security/advisories/new

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

We aim to acknowledge reports within 48 hours and patch confirmed issues as quickly as possible.

## Credential Handling

Nexus never uses the raw `Authorization` header as an identifier. The gateway
derives a deterministic, non-reversible fingerprint instead
(`safety.key_fingerprint.fingerprint_api_key`):

- An optional `Bearer` scheme is stripped, then the credential is hashed:
  `key_` + the first 16 hex characters of its SHA-256 digest.
- A missing or empty header maps to the shared `anonymous` identity.
- The same key always maps to the same id, so per-key rate limits, budgets,
  idempotency scopes, and response-cache namespaces stay isolated.

Only the fingerprint appears in HTTP 429/503 error details, `GET /v1/spend`
`by_tenant` keys, the SQLite spend ledger and idempotency store, and logs.

### Upgrading from builds before this fix

Earlier builds used the raw header (for example `Bearer sk-...`) as the
identifier, so it could be echoed in 429/503 responses and persisted in
`NEXUS_SPEND_LEDGER_PATH` / `NEXUS_IDEMPOTENCY_PATH`. If you ran such a build
with real keys, rotate the affected keys and delete or restrict access to those
SQLite files and any captured logs.
