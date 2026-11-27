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
- `Idempotency-Key` replays, and the 409 for an in-flight duplicate, are scoped
  to the key fingerprint plus `X-Tenant-Id` and the end-user id (`user`,
  `safety_identifier`, or `metadata.user_id`). Tenant and end-user ids are
  client-chosen, so they narrow a key's scope but never replace it: another key
  sending the same tenant or end-user id and `Idempotency-Key` is routed as a
  new request instead of receiving the stored response.
- On the Anthropic Messages endpoints (`/v1/messages*`), an `x-api-key` header
  is fingerprinted the same way (`fingerprint_request_credentials`), so
  `x-api-key: <key>` and `Authorization: Bearer <key>` map to the same id.
  Both headers carrying different keys is a 401 whose message names neither key.

Only the fingerprint appears in HTTP 429/503 error details, `GET /v1/spend`
`by_tenant` keys, the SQLite spend ledger and idempotency store, and logs.

With `NEXUS_AUTH_MODE=virtual_keys` the identity is the issued key's id
(`vk_...`, or `master` for the master key) instead of a fingerprint, every
endpoint accepts `Authorization: Bearer` or `x-api-key` (both must agree), and
unknown keys are refused with a 401 that never echoes them. Issued secrets are
stored only as HMAC-SHA-256 digests under `NEXUS_KEY_HASH_PEPPER`; see
[SAFETY.md](SAFETY.md#virtual-keys) and
[docs/guides/VIRTUAL_KEYS_GUIDE.md](docs/guides/VIRTUAL_KEYS_GUIDE.md#security-notes).

### Upgrading from builds before this fix

Earlier builds used the raw header (for example `Bearer sk-...`) as the
identifier, so it could be echoed in 429/503 responses and persisted in
`NEXUS_SPEND_LEDGER_PATH` / `NEXUS_IDEMPOTENCY_PATH`. If you ran such a build
with real keys, rotate the affected keys and delete or restrict access to those
SQLite files and any captured logs.

### Upgrading from builds with tenant-wide idempotency scopes

Earlier builds scoped `Idempotency-Key` replays by `X-Tenant-Id` alone, else by
the end-user id, else by the key fingerprint. Any API key that sent another
caller's tenant or end-user id with the same `Idempotency-Key` therefore received
that caller's stored response (or a 409 while it ran), and a response stored by
one key could be replayed to another.

No record stored by an earlier build replays after upgrading. Stored keys now
start with a line break, which neither the keys earlier builds stored nor any
client-sent key can contain (HTTP header values cannot hold line breaks). A
planted record whose client-chosen tenant spells another caller's scope can
therefore never match:

- A retry that spans the upgrade runs once more.
- Old records expire after `NEXUS_IDEMPOTENCY_TTL_SECONDS` (24 hours by default).
  Delete `NEXUS_IDEMPOTENCY_PATH` to drop them at once.
- During a rolling deploy, old and new instances sharing `NEXUS_IDEMPOTENCY_PATH`
  neither replay each other's records nor see each other's in-flight
  reservations. A duplicate that reaches the other version is routed again.

### Provider API keys

Adapters send provider credentials only in request headers (`Authorization`
for OpenAI and Moonshot, `x-api-key` for Anthropic, `x-goog-api-key` for
Google), never in URLs. httpx logs every request URL at INFO, and the gateway
writes INFO logs to stdout, so a key in a URL ends up in the logs.

Earlier builds sent the Gemini key as a `?key=` query parameter, so every Gemini
call logged a line like this one:

```text
HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key=<GOOGLE_API_KEY> "HTTP/1.1 200 OK"
```

If you ran such a build with `GOOGLE_API_KEY` set, rotate the key and delete or
restrict access to the captured logs.
