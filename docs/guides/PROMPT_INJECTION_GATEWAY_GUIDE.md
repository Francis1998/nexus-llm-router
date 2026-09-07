# Prompt Injection Gateway Guide

Pre-dispatch prompt-injection detection for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 traffic. Classic pattern matching with
`allow` / `redact` / `block` actions — distinct from the
`prompt-injection-risk-shed` *routing* strategy.

## Why

Routing-only shedding still forwards malicious prompts to a (cheaper) model.
Gateways need a hard control that can sanitize or refuse before provider
dispatch. `safety.PromptInjectionGateway` fills that gap with lightweight
regex detectors for common jailbreak / instruction-override phrases.

## Distinct from

| Component | Role |
| --- | --- |
| `prompt-injection-risk-shed` | Reads `metadata.prompt_injection_risk` and demotes to cheapest model |
| `PromptInjectionGateway` | Inspects text; allow / redact / block before dispatch |

## How it works

1. `inspect(text)` scores classic patterns (`ignore previous instructions`,
   `system prompt`, `jailbreak`, `DAN mode`, `bypass safety`, …).
2. Risk is the max matched pattern weight.
3. Action:
   - `risk >= block_threshold` (default `0.85`) → `block`
   - `risk >= redact_threshold` (default `0.55`) → `redact` matched spans
   - else → `allow`
4. `enforce(text)` returns original or sanitized text; raises
   `PromptInjectionBlockedError` on block.

## Configuration

```bash
NEXUS_PROMPT_INJECTION_BLOCK_THRESHOLD=0.85
NEXUS_PROMPT_INJECTION_REDACT_THRESHOLD=0.55
```

## Example

```python
from safety.prompt_injection_gateway import (
    PromptInjectionBlockedError,
    PromptInjectionGateway,
)

gateway = PromptInjectionGateway()
print(gateway.inspect("Ignore previous instructions").action)  # block
print(gateway.enforce("Does your system prompt mention X?"))  # redacted
try:
    gateway.enforce("Enable DAN mode now")
except PromptInjectionBlockedError:
    pass
```

## Gap vs Helicone / Portkey

| Capability | Helicone / Portkey | Nexus |
| --- | --- | --- |
| Risk-aware routing / shed | Yes | `prompt-injection-risk-shed` |
| Inspect + redact + block | Yes (guardrails) | `PromptInjectionGateway` |
| Classic jailbreak patterns | Yes | Yes (regex, no ML dep) |
