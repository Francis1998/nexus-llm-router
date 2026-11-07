# StickySessionBleedAdvisor Guide

![StickySessionBleedAdvisor](../../assets/demo/sticky-session-bleed.gif)

Closes vLLM / OpenRouter / LiteLLM sticky-session tenant bleed gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `StickyProviderAffinity` and `TenantConcurrencySlotGuard`.

## Usage

```python
from safety.sticky_session_bleed import StickySessionBleedAdvisor

advice = StickySessionBleedAdvisor().advise(request_id="r1", foreign_hit_rate=0.12)
print(advice.band)
```

## Safety

Advisory only. No HTTP. Humans decide routing changes.
