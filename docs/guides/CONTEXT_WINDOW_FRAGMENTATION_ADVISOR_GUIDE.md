# ContextWindowFragmentationAdvisor Guide

![ContextWindowFragmentationAdvisor HITL flow](../../assets/demo/context-window-fragmentation-advisor.gif)

Offline HITL advisor. Never auto-acts. Closes gaps vs OpenAI/Anthropic/Gemini context-window fragmentation advisors.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from ``ContextWindowFitAdvisor`` and ``KvCacheEvictionPressureAdvisor``.

## Usage

```python
from safety.context_window_fragmentation import ContextWindowFragmentationAdvisor

advice = ContextWindowFragmentationAdvisor().advise(request_id="r1", fragmentation_ratio=0.25)
print(advice.band)
```

## Safety

No HTTP. Humans decide. See `SAFETY.md`.
