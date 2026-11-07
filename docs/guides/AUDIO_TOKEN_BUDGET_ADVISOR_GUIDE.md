# AudioTokenBudgetAdvisor Guide

![AudioTokenBudgetAdvisor flow](../../assets/demo/audio-token-budget-advisor.gif)

Offline HITL advisor. Never network I/O.

Gap vs OpenAI Realtime/Gemini audio token budgets. Distinct from `MultimodalTokenTaxAdvisor` and `OutputTokenCeilingAdvisor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.audio_token_budget import AudioTokenBudgetAdvisor

advice = AudioTokenBudgetAdvisor().advise(request_id="r1", audio_token_ratio=0.1)
print(advice.band)
```

## Safety

Advisory bands only. Humans decide routing actions.
