# ModalityMixBudgetAdvisor Guide

![ModalityMixBudgetAdvisor](../../assets/demo/modality-mix-budget-advisor.gif)

Offline HITL advisor. Never network I/O.
Gap vs LiteLLM/OpenRouter multimodal modality-mix budget monitors. Distinct from `AudioTokenBudgetAdvisor` and `ReasoningTokenBudgetAdvisor`.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from safety.modality_mix_budget import ModalityMixBudgetAdvisor

advice = ModalityMixBudgetAdvisor().advise(request_id="r1", mix_ratio=0.45)
print(advice.band)
```
