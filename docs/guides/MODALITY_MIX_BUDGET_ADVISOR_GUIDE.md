# ModalityMixBudgetAdvisor Guide

![ModalityMixBudgetAdvisor](../../assets/demo/modality-mix-budget-advisor.gif)

Offline HITL advisor. Never network I/O.
Gap vs LiteLLM/OpenRouter multimodal modality-mix budget monitors. Distinct from `AudioTokenBudgetAdvisor` and `ReasoningTokenBudgetAdvisor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.modality_mix_budget import ModalityMixBudgetAdvisor

advice = ModalityMixBudgetAdvisor().advise(request_id="r1", mix_ratio=0.45)
print(advice.band)
```
