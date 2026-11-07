# ReasoningTraceLeakAdvisor Guide

![ReasoningTraceLeakAdvisor flow](../../assets/demo/reasoning-trace-leak-advisor.gif)

Offline HITL advisor. Never network I/O.

Gap vs vLLM/SGLang/OpenAI-compatible reasoning-trace leak gates. Distinct from `ReasoningTokenBudgetAdvisor` and `PromptInjectionGateway`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from safety.reasoning_trace_leak import ReasoningTraceLeakAdvisor

advice = ReasoningTraceLeakAdvisor().advise(request_id="r1", leak_score=0.1)
print(advice.band)
```

## Safety

Advisory bands only. Humans decide routing actions.
