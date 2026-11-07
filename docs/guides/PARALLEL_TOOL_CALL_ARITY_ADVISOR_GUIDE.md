# ParallelToolCallArityAdvisor Guide

![ParallelToolCallArityAdvisor](../../assets/demo/parallel-tool-call-arity.gif)

Offline advisory bands for parallel tool-call count. Never network I/O.
Closes OpenRouter / LiteLLM / Portkey parallel-tools fanout gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from request-hedging and tenant concurrency guards.

## Usage

```python
from safety.parallel_tool_call_arity import ParallelToolCallArityAdvisor

advice = ParallelToolCallArityAdvisor().advise(tool_call_count=6)
print(advice.band)
```
