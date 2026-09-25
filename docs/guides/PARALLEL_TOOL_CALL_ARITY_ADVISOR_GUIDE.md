# ParallelToolCallArityAdvisor Guide

![ParallelToolCallArityAdvisor](../../assets/demo/parallel-tool-call-arity.gif)

Offline advisory bands for parallel tool-call count. Never network I/O.
Closes OpenRouter / LiteLLM / Portkey parallel-tools fanout gaps.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from request-hedging and tenant concurrency guards.

## Usage

```python
from safety.parallel_tool_call_arity import ParallelToolCallArityAdvisor

advice = ParallelToolCallArityAdvisor().advise(tool_call_count=6)
print(advice.band)
```
