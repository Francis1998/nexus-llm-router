# ToolSchemaVersionSkewAdvisor Guide

![ToolSchemaVersionSkewAdvisor HITL flow](../../assets/demo/tool-schema-version-skew-advisor.gif)

Offline HITL advisor. Never network I/O. Closes gaps vs LiteLLM/OpenAI/Anthropic tool-schema version-skew monitors.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `ToolSchemaDriftAdvisor` / `ToolLoopDepthGateAdvisor`.

## Usage

```python
from safety.tool_schema_version_skew import ToolSchemaVersionSkewAdvisor

advice = ToolSchemaVersionSkewAdvisor().advise(request_id="r1", skew_score=0.15)
print(advice.band)
```
