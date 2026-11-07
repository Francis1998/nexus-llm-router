# ToolSchemaDriftGate Guide

![ToolSchemaDriftGate](../../assets/demo/tool-schema-drift.gif)

Offline advisory tool-schema hash drift bands. Closes OpenRouter / LiteLLM /
Portkey tool-schema drift gaps.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `OutputSchemaGuard` and `JsonSchemaRetryAdvisor`.

## Usage

```python
from safety.tool_schema_drift import ToolSchemaDriftGate

advice = ToolSchemaDriftGate().advise(tool_name="search", declared_hash="abc", observed_hash="xyz")
print(advice.band)
```
