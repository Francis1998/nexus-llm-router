# ProviderCanaryRolloutGuard Guide

![ProviderCanaryRolloutGuard demo](../../assets/demo/provider-canary.gif)

canary rollout off/canary/promote via canary_pct sampling (OpenRouter/Portkey/LiteLLM canary gap; distinct from ShadowTrafficMirrorGuard / ProviderHealthScoreboard)

Works with **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**. Never performs network I/O.

## Usage

```python
from safety.provider_canary import ProviderCanaryRolloutGuard
```
