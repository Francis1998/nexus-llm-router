# ProviderCanaryRolloutGuard Guide

![ProviderCanaryRolloutGuard demo](../../assets/demo/provider-canary.gif)

canary rollout off/canary/promote via canary_pct sampling (OpenRouter/Portkey/LiteLLM canary gap; distinct from ShadowTrafficMirrorGuard / ProviderHealthScoreboard)

Works with **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**. Never performs network I/O.

## Usage

```python
from safety.provider_canary import ProviderCanaryRolloutGuard
```
