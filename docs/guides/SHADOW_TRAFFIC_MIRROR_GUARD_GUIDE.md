# ShadowTrafficMirrorGuard Guide

![ShadowTrafficMirrorGuard demo](../../assets/demo/shadow-traffic.gif)

shadow mirror off/mirror/full via mirror_pct sampling (Portkey/Helicone/OpenRouter shadow-traffic gap; distinct from ProviderCanaryRolloutGuard)

Works with **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**. Never performs network I/O.

## Usage

```python
from safety.shadow_traffic import ShadowTrafficMirrorGuard
```
