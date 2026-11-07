# Cost Attribution Tag Ledger Guide

![Cost attribution tag ledger](../../assets/demo/cost-attribution-tag.gif)

Record Helicone-style cost attribution tags per request and roll up by tag.
Never performs network I/O. Closes the Helicone / Portkey / LiteLLM cost-tag gap.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `SpendLedgerGuard` and `TenantSpendQuotaGuard`.

## Usage

```python
from safety.cost_attribution_tag import CostAttributionTagLedger

ledger = CostAttributionTagLedger()
ledger.record("r1", tags=("team:search", "env:prod"), cost_usd=0.02)
print(ledger.rollup())
```
