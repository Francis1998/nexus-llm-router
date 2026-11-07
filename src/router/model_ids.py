"""Canonical provider model IDs for the built-in catalog and routing policy.

Refreshed for the November 2026 line-ups: OpenAI GPT-6, Anthropic Claude 5.5,
Google Gemini 4 / 3.8, and Kimi K3.1. Strategies and the default catalog import
from here so README, routing rules, and adapter cost tables stay aligned.
"""

OPENAI_FRONTIER_MODEL = "gpt-6-astra"
OPENAI_BALANCED_MODEL = "gpt-6-luna"
ANTHROPIC_SAFETY_MODEL = "claude-opus-5-5"
ANTHROPIC_FAST_MODEL = "claude-haiku-5-5"
GEMINI_PRO_MODEL = "gemini-4-argon"
GEMINI_FLASH_MODEL = "gemini-3.8-flash"
MOONSHOT_BALANCED_MODEL = "kimi-k3.1-pro"

# Provider limits shared by the catalog and adapter cost tables (2026-11-06).
# GPT-6 models accept at most 922K input tokens of their 1.05M window, and both
# GPT-6 (above 272K input tokens) and Claude Haiku 5.5 (above 100K prompt tokens)
# bill the whole request at long-prompt rates.
GPT6_MAX_INPUT_TOKENS = 922_000
GPT6_LONG_PROMPT_THRESHOLD_TOKENS = 272_000
HAIKU_5_5_LONG_PROMPT_THRESHOLD_TOKENS = 100_000
