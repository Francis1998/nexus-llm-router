"""List and inspect nexus models with the official OpenAI SDK (``GET /v1/models``).

Run against a live nexus (for example ``scripts/serve_mock.py``)::

    uv run --with openai python scripts/models_api_demo.py --base-url http://localhost:8000/v1
"""

import argparse

from openai import OpenAI


def main() -> None:
    """Print the model picker view and one model's nexus metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="gpt-6-astra")
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="local-dev")

    print(f"{'model':<18} {'owned_by':<10} {'context':>9} {'$in/1M':>7} {'$out/1M':>8}  routable")
    for model in client.models.list():
        nexus = (model.model_extra or {})["nexus"]
        pricing = nexus["pricing"]
        print(
            f"{model.id:<18} {model.owned_by:<10} {nexus['context_window']:>9,} "
            f"{pricing['input_per_1m_tokens']:>7} {pricing['output_per_1m_tokens']:>8}  "
            f"{nexus['health']['routable']}"
        )

    model = client.models.retrieve(args.model)
    nexus = (model.model_extra or {})["nexus"]
    print(f"\n{model.id}: max input {nexus['max_input_tokens']:,} tokens")
    print(f"  long-prompt pricing: {nexus['pricing']['long_prompt']}")
    print(f"  capabilities: {', '.join(nexus['capabilities'])}")


if __name__ == "__main__":
    main()
