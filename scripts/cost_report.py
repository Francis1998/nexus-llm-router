"""Print model catalog cost estimates."""

from router.config import default_model_catalog


def main() -> None:
    """Print cost report for a representative request."""
    input_tokens = 1000
    output_tokens = 500
    for candidate in default_model_catalog().values():
        estimated_cost = candidate.estimate_cost(input_tokens, output_tokens)
        domains = ",".join(sorted(domain.value for domain in candidate.supports_domains))
        print(
            f"{candidate.model:20} provider={candidate.provider:10} "
            f"quality={candidate.quality_score:.2f} cost=${estimated_cost:.6f} domains={domains}",
        )


if __name__ == "__main__":
    main()
