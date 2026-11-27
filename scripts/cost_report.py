"""Print model catalog cost estimates at the list prices in effect now."""

from router.config import catalog_at, default_model_catalog, utc_now


def main() -> None:
    """Print cost report for a representative request."""
    input_tokens = 1000
    output_tokens = 500
    for candidate in catalog_at(default_model_catalog(), utc_now()).values():
        estimated_cost = candidate.estimate_cost(input_tokens, output_tokens)
        domains = ",".join(sorted(domain.value for domain in candidate.supports_domains))
        print(
            f"{candidate.model:20} provider={candidate.provider:10} "
            f"quality={candidate.quality_score:.2f} cost=${estimated_cost:.6f} domains={domains}",
        )


if __name__ == "__main__":
    main()
