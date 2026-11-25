"""Show ``ModelDeprecationGuard`` verdicts on a given UTC date.

Prints each model's verdict (``ok``, ``warn``, or ``block``), retirement date,
and replacement, then the reason for every blocked model. ``--today`` previews
the guard on another date, for example the day a model retires::

    PYTHONPATH=src python scripts/model_deprecation_demo.py --today 2026-11-30 claude-sonnet-4-5
"""

import argparse
from datetime import date

from safety.model_deprecation import DeprecationVerdict, ModelDeprecationGuard

DEFAULT_MODELS = (
    "claude-sonnet-4-5",
    "claude-opus-4-5-20251101",
    "claude-3-5-sonnet-20241022",
    "gpt-4o",
    "kimi-k2",
    "claude-opus-5-5",
)


def main() -> None:
    """Print the verdict table and the reasons for blocked models."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("models", nargs="*", default=list(DEFAULT_MODELS), help="model IDs")
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=None,
        help="UTC date to check on, YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()
    pinned: date | None = args.today
    guard = ModelDeprecationGuard(today=None if pinned is None else lambda: pinned)

    width = max(len("model"), *(len(model) for model in args.models))
    print(f"{'model':<{width}}  verdict  retires on  replacement")
    blocked: list[DeprecationVerdict] = []
    for model in args.models:
        verdict = guard.check(model)
        severity = verdict.severity or "ok"
        retires_on = verdict.retires_on.isoformat() if verdict.retires_on else "-"
        print(f"{model:<{width}}  {severity:<7}  {retires_on:<10}  {verdict.replacement or '-'}")
        if verdict.severity == "block":
            blocked.append(verdict)
    for verdict in blocked:
        print(f"{verdict.model_id}: {verdict.reason}")


if __name__ == "__main__":
    main()
