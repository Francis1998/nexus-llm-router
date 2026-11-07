"""Guard user-facing prose against retired model-stack boilerplate.

Guides and module docstrings used to advertise "GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2" long after the default catalog moved to GPT-6, Claude 5.5,
Gemini 4, and Kimi K3.1. Lowercase model ids (``gpt-4o``) stay allowed because
docs use them as deliberate unknown-model or deprecation examples; historical
records (CHANGELOG, PR logs) and the deprecation table are exempt.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RETIRED_PROSE = re.compile(
    r"\bGPT-5\.5\b|\bGPT-4o\b|\bClaude\s+(?:Sonnet|Opus)\s+4\.\d\b"
    r"|\bSonnet\s+4\.6\b|\bKimi\s+K2\b(?!\.\d)"
)

EXEMPT = {
    Path("CHANGELOG.md"),
    Path("docs/PULL_SHARK_LOG.md"),
    Path("src/safety/model_deprecation.py"),
}


def _prose_files() -> list[Path]:
    """Return the user-facing docs and non-test sources to scan."""
    files = [*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md")]
    files += [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py")]
    return sorted(p for p in files if p.relative_to(ROOT) not in EXEMPT)


def test_scan_covers_guides_and_sources() -> None:
    """The guard must actually see the guide catalog and router sources."""
    rels = {p.relative_to(ROOT).as_posix() for p in _prose_files()}

    assert "README.md" in rels
    assert "src/router/strategies.py" in rels
    assert sum(r.startswith("docs/guides/") for r in rels) > 100


def test_no_retired_model_stack_in_prose() -> None:
    """Docs and docstrings name the current GPT-6 / Claude 5.5 / Gemini 4 / Kimi K3.1 stack."""
    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:120]}"
        for path in _prose_files()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if RETIRED_PROSE.search(line)
    ]

    assert offenders == [], "retired model names in prose:\n" + "\n".join(offenders[:40])
