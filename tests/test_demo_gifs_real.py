"""Demo GIFs embedded in the README and docs must be real animations, not placeholders."""

import re
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_WIDTH = 200
MIN_HEIGHT = 100
MIN_BYTES = 2 * 1024
_GIF_LINK = re.compile(r"""(?:\]\(|src=["'])\s*<?([^)"'\s>]+\.gif)\b""", re.IGNORECASE)


def _markdown_files() -> list[Path]:
    return [*sorted(REPO_ROOT.glob("*.md")), *sorted((REPO_ROOT / "docs").rglob("*.md"))]


def _referenced_gifs() -> dict[Path, list[str]]:
    """Map each GIF referenced from README/docs markdown to the files embedding it."""
    references: dict[Path, list[str]] = {}
    for markdown in _markdown_files():
        for match in _GIF_LINK.finditer(markdown.read_text(encoding="utf-8")):
            target = match.group(1)
            if target.startswith(("http://", "https://")):
                continue
            gif = (markdown.parent / target).resolve()
            references.setdefault(gif, []).append(markdown.relative_to(REPO_ROOT).as_posix())
    return references


def _placeholder_reason(gif: Path) -> str | None:
    """Return why ``gif`` is not a real demo animation, or ``None`` when it is."""
    data = gif.read_bytes()
    if data[:6] not in (b"GIF87a", b"GIF89a") or len(data) < 10:
        return "not a GIF file"
    width, height = struct.unpack("<HH", data[6:10])
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return f"{width}x{height} px is below {MIN_WIDTH}x{MIN_HEIGHT}"
    if len(data) < MIN_BYTES:
        return f"{len(data)} bytes is below {MIN_BYTES}"
    return None


def test_markdown_gif_scan_finds_embedded_demos() -> None:
    """The scanner must see README and guide embeds, so the checks below are not vacuous."""
    references = _referenced_gifs()
    assert REPO_ROOT / "assets" / "demo.gif" in references
    assert any(
        source.startswith("docs/guides/") for sources in references.values() for source in sources
    )


def test_referenced_demo_gifs_exist() -> None:
    """Every GIF embedded in README/docs markdown must exist in the repository."""
    missing = sorted(
        f"{gif.relative_to(REPO_ROOT).as_posix()} (from {', '.join(sorted(set(sources)))})"
        for gif, sources in _referenced_gifs().items()
        if not gif.is_file()
    )
    assert not missing, "missing demo GIFs:\n" + "\n".join(missing)


def test_referenced_demo_gifs_are_real_animations() -> None:
    """Embedded GIFs must be valid, at least 200x100 px, and at least 2 KB."""
    placeholders = sorted(
        f"{gif.relative_to(REPO_ROOT).as_posix()}: {reason}"
        for gif in _referenced_gifs()
        if gif.is_file() and (reason := _placeholder_reason(gif)) is not None
    )
    assert not placeholders, "placeholder demo GIFs:\n" + "\n".join(placeholders)


def test_asset_gifs_are_not_placeholders() -> None:
    """No placeholder GIF may be committed under assets/, referenced or not."""
    placeholders = sorted(
        f"{gif.relative_to(REPO_ROOT).as_posix()}: {reason}"
        for gif in (REPO_ROOT / "assets").rglob("*.gif")
        if (reason := _placeholder_reason(gif)) is not None
    )
    assert not placeholders, "placeholder GIFs under assets/:\n" + "\n".join(placeholders)
