"""Render a recorded terminal transcript as a demo GIF.

The API guides (Models, Responses, Anthropic Messages) show real terminal
sessions against ``scripts/serve_mock.py``. Run the session first and save its
verbatim output; this script only draws that text. Lines starting with ``$ ``
are commands, typed out over ``--type-steps`` frames, and the output lines that
follow appear in one frame. The window keeps the last ``--rows`` wrapped lines
visible, like a scrolling terminal.

Usage::

    PYTHONPATH=src python scripts/render_transcript_gif.py transcript.txt \\
        assets/demo/models-api.gif \\
        --title "nexus-llm-router: GET /v1/models (OpenAI-compatible model discovery)" \\
        --rows 22 --type-steps 4 --font-size 13
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from create_demo_gif import BACKGROUND, MUTED, PROMPT, TEXT, save_gif, wrap
from PIL import Image, ImageDraw, ImageFont

Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

# Monospace keeps the tables in recorded output aligned; the first one found wins.
MONOSPACE_FONTS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
)
TITLE_BAR = (30, 41, 59)
WINDOW_DOTS = ((255, 95, 86), (255, 189, 46), (39, 201, 63))
TITLE_BAR_HEIGHT = 36
PADDING = 16
FRAME_MS = 600
HOLD_FRAMES = 4
# Enough colors for anti-aliased text plus the small window dots.
PALETTE_COLORS = 64


@dataclass(frozen=True)
class Step:
    """One command and the output it printed."""

    command: str
    output: tuple[str, ...]


def monospace_font(size: int) -> Font:
    """Return the first available monospace font, or Pillow's default font."""
    for candidate in MONOSPACE_FONTS:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def parse_transcript(lines: Sequence[str]) -> list[Step]:
    """Group transcript lines into commands and their output.

    Raises:
        ValueError: If output appears before the first ``$ `` command.
    """
    steps: list[Step] = []
    output: list[str] = []
    command: str | None = None
    for line in lines:
        if line.startswith("$ "):
            if command is not None:
                steps.append(Step(command, tuple(output)))
            command, output = line, []
        elif command is None:
            raise ValueError("a transcript must start with a '$ ' command line")
        else:
            output.append(line)
    if command is not None:
        steps.append(Step(command, tuple(output)))
    return steps


def render_frames(
    steps: Sequence[Step],
    title: str,
    *,
    width: int,
    rows: int,
    type_steps: int,
    font_size: int,
) -> list[Image.Image]:
    """Return GIF frames that type each command and then reveal its output."""
    font = monospace_font(font_size)
    line_height = font_size + 7
    height = TITLE_BAR_HEIGHT + 2 * PADDING + rows * line_height
    text_width = width - 2 * PADDING

    def frame(shown: Sequence[tuple[str, bool]]) -> Image.Image:
        image = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, TITLE_BAR_HEIGHT), fill=TITLE_BAR)
        for index, color in enumerate(WINDOW_DOTS):
            x = PADDING + index * 20
            draw.ellipse((x, 12, x + 12, 24), fill=color)
        draw.text((PADDING + 70, 10), title, fill=MUTED, font=font)
        wrapped = [
            (chunk, is_command)
            for text, is_command in shown
            for chunk in wrap(text, font, text_width) or [""]
        ]
        y = TITLE_BAR_HEIGHT + PADDING
        for chunk, is_command in wrapped[-rows:]:
            draw.text((PADDING, y), chunk, fill=PROMPT if is_command else TEXT, font=font)
            y += line_height
        return image

    history: list[tuple[str, bool]] = []
    frames: list[Image.Image] = []
    for step in steps:
        for typed in range(1, type_steps + 1):
            partial = step.command[: max(2, round(len(step.command) * typed / type_steps))]
            frames.append(frame([*history, (partial, True)]))
        history.append((step.command, True))
        if step.output:
            history.extend((line, False) for line in step.output)
            frames.append(frame(history))
    if not frames:
        raise ValueError("the transcript has no commands to render")
    frames.extend([frames[-1]] * HOLD_FRAMES)
    palette = frames[-1].quantize(
        colors=PALETTE_COLORS, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE
    )
    return [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in frames]


def main_cli(argv: Sequence[str] | None = None) -> int:
    """Parse flags, render the transcript, and write the GIF."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("transcript", type=Path, help="recorded terminal session (UTF-8 text)")
    parser.add_argument("output", type=Path, help="GIF path to write")
    parser.add_argument("--title", default="nexus-llm-router", help="window title")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--rows", type=int, default=22, help="visible terminal rows")
    parser.add_argument("--type-steps", type=int, default=4, help="frames to type a command")
    parser.add_argument("--font-size", type=int, default=13)
    args = parser.parse_args(argv)
    steps = parse_transcript(args.transcript.read_text(encoding="utf-8").splitlines())
    frames = render_frames(
        steps,
        args.title,
        width=args.width,
        rows=args.rows,
        type_steps=args.type_steps,
        font_size=args.font_size,
    )
    save_gif(args.output, frames, FRAME_MS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
