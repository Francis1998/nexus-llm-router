"""Render a terminal-style demo GIF."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_PATH = Path("assets/demo.gif")
WIDTH = 1120
HEIGHT = 520
PADDING = 24
LINE_HEIGHT = 22
BACKGROUND = (8, 12, 18)
PROMPT = (106, 227, 183)
TEXT = (226, 232, 240)
MUTED = (148, 163, 184)


def demo_lines() -> list[tuple[str, tuple[int, int, int]]]:
    """Return terminal lines for the demo GIF.

    Returns:
        Lines with RGB colors.
    """
    return [
        ("$ PYTHONPATH=src python scripts/benchmark.py", PROMPT),
        (
            '{"request_id":"demo-medical","model":"claude-3-5-sonnet",'
            '"strategy":"rule-based","rationale":"medical domain requires highest safety prior"}',
            MUTED,
        ),
        (
            "medical: strategy=rule-based model=claude-3-5-sonnet "
            "cost=$0.000026 rationale=medical domain requires highest safety prior",
            TEXT,
        ),
        (
            '{"request_id":"demo-code","model":"gpt-4o","strategy":"classifier",'
            '"rationale":"classifier detected code domain"}',
            MUTED,
        ),
        (
            "code: strategy=classifier model=gpt-4o "
            "cost=$0.000028 rationale=classifier detected code domain",
            TEXT,
        ),
        (
            '{"request_id":"demo-cost","model":"gpt-4o-mini","strategy":"cost-optimal",'
            '"rationale":"LP objective minimized estimated cost with quality floor"}',
            MUTED,
        ),
        (
            "cost: strategy=cost-optimal model=gpt-4o-mini "
            "cost=$0.000027 rationale=LP objective minimized estimated cost $0.000309",
            TEXT,
        ),
        (
            '{"request_id":"demo-ab","model":"gpt-4o-mini","strategy":"ab",'
            '"rationale":"A/B bucket=0.3442 routed to gpt-4o-mini"}',
            MUTED,
        ),
        (
            "ab: strategy=ab model=gpt-4o-mini "
            "cost=$0.000023 rationale=A/B bucket=0.3442 routed to gpt-4o-mini",
            TEXT,
        ),
    ]


def render_frame(visible_lines: int) -> Image.Image:
    """Render one terminal frame.

    Args:
        visible_lines: Number of visible lines.

    Returns:
        Rendered frame.
    """
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rounded_rectangle(
        (12, 12, WIDTH - 12, HEIGHT - 12),
        radius=16,
        outline=(30, 41, 59),
        width=2,
    )
    draw.text((PADDING, PADDING), "nexus-llm-router demo", fill=PROMPT, font=font)
    for index, (line, color) in enumerate(demo_lines()[:visible_lines]):
        draw.text(
            (PADDING, PADDING + 38 + index * LINE_HEIGHT),
            line,
            fill=color,
            font=font,
        )
    return image


def main() -> None:
    """Write the demo GIF artifact."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = [render_frame(visible_lines) for visible_lines in range(1, len(demo_lines()) + 1)]
    frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=650,
        loop=0,
    )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
