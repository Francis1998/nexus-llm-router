"""Render terminal-style demo GIFs."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from router.model_ids import (
    ANTHROPIC_SAFETY_MODEL,
    GEMINI_FLASH_MODEL,
    OPENAI_BALANCED_MODEL,
    OPENAI_FRONTIER_MODEL,
)

WIDTH = 1120
HEIGHT = 520
PADDING = 24
LINE_HEIGHT = 22
BACKGROUND = (8, 12, 18)
PROMPT = (106, 227, 183)
TEXT = (226, 232, 240)
MUTED = (148, 163, 184)
ACCENT = (96, 165, 250)
WARN = (251, 191, 36)
SUCCESS = (52, 211, 153)
Color = tuple[int, int, int]
Line = tuple[str, Color]
Slide = tuple[str, list[Line]]


def demo_lines() -> list[tuple[str, tuple[int, int, int]]]:
    """Return terminal lines for the demo GIF.

    Returns:
        Lines with RGB colors.
    """
    return [
        ("$ PYTHONPATH=src python scripts/benchmark.py", PROMPT),
        (
            f'{{"request_id":"demo-medical","model":"{ANTHROPIC_SAFETY_MODEL}",'
            '"strategy":"rule-based","rationale":"medical domain requires highest safety prior"}',
            MUTED,
        ),
        (
            f"medical: strategy=rule-based model={ANTHROPIC_SAFETY_MODEL} "
            "cost=$0.000026 rationale=medical domain requires highest safety prior",
            TEXT,
        ),
        (
            f'{{"request_id":"demo-code","model":"{OPENAI_FRONTIER_MODEL}",'
            '"strategy":"classifier",'
            '"rationale":"classifier detected code domain"}',
            MUTED,
        ),
        (
            f"code: strategy=classifier model={OPENAI_FRONTIER_MODEL} "
            "cost=$0.000028 rationale=classifier detected code domain",
            TEXT,
        ),
        (
            f'{{"request_id":"demo-cost","model":"{OPENAI_BALANCED_MODEL}",'
            '"strategy":"cost-optimal",'
            '"rationale":"LP objective minimized estimated cost with quality floor"}',
            MUTED,
        ),
        (
            f"cost: strategy=cost-optimal model={OPENAI_BALANCED_MODEL} "
            "cost=$0.000027 rationale=LP objective minimized estimated cost $0.000309",
            TEXT,
        ),
        (
            f'{{"request_id":"demo-ab","model":"{OPENAI_BALANCED_MODEL}",'
            '"strategy":"ab",'
            f'"rationale":"A/B bucket=0.3442 routed to {OPENAI_BALANCED_MODEL}"}}',
            MUTED,
        ),
        (
            f"ab: strategy=ab model={OPENAI_BALANCED_MODEL} "
            f"cost=$0.000023 rationale=A/B bucket=0.3442 routed to {OPENAI_BALANCED_MODEL}",
            TEXT,
        ),
    ]


def use_case_slides() -> list[Slide]:
    """Return issue-to-solution slides.

    Returns:
        Slides for the use-case GIF.
    """
    return [
        (
            "Issue: frontier model spend is growing faster than traffic",
            [
                ("signal: simple summarization, realtime, low risk", MUTED),
                (f"decision: route to {GEMINI_FLASH_MODEL}", ACCENT),
                ("result: lower cost while preserving latency target", SUCCESS),
            ],
        ),
        (
            "Issue: medical/legal prompts need conservative defaults",
            [
                ("signal: domain=medical/legal, complexity=0.73", MUTED),
                (f"decision: route to {ANTHROPIC_SAFETY_MODEL}", ACCENT),
                ("result: deterministic rationale and audit trail", SUCCESS),
            ],
        ),
        (
            "Issue: provider p95 latency spikes during peak traffic",
            [
                ("signal: openai p95=2500ms, google p95=45ms", MUTED),
                ("decision: latency-aware penalty avoids slow provider", ACCENT),
                ("result: request meets realtime service objective", SUCCESS),
            ],
        ),
        (
            "Issue: one provider fails mid-incident",
            [
                ("signal: circuit opens after 3 consecutive failures", MUTED),
                ("decision: fallback chain walks to healthy provider", ACCENT),
                ("result: graceful degradation instead of outage", SUCCESS),
            ],
        ),
        (
            "Issue: teams need online eval without app rewrites",
            [
                ("signal: X-Router-Strategy: ab", MUTED),
                ("decision: stable request-id bucket selects model arm", ACCENT),
                ("result: model experiments stay reproducible", SUCCESS),
            ],
        ),
    ]


def decision_flow_slides() -> list[Slide]:
    """Return Observe-Decide-Act routing slides.

    Returns:
        Slides for the decision-flow GIF.
    """
    return [
        (
            "RECEIVED -> CLASSIFIED",
            [
                ("observe.complexity_score = 0.81", MUTED),
                ("observe.domain_tag = code", MUTED),
                ("observe.latency_requirement = realtime", MUTED),
            ],
        ),
        (
            "CLASSIFIED -> ROUTED",
            [
                ("strategy = classifier", MUTED),
                (f"chosen_model = {OPENAI_FRONTIER_MODEL}", ACCENT),
                ("rationale = classifier detected code domain", SUCCESS),
            ],
        ),
        (
            "ROUTED -> DISPATCHED",
            [
                ("budget guardrail: pass", SUCCESS),
                ("circuit breaker: provider available", SUCCESS),
                ("PII scrubber: optional pre-dispatch redaction", WARN),
            ],
        ),
        (
            "DISPATCHED -> RESPONDED",
            [
                ("provider = openai", ACCENT),
                ("latency_ms = 91.2", MUTED),
                ("audit log persisted request_id + cost + rationale", SUCCESS),
            ],
        ),
    ]


def render_terminal_frame(visible_lines: int) -> Image.Image:
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


def render_slide_frame(title: str, lines: list[Line], step: int, total_steps: int) -> Image.Image:
    """Render one slide-style frame.

    Args:
        title: Slide title.
        lines: Slide body lines.
        step: Current slide number.
        total_steps: Total slide count.

    Returns:
        Rendered slide frame.
    """
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rounded_rectangle(
        (12, 12, WIDTH - 12, HEIGHT - 12),
        radius=18,
        outline=(30, 41, 59),
        width=2,
    )
    draw.text((PADDING, PADDING), "nexus-llm-router", fill=PROMPT, font=font)
    draw.text((WIDTH - 130, PADDING), f"{step}/{total_steps}", fill=MUTED, font=font)
    draw.text((PADDING, 86), title, fill=TEXT, font=font)
    y_position = 150
    for line, color in lines:
        draw.rounded_rectangle(
            (PADDING, y_position - 10, WIDTH - PADDING, y_position + 36),
            radius=10,
            fill=(15, 23, 42),
            outline=(30, 41, 59),
        )
        draw.text((PADDING + 18, y_position + 4), line, fill=color, font=font)
        y_position += 72
    draw.text(
        (PADDING, HEIGHT - 48),
        "task-aware routing | cost controls | fallback safety | audit rationale",
        fill=MUTED,
        font=font,
    )
    return image


def save_gif(path: Path, frames: list[Image.Image], duration_ms: int) -> None:
    """Save rendered frames as a GIF.

    Args:
        path: Output GIF path.
        frames: Rendered frames.
        duration_ms: Frame duration in milliseconds.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"wrote {path}")


def main() -> None:
    """Write demo GIF artifacts."""
    terminal_frames = [
        render_terminal_frame(visible_lines) for visible_lines in range(1, len(demo_lines()) + 1)
    ]
    save_gif(Path("assets/demo.gif"), terminal_frames, 650)

    use_case_frames = [
        render_slide_frame(title, lines, index + 1, len(use_case_slides()))
        for index, (title, lines) in enumerate(use_case_slides())
    ]
    save_gif(Path("assets/use-cases.gif"), use_case_frames, 1350)

    decision_flow_frames = [
        render_slide_frame(title, lines, index + 1, len(decision_flow_slides()))
        for index, (title, lines) in enumerate(decision_flow_slides())
    ]
    save_gif(
        Path("assets/decision-flow.gif"),
        decision_flow_frames,
        1200,
    )


if __name__ == "__main__":
    main()
