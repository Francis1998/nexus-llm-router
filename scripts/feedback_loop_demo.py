"""Close the feedback loop against a running nexus: rate answers, summarize, route by feedback.

Start nexus with mock providers and a low cold-start threshold, then run the demo::

    NEXUS_FEEDBACK_MIN_SAMPLES=10 PYTHONPATH=src python scripts/serve_mock.py --port 8765 &
    PYTHONPATH=src python scripts/feedback_loop_demo.py --base-url http://127.0.0.1:8765

The session sends one ``feedback-aware`` request (cold start), twelve answers from
three models that a scripted reviewer rates with ``POST /v1/feedback``, reads
``GET /v1/feedback/summary``, routes six more ``feedback-aware`` requests, and
shows that another API key cannot rate the caller's requests. ``--gif`` renders
the printed transcript with the shared demo-GIF helpers in ``create_demo_gif``.
"""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from create_demo_gif import (
    ACCENT,
    BACKGROUND,
    HEIGHT,
    MUTED,
    PADDING,
    PROMPT,
    SUCCESS,
    TEXT,
    WARN,
    WIDTH,
    Line,
    load_font,
    save_gif,
    shared_palette,
    wrap,
)
from PIL import Image, ImageDraw, ImageFont

Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

HEADERS = {"Authorization": "Bearer sk-demo-local", "X-Tenant-Id": "acme"}
OTHER_TEAM = {"Authorization": "Bearer sk-other-team", "X-Tenant-Id": "acme"}
# A general-domain prompt that rule-based answers with an explicitly requested model.
PROMPT_TEMPLATE = (
    "Design a multi-region failover strategy for a {system}: compare active-active, "
    "active-passive, and pilot-light architectures, analyze cost, latency, and consistency "
    "trade-offs, and recommend one with a phased rollout plan."
)
SYSTEMS = (
    "payments platform",
    "ride-sharing app",
    "video streaming service",
    "online bookstore",
    "food delivery app",
    "travel booking site",
    "photo sharing app",
    "music streaming service",
    "banking portal",
    "ticketing system",
    "messaging app",
    "fitness tracker",
    "news site",
    "job board",
    "smart home hub",
    "school portal",
    "event planner",
    "recipe site",
    "parking app",
)
# The scripted reviewer: thumbs up (True) or down (False) for four answers per model.
REVIEWS = (
    ("gpt-6-luna", (False, False, True, False)),
    ("claude-opus-5-5", (True, True, True, True)),
    ("gpt-6-astra", (True, False, True, False)),
)
FEEDBACK_AWARE_REQUESTS = 6
# Printed widths of every summary-table column but the last.
TABLE_WIDTHS = (18, 4, 11, 7)
FRAME_MS = 1700
LINE_HEIGHT = 20


class DemoError(RuntimeError):
    """Raised when the server does not behave like nexus with mock providers."""


class Session:
    """HTTP client for the demo plus the transcript it prints."""

    def __init__(self, base_url: str) -> None:
        """Open the client.

        Args:
            base_url: Server root, for example ``http://127.0.0.1:8765``.
        """
        self._client = httpx.Client(base_url=base_url, timeout=30.0)
        self._prompts = iter(SYSTEMS)
        self.steps: list[list[Line]] = []

    def step(self, title: str) -> None:
        """Start a transcript section."""
        self.steps.append([])
        self.say(f"# {title}", MUTED)

    def say(self, text: str, color: tuple[int, int, int] = TEXT) -> None:
        """Print one transcript line and keep it for the GIF; tabs separate table cells."""
        *padded, last = text.split("\t")
        widths = zip(padded, TABLE_WIDTHS, strict=False)
        print("".join(cell.ljust(width) for cell, width in widths) + last)
        self.steps[-1].append((text, color))

    def chat(self, *, model: str | None = None, strategy: str | None = None) -> dict[str, Any]:
        """Send one chat completion and return its ``router`` block."""
        body: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": PROMPT_TEMPLATE.format(system=next(self._prompts))}
            ]
        }
        headers = dict(HEADERS)
        if model is not None:
            body["model"] = model
        if strategy is not None:
            headers["X-Router-Strategy"] = strategy
        response = self._client.post("/v1/chat/completions", json=body, headers=headers)
        if response.status_code != 200:
            raise DemoError(f"chat completion failed: HTTP {response.status_code} {response.text}")
        payload = response.json()
        if not payload["choices"][0]["message"]["content"].endswith("routed response"):
            raise DemoError("expected mock providers; start the server with scripts/serve_mock.py")
        router: dict[str, Any] = payload["router"]
        return router

    def feedback(self, request_id: str, value: bool, headers: dict[str, str]) -> httpx.Response:
        """Post one thumbs-up/down value for a request."""
        return self._client.post(
            "/v1/feedback", json={"request_id": request_id, "value": value}, headers=headers
        )

    def summary(self) -> dict[str, Any]:
        """Return the caller's quality summary grouped by model."""
        response = self._client.get(
            "/v1/feedback/summary", params={"group_by": "model"}, headers=HEADERS
        )
        if response.status_code != 200:
            raise DemoError(f"summary failed: HTTP {response.status_code} {response.text}")
        summary: dict[str, Any] = response.json()
        return summary


def run(session: Session) -> None:
    """Run the feedback loop and print what nexus answered at each step."""
    session.step("1) cold start: no feedback yet, so feedback-aware defers to rule-based")
    cold = session.chat(strategy="feedback-aware")
    session.say("POST /v1/chat/completions   X-Router-Strategy: feedback-aware")
    session.say(f"  -> {cold['model_used']}: {cold['rationale']}", WARN)

    total = sum(len(values) for _, values in REVIEWS)
    session.step(f"2) {total} answers from three models, each rated by a scripted reviewer")
    rated: list[str] = []
    for model, values in REVIEWS:
        ids: list[str] = []
        statuses: list[str] = []
        for value in values:
            routed = session.chat(model=model)
            if routed["model_used"] != model:
                raise DemoError(f"asked for {model}, served {routed['model_used']}")
            response = session.feedback(routed["request_id"], value, HEADERS)
            if response.status_code != 200:
                raise DemoError(f"feedback failed: HTTP {response.status_code} {response.text}")
            ids.append(routed["request_id"][:8])
            statuses.append(str(response.status_code))
            rated.append(routed["request_id"])
        votes = " ".join("up" if value else "down" for value in values)
        session.say(
            f"{model:<16} request ids {' '.join(ids)}  ->  POST /v1/feedback "
            f"[{votes}]  HTTP {' '.join(statuses)}"
        )

    summary = session.summary()
    session.step(
        f"3) GET /v1/feedback/summary?group_by=model   kind={summary['kind']}, "
        f"interval={summary['interval']} ({summary['confidence']:.0%}), n={summary['n']}"
    )
    session.say("key\tn\tpositives\tmean\t95% CI", ACCENT)
    for group in summary["groups"]:
        session.say(
            f"{group['key']}\t{group['n']}\t{group['positives']}\t{group['mean']:.2f}"
            f"\t[{group['ci_low']:.2f}, {group['ci_high']:.2f}]"
        )

    session.step("4) feedback-aware again: Thompson sampling, seeded by each request id")
    decisions = [session.chat(strategy="feedback-aware") for _ in range(FEEDBACK_AWARE_REQUESTS)]
    served = Counter(decision["model_used"] for decision in decisions)
    session.say(
        f"POST /v1/chat/completions x{FEEDBACK_AWARE_REQUESTS}   X-Router-Strategy: feedback-aware"
    )
    session.say(
        "  -> served " + ", ".join(f"{model} x{count}" for model, count in served.most_common()),
        SUCCESS,
    )
    session.say(f"  rationale: {decisions[0]['rationale']}")

    session.step("5) only the API key and tenant that made a request can rate it")
    foreign = session.feedback(rated[0], True, OTHER_TEAM)
    code = foreign.json()["error"]["code"] if foreign.status_code != 200 else "accepted"
    session.say(
        f"POST /v1/feedback  Authorization: Bearer sk-other-team  request {rated[0][:8]}"
        f"  -> HTTP {foreign.status_code} {code}",
        WARN,
    )


def _column_offsets(step: Sequence[Line], font: Font) -> list[int]:
    """Return x offsets that align the tab-separated cells of a section's table rows."""
    widths: list[int] = []
    for text, _ in step:
        if "\t" not in text:
            continue
        for index, cell in enumerate(text.split("\t")):
            width = int(font.getlength(cell))
            if index == len(widths):
                widths.append(width)
            else:
                widths[index] = max(widths[index], width)
    offsets, x = [], 0
    for width in widths:
        offsets.append(x)
        x += width + 28
    return offsets


def render_gif(path: Path, command: str, base_url: str, steps: Sequence[list[Line]]) -> None:
    """Render the transcript step by step into a terminal-style GIF.

    Args:
        path: Output GIF path.
        command: Command line that produced the transcript.
        base_url: Server the session ran against.
        steps: Transcript sections; each frame reveals one more section.
    """
    title_font, body_font = load_font(16), load_font(15)
    text_width = WIDTH - 2 * PADDING - 8
    top = PADDING + 58
    rows = sum(len(wrap(text, body_font, text_width)) for step in steps for text, _ in step)
    height = max(HEIGHT, top + rows * LINE_HEIGHT + len(steps) * 8 + PADDING)
    frames: list[Image.Image] = []
    for visible in range(1, len(steps) + 1):
        image = Image.new("RGB", (WIDTH, height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (12, 12, WIDTH - 12, height - 12), radius=16, outline=(30, 41, 59), width=2
        )
        draw.text((PADDING, PADDING), f"$ {command}", fill=PROMPT, font=title_font)
        subtitle = f"real session against scripts/serve_mock.py at {base_url} (mock providers)"
        draw.text((PADDING, PADDING + 26), subtitle, fill=MUTED, font=body_font)
        y = top
        for step in steps[:visible]:
            offsets = _column_offsets(step, body_font)
            for text, color in step:
                if "\t" in text:
                    for offset, cell in zip(offsets, text.split("\t"), strict=True):
                        draw.text((PADDING + offset, y), cell, fill=color, font=body_font)
                    y += LINE_HEIGHT
                    continue
                for chunk in wrap(text, body_font, text_width):
                    draw.text((PADDING, y), chunk, fill=color, font=body_font)
                    y += LINE_HEIGHT
            y += 8
        frames.append(image)
    frames = shared_palette(frames)
    save_gif(path, [*frames, frames[-1], frames[-1]], FRAME_MS)


def main_cli(argv: Sequence[str] | None = None) -> int:
    """Parse flags, run the session, and optionally render the GIF."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gif", type=Path, help="write the transcript as a GIF to this path")
    args = parser.parse_args(argv)
    session = Session(args.base_url)
    try:
        run(session)
    except (DemoError, httpx.HTTPError) as error:
        print(f"demo failed: {error}", file=sys.stderr)
        return 1
    if args.gif is not None:
        flags = " ".join(sys.argv[1:] if argv is None else argv)
        command = f"PYTHONPATH=src python scripts/feedback_loop_demo.py {flags}"
        render_gif(args.gif, command, args.base_url, session.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
