"""Issue, use, limit, and revoke a virtual key against a running nexus.

Start nexus in ``virtual_keys`` mode with mock providers, then run the demo::

    export NEXUS_AUTH_MODE=virtual_keys NEXUS_MASTER_KEY=sk-master-demo-0123456789abcdef
    PYTHONPATH=src python scripts/serve_mock.py --port 8766 &
    PYTHONPATH=src python scripts/virtual_keys_demo.py --base-url http://127.0.0.1:8766

The session issues a key with the master key (two models, a tiny daily budget),
calls the Chat Completions and Messages APIs with it, hits the model allowlist,
spends the budget until the forecast refuses a request with 402, reads
``GET /v1/key/info``, and revokes the key (401). ``--gif`` renders the printed
transcript with the shared demo-GIF helpers in ``create_demo_gif``. Secrets are
masked in the transcript.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from create_demo_gif import ACCENT, MUTED, SUCCESS, TEXT, WARN, Line
from feedback_loop_demo import render_gif

DEFAULT_MASTER_KEY = "sk-master-demo-0123456789abcdef"
KEY_FIELDS = {
    "alias": "search-team",
    "models": ["gpt-6.2-sol", "gemini-3.8-flash"],
    "max_budget_usd": 0.0001,
    "budget_duration": "1d",
    "rpm_limit": 30,
}
# Distinct prompts: identical ones would be answered from the response cache, at no cost.
QUESTIONS = (
    "Summarize the refund policy in one line.",
    "Summarize the shipping policy in one line.",
    "Summarize the privacy policy in one line.",
    "Summarize the returns policy in one line.",
    "Summarize the warranty policy in one line.",
    "Summarize the pricing policy in one line.",
    "Summarize the security policy in one line.",
    "Summarize the travel policy in one line.",
    "Summarize the support policy in one line.",
    "Summarize the leave policy in one line.",
)
MAX_BUDGET_CALLS = 6


class DemoError(RuntimeError):
    """Raised when the server does not behave like nexus in virtual_keys mode."""


class Session:
    """HTTP client for the demo plus the transcript it prints."""

    def __init__(self, base_url: str) -> None:
        """Open the client.

        Args:
            base_url: Server root, for example ``http://127.0.0.1:8766``.
        """
        self.client = httpx.Client(base_url=base_url, timeout=30.0)
        self.steps: list[list[Line]] = []
        self._questions = iter(QUESTIONS)

    def question(self) -> str:
        """Return the next distinct prompt."""
        return next(self._questions)

    def step(self, title: str) -> None:
        """Start a transcript section."""
        self.steps.append([])
        self.say(f"# {title}", MUTED)

    def say(self, text: str, color: tuple[int, int, int] = TEXT) -> None:
        """Print one transcript line and keep it for the GIF."""
        print(text)
        self.steps[-1].append((text, color))


def _mask(secret: str) -> str:
    """Show a secret as its first 13 characters (the listed prefix) plus an ellipsis."""
    return secret[:13] + "..."


def _chat(session: Session, headers: dict[str, str], model: str | None = None) -> httpx.Response:
    messages = [{"role": "user", "content": session.question()}]
    body: dict[str, Any] = {"messages": messages, "max_tokens": 8}
    if model is not None:
        body["model"] = model
    return session.client.post("/v1/chat/completions", json=body, headers=headers)


def _expect(response: httpx.Response, status: int) -> dict[str, Any]:
    if response.status_code != status:
        raise DemoError(f"expected HTTP {status}, got {response.status_code}: {response.text}")
    body: dict[str, Any] = response.json()
    return body


def run(session: Session, master_key: str) -> None:
    """Run the virtual-key session and record its transcript.

    Args:
        session: Demo HTTP session.
        master_key: The gateway's ``NEXUS_MASTER_KEY``.
    """
    admin = {"Authorization": f"Bearer {master_key}"}
    session.step("1) the admin issues a key: two models, $0.0001 per UTC day, 30 RPM")
    created = _expect(session.client.post("/v1/keys", json=KEY_FIELDS, headers=admin), 200)
    secret, key_id = created["key"], created["id"]
    session.say(f"POST /v1/keys  Authorization: Bearer {_mask(master_key)}  -> HTTP 200", ACCENT)
    session.say(f"  id={key_id}  key={_mask(secret)} (shown once)  prefix={created['key_prefix']}")
    session.say(f"  models={','.join(created['models'])}  budget=$0.0001/1d  rpm_limit=30")
    caller = {"Authorization": f"Bearer {secret}"}

    session.step("2) the team calls nexus with its key (OpenAI and Anthropic APIs)")
    chat = _expect(_chat(session, caller), 200)
    if not chat["choices"][0]["message"]["content"].endswith("routed response"):
        raise DemoError("expected mock providers; start the server with scripts/serve_mock.py")
    session.say(
        f"POST /v1/chat/completions  -> 200  model={chat['model']}"
        f"  cost=${chat['router']['cost_usd']:.6f}",
        SUCCESS,
    )
    anthropic_body = {
        "model": "auto",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": session.question()}],
    }
    message = session.client.post(
        "/v1/messages", json=anthropic_body, headers={"x-api-key": secret}
    )
    served = _expect(message, 200)["model"]
    session.say(f"POST /v1/messages  x-api-key  -> 200  model={served}", SUCCESS)

    session.step("3) the model allowlist: other catalog models are refused and hidden")
    denied = _chat(session, caller, model="claude-opus-5-5")
    detail = _expect(denied, 403)["detail"]
    session.say(f"POST /v1/chat/completions  model=claude-opus-5-5  -> 403  {detail}", WARN)
    listed = _expect(session.client.get("/v1/models", headers=caller), 200)["data"]
    session.say(f"GET /v1/models  -> {', '.join(model['id'] for model in listed)}")

    session.step("4) spend settles from each answer's real cost until the forecast refuses")
    for _ in range(MAX_BUDGET_CALLS):
        response = _chat(session, caller)
        if response.status_code != 200:
            break
        info = _expect(session.client.get("/v1/key/info", headers=caller), 200)["spend"]
        session.say(
            f"chat  -> 200  spent=${info['spend_usd']:.6f}  remaining=${info['remaining_usd']:.6f}",
            SUCCESS,
        )
    refusal = _expect(response, 402)["detail"]
    session.say(f"chat  -> 402  {refusal}", WARN)

    session.step("5) GET /v1/key/info: the key's own limits, spend, and reset time")
    info = _expect(session.client.get("/v1/key/info", headers=caller), 200)
    spend = info["spend"]
    resets = datetime.fromtimestamp(spend["resets_at"], UTC).strftime("%Y-%m-%dT%H:%MZ")
    session.say(
        f"status={info['status']}  requests={spend['request_count']}"
        f"  window spend=${spend['spend_usd']:.6f}  remaining=${spend['remaining_usd']:.6f}"
        f"  resets_at={resets}"
    )

    session.step("6) the admin revokes the key: 401 at once, still listed for audit")
    revoked = _expect(session.client.delete(f"/v1/keys/{key_id}", headers=admin), 200)
    session.say(f"DELETE /v1/keys/{key_id}  -> status={revoked['status']}", ACCENT)
    after = _chat(session, caller)
    session.say(f"chat  -> {after.status_code}  {_expect(after, 401)['detail']}", WARN)
    audit = _expect(session.client.get("/v1/keys?revoked=true", headers=admin), 200)["data"]
    session.say(
        f"GET /v1/keys?revoked=true  -> {audit[0]['id']}  status={audit[0]['status']}"
        f"  spend=${audit[0]['spend']['total_spend_usd']:.6f}",
        TEXT,
    )


def main_cli(argv: Sequence[str] | None = None) -> int:
    """Parse flags, run the session, and optionally render the GIF."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gif", type=Path, help="write the transcript as a GIF to this path")
    args = parser.parse_args(argv)
    master_key = os.environ.get("NEXUS_MASTER_KEY", DEFAULT_MASTER_KEY)
    session = Session(args.base_url)
    try:
        run(session, master_key)
    except (DemoError, httpx.HTTPError, KeyError) as error:
        print(f"demo failed: {error}", file=sys.stderr)
        return 1
    if args.gif is not None:
        flags = " ".join(sys.argv[1:] if argv is None else argv)
        command = f"PYTHONPATH=src python scripts/virtual_keys_demo.py {flags}"
        render_gif(args.gif, command, args.base_url, session.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
