"""The new UI against a real backend, end to end.

Everything else in the UI suite runs against a scripted peer, which is the
right default: it is fast, hermetic, and it proves the translation. What it
cannot prove is that the layers *join up* — that `boot` opens the databases in
an order the service accepts, that a real graph's state survives
`build_entries` and the wire, and that what comes back is drawable. This is the
one test that runs the whole stack, so it is the one that catches a seam nobody
owns.

Opt in with ``pixi run -e dev test-live``. It runs a real generation.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from hpca.config import Settings
from hpca.protocol import SessionNew, SessionOpen, TurnSubmit
from hpca.transport import InProcessConnection
from hpca.ui.app import RowUI
from hpca.ui.boot import Core
from hpca.ui.client import UIClient

# ---- integration tests
from tests.live_backend import LIVE_KEY, LIVE_MODEL, LIVE_URL, integration  # noqa: E402

SGR = re.compile(r"\x1b\[[0-9;]*m")

# The model is asked for one word so the assertion is about the plumbing and
# not about the model's prose. Long enough to be unmistakable in a frame,
# short enough that a slow backend does not turn this into a five-minute test.
ASK = "Reply with exactly the word: pineapple. Nothing else."
ANSWER = "pineapple"


async def _await_event(seen: list, kind: str, timeout: float) -> object:
    """Wait for one event type, and say what did arrive when it never comes."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for env in seen:
            if env.type == kind:
                return env
        await asyncio.sleep(0.05)
    raise AssertionError(f"no {kind} within {timeout}s; saw {[e.type for e in seen]}")


class _Live:
    """A booted core, a client and a UI, wired the way `hpca` wires them."""

    def __init__(self, core: Core, ui: RowUI, client: UIClient, seen: list) -> None:
        self.core, self.ui, self.client, self.seen = core, ui, client, seen

    async def do(self, command) -> None:
        self.client.command(command)
        await self.client.flush()

    async def wait(self, kind: str, timeout: float = 300.0):
        return await _await_event(self.seen, kind, timeout)

    def frame(self, width: int = 100, height: int = 24) -> list[str]:
        return [SGR.sub("", line) for line in self.ui.render(width, height)]


@pytest.fixture
async def live(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    settings = Settings()
    settings.llm.base_url = LIVE_URL
    settings.llm.model = LIVE_MODEL
    settings.llm.api_key = LIVE_KEY

    ui_end, core_end = InProcessConnection.pair()
    core = await Core.start(settings=settings, app_dir=tmp_path, wire=core_end)
    core.run()

    ui = RowUI()
    client = UIClient(ui, ui_end)
    seen: list = []
    inner = client.apply

    def spy(env):
        seen.append(env)
        return inner(env)

    client.apply = spy
    pump = asyncio.create_task(client.run())
    try:
        yield _Live(core, ui, client, seen)
    finally:
        pump.cancel()
        await core.stop(say=lambda _text: None)


@integration
async def test_a_real_turn_reaches_the_screen(live):
    await live.do(SessionNew(profile="default"))
    created = await live.wait("session.created", 60)
    session_id = created.payload["row"]["session_id"]

    await live.do(SessionOpen(session_id=session_id))
    await live.wait("chat.reset", 60)

    await live.do(TurnSubmit(session_id=session_id, text=ASK))
    await live.wait("turn.started", 120)
    finished = await live.wait("turn.finished", 300)
    assert ANSWER in (finished.payload.get("reply") or "").lower()

    # Re-open rather than reading the live frame: nothing emits chat.append for
    # turn content yet, so the reply reaches the screen only through a reset.
    # When M4 lands this re-open becomes unnecessary and the assertion below
    # should be moved to the frame taken straight after `turn.finished` — that
    # difference is the whole of what M4 buys.
    live.seen.clear()
    await live.do(SessionOpen(session_id=session_id))
    await live.wait("chat.reset", 60)

    frame = live.frame()
    assert any(ANSWER in line.lower() for line in frame), frame
    assert any(ASK[:20] in line for line in frame), frame


@integration
async def test_every_row_of_a_real_frame_is_exactly_the_terminal_width(live):
    """A real reply carries whatever the model wrote — the padding must hold.

    The hermetic tests draw content this repo chose. This one draws content a
    model chose, which is the case that finds a width bug.
    """
    await live.do(SessionNew(profile="default"))
    created = await live.wait("session.created", 60)
    session_id = created.payload["row"]["session_id"]

    await live.do(TurnSubmit(session_id=session_id, text=ASK))
    await live.wait("turn.finished", 300)

    await live.do(SessionOpen(session_id=session_id))
    await live.wait("chat.reset", 60)

    for width in (80, 100, 137):
        assert {len(line) for line in live.frame(width, 24)} == {width}
