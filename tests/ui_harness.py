"""Helpers the `hpca.ui` tests share.

`RowUI` is synchronous and does no I/O, so the whole harness is "construct it,
feed keys through `handle()`, read frames back from `render()`". The first four
helpers are everything on top of that: strip the SGR so a frame can be searched
as plain text, drive the escape-stop clock by hand instead of sleeping through
its window, and find a row worth aiming at.

`Peer` and `Wire`, at the bottom, add the one thing M2 needs beyond that: a
scripted core at the far end of a real `InProcessConnection.pair()`, so a test
can say what the core said and then read the frame, or press a key and then
read the command that went out.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass

from hpca import protocol
from hpca.transport import InProcessConnection
from hpca.ui import state
from hpca.ui.ansi import cell_width
from hpca.ui.app import CHAT, RowUI
from hpca.ui.client import UIClient

# Every style is an SGR sequence, so one pattern strips a frame back to what a
# terminal would actually show — which is what the width assertions count.
SGR = re.compile(r"\x1b\[[0-9;]*m")


def plain(line: str) -> str:
    return SGR.sub("", line)


def frame(ui: RowUI, width: int, height: int) -> list[str]:
    """One rendered frame, styles removed."""
    return [plain(x) for x in ui.render(width, height)]


def widths(lines: list[str]) -> set[int]:
    """The distinct visible widths in a frame — `{width}` if it is well formed.

    Counted in terminal *cells*, not characters: an emoji occupies two of them
    and a combining mark none, so a frame full of ASCII and a frame full of CJK
    are only comparable on this scale — and cells are what the differential
    repaint is addressed in.
    """
    return {cell_width(plain(x)) for x in lines}


def clocked(ui: RowUI) -> RowUI:
    """Drive the escape window by hand instead of sleeping through it."""
    ui._now = 100.0
    ui.clock = lambda: ui._now
    return ui


def at_wall(ui: RowUI, when: float) -> RowUI:
    """Put the *wall* clock at an instant, which is a different clock.

    The escape window is measured with `clock` (monotonic, a gesture); how
    long a turn has been running is measured with `wall` against a stamp the
    core took (`TurnActivity.started_at`), and the two processes share a wall
    clock and not a monotonic one.
    """
    ui.wall = lambda: when
    return ui


def recorded(ui: RowUI) -> RowUI:
    """Keep a copy of every intent, whatever else is listening for them.

    `RowUI` records intents in ``ui.intents`` only when nothing else is wired
    to `send` — and a UI built by `demo.build` has a client on the other end.
    This keeps both, so a test can drive the demo *and* say what a keypress
    asked the core to do without going through the wire to find out.
    """
    original = ui.send

    def record(intent) -> None:
        ui.intents.append(intent)
        original(intent)

    ui.send = record
    return ui


def served(ui: RowUI, bodies=None, skills=None) -> RowUI:
    """Answer the read paths the screens fetch when they open.

    Every editable body is a round trip now (`profile.get`, `skill.get`,
    `settings.get`), so a `RowUI` with no core behind it opens its editors on
    "fetching…" and never fills them. This is the smallest thing that can
    answer: the intents still land in ``ui.intents``, and each `Fetch` is
    answered in place — synchronously, which is what lets a test press the key
    and read the frame in the next line.

    A key with no entry in ``bodies`` is answered with an *error*, which is the
    core's way of saying a file could not be read and the only thing that
    stops an editor opening over it.
    """
    bodies = {} if bodies is None else bodies
    skills = {} if skills is None else skills
    original = ui.send
    # What `UIClient` wires: the menu asks for a profile's skills the first
    # time it wants them, and the answer arrives as an intent this handles.
    # The *visible* scope, as the client asks for it — everything callable,
    # the shipped skills included.
    ui.skills_loader = lambda profile: ui.send(
        state.FetchSkills(profile, scope="visible")
    )

    def answer(intent) -> None:
        original(intent)
        if isinstance(intent, state.Fetch):
            key = (intent.what, tuple(intent.key))
            text = bodies.get(key)
            ui.body_arrived(
                key, text or "", "" if text is not None else "could not be read"
            )
        elif isinstance(intent, state.FetchSkills):
            # Answered by scope, as the core answers it: the menu's list and
            # the editable one are two different questions, and a harness that
            # answered both with one list would hide exactly that.
            rows = list(skills.get(intent.profile, []))
            if intent.scope == "visible":
                ui.skills_listed(intent.profile, rows)
                return
            own = [x for x in rows if x.removable]
            ui.own_skills_listed(intent.profile, own)
            ui.list_arrived(("skills", intent.profile), own)

    ui.send = answer
    return ui


def on_own_message(ui: RowUI, nth: int = 4) -> int:
    """Put the chat cursor on the nth message the user wrote — not the first,
    so that a fork or a rollback actually has a conversation to cut."""
    ui.focus = CHAT
    ui.chat.cursor = 0
    seen = 0
    while True:
        index = ui.chat.current(118)
        if ui.chat.items[index].kind == "user":
            seen += 1
            if seen == nth:
                assert index > 0, "the cut has to be mid-conversation to mean anything"
                return index
        was = ui.chat.cursor
        ui.chat.move(1, 20, 118)
        if ui.chat.cursor == was:
            raise AssertionError("ran out of chat before finding one")


# ------------------------------------------------- driving a scripted core

# The M2 harness. `RowUI` is still synchronous and still pure, so the only
# thing the client adds is a wire to pump: a real `InProcessConnection.pair()`
# rather than a mocked transport, because the two ends' behaviour at the edges
# — a closed peer, a dropped frame — is exactly what a mock would get wrong.


async def settle(turns: int = 20) -> None:
    """Let the queues drain.

    `InProcessConnection` is a pair of asyncio queues, so a frame put in one is
    delivered the moment the reader task next runs. Yielding a few times is
    deterministic and enough — nothing here sleeps or polls, and a test that
    needed more turns than this would be waiting on something that is not the
    wire.
    """
    for _ in range(turns):
        await asyncio.sleep(0)


class Peer:
    """The core, scripted: it says what a test tells it to and keeps what it
    hears, so an assertion can be made about either direction."""

    def __init__(self, conn) -> None:
        self.conn = conn
        self.commands: list[protocol.Message] = []

    async def listen(self) -> None:
        async for env in self.conn:
            self.commands.append(protocol.parse(env))

    def took(self, kind: type | None = None) -> list[protocol.Message]:
        return [c for c in self.commands if kind is None or isinstance(c, kind)]

    def last(self, kind: type) -> protocol.Message:
        found = self.took(kind)
        assert found, f"no {kind.__name__} was sent: {self.kinds()}"
        return found[-1]

    def kinds(self) -> list[str]:
        return [type(c).__name__ for c in self.commands]

    def clear(self) -> None:
        self.commands.clear()


@asynccontextmanager
async def connected(ui: RowUI | None = None):
    """A UI, a client and a scripted core, over a real pair of connections.

    The transport is not mocked: `InProcessConnection.pair()` is the same
    object `coreproc` hands the client, so the edges — a closed peer, a
    dropped frame — behave here as they do in a real run.
    """
    ui = RowUI() if ui is None else ui
    ours, theirs = InProcessConnection.pair()
    client = UIClient(ui, ours)
    peer = Peer(theirs)
    tasks = [asyncio.create_task(client.run()), asyncio.create_task(peer.listen())]
    try:
        yield Wire(ui, client, peer)
    finally:
        await ours.close()
        await theirs.close()
        for task in tasks:
            task.cancel()


@dataclass
class Wire:
    """One UI, one client, one scripted core, and the two verbs between them.

    `tell` is the core speaking and `press` is the user typing; both end by
    flushing whatever the other side now has to say, which is the shape of
    `run.py`'s loop with the terminal and the event loop taken out.
    """

    ui: RowUI
    client: UIClient
    peer: Peer
    width: int = 120
    height: int = 40

    async def tell(self, *events: protocol.Event) -> None:
        for event in events:
            await self.peer.conn.send(event)
        await settle()
        await self.client.flush()
        await settle()

    async def press(self, *keys: str) -> None:
        for key in keys:
            self.ui.handle(key, self.width, self.height)
        await self.client.flush()
        await settle()

    def frame(self) -> list[str]:
        return frame(self.ui, self.width, self.height)

    def screen(self) -> str:
        return "\n".join(self.frame())

    @property
    def inner(self) -> int:
        """The width a pane flattens at: the terminal less its gutter."""
        return max(8, self.width - 2)
