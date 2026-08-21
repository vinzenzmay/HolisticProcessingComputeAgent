"""What the UI holds between frames, and what it asks the core to do.

Plain dataclasses and one class with panes in it. No pydantic, no protocol, no
I/O — `client.py` is the only module that has heard of a wire, and it fills
these in from events; `app.py` renders them and never sees anything else
(specs-ui-replacement.md §3.1).

Two shapes cross the seam in each direction:

* **state**, below the fold: a `SessionState` per conversation, holding the
  chat, the draft, the watchers, the turn and the context meter. The chat is
  appended to and revised in place, never rebuilt — that invariant (§3.2) is
  the reason `ChatEntry.seq` is carried all the way through to `Item.key`
  rather than the UI numbering its own rows;
* **intents**, at the bottom: what a keypress *means*, with no opinion about
  which command carries it. `app.py` builds one of these and hands it to a
  callable; `client.py` turns it into a `protocol.Command`. That is the whole
  of what keeps the key dispatch free of the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hpca.ui.ansi import BLUE, DIM, RED, YELLOW
from hpca.ui.editor import Editor
from hpca.ui.pane import Item, Pane

# The kinds of chat entry that are the user's own words, and so the ones Enter
# offers the rewind on. `queued` counts: it is a message the user wrote, drawn
# ahead of the turn that will send it.
OWN_MESSAGE_KINDS = ("user", "queued")

# What a step's label is padded to in an opened entry, so tool names line up.
TOOL_COLUMN = 14


# ------------------------------------------------------------------- the chat


@dataclass
class ChatPart:
    """One piece of a turn's working: reasoning, or a tool call and its result.

    The plain twin of `protocol.Part`, which is itself the wire twin of
    `hpca.transcript.Step`. Three copies of six fields looks like duplication
    and is the price of the two boundaries: the wire may not import the agent,
    and the renderer may not import pydantic.
    """

    kind: str = ""
    text: str = ""
    tool: str = ""
    target: str = ""
    result: str = ""
    done: bool = False
    failed: bool = False


@dataclass
class ChatEntry:
    """One row of chat, as the UI holds it.

    ``seq`` is the core's name for this row and the only thing the UI keys it
    on — not its position, which every fold and every rollback changes. 0 means
    the core did not number it (an error the UI wrote itself), and such a row
    can never be revised.

    ``index`` is a different number and must not be confused with ``seq``: it
    is the thread *message* this entry is, or -1, and it is what the rewind
    names when it asks the core to fork or roll back.
    """

    kind: str = ""
    text: str = ""
    seq: int = 0
    index: int = -1
    steps: int = 0
    reasoning_chars: int = 0
    parts: list[ChatPart] = field(default_factory=list)


def _one_line(text: str) -> str:
    """A head line is one line; a chat message is not. Collapse it."""
    return " ".join(text.split())


def _part_lines(part: ChatPart) -> list[str]:
    """One step, opened: what was called, and what came back."""
    label = part.tool or part.kind or "step"
    detail = part.target or _one_line(part.text)
    head = f"{label:<{TOOL_COLUMN}}{detail}".rstrip()
    if not part.result:
        # A call whose result has not landed says so, because the alternative
        # is a row that looks finished and is not (`Part.done`).
        return [head] if part.done else [f"{head} …"]
    return [head] + [f"{' ' * TOOL_COLUMN}{line}" for line in part.result.split("\n")]


def entry_item(entry: ChatEntry) -> Item:
    """The row an entry draws as.

    The one place a `ChatEntry` becomes something with colours in it, so that
    every path into the chat — reset, append, update — produces the same row
    for the same entry, and an update genuinely replaces what it revises.
    """
    said = _one_line(entry.text)
    body = entry.text.split("\n") if "\n" in entry.text else []
    # Every row carries the core's name for it, which is what `chat.update`
    # addresses and what `Pane.expanded` remembers.
    row = dict(kind=entry.kind, text=entry.text, key=str(entry.seq))
    if entry.kind == "user":
        return Item(head=f"you   {said}", accent=BLUE, **row)
    if entry.kind == "queued":
        return Item(head=f"…     {said}", accent=DIM, **row)
    if entry.kind == "error":
        return Item(head=f"!     {said}", body=body, accent=RED, **row)
    if entry.kind == "thinking":
        steps = entry.steps or len(entry.parts)
        names = [x.tool or x.kind for x in entry.parts if x.tool or x.kind]
        summary = " → ".join(names[:3]) + (" …" if len(names) > 3 else "")
        lines: list[str] = []
        for part in entry.parts:
            lines += _part_lines(part)
        return Item(
            head=f"      {steps} steps" + (f" · {summary}" if summary else ""),
            body=lines,
            **row,
        )
    if entry.kind in ("event", "recall"):
        mark = "↺" if entry.kind == "recall" else "·"
        return Item(head=f"      {mark} {said}", body=body, accent=DIM, **row)
    # Anything else is drawn as the agent talking, including a kind this
    # renderer has never heard of: the text is what matters and dropping the
    # row would lose it.
    return Item(
        head=f"hpca  {said}",
        body=body or [entry.text],
        accent=YELLOW,
        **{**row, "kind": entry.kind or "assistant"},
    )


# ------------------------------------------------------------- the turn and it


@dataclass
class Turn:
    """What a session's turn is doing, as far as the UI is concerned.

    ``started_at`` is the core's stamp rather than a local clock, so the
    elapsed count survives the UI being slow — and it is only taken when the
    *activity* changes, because a repeated `turn.activity` for the same work is
    a heartbeat, not a restart (§3.2).
    """

    working: bool = False
    activity: str = ""
    started_at: str = ""

    def activity_is(self, activity: str, started_at: str) -> None:
        if activity and activity == self.activity:
            return  # the same work, said again: the clock keeps running
        self.activity = activity
        self.started_at = started_at


@dataclass
class Context:
    """The context meter: how full the window is, and how sure we are.

    ``measured`` is what `turn.usage` sets and `context.estimate` respects — a
    real prompt_tokens from the backend beats a local estimate, and the
    estimate only speaks again once the thread it described is gone (a reset).
    """

    used: int = 0
    window: int = 0
    measured: bool = False

    @property
    def percent(self) -> int:
        return round(100 * self.used / self.window) if self.window else 0

    def label(self) -> str:
        """`31% ctx`, or `~31% ctx` while it is only an estimate."""
        if not self.window:
            return ""
        return f"{'' if self.measured else '~'}{self.percent}% ctx"


@dataclass
class Toast:
    text: str
    severity: str = "information"
    timeout: float | None = None


@dataclass
class Confirm:
    """A yes/no question that is not a tool approval (`confirm.requested`)."""

    id: str
    question: str


@dataclass
class Proposal:
    scope: str
    kind: str
    text: str


# ---------------------------------------------------------------- the session


class SessionState:
    """Everything that belongs to one conversation rather than to the app.

    Switching sessions swaps this and nothing else, which is why the chat's
    cursor line, its open entries and a half-typed message all survive going
    away and coming back — the same property each row has, one level up.

    A session's chat is *given* to it, by `chat.reset` when the core opens the
    conversation and by `chat.append` after that. It is never manufactured
    here: a model layer that can generate its own content is a model layer that
    can disagree with the core about what was said.
    """

    def __init__(
        self,
        session_id: str,
        *,
        title: str = "",
        profile: str = "",
        mode: str = "",
        flags: tuple[str, ...] = (),
        model: str = "",
    ) -> None:
        self.session_id = session_id
        self.title = title
        self.profile = profile
        self.mode = mode
        self.flags = list(flags)
        # Not on the wire: `protocol.SessionRow` carries no model. Kept so the
        # message row has somewhere to read one from when it does (M4).
        self.model = model
        self.draft = Editor(wrap=True)
        self.chat = Pane("chat", [])
        self.watchers = Pane("watchers", [])
        self.turn = Turn()
        self.context = Context()
        self.decision: dict | None = None
        self.proposals: list[Proposal] = []
        self.entries: list[ChatEntry] = []
        # seq -> position in `entries` / `chat.items`, which are parallel.
        # The map, not a scan: a `chat.update` during a long turn arrives once
        # per tool call, and a linear search per update is O(chat) per step.
        self._rows: dict[int, int] = {}
        self.loaded = False

    # -------------------------------------------------------------- the chat

    def reset(self, entries: list[ChatEntry]) -> None:
        """Replace the transcript — the snapshot the deltas build on.

        The only path that may throw rows away, and it throws away what was
        open with them: a reset re-bases the numbering, so a `seq` that is
        still in `expanded` afterwards would be naming a different row.
        """
        self.entries = list(entries)
        # The thread this described is gone, so a fresh estimate may speak
        # again (`protocol.SessionRollback`).
        self.context.measured = False
        self.chat.items = [entry_item(entry) for entry in self.entries]
        self._rows = {
            entry.seq: i for i, entry in enumerate(self.entries) if entry.seq
        }
        self.chat.expanded.clear()
        self.chat.invalidate()
        self.chat.cursor = 10**9  # open at the newest, as the old app does
        self.loaded = True

    def append(self, entry: ChatEntry) -> None:
        """One new row. The only path by which a chat grows (§3.2)."""
        if entry.seq:
            self._rows[entry.seq] = len(self.entries)
        self.entries.append(entry)
        self.chat.items.append(entry_item(entry))
        self.chat.invalidate()
        self.chat.cursor = 10**9
        self.loaded = True

    def update(self, entry: ChatEntry) -> bool:
        """Revise a row already on screen. False if there is no such row.

        An update for a `seq` this session does not have is dropped rather than
        appended: it means the client is out of step with a reset, and
        inventing a row would hide that instead of letting the next reset fix
        it (`protocol.ChatUpdate`).
        """
        row = self._rows.get(entry.seq) if entry.seq else None
        if row is None:
            return False
        self.entries[row] = entry
        self.chat.items[row] = entry_item(entry)
        self.chat.invalidate()
        return True

    def entry_at(self, position: int) -> ChatEntry | None:
        """The entry a pane position is showing, if it is showing one."""
        if 0 <= position < len(self.entries):
            return self.entries[position]
        return None

    def entry_of(self, seq: int) -> ChatEntry | None:
        row = self._rows.get(seq)
        return None if row is None else self.entries[row]

    # ------------------------------------------------------------ the panels

    def set_watchers(self, items: list[Item]) -> None:
        """Repaint the right column without moving the cursor off its row."""
        self.watchers.replace(items)

    @property
    def watch_count(self) -> int:
        return len(self.watchers.items)

    @property
    def working(self) -> bool:
        return self.turn.working

    def invalidate(self) -> None:
        for pane in (self.chat, self.watchers):
            pane.invalidate()


# ---------------------------------------------------------------- the intents


@dataclass(frozen=True)
class SidebarRow:
    """One line of the session list, as the core last described it.

    A snapshot of `protocol.SessionRow` with the pydantic taken off, and the
    only thing `sync_sessions` is allowed to change about a session it already
    holds: the chat, the draft and the cursor line are the UI's, not the
    core's, and a repaint of the sidebar must not touch them.
    """

    session_id: str
    title: str = ""
    profile: str = ""
    mode: str = ""
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpenSession:
    """Show me this conversation."""

    session_id: str


@dataclass(frozen=True)
class Submit:
    """Send this message."""

    session_id: str
    text: str


@dataclass(frozen=True)
class Interrupt:
    """Stop whatever this session is doing."""

    session_id: str


@dataclass(frozen=True)
class Fork:
    """Branch the conversation just before the row named by ``seq``.

    A row, not a message index: the UI knows its rows by the name the core gave
    them and by nothing else. Turning that into the `Entry.index` the wire
    wants is `client.py`'s job, because it is the side that knows what an
    index is.
    """

    session_id: str
    seq: int


@dataclass(frozen=True)
class Rollback:
    """Trim the conversation back to just before the row named by ``seq``."""

    session_id: str
    seq: int


@dataclass(frozen=True)
class Peek:
    """What is this watch box saying right now?"""

    ref: str


@dataclass(frozen=True)
class Drop:
    """Stop watching this."""

    ref: str


Intent = (
    OpenSession | Submit | Interrupt | Fork | Rollback | Peek | Drop
)
