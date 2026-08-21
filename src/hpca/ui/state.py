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
from datetime import datetime

from hpca.ui.ansi import BLUE, DIM, GREEN, RED, YELLOW
from hpca.ui.approval import Decision
from hpca.ui.editor import Editor
from hpca.ui.meter import render_bar, severity
from hpca.ui.pane import Fold, Item, Pane

# The kinds of chat entry that are the user's own words, and so the ones Enter
# offers the rewind on. `queued` counts: it is a message the user wrote, drawn
# ahead of the turn that will send it.
OWN_MESSAGE_KINDS = ("user", "queued")

# What a step's label is padded to in an opened entry, so tool names line up.
TOOL_COLUMN = 14

# The mode line's copy, lifted from `tui/mode_bar.py` — the hint is the whole
# value of the row: "auto" and "full-auto" differ by whether a destructive
# operation stops to ask, which is not something a user should have to
# remember from the name.
MODE_HINTS = {
    "manual": "scripts run only with your approval",
    "auto": "works until the task is done",
    "full-auto": "asks for nothing, destructive ops included",
}
# ctrl+m is carriage return in most terminals, so shift+tab is the binding
# that always works (§5).
MODE_SWITCH_HINT = "shift+tab to switch"
MODE_COLOURS = {"manual": YELLOW, "auto": GREEN, "full-auto": RED}


def mode_line(mode: str, *, hint: bool = True, switch: bool = True) -> str:
    """`mode: full auto — asks for nothing…`, in one of three lengths.

    The two suffixes come off in the order they can be spared: the key that
    changes it is discoverable from `?`, and the sentence is a reminder rather
    than news, but *which mode is on* is a safety fact and never drops.
    """
    if not mode:
        return ""
    label = mode.replace("-", " ")  # "full-auto" reads as "full auto"
    text = f"mode: {label}"
    if hint and MODE_HINTS.get(mode):
        text += f" — {MODE_HINTS[mode]}"
    if switch:
        text += f" · {MODE_SWITCH_HINT}"
    return text


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


def part_fold(part: ChatPart) -> Fold:
    """One step, as a row of its own inside the turn's fold.

    The head is the call — the tool and what it was aimed at — and the body is
    what came back. Two levels rather than one because a tool result is
    routinely a whole file: the steps of a turn have to stay readable as a
    list, and the four hundred lines `read_file` returned must be one more
    keypress away rather than in between the steps either side of it.

    A call whose result has not landed says so with a trailing "…", because
    the alternative is a row that looks finished and is not (`Part.done`) —
    and that mark is exactly what a live step row is: the call, on screen,
    while the tool is still running.
    """
    label = part.tool or part.kind or "step"
    detail = part.target or _one_line(part.text)
    head = f"{label:<{TOOL_COLUMN}}{detail}".rstrip()
    if not part.done and not part.result:
        head = f"{head} …"
    if part.failed:
        head = f"{head}  ✗"
    return Fold(head=head, body=part.result.split("\n") if part.result else [])


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
        # One collapsed box per turn, opening into its steps: the shape
        # `tui/app.py`'s ThinkingBox and StepBox had between them, minus the
        # two widget classes.
        return Item(
            head=f"      {steps} steps" + (f" · {summary}" if summary else ""),
            folds=[part_fold(part) for part in entry.parts],
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


# The spinner, and how fast it turns. 0.1s is 10 frames a second: fast enough
# to read as motion, and slow enough that a session over a loaded SSH link
# spends a tenth of the repaints a 0.08s spinner would. Nothing else on the
# screen changes by the clock, so this interval *is* the UI's idle cost while
# a turn runs — see `RowUI.next_wake`, which books exactly one wake per frame
# rather than reintroducing a poll.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL = 0.1

# What the spinner says when the core has not named a step yet. The graph's own
# wording for the wait that is not a tool (`agent/graph.py` reports "LLM
# processing", "running <tool>", "compacting context"), so the two never
# disagree about what to call it.
DEFAULT_ACTIVITY = "LLM processing"

# Both routes are named because they are not interchangeable: enter has to be
# aimed at this line, which is the harder thing to do precisely when the agent
# is filling the log, while esc esc works from wherever the user already is.
INTERRUPT_HINT = "(enter or esc esc to interrupt)"

# What the working row is called, in the two namespaces a row has: the key a
# pane remembers it by, and the kind a keypress recognises it as. Neither can
# collide with an entry's — a `seq` is a number and an entry kind comes from
# the protocol's list.
WORKING_KEY = "#working"
WORKING_KIND = "working"


def _epoch(stamp: str) -> float | None:
    """The core's ISO stamp as seconds, or None if it did not send one.

    None rather than "now": a spinner with no stamp counts from zero and says
    nothing false, whereas seeding it with the local clock would claim the
    turn started when this frame was drawn.
    """
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except ValueError:
        return None


@dataclass
class Turn:
    """What a session's turn is doing, as far as the UI is concerned.

    ``started_at`` is the core's stamp rather than a local clock, so the
    elapsed count survives the UI being slow — and it is only taken when the
    *activity* changes, because a repeated `turn.activity` for the same work is
    a heartbeat, not a restart (§3.2). Holding it here rather than on the
    spinner is also what makes "how long since I sent it" survive a session
    switch: leaving redraws the row, and a clock owned by the row would start
    again from zero on the way back.

    ``working`` and ``activity`` are separately meaningful, and the difference
    is what the interrupt hint is read off. A `turn.started` means there is a
    turn to stop; a `turn.activity` on its own is a backend call that is not a
    turn — a silent `/conclude`, a compaction, the titler — which still has to
    time itself and still has nothing to abort.
    """

    working: bool = False
    activity: str = ""
    started_at: str = ""
    # Waiting for the user to approve or refuse a tool call. A turn parked
    # there is *working* and is not stoppable: the scheduler emits
    # `decision.requested` and returns without a `turn.finished`, having
    # already popped its `TurnState` — so `Interrupt` finds nothing to
    # interrupt and the core answers by emitting nothing at all. Kept on the
    # turn rather than read off the session's `Decision` because it is what
    # the working row's own hint is drawn from (specs-ui-coverage.md §4).
    parked: bool = False
    # `started_at` parsed once, because the alternative is parsing an ISO
    # string ten times a second for as long as a turn runs.
    started_epoch: float | None = None

    def activity_is(self, activity: str, started_at: str) -> None:
        if activity and activity == self.activity:
            return  # the same work, said again: the clock keeps running
        self.activity = activity
        self.started_at = started_at
        self.started_epoch = _epoch(started_at)

    @property
    def busy(self) -> bool:
        """Whether anything is in flight worth drawing a spinner for."""
        return self.working or bool(self.activity)

    @property
    def interruptible(self) -> bool:
        """Whether stopping it is a thing that can be done.

        Not merely whether something is running: the core will only stop a
        turn it still holds a `TurnState` for, and a turn parked on an
        approval is one it has already let go of. Saying so here is what keeps
        the gesture honest — the alternative is a footer claiming a stop that
        never happened, on the one path where a turn most often waits.
        """
        return self.working and not self.parked

    @property
    def label(self) -> str:
        return self.activity or DEFAULT_ACTIVITY

    def elapsed(self, now: float) -> int:
        """Whole seconds since the turn started — not since this step did.

        The number answers "how long since I asked?", which is the question
        the user actually has while waiting. Timing each step separately read
        better in theory but hid the total: a turn that spent a minute across
        four steps never showed a number above twenty.
        """
        if self.started_epoch is None:
            return 0
        return max(0, int(now - self.started_epoch))

    def frame(self, now: float) -> str:
        """The braille glyph for this instant.

        Derived from the clock rather than advanced by a tick, so the spinner
        needs nothing to drive it: any frame drawn at time *t* shows the same
        glyph, and a UI that repaints only when something changed can work out
        when this one next will (`next_frame`).
        """
        base = now - (self.started_epoch or 0.0)
        return SPINNER_FRAMES[int(base / SPINNER_INTERVAL) % len(SPINNER_FRAMES)]

    def next_frame(self, now: float) -> float:
        """Seconds until `frame` would answer differently."""
        base = now - (self.started_epoch or 0.0)
        return SPINNER_INTERVAL - (base % SPINNER_INTERVAL)

    def line(self, now: float) -> str:
        """The working row: spinner, step, clock, and how to stop it."""
        seconds = self.elapsed(now)
        elapsed = f" {seconds}s" if seconds else ""
        hint = f"  {INTERRUPT_HINT}" if self.interruptible else ""
        return f"{self.frame(now)} {self.label}…{elapsed}{hint}"


@dataclass
class Context:
    """The context meter: how full the window is, and how sure we are.

    ``measured`` is what `turn.usage` sets and `context.estimate` respects — a
    real prompt_tokens from the backend beats a local estimate, and the
    estimate only speaks again once the thread it described is gone (a reset).

    ``known`` is a different question from ``measured`` and both are needed: a
    session with no reply yet has no number at all, and drawing that as a
    precise zero would say the window is empty when what is true is that
    nobody has counted. The bar is only drawn once something has.
    """

    used: int = 0
    window: int = 0
    measured: bool = False
    known: bool = False
    # Completion tokens over the last request's wall clock, and the session's
    # thinking level. Both arrive from the core: `turn.usage` carries
    # ``completion_tokens`` and ``request_seconds`` as the pair rather than as
    # a rate — dividing them is a rendering decision, and a client that wanted
    # "3.4s for 210 tokens" instead could not get that back out of a rate —
    # and the effort rides on `session.rows`. None until something has been
    # generated; never zero, because "nothing was counted" and "nothing was
    # produced" are different facts and only one of them is worth drawing.
    speed: float | None = None
    effort: str | None = None

    @property
    def percent(self) -> int:
        return round(100 * self.used / self.window) if self.window else 0

    @property
    def severity(self) -> str:
        """ok / warn / danger / unknown — what the meter is coloured by."""
        if not self.known:
            return "ok"
        return severity(self.used, self.window)

    def label(self) -> str:
        """`31% ctx`, or `~31% ctx` while it is only an estimate."""
        if not self.window:
            return ""
        return f"{'' if self.measured else '~'}{self.percent}% ctx"

    def bar(self, cells: int = 28, *, speed: bool = True, effort: bool = True) -> str:
        """The meter as one line, as long as the room allows.

        ``cells`` at 0 drops the picture and keeps the numbers; the two
        suffixes come off after that. Order matters: the fill is the thing
        that changes what the user should do next, and it is the last to go.
        """
        if not self.known:
            window = f"{self.window:,}" if self.window else "unknown"
            text = f"context: window {window} · no reply yet"
        else:
            text = render_bar(
                self.used, self.window, cells=cells, estimated=not self.measured
            )
            if speed and self.speed:
                # One decimal only where it carries information (slow turns).
                rate = (
                    f"{self.speed:.1f}" if self.speed < 10 else f"{self.speed:,.0f}"
                )
                text += f" · {rate} tok/s"
        # Last, and abbreviated: on a narrow terminal the right end of this
        # line is the first thing to go, and the fill is what must survive.
        # Shown before the first reply too — a session left on xhigh looks
        # identical to one on off until the wait.
        if effort and self.effort:
            text += f" · think {self.effort}"
        return text

    def measure(
        self,
        used: int,
        window: int | None = None,
        *,
        completion_tokens: int = 0,
        request_seconds: float | None = None,
    ) -> None:
        """What the backend said the last prompt cost (`turn.usage`).

        The rate is worked out here rather than sent as one, which is what
        `protocol.TurnUsage` asks for: the token count is the backend's and
        the wall clock is ours, and a front-end that wanted to show the two
        numbers could not recover them from a ready-made rate.

        A restatement carries neither — a session whose backend was switched
        has a prompt size and no fresh generation behind it — and leaves the
        last rate alone rather than replacing it with a zero, since "nothing
        was generated just now" is not "this backend generates nothing".
        """
        self.used = used
        if window:
            self.window = window
        self.measured = True
        self.known = True
        if completion_tokens and request_seconds:
            self.speed = completion_tokens / request_seconds

    def estimate(self, used: int, window: int) -> None:
        """A character-derived figure for a session with no reply yet.

        Reopening a long session should show it is nearly full *before* the
        next message is sent, not after the reply that overflows it. A
        measured number always supersedes this — and says so here rather than
        at the call site, because `context.estimate` carries no flag to tell
        the two apart.
        """
        if self.measured:
            return
        self.used = used
        self.window = window
        self.known = True

    def superseded(self) -> None:
        """This measurement is no longer about this thread (`/compact`).

        A fold rewrites the conversation without taking a message out of it,
        so the core deliberately sends no `chat.reset` — and without one the
        measured 92% from before the fold would sit there, unmarked and
        wrong, until the next reply. `reset` is too strong: the number is
        stale rather than gone, and blanking the bar between the command and
        the core's fresh `context.estimate` would flicker "no reply yet" onto
        a conversation that has had plenty of them.

        So the fill stays and stops claiming to be measured — it draws with
        the `~` — and the estimate that follows is allowed to speak again.
        """
        self.measured = False

    def reset(self) -> None:
        """A different thread is a different number; showing the previous one
        until the next reply would be a lie. The window survives, because it
        belongs to the backend rather than to the conversation."""
        self.used = 0
        self.measured = False
        self.known = False
        self.speed = None


@dataclass
class Toast:
    """One `notify`, as the thing that draws it needs it (`ui/toasts.py`).

    The text is kept raw: sanitising on the way in would make the state depend
    on the renderer's rules, and a toast is also the footer's note, which cuts
    it differently. `toasts.one` is where it is made safe to draw.
    """

    text: str
    severity: str = "information"
    timeout: float | None = None
    # `protocol.Notify.title`: a heading for the body, or empty. Carried
    # separately rather than glued onto the front of the text, which is the
    # whole reason the field exists on the wire — a renderer can no longer
    # tell a heading from the body it introduces once they are one string.
    title: str = ""
    # When it was raised, on `RowUI.clock`. What decides when it goes.
    at: float = 0.0


@dataclass
class Confirm:
    """A yes/no question, from the core or from the UI itself.

    ``id`` is `confirm.requested`'s: the core holds the continuation (today,
    the coroutine that writes a learned log signature) and only the yes/no
    crosses back. It is empty for the eleven local questions — really quit,
    delete this session, interrupt this turn — which have no core state
    waiting on them and are answered by ``on_answer`` alone.
    """

    id: str = ""
    question: str = ""
    # What to do with the answer here, when the answer is this side's business.
    # A callable rather than a verdict flag because the eleven call sites do
    # eleven different things, and the alternative is the UI holding a little
    # enum of what it is currently asking about.
    on_answer: object = None


@dataclass
class Proposal:
    scope: str
    kind: str
    text: str


@dataclass
class BackendInfo:
    """One LLM the UI can draw and name — the plain twin of `protocol.LLMEntry`.

    ``label`` is the identity and the only field a command may name an entry
    by (`protocol.SessionNew.backend`); it is minted by the core so both sides
    cannot disagree about what a backend is called.

    ``reachable`` is deliberately three-state. A probe costs a round trip to a
    cluster node, so the catalog arrives first with nothing known and is
    restated once the probes land — and a client that drew ○ for "not asked"
    would libel every backend for as long as the scan takes.

    There is no api key here and there will not be one: the key never crosses
    the socket, and ``needs_key`` is all the manage-LLMs line ever drew.
    """

    label: str = ""
    model: str = ""
    base_url: str = ""
    context: int = 0
    needs_key: bool = False
    active: bool = False
    reachable: bool | None = None
    # Found by a scan rather than configured. One list with a flag rather than
    # two, because everything else about the two rows is identical.
    discovered: bool = False


@dataclass
class SkillInfo:
    """One procedure file, as the screens and the "/" menu draw it.

    The body travels with the name because the skill editor is a raw text box
    over that file (`tui/profiles_screen.py`'s `ProfileSkillsScreen` read it
    off disk; rule 2 of §4.2 puts disk out of the UI's reach, so it is handed
    in instead).

    ``level`` is where the file lives — "builtin", "global", "profile" or
    "project" (`protocol.SkillRow`). It is what tells the two lists apart: a
    menu offers every level, and only the profile's own and the project's may
    be offered for removal. It defaults to "profile" because that is what the
    other three levels are the exception to, and what a skill invented on this
    side (the creator's form) is until the user says otherwise.
    """

    name: str
    description: str = ""
    text: str = ""
    level: str = "profile"

    @property
    def removable(self) -> bool:
        """Whether `skill.delete` can take this one.

        The profile's own and the project's. A shipped or global skill is
        visible and callable and is nobody's single profile to delete —
        removing one would change every other profile that sees it, which is
        the same rule the core enforces on the way in.
        """
        return self.level in ("profile", "project")


@dataclass
class ProfileInfo:
    """One profile, as the profiles screen and the pickers need it.

    Everything the Textual screen read off disk in `refresh_profiles` —
    the memory count, the star, the provenance — plus the two file bodies its
    editors opened. One record rather than four parallel dicts keyed by name,
    because the screens ask about *a profile*, and a name that is in three of
    the four dicts is a bug that only shows up on the fourth screen.

    Filled from `protocol.ProfileRow` and from nothing else. The *bodies* are
    deliberately not here: `profile.rows` carries a memory count, and each of
    the two editable files is fetched when its editor opens (`profile.get`,
    `skill.get`) so that what is edited is what is on disk now rather than
    what was on disk when the list was drawn — which, with a verbatim save
    behind it, is how an edit made elsewhere gets silently reverted.
    """

    name: str
    memories: int = 0
    sessions: int = 0
    copied_from: str = ""
    # The ★: where a deleted profile's sessions land, and so the one profile
    # that cannot be deleted. A different question from ``working``, which is
    # the one the core is currently running under — usually another profile.
    default: bool = False
    working: bool = False
    # The profile's own skills, name and description, as `skill.rows` sent
    # them: what the skills screen lists and what the "/" menu offers. Empty
    # until something asks (`skill.list`), which is not the same as "none".
    skills: list[SkillInfo] = field(default_factory=list)


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
        thinking: str = "",
    ) -> None:
        self.session_id = session_id
        self.title = title
        self.profile = profile
        self.mode = mode
        self.flags = list(flags)
        # `protocol.SessionRow.thinking`: how hard this conversation reasons,
        # empty when it never chose and follows the configured default. Held
        # here and not only on the meter because `/thinking` opens with the
        # current level preselected, and "" is a different answer from "off".
        self.thinking = thinking
        # `protocol.SessionRow.model`: the backend this conversation is pinned
        # to, as a bare name. Drawn on the message row (§4.3 item 20) and
        # empty for a session that talks to the bootstrap client.
        self.model = model
        self.draft = Editor(wrap=True)
        # Which row of the "/" menu is highlighted, for as long as this
        # session's draft is a command being named. Held here and not on the
        # app because the draft is held here: a parked `/…` draft has to bring
        # its menu back with it (specs-ui-acceptance.md, "Drafts"), and the
        # menu's *contents* need no parking at all — they are a function of the
        # draft, so restoring the draft restores them. Only the cursor is
        # state, and this is the one place it can belong to the same session.
        self.menu_at = 0
        self.chat = Pane("chat", [])
        # The folds this UI opened by itself, because their steps were
        # arriving while the user watched. Remembered so that the end of the
        # turn can close exactly those and leave alone whatever the user
        # opened by hand.
        self._live: set[str] = set()
        self.watchers = Pane("watchers", [])
        self.turn = Turn()
        self.context = Context()
        # The approval this conversation is parked on, if it is parked on one.
        # Per session and held here rather than in one shared bar, which is
        # what parks the half-typed refusal across a switch (§4.4).
        self.decision: Decision | None = None
        self.proposals: list[Proposal] = []
        # A reply landed here while the user was looking at another
        # conversation (§4.3 item 15). Local, because the core has no flag for
        # it and could not have one: "you have not read this" is a fact about
        # which session is on *this* screen, and a second front-end watching
        # the same core has a different answer to it. Cleared by opening the
        # session, which is the only thing that can be read as having read it.
        self.updated = False
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
        # again (`protocol.SessionRollback`), and the number that described it
        # is not this conversation's any more.
        self.context.reset()
        self._live.clear()
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
        if self.turn.busy and entry.kind == "thinking" and entry.seq:
            # A turn's steps arrive while it works, and a fold that opened
            # only after the turn ended would show them all at once, after the
            # fact — "the call is on screen *while* the tool runs" is the
            # claim, and it is what makes a long turn legible rather than
            # merely animated. The turn's end closes it again.
            self._live.add(str(entry.seq))
            self.chat.expanded.add(str(entry.seq))
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

    def remove(self, seq: int) -> ChatEntry | None:
        """Take one row back off the chat. The single exception to §3.2.

        Append-only "between resets" has exactly one hole in it, and the
        protocol is the one that cuts it: `turn.unqueued` says "drop its row,
        keep its text". A message that was never in the graph cannot be
        removed by a `chat.reset` — there is nothing for the core to re-read
        that would leave it out — so the event names the row and the UI drops
        it. Nothing else may use this: every other row on screen is a record
        of something that happened.
        """
        row = self._rows.pop(seq, None) if seq else None
        if row is None:
            return None
        entry = self.entries.pop(row)
        self.chat.items.pop(row)
        # The map is positions into a list that just got shorter.
        self._rows = {k: (v - 1 if v > row else v) for k, v in self._rows.items()}
        self.chat.expanded.discard(str(seq))
        self._live.discard(str(seq))
        self.chat.invalidate()
        return entry

    def entry_at(self, position: int) -> ChatEntry | None:
        """The entry a pane position is showing, if it is showing one."""
        if 0 <= position < len(self.entries):
            return self.entries[position]
        return None

    def entry_of(self, seq: int) -> ChatEntry | None:
        row = self._rows.get(seq)
        return None if row is None else self.entries[row]

    # ---------------------------------------------------------- the decision

    def request_decision(self, payload: dict) -> Decision:
        """`decision.requested` for this conversation.

        The same payload arriving again is the same question — the core
        re-emits a parked decision on subscribe (§4.4) — so the stage and the
        half-typed reason are left alone. Rebuilding them would throw away a
        refusal someone was in the middle of writing every time the client
        reconnected.
        """
        # Whatever else it is, this turn is now waiting on a person, and the
        # core has let go of it: nothing about it can be stopped until it is
        # answered.
        self.turn.parked = True
        if self.decision is not None and self.decision.payload == payload:
            return self.decision
        self.decision = Decision(payload=payload)
        return self.decision

    def clear_decision(self) -> None:
        self.decision = None
        # Answered, or its turn died. Either way the wait is over: the resume
        # re-binds the anchor and the turn is a turn again, and where there is
        # no turn `working` is already False.
        self.turn.parked = False

    # -------------------------------------------------------------- the turn

    def start_turn(self, started_at: str = "") -> None:
        """`turn.started`: there is now something to stop, and a clock to run.

        The stamp is the core's, and it is taken here rather than waiting for
        the first `turn.activity`: the two carry the same instant (the
        scheduler stamps the turn once, `TurnState.started_at`), and the gap
        between them is exactly the silent wait on the backend's first
        answer — the part of a slow turn the user most wants a number for.
        """
        self.turn.working = True
        if started_at and not self.turn.started_at:
            self.turn.started_at = started_at
            self.turn.started_epoch = _epoch(started_at)

    def end_turn(self) -> None:
        """`turn.finished` / `turn.failed`: the spinner goes and the live steps
        fold back into the one box per turn that the transcript keeps.

        Only the folds this UI opened by itself are closed. A step the user
        opened to read stays open — the turn ending is not a reason to take
        away what someone was in the middle of reading.
        """
        self.turn = Turn()
        if self._live:
            self.chat.expanded -= self._live
            self._live.clear()
            self.chat.invalidate()
        self.chat.set_tail(None)

    def tick(self, now: float) -> None:
        """Put the working row where it belongs for this instant, or take it away.

        Called once per frame, and deliberately the only thing that is: the
        spinner is a function of the clock (`Turn.frame`), so a frame of it
        rewrites one cached line rather than relaying out the log — the
        performance requirement the Textual `WorkingIndicator` paid 44ms of
        loop lag to learn (specs-ui-acceptance.md, "The spinner").

        It is a *row*, after the last message, because Enter on it is the
        interrupt gesture and because typing ahead must never bury it: rows
        arriving push it down the log, and it is still the last of them.
        """
        if not self.turn.busy:
            if self.chat.tail is not None:
                self.chat.set_tail(None)
            return
        self.chat.set_tail(
            Item(head=self.turn.line(now), kind=WORKING_KIND, key=WORKING_KEY)
        )

    def next_wake(self, now: float) -> float | None:
        """When this session's frame stops being true on its own."""
        return self.turn.next_frame(now) if self.turn.busy else None

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
    model: str = ""
    thinking: str = ""
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class NewSession:
    """Start a conversation, under this profile and this backend.

    The one intent that names no session, because the point of it is that
    there is not one yet — the core makes it and says which it is
    (`protocol.SessionCreated`), and the UI opens what comes back rather than
    guessing an id.

    ``backend`` is opaque here on purpose: what identifies a backend is the
    core's business (`core.backends.backend_for_new_session`), and the UI only
    ever hands back the string the catalog it was given handed it. Empty means
    "whatever the core would have picked", which is what a run with no
    configured backends gets.
    """

    profile: str = ""
    backend: str = ""


@dataclass(frozen=True)
class Rename:
    """Give this conversation the name the user typed."""

    session_id: str
    title: str


@dataclass(frozen=True)
class Retitle:
    """Ask the model to name this conversation.

    No title travels with it: the UI is asking for one, not supplying one, and
    which model is asked is the core's decision — a session pinned to a
    backend must not be named by whichever model the core happens to hold.
    """

    session_id: str


@dataclass(frozen=True)
class DeleteSession:
    """Drop this conversation. Asked first — `RowUI.ask` owns the question."""

    session_id: str


@dataclass(frozen=True)
class OpenSession:
    """Show me this conversation."""

    session_id: str


@dataclass(frozen=True)
class Submit:
    """Send this message.

    ``forced_skill`` is what makes ``/<skill> …`` an ordinary turn rather than
    a command: the skill's procedure goes into the *model's* copy of the
    message and not into the stored transcript, which is the core's job
    (`protocol.TurnSubmit.forced_skill`), so all this side carries is the name.
    """

    session_id: str
    text: str
    forced_skill: str = ""


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
class CycleMode:
    """Move this session to the next agent mode.

    "The next one", not "this one": which modes exist and what follows what is
    the agent's business (`hpca.agent.next_mode`), and a key that had to name
    the destination would be the key deciding it. `client.py` resolves it and
    sends the `mode.set` that persists it.
    """

    session_id: str


@dataclass(frozen=True)
class Decide:
    """Answer the approval this session is parked on.

    One intent for both verdicts, carrying the reason with the refusal,
    because they are one answer: the turn resumes with a verdict either way
    and the model must not be told "no" twice. ``reason`` is empty for an
    approval and for a refusal nobody explained.
    """

    session_id: str
    approved: bool
    reason: str = ""


@dataclass(frozen=True)
class Unqueue:
    """Take a typed-ahead message back out of this session's queue.

    By ``seq``, the core's name for the row, and never by position: the turn
    ahead of it can finish while the dialog is open, and every position behind
    it then shifts by one (`protocol.TurnUnqueue`).
    """

    session_id: str
    seq: int


@dataclass(frozen=True)
class Answer:
    """The yes/no to a `confirm.requested`, by the id it asked under."""

    id: str
    confirmed: bool


@dataclass(frozen=True)
class Peek:
    """What is this watch box saying right now?"""

    ref: str


@dataclass(frozen=True)
class Drop:
    """Stop watching this."""

    ref: str


@dataclass(frozen=True)
class SaveSettings:
    """The whole settings file, as the config editor left it (§4.3 item 26).

    The text and not a parsed object: the UI validated it to know whether it
    could close, but what a settings *file* means belongs to `hpca.config`,
    and a UI that shipped a half-understood dict would be the second place the
    model is defined.
    """

    text: str


@dataclass(frozen=True)
class SetThinking:
    """How hard this conversation reasons (`hpca.thinking`).

    Per session and not per app, like the mode and the backend: two
    conversations keep their own levels, and the level is a stable property of
    one thread because changing it throws away that thread's prefix cache.
    """

    session_id: str
    effort: str


@dataclass(frozen=True)
class SetBackend:
    """Which model answers — this session's, or, with no session, everyone's.

    ``backend`` is an opaque blob the way `NewSession.backend` is a string:
    what identifies a backend is the core's business, and the UI hands back
    what the catalog it was given handed it. `protocol.BackendSet` validates
    it against the real settings model at the far end.
    """

    backend: dict
    session_id: str = ""
    # The catalog entry's `LLMEntry.label`, which is the form a picker uses:
    # the catalog carries no api_key, so a rebuilt blob cannot name a
    # key-locked backend and a label can. A label wins where both are set.
    label: str = ""


@dataclass(frozen=True)
class SetProfile:
    """The profile the core works under when nothing else narrows it."""

    name: str


@dataclass(frozen=True)
class SaveProfile:
    """Write back what the user edited: a profile's memories, or its archive.

    Two kinds down one intent because they are one screen — the memory editor
    is content-agnostic and the caller decides where the text lands, which is
    exactly what `tui/profiles_screen.py` said about it.
    """

    name: str
    kind: str  # "memories" or "archive"
    text: str


@dataclass(frozen=True)
class CreateProfile:
    """Make a profile under this name. The name is a filename, and whether it
    is a usable one is the profile store's rule and its wording — so a bad
    name comes back as a `notify`, not as a refusal drawn here."""

    name: str


@dataclass(frozen=True)
class CopyProfile:
    """Fork a profile under a new name: same learnings, its own future."""

    source: str
    name: str


@dataclass(frozen=True)
class DeleteProfile:
    """Drop a profile and everything it learned.

    The three refusals — the default cannot go, a gone profile, a profile with
    work in flight — are the core's (`_delete_profile`), because two of them
    are about state the UI cannot see. Only the first is also checked here,
    so the key is inert rather than a round trip that comes back "no".
    """

    name: str


@dataclass(frozen=True)
class SaveSkill:
    """Persist a skill file verbatim, front matter and all.

    ``level`` is where it lands: the profile's own directory, the global one
    every profile sees, or this project's. The creator asks; every other path
    that saves a skill is editing a file that already exists, and leaves the
    default alone (`protocol.SkillSave`).
    """

    profile: str
    name: str
    text: str
    level: str = "profile"


@dataclass(frozen=True)
class DeleteSkill:
    """Remove one of a profile's own skills. Asked first."""

    profile: str
    name: str


@dataclass(frozen=True)
class ResolveMemory:
    """The verdicts on a `memory.proposals` offer, in the order offered.

    Positional, because the core holds the authoritative proposals and a
    front-end must not be able to smuggle an edited memory back in an
    approval (`protocol.MemoryResolve`). A short list rejects the rest, which
    is what escaping the review half-way through means.
    """

    session_id: str
    approved: tuple[bool, ...] = ()


@dataclass(frozen=True)
class Fetch:
    """Ask for a body an editor is about to open — and about to overwrite.

    One intent for the three read paths (`profile.get`, `skill.get`,
    `settings.get`) because the screens do one thing with all three: open
    empty, wait, and fill. ``what`` names which, ``key`` is whatever that read
    path is addressed by, and the same ``key`` comes back on the answer so the
    screen still waiting for it can recognise its own — a `profile.body` that
    arrived after the user escaped and opened a different profile must fill
    nothing.
    """

    what: str  # "profile", "skill" or "settings"
    key: tuple = ()


@dataclass(frozen=True)
class FetchSkills:
    """`skill.list`: a profile's skills, at one of the two scopes.

    ``own`` is what a screen that edits and deletes them may show; ``visible``
    is everything the profile can call — the shipped skills included — which
    is what the "/" menu needs and what makes `/plan` work on a fresh install.
    The answer says which scope it is, so the two lists cannot fill each
    other (`protocol.SkillList`).
    """

    profile: str
    scope: str = "own"


@dataclass(frozen=True)
class DraftSkill:
    """`skill.draft`: ask the core for a model-written first draft.

    `/skill-creator <what it should do>`. The form is this side's and the
    draft cannot be: a draft is a model call. Nothing is written by it — what
    comes back fills the same form, which the user still edits and confirms,
    and the confirmation leaves as an ordinary `SaveSkill`.

    ``session_id`` is the conversation the request came out of, empty when
    there is none: the transcript goes to the drafter with the request, and
    the session is where the core reports the wait.
    """

    profile: str
    request: str
    session_id: str = ""


@dataclass(frozen=True)
class ScanBackends:
    """`backend.scan`: look for endpoints nobody has configured yet.

    No arguments, because there is nothing to narrow: the core runs both
    searches (the localhost port sweep and the cluster's manifest dir) and
    neither finds the other's hits. The answer is a sequence — `llm.catalog`
    again for every hit, then `backend.scanned` with the verdict — so the
    screen that asked fills as it goes and stays closable throughout.
    """


@dataclass(frozen=True)
class ProbeBackend:
    """`backend.probe`: what does this endpoint serve, and is this key good?

    The connection form cannot answer it itself (rule 2 of §4.2, and the
    endpoint may be on a node this process cannot reach). ``api_key`` is the
    one thing that travels UI → core, and only because the user has just
    typed it into the form: nothing sends it back, and it never joins the
    catalog (`protocol.BackendProbe`).
    """

    base_url: str
    api_key: str = ""


@dataclass(frozen=True)
class RemoveBackend:
    """`backend.remove`: drop a configured entry, by the name the picker knows.

    By label rather than by value, for `protocol.BackendRemove`'s reason: what
    the screen is holding is a catalog row, which has no key on it, and an
    entry named by a rebuilt blob is one a missing key could fail to match.
    """

    label: str


@dataclass(frozen=True)
class RunCommand:
    """A slash command: `/compact`, `/thinking`, `/skills-list`, `/<skill>`.

    ``session_id`` is empty for the profile-scoped ones, which is the
    difference `protocol.CommandRun` spells as None: `/skills-list` acts on the
    profile and would be wrong to aim at whatever session happened to be open.
    """

    name: str
    args: str = ""
    session_id: str = ""


Intent = (
    NewSession
    | Rename
    | Retitle
    | DeleteSession
    | OpenSession
    | Submit
    | Interrupt
    | Fork
    | Rollback
    | CycleMode
    | Decide
    | Unqueue
    | Answer
    | Peek
    | Drop
    | SaveSettings
    | SetThinking
    | SetBackend
    | SetProfile
    | SaveProfile
    | CreateProfile
    | CopyProfile
    | DeleteProfile
    | SaveSkill
    | DeleteSkill
    | ResolveMemory
    | RunCommand
    | Fetch
    | FetchSkills
    | DraftSkill
    | ScanBackends
    | ProbeBackend
    | RemoveBackend
)
