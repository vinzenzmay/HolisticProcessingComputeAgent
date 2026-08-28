"""What the UI holds between frames, and what it asks the core to do.

Plain dataclasses and one class with panes in it. No pydantic, no protocol, no
I/O — `client.py` is the only module that has heard of a wire, and it fills
these in from events; `app.py` renders them and never sees anything else
(specs/specs-ui-replacement.md §3.1).

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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from hpca.ui import rain, theme
from hpca.ui.ansi import PULSE_PERIOD, RESET
from hpca.ui.approval import Decision
from hpca.ui.compaction import CompactProposal, CompactReview
from hpca.ui.editor import Editor
from hpca.ui.meter import render_bar, severity
from hpca.ui.pane import SPACER_LINES, Fold, Item, Pane
from hpca.ui.rain import FPS as RAIN_FPS
from hpca.ui.theme import FLASH_HOLD

# The kinds of chat entry that are the user's own words, and so the ones Enter
# offers the rewind on. `queued` counts: it is a message the user wrote, drawn
# ahead of the turn that will send it.
OWN_MESSAGE_KINDS = ("user", "queued")

# What a step's label is padded to in an opened entry, so tool names line up.
TOOL_COLUMN = 14

# What separates a call from what it returned inside an opened step. The same
# string `transcript.RESULT_RULE` writes into the session log, so a step reads
# the same on screen as it does in the file — copied for the reason the whole
# of `ChatPart` is copied: this module may not import the agent side.
RESULT_RULE = "── result ──"

# How the UI writes an instant, wherever it writes one: a chat row's label and
# a sidebar row's activity. Day first and seconds included — the seconds are
# not decoration here, because a turn's question and its answer routinely land
# in the same minute and the log is read to tell them apart.
STAMP_FORMAT = "%d-%m-%Y %H:%M:%S"


def when(stamp: str) -> str:
    """A core ISO stamp as local wall-clock, or "" if there is not one.

    Local, and that is the whole reason the conversion lives on this side of
    the wire: the core stamps in UTC because it need not be on the same
    machine as the front-end (specs/specs-core-process.md), and only the front-end
    knows which clock a person is reading.

    Empty rather than a placeholder when the core sent nothing — a row from a
    thread written before there were stamps, or one the UI wrote itself. A
    label with no time on it says "not known", where a zero would say
    something false.
    """
    if not stamp:
        return ""
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime(STAMP_FORMAT)
    except ValueError:
        return ""


def _label(who: str, at: str, stamps: bool = True) -> str:
    """`you 21-08-2026 19:04:11` — who said it, and when.

    ``stamps`` is `Display.chat_stamps` and turning it off leaves the bare
    `you`, which is the same shape a row with no ``at`` already draws — so the
    two answers to "no time on this row" produce one label rather than a
    second layout nothing else in the pane has to line up against.
    """
    stamp = when(at) if stamps else ""
    return f"{who} {stamp}" if stamp else who


@dataclass(frozen=True)
class Display:
    """What this front-end draws with, as the core last said (§4.2 rule 2).

    The plain twin of `protocol.DisplaySettings` — `client.py` is the only
    module that has seen the wire one, so this is the shape the renderer takes
    it in. One object rather than loose attributes because the two arrive
    together, twice: on `hello`, and again whenever the config editor saves.

    Frozen, and replaced rather than edited. A `SessionState` holds the same
    instance the `RowUI` does, so a mutable one would let a session's chat be
    rebuilt against a value the app no longer believes — the divergence would
    be invisible, since both copies stay individually consistent.
    """

    # Whether a chat row's label carries the instant it was said.
    chat_stamps: bool = True
    # Whether the screen the quit question cleared rains (`ui.rain`), and
    # how many frames a second it falls at. The module default stands in until
    # `hello` lands, which is long before there is a quit dialog to draw.
    quit_rain: bool = True
    quit_rain_fps: int = RAIN_FPS
    # One breath of the decision prompt's answer line, in seconds. The module
    # default stands in until `hello` lands, which is before there is a
    # decision on screen to pulse (`client._hello` is the first frame).
    decision_pulse_seconds: float = PULSE_PERIOD
    # The colours, as colour *specs* rather than escape sequences — an xterm
    # index or a hex triple, the way the settings file writes them. They are
    # turned into sequences once, by `ui.theme.apply`, when this is adopted;
    # holding them resolved here would put the same palette in two places and
    # make "which one is on screen" a question with two answers.
    #
    # A mapping and not ten fields, because nothing in this module reads an
    # individual colour: `RowUI.set_display` hands the whole thing to the theme
    # and the theme is what the drawing code asks. Every key is optional, and
    # one that is absent keeps the built-in — which is what lets a settings
    # file name three colours and mean exactly that.
    palette: Mapping[str, object] = field(default_factory=dict)
    # How long the pane that just took focus is washed in `palette["flash"]`.
    # Zero is off.
    focus_flash_seconds: float = FLASH_HOLD
    # Blank rows under each chat row, and so the gap above the message box —
    # the chat hangs from the bottom of its pane, so its last row's blanks are
    # what stands between the conversation and what you type into it. Zero is
    # off. The module default stands in until `hello` lands, which is before
    # there is a conversation to space out.
    spacer_lines: int = SPACER_LINES


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
# Which palette role each mode is drawn in — the role's *name*, resolved when
# the row is drawn. A table of finished escape sequences would be built at
# import and go on being right about a palette the settings editor had already
# replaced, which is the one thing `ui.theme` exists to prevent.
MODE_ROLES = {"manual": "warn", "auto": "ok", "full-auto": "danger"}


def mode_colour(mode: str) -> str:
    """The colour `mode` is drawn in, or the quiet one if it is not a mode."""
    return getattr(theme, MODE_ROLES.get(mode, "faint"))


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
    # When it happened, as the core stamped it (ISO-8601 UTC). Rendered by
    # `when`, which is the only thing that knows about local clocks.
    at: str = ""


def _one_line(text: str) -> str:
    """A head line is one line; a chat message is not. Collapse it."""
    return " ".join(text.split())


def _first_line(text: str) -> str:
    """The first line with anything on it — a head's worth of a call.

    Not `_one_line`, which would run a fifty-line script together into one
    smear: the lines of a script are separate commands, and joining them makes
    a line that reads as a command nobody wrote. One line of it, and the fold
    marker says the rest is behind the row.
    """
    for line in text.split("\n"):
        if line.strip():
            return _one_line(line)
    return ""


def part_body(part: ChatPart) -> list[str]:
    """What a step opens into: the call itself, and then what it returned.

    Both halves, not just the result. A head has one line and a `run_bash`
    call's is a fifty-line script — so a row that opened into its output alone
    showed the user the stdout of a command they could not read. The same
    choice `transcript.Step.body` makes for the log, and the rule between the
    halves is its string; copied rather than imported because this module may
    not pull in the agent side (see `ChatPart`).

    A call still in flight has only its half, and gets no heading over an
    empty result: that reads as a tool that answered with nothing.
    """
    call = part.text.strip()
    result = part.result.strip()
    if not call:
        return result.split("\n") if result else []
    if not result:
        return call.split("\n")
    return [*call.split("\n"), "", RESULT_RULE, *result.split("\n")]


def part_fold(part: ChatPart) -> Fold:
    """One step, as a row of its own inside the turn's fold.

    The head is the call — the tool and what it was aimed at — and the body is
    the call in full with the result under it. Two levels rather than one
    because a tool result is routinely a whole file: the steps of a turn have
    to stay readable as a list, and the four hundred lines `read_file`
    returned must be one more keypress away rather than in between the steps
    either side of it.

    A call whose result has not landed says so with a trailing "…", because
    the alternative is a row that looks finished and is not (`Part.done`) —
    and that mark is exactly what a live step row is: the call, on screen,
    while the tool is still running.
    """
    label = part.tool or part.kind or "step"
    detail = part.target or _first_line(part.text)
    head = f"{label:<{TOOL_COLUMN}}{detail}".rstrip()
    if not part.done and not part.result:
        head = f"{head} …"
    if part.failed:
        head = f"{head}  ✗"
    return Fold(head=head, body=part_body(part))


def entry_item(entry: ChatEntry, *, stamps: bool = True) -> Item:
    """The row an entry draws as: who said it, and underneath, what they said.

    ``stamps`` is `Display.chat_stamps`, passed in by the session rather than
    read from anywhere: this is a pure function of an entry and a setting, and
    a module-level flag would make two sessions' rows depend on the order they
    were built in.

    The one place a `ChatEntry` becomes something with colours in it, so that
    every path into the chat — reset, append, update — produces the same row
    for the same entry, and an update genuinely replaces what it revises.

    **Why the words are not on the same line as the label.** They used to be:
    a row read `you   annotate the cohort BAMs`, indented under a gutter and a
    fold marker. That is four columns of furniture in front of every line of
    the conversation, and a terminal's drag-to-select takes all four along
    with the text — which matters here because selecting out of the chat is
    the terminal's job in this UI and not the UI's own
    (specs/specs-ui-replacement.md §4.1 item 6). So the label is a line of its own
    and the message is `body`, which `Pane(flush=True)` draws at column 0 with
    nothing in front of it: what you drag across is what you paste.

    **And why the row still closes.** A conversation whose every message shows
    its whole self is a conversation you scroll rather than read: three
    exchanges fill the pane and the shape of the thing is gone. So a closed
    row is the label and *one* line — `preview`, clipped to the terminal with
    a ``[...]`` that says there is more and that → will show it. Two lines a
    message, which is what a log you can skim costs; the wrapped text and the
    clean selection are one keypress in.

    The rows that are *not* somebody's words have no preview and close to
    their head alone — a turn's working already summarises itself ("8 steps ·
    edit_file → run_bash …"), and a notice is one line to begin with.
    """
    said = _one_line(entry.text)
    # Empty rather than one blank line: an assistant row is appended before
    # its first token arrives, and `[""]` would make it openable and give it a
    # marker pointing at nothing.
    body = entry.text.split("\n") if entry.text else []
    # Every row carries the core's name for it, which is what `chat.update`
    # addresses and what `Pane.expanded` remembers.
    row = dict(kind=entry.kind, text=entry.text, key=str(entry.seq))
    if entry.kind == "user":
        return Item(
            head=_label("you", entry.at, stamps),
            body=body,
            preview=said,
            accent=theme.user,
            label=True,
            **row,
        )
    if entry.kind == "queued":
        # Still the user's own words, and still copyable as such — the label
        # is what says they have not been sent yet.
        return Item(
            head=f"{_label('you', entry.at, stamps)} · queued",
            body=body,
            preview=said,
            accent=theme.faint,
            label=True,
            **row,
        )
    if entry.kind == "error":
        return Item(
            head=_label("error", entry.at, stamps),
            body=body,
            preview=said,
            accent=theme.danger,
            label=True,
            **row,
        )
    if entry.kind == "thinking":
        steps = entry.steps or len(entry.parts)
        names = [x.tool or x.kind for x in entry.parts if x.tool or x.kind]
        summary = " → ".join(names[:3]) + (" …" if len(names) > 3 else "")
        # One collapsed box per turn, opening into its steps: the shape
        # `tui/app.py`'s ThinkingBox and StepBox had between them, minus the
        # two widget classes.
        #
        # Dim, and dim all the way down. A turn's working is the machinery
        # behind the answer rather than the answer, and it is the bulkiest
        # thing in the log — an opened turn is thirty rows of tool names
        # between two paragraphs of prose. Saying so with the accent rather
        # than by leaving it unset is what makes the *head* grey too: an
        # accentless row falls through to `Pane.render`'s default, which dims
        # the body lines and leaves the head at full weight, so a closed turn
        # stood out from the conversation exactly as loudly as a reply.
        return Item(
            head=f"{steps} steps" + (f" · {summary}" if summary else ""),
            folds=[part_fold(part) for part in entry.parts],
            accent=theme.faint,
            **row,
        )
    if entry.kind == "compaction":
        # A divider, not a message. The head names the moment the model's view
        # was cut and the rest of the line is drawn out to the edge of the
        # pane, because what this row marks is a *boundary* across the whole
        # conversation rather than an event at a point in it — the one row
        # here whose meaning is "everything above this is different from
        # everything below".
        #
        # It opens into the summary that stands in for what is above it, which
        # is the question a boundary immediately raises: the user can see that
        # the context was cut, and the only useful next thing is what the model
        # kept of it. No preview — a closed divider is a line, and one clipped
        # line of a thousand-character summary hanging under it would read as
        # a message somebody sent.
        return Item(
            head=f'{_label("compacted", entry.at, stamps)} ',
            body=body,
            # Dashed, not solid: a gapless rule here is the same line the
            # pane draws over every panel, and this is not a panel edge — it
            # is a mark inside the conversation. The gaps are what say so at
            # a glance, and they are made of the rule's own character so the
            # two read as the same weight rather than as two kinds of line.
            fill="── ",
            accent=theme.faint,
            **row,
        )
    if entry.kind in ("event", "recall"):
        mark = "↺" if entry.kind == "recall" else "·"
        return Item(head=f"{mark} {said}", accent=theme.faint, **row)
    # Anything else is drawn as the agent talking, including a kind this
    # renderer has never heard of: the text is what matters and dropping the
    # row would lose it.
    return Item(
        head=_label("hpca", entry.at, stamps),
        body=body,
        preview=said,
        accent=theme.agent,
        label=True,
        **{**row, "kind": entry.kind or "assistant"},
    )


# ------------------------------------------------------------- the turn and it


# How fast the spinner turns. 0.1s is 10 frames a second: fast enough to read
# as motion, and slow enough that a session over a loaded SSH link spends a
# tenth of the repaints a 0.08s spinner would. Nothing else on the screen
# changes by the clock, so this interval *is* the UI's idle cost while a turn
# runs — see `RowUI.next_wake`, which books exactly one wake per frame rather
# than reintroducing a poll.
#
# The glyphs themselves are `ui.rain`'s: four cells of the same katakana the
# quit screen rains, with the drop bouncing between the walls and its trail
# fading behind it. A braille wheel turned here for a long time and said only
# "not hung"; the effect the rest of the UI already uses for "alive and
# waiting" says the same thing in the same language, and the working row is
# the other place the user is doing nothing but waiting.
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
    # the working row's own hint is drawn from (specs/specs-ui-coverage.md §4).
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
        """The spinner's four cells for this instant, plain.

        Derived from the clock rather than advanced by a tick, so the spinner
        needs nothing to drive it: any frame drawn at time *t* shows the same
        glyphs, and a UI that repaints only when something changed can work out
        when this one next will (`next_frame`). The churn rides the same
        counter, so the glyphs are swapped only on frames that were going to be
        painted anyway — a second clock for them would be a second reason to
        wake up, for a change nobody asked to see sooner.
        """
        return rain.spinner(self._tick(now))[0]

    def _tick(self, now: float) -> int:
        """Which frame of the animation this instant is."""
        return int((now - (self.started_epoch or 0.0)) / SPINNER_INTERVAL)

    def paint(self, now: float) -> Callable[[str, str], str]:
        """Put the trail's colours back on a row that was measured without them.

        Handed to the working row as `pane.Item.paint`, and given the finished,
        padded line: the spinner is found in it by its own glyphs, which are
        katakana and cannot collide with the activity text beside them. A row
        too narrow to have kept them is left alone rather than guessed at —
        `ansi.pad` truncates, and the answer to "the spinner was cut off" is a
        line with no colour in it, not one with colour in the wrong cells.
        """
        glyphs, styles = rain.spinner(self._tick(now))

        def paint(row: str, style: str) -> str:
            at = row.find(glyphs)
            if at < 0:
                return row
            lit = "".join(s + g for g, s in zip(glyphs, styles))
            # Back into the style the row is being drawn in, not out of it: on
            # the cursor's own row that is REVERSE, and a bare RESET here would
            # end the highlight four cells in.
            return f"{row[:at]}{lit}{RESET}{style}{row[at + len(glyphs) :]}"

        return paint

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
            text += f" · reason {self.effort}"
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
class Offer:
    """A `confirm.requested`, waiting in the session it was raised about.

    The core holds the continuation under ``id`` — today, the coroutine that
    writes a learned log signature — and only the yes/no crosses back.

    Held on the `SessionState` and not on the app, which is the whole point of
    it: this is raised by a poll rather than by a keypress, so it belongs to
    the conversation whose job failed rather than to whatever happens to be on
    screen when it lands. Switching session carries it, exactly as the parked
    decision and the half-typed draft are carried (§4.4).
    """

    id: str = ""
    question: str = ""


@dataclass
class Confirm:
    """A yes/no the UI asks itself, over everything else.

    Really quit, interrupt this turn, delete this session: local questions
    with no core state waiting on them, answered by ``on_answer`` alone. What
    the *core* asks is an `Offer` above, which is a different thing in every
    way that matters — it is about one session, it can arrive while the user
    is elsewhere, and so it waits in that session rather than taking the
    screen.
    """

    question: str = ""
    # Whether the screen this one cleared falls (`ui.rain`). Per question and
    # not per confirmation: leaving is the one of these you are not coming
    # back from, so it is the one that gets a send-off. Stopping a turn or
    # deleting a session are things you do in the middle of working, and an
    # animation over the top of them would be a flourish charged to somebody
    # who is busy.
    rain: bool = False
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
    the socket. ``needs_key`` is the one thing the manage-LLMs line draws about
    keys, and it means *still* locked — the endpoint asked for a key and none
    we hold was accepted (`protocol.LLMEntry`). A backend the user has keyed
    draws no hint.
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
        display: Display | None = None,
    ) -> None:
        self.session_id = session_id
        # What this conversation's rows are drawn with (`RowUI.set_display`
        # hands the same object to every session, and swaps them all at once).
        # Defaulted rather than required so that a `SessionState` built by a
        # test or by the blank stand-in draws what the settings' own defaults
        # would have said.
        self.display = display or Display()
        self.title = title
        self.profile = profile
        self.mode = mode
        self.flags = list(flags)
        # `protocol.SessionRow.thinking`: how hard this conversation reasons,
        # empty when it never chose and follows the configured default. Held
        # here and not only on the meter because `/reasoning` opens with the
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
        # its menu back with it (specs/specs-ui-acceptance.md, "Drafts"), and the
        # menu's *contents* need no parking at all — they are a function of the
        # draft, so restoring the draft restores them. Only the cursor is
        # state, and this is the one place it can belong to the same session.
        self.menu_at = 0
        # The messages ↑/↓ walks, while they are being walked. Per session for
        # the same reason `menu_at` is: it is about this conversation's draft,
        # and switching away and back must not leave a browse half-done.
        #
        # Snapshotted when browsing starts rather than read live, because a
        # turn can append to the chat mid-browse — a queued message of yours
        # becoming a real one is exactly that — and an index into a list that
        # grew under it points at the wrong message. The draft that was in the
        # box when browsing began is the last element, so walking forward off
        # the end restores it: nothing typed is lost to a stray ↑.
        self.history: list[str] | None = None
        self.history_at = 0
        self.history_home = (0, 0)  # the stashed draft's cursor
        # The one flush pane: its rows are prose somebody is going to select
        # with the mouse, so it spends no columns in front of them. And the
        # one spaced pane, for the same reason one level on: the rows are
        # paragraphs, and paragraphs are told apart by the blank between them
        # (`pane.SPACER_LINES`).
        self.chat = Pane("chat", [], flush=True, spacer=self.display.spacer_lines)
        # The folds this UI opened by itself, because their steps were
        # arriving while the user watched. Remembered so that the end of the
        # turn can close exactly those and leave alone whatever the user
        # opened by hand.
        self._live: set[str] = set()
        # The row this UI opened by itself for the other reason: it is the
        # newest one there is. Same bargain as `_live` above, and kept apart
        # from it because the two are undone by different events — a live fold
        # closes when its turn ends, and this one closes when something newer
        # arrives to take the title off it. One key, never a set: there is
        # only ever one last row.
        self._auto = ""
        self.watchers = Pane("watchers", [])
        self.turn = Turn()
        self.context = Context()
        # The approval this conversation is parked on, if it is parked on one.
        # Per session and held here rather than in one shared bar, which is
        # what parks the half-typed refusal across a switch (§4.4).
        self.decision: Decision | None = None
        # Questions the core raised about this conversation's own work, oldest
        # first (`Offer`). A list rather than one slot: two background jobs can
        # fail before either question is answered, and the core is holding a
        # continuation per id — a second offer overwriting the first would
        # strand that one with nothing left on any screen able to answer it.
        self.offers: list[Offer] = []
        self.proposals: list[Proposal] = []
        # The summary this conversation is being asked to accept, and what has
        # been done about it so far (`compaction.CompactReview`). One slot,
        # unlike `offers` above: a second `/compact` on a session that is
        # already holding one either re-offers it or replaces it, so there is
        # never a queue of summaries of the same history. Held here rather
        # than in one shared prompt for the reason `decision` is — the review
        # is per conversation, and so is the half-typed complaint in its box.
        self.review = CompactReview()
        # A reply landed here while the user was looking at another
        # conversation (§4.3 item 15). Local, because the core has no flag for
        # it and could not have one: "you have not read this" is a fact about
        # which session is on *this* screen, and a second front-end watching
        # the same core has a different answer to it. Cleared by opening the
        # session, which is the only thing that can be read as having read it.
        self.updated = False
        # When something last happened here, as the core stamped it. Not to be
        # read as a companion to `updated` above, which is a different fact
        # entirely: that one is "there is something here you have not seen",
        # this one is "this is when it was last worked in", and a session can
        # be either without the other.
        self.last_active = ""
        self.entries: list[ChatEntry] = []
        # seq -> position in `entries` / `chat.items`, which are parallel.
        # The map, not a scan: a `chat.update` during a long turn arrives once
        # per tool call, and a linear search per update is O(chat) per step.
        self._rows: dict[int, int] = {}
        # The `seq` of the compaction boundary row, or 0 for a conversation
        # that has never been folded. Named by seq rather than by position so
        # it survives `remove` shifting the rows under it — `_rows` is what
        # keeps positions honest, and this rides it.
        #
        # One, not a list: the graph keeps a single fold record and moves it
        # (`AgentState.compacted`), so a second compaction replaces the first
        # rather than adding to it.
        self._fold_seq = 0
        self.loaded = False

    # -------------------------------------------------------------- the chat

    def _item(self, entry: ChatEntry) -> Item:
        """One row, drawn with this session's display settings.

        The single place `entry_item` is called from, which is what keeps the
        three paths into the chat — reset, append, update — producing the same
        row for the same entry after a setting changes under them.
        """
        return entry_item(entry, stamps=self.display.chat_stamps)

    def restyle(self, display: Display) -> None:
        """Redraw every row, because what a row looks like has changed.

        Not `reset`: nothing about the *conversation* changed, so the rows
        that are open stay open, the cursor stays on the line it was on and
        the numbering is untouched — this rebuilds the `Item`s the entries
        already produced and nothing else. Which is why it is here rather than
        in `RowUI`: `entries` and `chat.items` are parallel and this is the
        only side that knows it.
        """
        self.display = display
        # Assigned before the rebuild, not after: the setter invalidates only
        # when the count actually changed, and a `spacer` handed over after
        # `invalidate` would leave the cache holding the old spacing until the
        # next thing that happened to throw it away.
        self.chat.spacer = display.spacer_lines
        self.chat.items = [self._item(entry) for entry in self.entries]
        self.chat.invalidate()

    def reset(self, entries: list[ChatEntry]) -> None:
        """Replace the transcript — the snapshot the deltas build on.

        The only path that may throw rows away, and it throws away what was
        open with them: a reset re-bases the numbering, so a `seq` that is
        still in `expanded` afterwards would be naming a different row. What
        is open when it returns is decided here, after the renumbering, and it
        is the one row `_open_last` opens — a conversation you have just
        opened is one whose last message you want to read.
        """
        self.entries = list(entries)
        # The thread this described is gone, so a fresh estimate may speak
        # again (`protocol.SessionRollback`), and the number that described it
        # is not this conversation's any more.
        self.context.reset()
        self._live.clear()
        self.chat.items = [self._item(entry) for entry in self.entries]
        self._rows = {
            entry.seq: i for i, entry in enumerate(self.entries) if entry.seq
        }
        # Only a reset can bring one in: `build_entries` draws the boundary
        # for whole-chat builders alone, and an append arrives below it by
        # construction. So this is the one place it needs finding.
        self._fold_seq = next(
            (e.seq for e in reversed(self.entries) if e.kind == "compaction"), 0
        )
        self.chat.expanded.clear()
        self.chat.invalidate()
        self._auto = ""
        self._open_last()
        self.chat.to_end()  # open at the newest, as the old app does
        self.loaded = True

    def _open_last(self) -> None:
        """Show the newest row whole, and remember that nobody asked for it.

        Exactly one row is open by itself, and it is the last one: whatever
        was said most recently is what is being read, and having to press → to
        see the reply that just landed is a keypress on every single turn. The
        rows above it stay a label and a line, which is the folded form the
        chat is legible in (`entry_item`) — so the log cleans up behind itself
        as it grows instead of becoming a wall to scroll.

        The key goes in ``_auto`` because the next row to arrive has to undo
        *this* and nothing else. A row the user opened by hand is a row
        somebody is reading, and a reply landing is not a reason to take it
        away — the same distinction `_live` draws for a turn's steps, and the
        reason neither can be an "everything that is open" set.

        A row with nothing behind it opens into nothing and is left alone: an
        assistant row is appended before its first token arrives, and opening
        it would only give it a marker pointing at nothing. `update` picks it
        up when the text lands.
        """
        if not self.chat.items:
            return
        index = len(self.chat.items) - 1
        if not self.chat.items[index].openable:
            return
        self._auto = self.chat.key_at(index)
        self.chat.expanded.add(self._auto)
        # The row is the last one, so its lines are the tail of the flattened
        # cache and can be rebuilt in place. `invalidate` would be correct and
        # would re-flatten the conversation behind it, once per arriving row.
        self.chat.reflow_last()

    def _close_auto(self) -> None:
        """The row that opened because it was the last one is not, any more.

        Skipped while the turn that is writing it is still running: a live
        thinking fold is open for the *other* reason and `end_turn` owns it,
        and closing it here would fold the steps away halfway through the turn
        they belong to.
        """
        if self._auto and self._auto not in self._live:
            self.chat.expanded.discard(self._auto)
            self.chat.reflow_last()
        self._auto = ""

    def append(self, entry: ChatEntry) -> None:
        """One new row. The only path by which a chat grows (§3.2)."""
        if entry.seq:
            self._rows[entry.seq] = len(self.entries)
        self.entries.append(entry)
        if self.turn.busy and entry.kind == "thinking" and entry.seq:
            # A turn's steps arrive while it works, and a fold that opened
            # only after the turn ended would show them all at once, after the
            # fact — "the call is on screen *while* the tool runs" is the
            # claim, and it is what makes a long turn legible rather than
            # merely animated. The turn's end closes it again.
            self._live.add(str(entry.seq))
            self.chat.expanded.add(str(entry.seq))
        # Whatever was last is not last now, and the row it is losing the
        # title to is about to be built — so it is closed first, while its
        # lines are still the tail of the cache.
        self._close_auto()
        # `extend`, not `items.append` + `invalidate`: whether this row opens
        # itself has just been decided, so its lines can be built now and
        # added to the cache rather than the whole conversation re-flattened
        # on the next frame. This is the append-only invariant (§3.2) being
        # spent rather than merely kept.
        self.chat.extend(self._item(entry))
        self._open_last()
        if self.chat.follow:
            # Only if the newest line is what is being read. Somebody who has
            # scrolled up is reading something else, and a reply landing is
            # not a reason to take it away from them — the same distinction
            # `_close_auto` draws about what is open.
            self.chat.to_end()
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
        self.chat.items[row] = self._item(entry)
        self.chat.invalidate()
        # A row that had nothing to open when it arrived and has something now
        # — an assistant row is appended empty and filled token by token — is
        # the last row finally becoming showable, so it opens here instead.
        # Only when nothing has auto-opened yet, which is what keeps a stream
        # of updates from re-opening a row the user has just folded away: once
        # `_auto` names this row, it names it whether or not it is still open.
        if not self._auto and row == len(self.entries) - 1:
            self._open_last()
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
        # An unqueued message is routinely the newest row there is, and the
        # row it leaves behind is the last one now: it inherits the opening
        # rather than the chat ending up with none.
        if self._auto == str(seq):
            self._auto = ""
            self._open_last()
        return entry

    def folded_away(self, seq: int) -> bool:
        """Is that row above this conversation's compaction boundary?

        The rewind is not offered above the fold, and this is the question it
        asks. Both halves of it — cutting the thread back to a message, and
        forking a new one from it — reach for history that the model no longer
        holds: rolling back past the boundary drops the summary standing for
        it, and a fork from up there starts a conversation the summary does
        not describe. Both are *defined* (`graph.rollback_thread` /
        `fork_thread` handle the fold deliberately), and neither is a thing to
        offer behind a keystroke on a row that looks like every other one.

        Positions rather than seqs, because seqs are assigned in row order but
        the comparison must survive a `remove` renumbering the map under it.
        """
        if not self._fold_seq:
            return False
        row, fold = self._rows.get(seq), self._rows.get(self._fold_seq)
        if row is None or fold is None:
            return False
        return row < fold

    def entry_at(self, position: int) -> ChatEntry | None:
        """The entry a pane position is showing, if it is showing one."""
        if 0 <= position < len(self.entries):
            return self.entries[position]
        return None

    def entry_of(self, seq: int) -> ChatEntry | None:
        row = self._rows.get(seq)
        return None if row is None else self.entries[row]

    # ------------------------------------------------------- message history

    def start_history(self, draft: str, cursor: tuple[int, int]) -> bool:
        """Open a browse over this session's own messages, oldest first, with
        ``draft`` stashed on the end. False when there is nothing to walk.

        The chat log *is* the history — no second copy, no new protocol, and
        it comes back with the conversation because reopening one replays the
        log from the core. What it costs is that `/`-commands are not in it:
        only a real submission becomes a chat entry, so ↑ recalls messages,
        which is the thing the box is for.
        """
        texts: list[str] = []
        for entry in self.entries:
            if entry.kind not in OWN_MESSAGE_KINDS or not entry.text:
                continue
            if texts and texts[-1] == entry.text:
                # The same message twice running is one thing to recall.
                # Walking through three identical rows is not history, it is
                # a key that looks broken.
                continue
            texts.append(entry.text)
        if not texts:
            return False
        self.history = texts + [draft]
        self.history_at = len(self.history) - 1
        self.history_home = cursor
        return True

    def end_history(self) -> None:
        """Stop browsing; whatever is in the box stays in it."""
        self.history = None

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

    # ------------------------------------------------------ the compaction

    @property
    def compaction(self) -> CompactProposal | None:
        """The summary this conversation is being asked to accept, or None.

        A property over `review.proposal` rather than a field beside it, so
        that "is there a summary waiting here?" has one answer and the drawing
        state cannot drift from the offer it belongs to.
        """
        return self.review.proposal

    @compaction.setter
    def compaction(self, proposal: CompactProposal | None) -> None:
        """A new offer, and a review of it that has had nothing done to it yet.

        Assigning replaces the whole review rather than only its proposal,
        which is what makes a retry's second summary arrive scrolled to the
        top with an empty comment box: the complaint that produced it has been
        sent, and leaving it in the box would invite sending it twice.
        """
        self.review = CompactReview(proposal) if proposal is not None else CompactReview()

    # ---------------------------------------------------------- the offers

    @property
    def offer(self) -> Offer | None:
        """The one being asked, which is the oldest one still unanswered."""
        return self.offers[0] if self.offers else None

    def add_offer(self, offer_id: str, question: str) -> Offer:
        """`confirm.requested` for this conversation.

        The same id twice is the same question — a re-emit on subscribe would
        otherwise ask it twice and leave the second copy unanswerable, since
        the core frees the continuation on the first answer.
        """
        for existing in self.offers:
            if existing.id == offer_id:
                return existing
        offer = Offer(id=offer_id, question=question)
        self.offers.append(offer)
        return offer

    def drop_offer(self, offer_id: str) -> None:
        """Answered. Nothing about the turn changes — this never held one."""
        self.offers = [o for o in self.offers if o.id != offer_id]

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
        loop lag to learn (specs/specs-ui-acceptance.md, "The spinner").

        It is a *row*, after the last message, because Enter on it is the
        interrupt gesture and because typing ahead must never bury it: rows
        arriving push it down the log, and it is still the last of them.
        """
        if not self.turn.busy:
            if self.chat.tail is not None:
                self.chat.set_tail(None)
            return
        self.chat.set_tail(
            Item(
                head=self.turn.line(now),
                kind=WORKING_KIND,
                key=WORKING_KEY,
                paint=self.turn.paint(now),
            )
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
    last_active: str = ""
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
class MoveSession:
    """alt+↑ / alt+↓ on a sidebar row: shift it one place, for good.

    An offset and not a slot, and a row named by id rather than by position,
    because that is the whole gesture the user has and because both halves of
    that survive the round trip: the core swaps with whichever row is the
    neighbour *when it arrives*, so two presses in quick succession walk one
    session past two others even though the second was sent before the first
    was answered.

    Which is also why nothing is reordered here. The arrangement is a fact
    about the database (rule 2 of §4.2), the answer is a whole new
    `session.rows`, and a sidebar that shuffled itself first would only be
    overwritten by it — which is exactly the bug this replaced: the list moved,
    and the next frame from the core put it straight back.
    """

    session_id: str
    delta: int


@dataclass(frozen=True)
class MoveWatch:
    """alt+↑ / alt+↓ on a watch box: the right column's half of `MoveSession`.

    Same shape and same reasons — see there. ``ref`` is the string every panel
    row carries; a watch id is an int, and that conversion is the sender's
    errand (`protocol.PanelRow`), which is `client.py`'s side of the line.
    """

    ref: str
    delta: int


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
class EditProfile:
    """Open one of a profile's two files in `$EDITOR`, and save what comes back.

    The counterpart of `SaveProfile` for the profiles screen's other editor:
    the same two kinds, the same profile named by name, and no text — because
    the whole point of this one is that the text is fetched, edited outside
    the app and written back by the side holding the wire
    (`UIClient.edit_profile`).

    The profile is named rather than assumed. A screen sends this about the
    row under the cursor, which is not usually the profile the core is
    working under, and a command that meant "the active one" is what made the
    old ctrl+e edit a file nobody had selected.
    """

    name: str
    kind: str  # "memories" or "archive"


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
class ResolveCompact:
    """The verdict on a `compact.proposed` offer.

    Three answers rather than a yes/no, because a summary is a thing that can
    be *nearly* right: `retry` sends it back with ``comment`` saying what it
    has to do differently, which is the one field here carrying text. The
    summary itself never travels back — the core holds it, so what lands is
    what the model wrote (`protocol.CompactResolve`).

    Closing the review without answering sends none of these: the core goes on
    holding the offer and `/compact` brings it back.
    """

    session_id: str
    action: str = "discard"  # "accept", "retry", "discard"
    comment: str = ""


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
    """A slash command: `/compact`, `/reasoning`, `/skills-list`, `/<skill>`.

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
    | MoveSession
    | MoveWatch
    | SaveSettings
    | SetThinking
    | SetBackend
    | SetProfile
    | SaveProfile
    | EditProfile
    | CreateProfile
    | CopyProfile
    | DeleteProfile
    | SaveSkill
    | DeleteSkill
    | ResolveMemory
    | ResolveCompact
    | RunCommand
    | Fetch
    | FetchSkills
    | DraftSkill
    | ScanBackends
    | ProbeBackend
    | RemoveBackend
)
