"""The wire between the UI process and the core process (specs-core-process §4).

Both processes import this module, and it is the *only* thing they must agree
on, so it deliberately imports nothing of hpca's own: no Textual, no langgraph,
no sqlite. A front-end that renders chat has no business importing the agent
runtime, and a core that runs the graph has no business importing a widget
library — that separation is the whole point of the split, and it starts here.

Framing is newline-delimited JSON. A stream socket delivers bytes, not
messages, so something has to say where a frame ends; a delimiter needs no
length prefix to get wrong, and JSON escapes every newline that occurs inside a
string, so the delimiter provably cannot appear inside a frame. That is worth a
test (and has one): a pasted traceback in a chat message is exactly the input
that would break a hand-rolled framing.

Payloads are typed models rather than bare dicts because the far side of this
socket is not to be trusted — it is a `run_bash`-shaped endpoint (§3). A
malformed frame must fail as a `ProtocolError` at the boundary, where the
reader loop can log it and carry on, rather than as a `KeyError` three layers
into a handler that has already started doing something.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Asserted during the stdout handshake (§2.1). Both ends normally come from one
# install, so this is not about version skew between releases: it is about a
# stale core, left listening by a crashed run, being attached to later.
PROTOCOL_VERSION: int = 1


class ProtocolError(Exception):
    """A frame that cannot be trusted: bad JSON, unknown type, wrong shape.

    One exception for the whole module, so a reader loop needs one ``except``
    and pydantic's ``ValidationError`` never escapes into calling code.
    """


class _Model(BaseModel):
    # extra="forbid" everywhere: the handshake asserts PROTOCOL_VERSION and
    # both ends ship from one install, so an unexpected key is a bug on the
    # sending side rather than a newer peer being polite. Rejecting it loudly
    # beats dropping it silently and debugging the missing field later.
    model_config = ConfigDict(extra="forbid")


class Envelope(_Model):
    """The one frame shape, in both directions (§4)."""

    # Monotonic per sender, from 1. Nothing reads it yet; it is here because
    # numbering cannot be retrofitted onto a stream that never had it, and any
    # future reconnect or replay needs it.
    seq: int = 0
    # Correlation id, set by the sender of a command. A core reply that answers
    # one command echoes it in ``payload.reply_to`` (see `Event`); most
    # commands are answered by ordinary events instead.
    id: str | None = None
    # The routing field, and the only mandatory one — an empty type is as
    # useless as a missing one, hence min_length.
    type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


def encode(env: Envelope) -> bytes:
    """One NDJSON frame: compact JSON, UTF-8, one trailing newline."""
    try:
        line = env.model_dump_json()
    except (TypeError, ValueError) as e:
        # Only reachable for a hand-built payload holding a live object;
        # anything built through `Message.to_envelope` is already JSON-safe.
        raise ProtocolError(f"cannot serialise {env.type!r}: {e}") from e
    return line.encode("utf-8") + b"\n"


def decode(line: bytes | str) -> Envelope:
    """One NDJSON frame back into an `Envelope`, or `ProtocolError`.

    Everything a stranger can put on the socket is a `ProtocolError` here:
    undecodable bytes, broken JSON, JSON that is not an object, a missing or
    empty type. The caller decides whether that is worth dropping the
    connection over.
    """
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ProtocolError(f"frame is not UTF-8: {e}") from e
    try:
        raw = json.loads(line)
    except ValueError as e:
        raise ProtocolError(f"frame is not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ProtocolError(f"frame is not a JSON object: {type(raw).__name__}")
    try:
        return Envelope.model_validate(raw)
    except ValidationError as e:
        raise ProtocolError(f"not an envelope: {e}") from e


# The registries. Populated by `__init_subclass__` below rather than by hand:
# a table maintained next to the classes is a table that eventually disagrees
# with them, and this one is what dispatch runs on.
COMMANDS: dict[str, type[Message]] = {}
EVENTS: dict[str, type[Message]] = {}


class Message(_Model):
    """A typed body that knows which envelope type it belongs in.

    Subclasses declare `TYPE` and land in `COMMANDS` or `EVENTS` by virtue of
    descending from `Command` or `Event`; a subclass without a `TYPE` is an
    abstract intermediate and is not registered.
    """

    TYPE: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _register(cls)

    def to_envelope(self, *, seq: int = 0, id: str | None = None) -> Envelope:
        return Envelope(
            seq=seq,
            id=id,
            type=self.TYPE,
            # mode="json" so nested models and enums are already primitives:
            # `encode` must not be the first place a payload can fail.
            payload=self.model_dump(mode="json"),
        )

    @classmethod
    def from_envelope(cls, env: Envelope) -> Self:
        expected = getattr(cls, "TYPE", "")
        if expected and env.type != expected:
            raise ProtocolError(f"{cls.__name__} cannot read a {env.type!r} frame")
        try:
            return cls.model_validate(env.payload)
        except ValidationError as e:
            raise ProtocolError(f"bad payload for {env.type!r}: {e}") from e


def _register(cls: type[Message]) -> None:
    """Put a concrete message class in the table its direction dispatches on.

    Defined before `Command` and `Event` because it runs while they are being
    created; it only looks them up for classes that carry a TYPE, and those
    are all declared further down.

    A clash is raised at import time on purpose: two classes claiming one type
    string means one of them is unreachable, and finding that out from a
    mis-routed frame in production is far more expensive than a failed import.
    """
    # `__dict__`, not `getattr`: an intermediate base that merely inherits a
    # TYPE is not a second claim on it.
    type_name = cls.__dict__.get("TYPE", "")
    if not type_name:
        return
    existing = COMMANDS.get(type_name) or EVENTS.get(type_name)
    if existing is not None:
        raise ProtocolError(
            f"duplicate message type {type_name!r}: "
            f"{existing.__name__} and {cls.__name__}"
        )
    if issubclass(cls, Command):
        COMMANDS[type_name] = cls
    elif issubclass(cls, Event):
        EVENTS[type_name] = cls
    else:
        raise ProtocolError(
            f"{cls.__name__} must subclass Command or Event to carry a TYPE"
        )


class Command(Message):
    """UI → core (§4.1). An instruction; the answer, if any, is an event."""


class Event(Message):
    """Core → UI (§4.2). State the UI renders, or the answer to a command."""

    # Set only when this event answers a specific command, echoing the id the
    # UI put on it (§4). Every event carries the field because any of them may
    # be pressed into service as a reply.
    reply_to: str | None = None


def parse(env: Envelope) -> Message:
    """The envelope's typed body, or `ProtocolError` if it has no place here."""
    model = COMMANDS.get(env.type) or EVENTS.get(env.type)
    if model is None:
        raise ProtocolError(f"unknown message type {env.type!r}")
    return model.from_envelope(env)


# --------------------------------------------------------------- payload shapes


class Part(_Model):
    """One ordered piece of a turn's working: a block of reasoning, or one tool
    exchange — the call and the result it returned, held together.

    The wire twin of `hpca.transcript.Step`. Not that class: this module stays
    importable without the agent side, and `transcript` pulls in the LLM types.

    Both halves travel in one part because a client renders them as one row,
    and `done` is what tells it whether the second half has arrived: a call sent
    while the tool is still running has an empty `result`, and the same part
    comes again filled in rather than a second part arriving after it.
    """

    kind: str  # reasoning | call | step
    text: str
    tool: str = ""  # call: the tool named
    target: str = ""  # the file or key the call is about, if it has one
    result: str = ""  # what came back, framing for the model taken off
    done: bool = False  # whether the result has landed
    failed: bool = False  # the tool raised, or the user refused the call


class Entry(_Model):
    """One rendered line of chat — the wire twin of `hpca.transcript.Entry`.

    Copied rather than imported for the same reason as `Part`. The copy is
    only safe while it stays in step with the original, which a test asserts.
    """

    # queued is the one kind the thread has never seen: a message typed ahead
    # of a running turn, drawn as chat because that is what it will become.
    kind: str  # user | assistant | thinking | error | event | recall | queued
    text: str
    steps: int = 0
    reasoning_chars: int = 0
    parts: list[Part] = Field(default_factory=list)
    # The message index this entry IS, or -1 (see the transcript original).
    # What a UI needs to offer the chat rewind on a message the user sent.
    index: int = -1
    # What this ROW is called, so a later frame can revise it in place: see
    # `ChatUpdate`. Assigned by the core, per session, monotonically from 1 —
    # 0 means "not numbered", the way `Envelope.seq` uses it, and marks an
    # entry nobody can send an update for.
    #
    # Not `index`, and the two must not be confused. `index` says which thread
    # *message* an entry is and is -1 for the many entries that are not one
    # (a thinking entry folds several, a queued entry has no message yet);
    # `seq` names the row on screen and every row the core sends has one. A
    # UI keys its rows on this and on nothing else — least of all on list
    # position, which every fold and every rollback changes.
    #
    # Only meaningful within the current `ChatReset` generation: a reset
    # renumbers from 1 and the UI drops what it had, which is what keeps a
    # rollback or a fold from leaving the two sides numbering different rows.
    seq: int = 0


class SessionRow(_Model):
    """One line of the sidebar."""

    session_id: str
    title: str
    profile: str = ""
    mode: str = ""
    # Render markers the core owns because it owns the state behind them —
    # "working", "decision pending". Open-ended: the UI ignores what it does
    # not know how to draw.
    flags: list[str] = Field(default_factory=list)


# What a panel row stands for. The UI turns `kind` plus `ref` into the command
# a keypress sends, so both ends have to agree on the string.
#
# Only one, now that the right column is watch boxes and nothing else. There
# were four — `process`, `job` and an inert `heading` alongside this — for the
# run history the column used to carry underneath (project.md §3.3). The field
# stays rather than being inlined: it is what tells the UI which command a
# keypress becomes, and a column that grows a second kind of row should have to
# say so on the wire rather than infer it from a key prefix.
PANEL_WATCH = "watch"


class PanelRow(_Model):
    """One row of the right column (§4.3), stripped of its widget payloads."""

    # Stable across repaints, so the UI can update text in place instead of
    # rebuilding the column under the user's cursor.
    key: str
    text: str
    classes: str = ""
    title: str = ""  # box border title; watches only
    kind: str = ""  # one of the PANEL_* strings above
    # The pid, job id or watch id behind the row. A string even for the two
    # numeric ids, because a Slurm array task ("1234_7") is not an int and one
    # field beats three; the sender of `process.kill` / `watch.drop` converts.
    ref: str = ""


class Proposal(_Model):
    """A memory the agent suggests keeping; mirrors `agent.conclude`'s model."""

    scope: str
    kind: str
    text: str


# -------------------------------------------------------------------- commands


class SessionList(Command):
    TYPE: ClassVar[str] = "session.list"


class SessionNew(Command):
    TYPE: ClassVar[str] = "session.new"
    profile: str
    # The backend *label* a session is pinned to, which survives that backend
    # being dropped from the catalog (see `sessions.Session.backend`).
    backend: str | None = None


class SessionOpen(Command):
    TYPE: ClassVar[str] = "session.open"
    session_id: str


class SessionClose(Command):
    TYPE: ClassVar[str] = "session.close"


class SessionRename(Command):
    TYPE: ClassVar[str] = "session.rename"
    session_id: str
    title: str


class SessionRetitle(Command):
    """Ask the model for a title, as opposed to being handed one."""

    TYPE: ClassVar[str] = "session.retitle"
    session_id: str


class SessionDelete(Command):
    TYPE: ClassVar[str] = "session.delete"
    session_id: str


class _Rewind(Command):
    """The shape both halves of the chat rewind share (§ chat rewind).

    ``index`` is the entry index the cut is made at — the ``Entry.index`` the
    core itself put on that message — and deliberately *not* the ``keep``
    count `agent.graph.rollback_thread` and `fork_thread` take. The two are
    the same number today, which is exactly why the conversion needs an owner
    rather than happening twice: the UI is naming a message it was shown, the
    graph wants a length, and if a turn has appended to the thread since the
    rewind dialog opened, only the core can say whether that name still points
    where the user thinks it does. So the wire carries the name, the core
    resolves it into a length, and an index the thread no longer has is
    refused rather than silently truncating somewhere else.

    A command that cannot be carried out is refused with a warning `notify`
    and no state change — the same channel every un-carry-out-able command
    answers on. There is no "rejected" event, because a refusal is something
    to tell the user, not state to render. Which conditions refuse *which* of
    the two is not shared, and is stated on each below.
    """

    session_id: str
    index: int


class SessionFork(_Rewind):
    """Branch this conversation into a new session, cut before ``index``.

    Non-destructive: the source keeps its whole history. The new session's
    title, profile, backend and mode are copied from the source by the core
    and are not on the wire — a front-end that could name them could fork a
    conversation into a profile the user never chose.

    Deliberately allowed while the source is busy, unlike the rollback below.
    Both can be aimed at a stale cut point, so that risk does not separate
    them; the consequence does. A fork at a stale index leaves an extra
    session the user deletes, and branching off *while* the agent works is the
    case forking exists for: it only ever reads a checkpoint snapshot of the
    source and only ever writes to a thread nothing else has touched.

    Answered by `session.created`, which carries the new row, because the UI
    has to *open* the fork: `session.rows` alone re-states the sidebar without
    saying which line is new, and diffing two sidebars gets it wrong the
    moment a background turn retitles something in between.
    """

    TYPE: ClassVar[str] = "session.fork"


class SessionRollback(_Rewind):
    """Trim this conversation back to before ``index``, in place.

    Destructive and irreversible, which is why this half is gated and the
    fork is not: refused while a turn is running on the session, while a
    decision is unanswered, or while a message is queued behind it — see
    `core.scheduler.TurnScheduler.rewind_blocker`, which is the core's answer
    to "is it safe", and phrases it to be shown.

    No reply of its own: what the UI needs afterwards is the trimmed
    transcript, which is the ordinary `chat.reset` for that session, plus a
    fresh `context.estimate` — the measured fill described a thread that no
    longer exists.
    """

    TYPE: ClassVar[str] = "session.rollback"


class SessionFocus(Command):
    """Which session the user is looking at; null when none is (§4.4).

    An implicit coupling made explicit: `_deliver_event` decides whether a
    background completion starts a turn from whether its session is on screen,
    and the core cannot know that unless the UI says so.
    """

    TYPE: ClassVar[str] = "session.focus"
    session_id: str | None = None


class TurnSubmit(Command):
    TYPE: ClassVar[str] = "turn.submit"
    session_id: str
    text: str
    forced_skill: str | None = None


class TurnInterrupt(Command):
    TYPE: ClassVar[str] = "turn.interrupt"
    session_id: str


class TurnUnqueue(Command):
    """Take a message back out of a session's type-ahead queue.

    ``seq`` is the row: the `Entry.seq` the core put on the ``queued``
    `chat.append` it sent when the message was accepted. Not the text (the
    same message queued twice must lose one copy rather than both) and not a
    position in the queue — the turn ahead can finish while the user is still
    deciding, and every position behind it then shifts by one, so a position
    that looks valid quietly names the neighbour. A core-issued row name
    cannot drift: it either still refers to a waiting message or it does not.

    A ``seq`` that is no longer queued is refused with a warning `notify` —
    the message has already started, and stopping *that* is `turn.interrupt`,
    a different question asked from a different place.
    """

    TYPE: ClassVar[str] = "turn.unqueue"
    session_id: str
    seq: int = 0


class DecisionResolve(Command):
    """The answer to a parked `interrupt()`.

    The decision is core state; the half-typed reason is a UI draft until it
    is sent, which is why only the finished string crosses (§4.4).
    """

    TYPE: ClassVar[str] = "decision.resolve"
    session_id: str
    approved: bool
    reason: str = ""


class CommandRun(Command):
    """A slash command: `/compact`, `/memorize`, `/conclude`, `/skill-*`."""

    TYPE: ClassVar[str] = "command.run"
    name: str
    # The rest of the typed line, unsplit — the UI parses `/name rest` and the
    # handler decides what `rest` means (a note, a skill name, nothing).
    args: str = ""
    # Which conversation the command acts on. Session-scoped commands
    # (`/compact`, `/conclude`, `/memorize`) carry it; profile-scoped ones
    # (`/skills-list`, `/skill-creator`, `/skill-remove`) leave it None. Sent
    # explicitly rather than resolved from the last `session.focus`: the core
    # would otherwise infer the target of a destructive fold from UI state
    # that may have moved on between the keystroke and the frame arriving.
    session_id: str | None = None


class ConfirmResolve(Command):
    """The answer to a `confirm.requested` question, by its id.

    Same shape as the other two round trips, and for the same reason: the core
    holds the continuation — here, the coroutine that writes a learned log
    signature — and only the yes/no crosses. An id rather than a session_id
    because the question need not belong to a conversation; a triage offer
    comes from a poll.
    """

    TYPE: ClassVar[str] = "confirm.resolve"
    id: str
    confirmed: bool = False


class MemoryResolve(Command):
    """The answer to a `memory.proposals` offer.

    The same produce-then-apply shape as `decision.resolve`, and for the same
    reason: nothing in the core may block waiting for a person. The core holds
    the authoritative proposal objects and this only says which of them were
    approved — positionally, against the order they were offered in. A
    front-end therefore cannot smuggle an edited memory back in an approval,
    which it could if the answer carried the text.

    A short list rejects the rest: an answer that never arrived is not an
    approval.
    """

    TYPE: ClassVar[str] = "memory.resolve"
    session_id: str
    approved: list[bool] = Field(default_factory=list)


class ModeSet(Command):
    TYPE: ClassVar[str] = "mode.set"
    session_id: str
    # Not an enum here: the set of modes is the agent's business, and the core
    # rejects one it does not know. The protocol only has to carry the string.
    mode: str


class ThinkingSet(Command):
    """The other per-session dial (hpca.thinking), carried like ModeSet.

    Same shape and the same reason for it: the level a session reasons at is
    the front-end's to change, and a remote front-end has no other way to say
    so. A plain string for the same reason as ``mode`` above — which levels
    exist is decided by the served model, so the protocol only carries it.
    """

    TYPE: ClassVar[str] = "thinking.set"
    session_id: str
    effort: str


class BackendSet(Command):
    TYPE: ClassVar[str] = "backend.set"
    # An `LLMBackend` as JSON. Opaque on purpose: duplicating a settings model
    # in the protocol would give it two definitions to drift apart, and the
    # core validates it against the real one anyway.
    backend: dict[str, Any] = Field(default_factory=dict)
    # None sets the default backend rather than one session's.
    session_id: str | None = None


class ProfileSet(Command):
    TYPE: ClassVar[str] = "profile.set"
    name: str


class ProfileSave(Command):
    """Write back what the user edited in $EDITOR (§4.4)."""

    TYPE: ClassVar[str] = "profile.save"
    name: str
    kind: Literal["memories", "archive"]
    text: str


class _ProfileEdit(Command):
    """The shape §4.1 gives the three profile lifecycle commands."""

    name: str
    source: str | None = None  # duplicate: the profile copied from


class ProfileCreate(_ProfileEdit):
    TYPE: ClassVar[str] = "profile.create"


class ProfileDelete(_ProfileEdit):
    TYPE: ClassVar[str] = "profile.delete"


class ProfileDuplicate(_ProfileEdit):
    TYPE: ClassVar[str] = "profile.duplicate"


class _SkillEdit(Command):
    """The shape §4.1 gives both skill commands; `text` is unused by delete."""

    profile: str
    name: str
    text: str | None = None


class SkillSave(_SkillEdit):
    TYPE: ClassVar[str] = "skill.save"


class SkillDelete(_SkillEdit):
    TYPE: ClassVar[str] = "skill.delete"


class ProcessKill(Command):
    TYPE: ClassVar[str] = "process.kill"
    pid: int


class JobCancel(Command):
    TYPE: ClassVar[str] = "job.cancel"
    job_id: str


class WatchPeek(Command):
    """Enter on a watch box: what is this thing saying right now?

    The only read-only command in §4.1, and it is still a command rather than
    something the UI could work out for itself: the tail lives in a file on a
    node the UI may not share, and a job's state costs an squeue call. Rule 2
    of §4.2 (the UI never reads the database) makes that the core's errand.
    """

    TYPE: ClassVar[str] = "watch.peek"
    watch_id: int


class WatchDrop(Command):
    TYPE: ClassVar[str] = "watch.drop"
    watch_id: int


class Shutdown(Command):
    TYPE: ClassVar[str] = "shutdown"


# ---------------------------------------------------------------------- events


class Hello(Event):
    """First frame on connect: who the core is and what it is configured as."""

    TYPE: ClassVar[str] = "hello"
    version: int = PROTOCOL_VERSION
    profile: str = ""
    # Lets the UI notice that settings changed under a reconnect without
    # shipping the whole settings tree to a process that cannot use it.
    settings_digest: str = ""


class SessionRows(Event):
    """The sidebar contents — the answer to a `session.list` command.

    Named apart from the command deliberately: §4.2 reuses the string
    `session.list` for this event, but one type string that maps to two
    payload shapes cannot be dispatched by `parse`, which sees only a frame
    and not the direction it travelled.
    """

    TYPE: ClassVar[str] = "session.rows"
    rows: list[SessionRow] = Field(default_factory=list)


class SessionCreated(Event):
    """A session the core has just made, and that the UI is expected to open.

    The reply to `session.fork` (and the one `session.new` needs for the same
    reason): both produce a conversation that exists only because the user
    asked for it, and neither is finished until the front-end is looking at
    it. Carries the whole `SessionRow` rather than a bare id so one frame is
    enough to both open it and put it in the sidebar.

    The `session.rows` that follows re-states the sidebar; this says which row
    is the new one, which a list cannot — two forks of one conversation have
    the same title, and a UI diffing sidebars would pick whichever it saw
    first.
    """

    TYPE: ClassVar[str] = "session.created"
    row: SessionRow


class ChatReset(Event):
    """The whole transcript, on open only — the snapshot before the deltas.

    Also re-bases the row numbering: the entries arrive carrying the `seq`
    values every later `chat.append` and `chat.update` will use, and the UI
    forgets the ones it held. That is what makes a rollback or a fold safe to
    express — the rows it removed cannot be addressed afterwards by either
    side.
    """

    TYPE: ClassVar[str] = "chat.reset"
    session_id: str
    entries: list[Entry] = Field(default_factory=list)


class ChatAppend(Event):
    """One new entry. The reason the history stops crossing the socket every
    turn, and so the reason this protocol is not slower than what it replaces
    (§4.2)."""

    TYPE: ClassVar[str] = "chat.append"
    session_id: str
    entry: Entry


class ChatUpdate(Event):
    """A row that is already on screen, again — revised, in place.

    The other half of "snapshot then delta" (§4.2 property 1), and what makes
    the chat genuinely append-only between resets rather than append-mostly.
    Three things need it, and they are one problem:

    * a tool call that appears the moment it is made and gains its result
      afterwards — `Part.done` already describes exactly this ("the same part
      comes again filled in rather than a second part arriving after it"), and
      until now there was no frame that could say so;
    * a ``queued`` entry becoming an ordinary ``user`` one when its turn
      starts, rather than a second row appearing and the first being guessed
      away;
    * anything a re-opened session's `chat.reset` has to keep addressable.

    Carries the whole entry rather than a patch: rows are small, a diff would
    need the core to model what the UI drew (the same argument `panel.update`
    makes), and the entry names itself through `Entry.seq` — so there is one
    place to get the row identity right instead of two that can disagree.

    An update for a ``seq`` the UI does not have is to be dropped, never
    turned into a new row: it means the client is out of step with a reset,
    and inventing a row would hide that instead of letting the next reset fix
    it.
    """

    TYPE: ClassVar[str] = "chat.update"
    session_id: str
    entry: Entry


class TurnStarted(Event):
    TYPE: ClassVar[str] = "turn.started"
    session_id: str


class TurnActivity(Event):
    """What the turn is doing now, and since when.

    One event replaces show_working / hide_working / report_activity: the
    started_at stamp is what lets the UI run its own clock without the core
    having to tick it.
    """

    TYPE: ClassVar[str] = "turn.activity"
    session_id: str
    activity: str = ""  # empty means the turn is no longer working
    started_at: str = ""  # ISO 8601, as everything stored in hpca is


class TurnUsage(Event):
    TYPE: ClassVar[str] = "turn.usage"
    session_id: str
    prompt_tokens: int
    max_model_len: int | None = None


class TurnFinished(Event):
    TYPE: ClassVar[str] = "turn.finished"
    session_id: str
    reply: str | None = None


class TurnFailed(Event):
    TYPE: ClassVar[str] = "turn.failed"
    session_id: str
    error: str


class TurnUnqueued(Event):
    """A typed-ahead message was taken back: drop its row, keep its text.

    ``seq`` is the row to remove, echoed rather than left for the UI to
    remember; a cancel that could not be carried out is a warning `notify` and
    never this event, so seeing this at all means that row is gone.

    ``text`` rides along because cancelling lands where an interrupt lands —
    the message back in the entry box, to edit and send again — and the UI
    cannot be trusted to still hold it: the answer can arrive after the user
    has switched sessions, and the row it came from may have been redrawn
    since.
    """

    TYPE: ClassVar[str] = "turn.unqueued"
    session_id: str
    seq: int
    text: str


class DecisionRequested(Event):
    """A turn parked at `interrupt()` and needs an answer.

    Re-emitted on subscribe, which is the fix for a decision surviving a
    restart with nothing on screen to answer it (§4.4).
    """

    TYPE: ClassVar[str] = "decision.requested"
    session_id: str
    # The graph's interrupt value, passed through untouched: its shape is the
    # agent's, and the protocol has no reason to have an opinion about it.
    payload: dict[str, Any] = Field(default_factory=dict)


class DecisionCleared(Event):
    TYPE: ClassVar[str] = "decision.cleared"
    session_id: str


class PanelUpdate(Event):
    """The right column, whole. Rows are cheap and diffing them is the UI's
    job — a partial update would need the core to model what the UI drew."""

    TYPE: ClassVar[str] = "panel.update"
    profile: str = ""
    session_id: str | None = None
    rows: list[PanelRow] = Field(default_factory=list)


class WatchPeeked(Event):
    """The answer to a `watch.peek`: a log's tail, or a job's state line.

    Deliberately not a `notify`, though the front-end may well draw it as a
    toast. Three reasons. It answers a keypress rather than announcing
    something the core decided to say, so it echoes ``reply_to``. It names the
    watch: a job peek costs an squeue call and a database read, so two peeks
    in a row genuinely can cross, and a bare string of text could then be
    attributed to the wrong box. And `notify` carries a ``timeout`` — how long
    a tail should stay on screen is a rendering decision, and a core reading a
    file has no business making it.

    A read that failed comes back as ordinary ``text`` ("(could not read …)",
    see `watches.peek`) rather than as an error: what happened to the log is
    exactly what the user pressed Enter to find out, and it belongs in the
    same place the tail would have been.
    """

    TYPE: ClassVar[str] = "watch.peeked"
    watch_id: int
    title: str = ""  # the box's border title — what this is the tail *of*
    # Already trimmed to a glance by the core (`watches.PEEK_CHARS`): the
    # whole point of peeking is not to ship a gigabyte of progress bars.
    text: str = ""


class MemoryProposals(Event):
    TYPE: ClassVar[str] = "memory.proposals"
    session_id: str
    proposals: list[Proposal] = Field(default_factory=list)


class ConfirmRequested(Event):
    """A yes/no question that is not a tool approval.

    One case today: triage proposing a log signature it just learned (§5.5
    tier 3). It is deliberately not `decision.requested` — that one parks a
    graph thread and its answer resumes a turn, whereas this one comes from a
    poll and nothing is waiting on it. Conflating them would let an unanswered
    triage offer look like a stalled session.
    """

    TYPE: ClassVar[str] = "confirm.requested"
    id: str
    question: str


class ContextEstimate(Event):
    TYPE: ClassVar[str] = "context.estimate"
    session_id: str
    used: int
    window: int


class Notify(Event):
    """A toast. The three severities are Textual's, so an invalid one is
    caught here rather than raising inside the render."""

    TYPE: ClassVar[str] = "notify"
    severity: Literal["information", "warning", "error"] = "information"
    text: str
    # Seconds to keep it up, or None for the renderer's default. Carried
    # because a few warnings — a matched past struggle, the memory budget, a
    # curation run — are ones the user is meant to actually read, and the core
    # is what knows which those are. How long a second is remains the
    # renderer's business; this is a hint, not a layout instruction.
    timeout: float | None = None
