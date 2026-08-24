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
    # When this happened, ISO-8601 UTC — see the transcript original. UTC on
    # the wire and not the core's local rendering of it, because a core and a
    # front-end need not be on the same machine (specs-core-process.md) and
    # only the front-end knows which clock a person is reading.
    at: str = ""


class SessionRow(_Model):
    """One line of the sidebar."""

    session_id: str
    title: str
    profile: str = ""
    mode: str = ""
    # When something last happened in this conversation, ISO-8601 UTC: a turn
    # submitted, or one recorded. Stored on the session rather than derived
    # from the thread, because the sidebar draws every row and reading each
    # one's last message would mean opening every thread to paint a list.
    #
    # Named for what it is rather than `updated_at`, which would sit one
    # character away from the front-end's own `updated` — the unread marker —
    # in code that draws both on the same row.
    last_active: str = ""
    # The model this conversation is pinned to; empty when it talks to the
    # bootstrap client. On the wire because two things drew it — the sidebar
    # and the message row (§4.3 item 20) — and a front-end cannot work it out:
    # the backend a session uses is a JSON blob in the database, which rule 2
    # of §4.2 puts out of its reach, so the first real UI consumer had to drop
    # the model line for want of this field. A bare name rather than the
    # backend entry, because a label is all that is being drawn and shipping
    # the entry would put an api_key on the wire to render one.
    model: str = ""
    # The session's thinking level (hpca.thinking), empty when it never chose
    # one and follows the configured default.
    #
    # Here rather than in an event of its own, which was the other candidate
    # for closing the "`ThinkingSet` is a command with no answer" gap. Three
    # reasons, and the third is the deciding one. It is the same class of thing
    # as `mode` — a per-session dial the core stores and the front-end changes
    # — and `mode` is already here. Every path that can change it already
    # restates the sidebar (`thinking.set`, `command.run` of `/thinking`, a new
    # session, a fork), so an event would be a second announcement of a change
    # that has just been announced. And a `thinking.changed` event could only
    # ever describe the session that just changed, whereas the meter has to
    # draw the level of whichever session is opened next: a session left on
    # xhigh looks identical to one on off until the first wait, which is
    # exactly the case the meter's `· think medium` exists to prevent.
    thinking: str = ""
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


class LLMEntry(_Model):
    """One line of the LLM catalog: what a front-end draws, and what it sends
    back to act on it.

    Not `config.LLMBackend`, and the difference is the point. That model holds
    an ``api_key``, and a catalog crosses this socket precisely so that a
    picker can be drawn — `SessionRow.model` set the precedent (a bare name
    rather than the entry, "because shipping the entry would put an api_key on
    the wire to render one"). So the key never travels.

    ``needs_key`` is what is *still missing*: the endpoint asked for a key and
    no key we hold has been accepted. It is not "a key is involved" — a row
    that a stored or pool key already unlocked draws no hint, because "api key
    required" next to a backend the user has just given a working key to reads
    as the key not having taken.

    ``label`` is the identity, and the one field a command may name an entry
    by (`SessionNew.backend`). Minted by the core (`BackendRegistry.catalog`)
    so both sides cannot disagree about what a backend is called: the model
    name where it is unique in the catalog, and ``model @ host:port`` where it
    is not, because two vLLMs serving one model on two nodes are two entries a
    user has to be able to tell apart.

    The three render markers the manage-LLMs and switch-LLM screens draw:

    * ``active`` — the ★: what a session that pinned nothing talks to.
    * ``reachable`` — ● connected / ○ disconnected, and **None for neither**.
      A probe costs a round trip to a cluster node, so the catalog is answered
      first with nothing known and restated once the probes land; a client
      that drew "disconnected" for unknown would libel every backend for as
      long as the scan takes.
    * ``discovered`` — this entry was found by a scan rather than configured.
      One list with a flag rather than two lists, because everything else
      about the two rows is identical and a second field would have to be kept
      in step with the first.
    """

    label: str
    model: str
    base_url: str = ""
    max_model_len: int | None = None
    needs_key: bool = False
    active: bool = False
    reachable: bool | None = None
    discovered: bool = False


# Where a skill lives, and so who sees it and who may change it. Spelled out
# here rather than imported from `hpca.skills` because this module imports
# nothing of hpca's own (see the module docstring); the two lists are the same
# names that module uses, and `tests/test_protocol.py` holds them to it.
#
# Two literals rather than one, and the difference is the point: a listing may
# report all four levels, and a *write* may only name the three a user owns.
# The shipped level is read-only — it is package data, and a `skill.save` that
# could name it would write into the install.
SkillOrigin = Literal["builtin", "global", "profile", "project"]
SkillLevel = Literal["global", "profile", "project"]

# Which set `skill.list` answers with. ``own`` is the editable set — what
# `skill.save` overwrites and `skill.delete` removes for that profile, and so
# what a screen offering those two may draw. ``visible`` is the callable set —
# everything `/<skill>` can name, the shipped and shared skills included, each
# row saying which level it resolved from. A menu wants the second and an
# editor wants the first, which is why one command answers both and the answer
# says which it is.
SkillScope = Literal["own", "visible"]


class SkillRow(_Model):
    """One line of a profile's skill menu (`skill.rows`).

    What the menu draws and nothing else. The body is `skill.get`'s answer,
    fetched when an editor opens rather than carried here — a listing has no
    use for it, and one carried here would be stale by the time it is edited.

    ``level`` is where the skill resolved from, and it is what tells a
    front-end which rows it may offer to remove: a ``visible`` listing mixes
    the shipped and shared skills — which are not one profile's to delete —
    with the profile's and the project's, which are.
    """

    name: str
    description: str = ""
    level: SkillOrigin = "profile"


class ProfileRow(_Model):
    """One line of the profiles screen (`tui/profiles_screen.py`).

    Everything that screen puts on a row and nothing else: the name, how much
    the profile has learned, and — for a copy — what it was copied from, which
    is "the thing you need to know when they start disagreeing with each
    other". The memories are counted rather than sent: the screen shows a
    count, and the text of one is a `profile.save` round trip away.

    ``sessions`` is counted the same way and for the same reason, and it is
    the core's answer rather than something a front-end can total up from the
    sidebar: the sidebar holds the sessions it has been sent, and a profile
    that has conversations under it is still a profile the user may be about
    to delete.

    Two flags rather than one, because they answer different questions. The
    ★ marks the *default* profile, which is where a deleted profile's sessions
    land and so the one that cannot be deleted; ``working`` marks the one the
    core is currently running under, which is where the picker's cursor
    starts. They are usually different profiles.
    """

    name: str
    memories: int = 0
    # How many conversations are filed under it. Zero is both "none" and "the
    # core could not count them"; the row simply says nothing about sessions
    # in either case, which is what it did before it could count at all.
    sessions: int = 0
    copied_from: str = ""
    is_default: bool = False
    working: bool = False


# -------------------------------------------------------------------- commands


class SessionList(Command):
    TYPE: ClassVar[str] = "session.list"


class SessionNew(Command):
    """Make a conversation under this profile, talking to this backend.

    ``backend`` is an `LLMEntry.label` — a name out of the catalog the core
    answered `llm.list` with — and null means "whatever the core would have
    picked", which is what a run with no configured backends gets.

    It was documented as a label and implemented as serialised `LLMBackend`
    JSON, which is worse than either: a front-end sending what this docstring
    described pinned nothing at all, silently, because an unparseable value
    fell back to the bootstrap client. A label settles it in the direction the
    documentation already pointed, and it is now answerable — `llm.catalog`
    gives a front-end the names to use, and one it does not recognise comes
    back as a warning `notify` rather than a session quietly created against
    the wrong model.

    The blob does not survive as an alternative spelling. Two accepted forms
    would mean a typo'd label that happened to parse as JSON pinning something
    nobody chose, and the one thing that genuinely needs to name a backend
    that is not in the catalog — a hand-filled connection form — is
    `backend.set`, which still carries the whole entry.
    """

    TYPE: ClassVar[str] = "session.new"
    profile: str
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


class SessionMove(Command):
    """alt+↑ / alt+↓ on a sidebar row: shift it one place, for good.

    ``delta`` is an offset and not a slot — ``-1`` for up, ``+1`` for down —
    because that is the whole gesture the user has. Two presses walk a row
    past two others; nothing in a keypress can say "third from the top", so
    nothing in the command tries to.

    The core answers with `session.rows` (and `watch.move` with
    `panel.update`), which is the point of routing a keypress through the
    socket at all rather than letting the front-end shuffle its own list. The
    arrangement is a fact about the database, which §4.2 rule 2 puts out of a
    front-end's reach, so the only order that survives a restart is the one
    the core sends back — and a UI that reordered locally would be overwritten
    by the very next frame anyway.

    A move at the end of the list changes nothing and is answered the same
    way, with the unchanged list. That is not a refusal worth a message: it is
    what holding the key down looks like once the row has arrived.
    """

    TYPE: ClassVar[str] = "session.move"
    session_id: str
    delta: int


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


class CompactResolve(Command):
    """The answer to a `compact.proposed` offer.

    The third round trip of the same shape as `memory.resolve` and
    `confirm.resolve`, and for the same reason: the core holds the thing being
    decided — here the candidate summary and the position in the history it
    covers — and only the verdict crosses. A front-end therefore cannot hand
    back an edited summary, which is the one way a fold could quietly become
    something the model never wrote.

    Three answers, because a summary is not a yes/no. `accept` lands the fold.
    `retry` throws this attempt away and asks for another one, and `comment` is
    what the user said was wrong with it — the whole point of the exchange, and
    why this carries text at all. `discard` leaves the conversation as it was.

    Saying nothing is also allowed: a review closed without an answer sends no
    command, and the core keeps holding the offer, so `/compact` brings it back
    instead of paying for a second summary.
    """

    TYPE: ClassVar[str] = "compact.resolve"
    session_id: str
    action: Literal["accept", "retry", "discard"] = "discard"
    # What the summary has to do differently. Only `retry` reads it.
    comment: str = ""


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
    """Point a session — or everything without one — at a backend.

    Two ways of naming it, because there are two situations and only one of
    them can be named by value.

    ``label`` names an entry of the catalog the core just sent
    (`LLMEntry.label`). This is the form a picker uses, and it exists because
    the catalog deliberately carries no api_key: with only the blob form, a
    front-end could not switch a session to a key-locked backend at all — it
    would have to send back an entry it was never given the key for, and the
    core would build a client that 401s. Naming the entry lets the key stay
    where it already is.

    ``backend`` is a whole `LLMBackend` as JSON, for the one case a label
    cannot cover: a hand-filled connection form names a backend that is not in
    the catalog yet, so there is no label to name it by. Opaque on purpose —
    duplicating a settings model in the protocol would give it two definitions
    to drift apart, and the core validates it against the real one anyway.

    A label wins when both are set: it is the more specific of the two, and a
    front-end that sends both has already been handed the entry.
    """

    TYPE: ClassVar[str] = "backend.set"
    backend: dict[str, Any] = Field(default_factory=dict)
    label: str = ""
    # None sets the default backend rather than one session's.
    session_id: str | None = None


class BackendProbe(Command):
    """Ask one endpoint what it serves — the connection form's check.

    The form cannot answer this itself (§4.2 rule 2, and the endpoint may be
    on a node the front-end cannot reach), and the answer does three jobs at
    once: it says whether anything OpenAI-shaped is there, it validates the
    key that was typed, and it names the models so a blank model field can be
    auto-filled — or a picker offered when the endpoint serves several.

    ``api_key`` is the one place a key travels UI → core, and it is not a leak
    of the rule that keeps keys off the wire: the user has just typed it into
    the form, the core is the only side that can spend it, and nothing sends
    it back. Answered by `backend.probed`.
    """

    TYPE: ClassVar[str] = "backend.probe"
    base_url: str
    api_key: str | None = None


class BackendScan(Command):
    """Look for endpoints nobody has configured yet — manage-LLMs' rescan.

    Two searches under one command, because a backend can be reached two ways
    and neither search finds the other's hits: a localhost port sweep finds
    the SSH-tunnelled ones, and the cluster's manifest dir declares the ones
    living on a compute node's own IP, which no localhost scan can see.

    Slow — the sweep is tens of thousands of ports — so it is answered
    incrementally: every hit restates `llm.catalog` with the new row flagged
    `discovered`, and `backend.scanned` closes the run. A front-end may drop
    the whole thing by simply not drawing further frames; the scan does not
    hold a screen open.
    """

    TYPE: ClassVar[str] = "backend.scan"


class BackendRemove(Command):
    """Drop a catalog entry, named the way a picker knows it.

    The other half of `backend.set`, which is what adds one. By label rather
    than by value for the same reason the by-label set exists: what the screen
    is holding is an `LLMEntry`, which has no key to send back, and an entry
    identified by a re-sent blob would be one a wrong key could fail to match.
    """

    TYPE: ClassVar[str] = "backend.remove"
    label: str


class LLMList(Command):
    """Ask for the LLM catalog — what a picker and the manage-LLMs screen draw.

    A command rather than a handshake frame, and the same shape as
    `session.list` for the same reason: the catalog is a screen's worth of
    state a client may not need at all, and a front-end that wants it at
    startup asks for it in the same breath as the sidebar.

    Answered by `llm.catalog`, usually twice — once with what settings say and
    once with the probes filled in (`LLMEntry.reachable`).
    """

    TYPE: ClassVar[str] = "llm.list"
    # Whether to go and ask each endpoint whether it is up. Off is for a
    # client that only needs names (the new-session picker), on for the
    # screen that draws ● / ○; the first frame is identical either way, so a
    # client that asks for probes has nothing extra to wait for before it can
    # draw.
    probe: bool = False


class ProfileList(Command):
    """Ask for the profiles — the answer is `profile.rows`.

    The profiles screen used to be assembled by inference: `hello` named one
    profile, the sidebar rows named the others, and a picker was handed
    whatever the last screen happened to hold. That works and is inference
    where an event belongs — none of those sources knows how much a profile
    has learned or what it was copied from, and a profile with no session in
    the sidebar appeared in none of them.
    """

    TYPE: ClassVar[str] = "profile.list"


class ProfileSet(Command):
    TYPE: ClassVar[str] = "profile.set"
    name: str


class ProfileGet(Command):
    """Fetch the body an editor is about to open — and about to overwrite.

    The read half of `profile.save`, and it exists because the write half
    writes *verbatim*: an editor opened over a body it could not fetch would
    save an empty buffer over the file. `profile.rows` carries a memory
    *count*, which is what a listing draws and is no use to an editor.

    Named by the same two words the save is (``name``, ``kind``) rather than
    by a path, so a front-end can neither read nor write anything but the two
    files it is allowed to edit; a general "give me this file" command would
    be exactly the `run_bash`-shaped endpoint this protocol is careful not to
    be. Answered by `profile.body`.
    """

    TYPE: ClassVar[str] = "profile.get"
    name: str
    kind: Literal["memories", "archive"]


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
    """Write one skill file, verbatim, at one level.

    ``level`` is where it lands — the profile's own directory, the shared
    `_shared/` one every profile sees, or the project's `.hpca/skills` in the
    working directory. It defaults to `profile`, which is where a hand-edited
    file came from and where a self-review patch goes, so an editor saving
    back what `skill.get` handed it needs no opinion about levels at all.

    Deliberately narrower than what `skill.list` can *report*: the shipped
    level is missing from `SkillLevel` because it is package data. Listing and
    writing have different scopes on purpose — a front-end may see a skill it
    may not overwrite.
    """

    TYPE: ClassVar[str] = "skill.save"
    level: SkillLevel = "profile"


class SkillDelete(_SkillEdit):
    """Remove one skill file — the profile's own, or this project's.

    The two removable levels, and only those: deleting a shared or shipped
    skill from one profile would silently change every other profile that
    sees it. Named rather than levelled, because a name resolves to at most
    one removable file (`skills.delete_own_skill`).
    """

    TYPE: ClassVar[str] = "skill.delete"


class SkillList(Command):
    """What skills a profile has — answered by `skill.rows`.

    ``scope`` decides which set, and the two exist because two screens ask
    different questions of the same command:

    * ``own`` — the profile's own directory, which is exactly what the editor
      may open (`skill.get`), overwrite (`skill.save`) and delete
      (`skill.delete`) for that profile. The default, because a screen that
      offers those three must not list a file it cannot touch.
    * ``visible`` — everything the profile can *call*: the shipped skills,
      `_shared/`, its own, and the project's, each row tagged with its level.
      This is what a "/" menu needs — HPCA ships skills, so a menu built from
      `own` reports `/plan` unknown on a fresh install.

    So listing is wider than writing, on purpose. A `visible` row is not a
    permission: what may be written is `SkillLevel`, and what may be removed
    is what `skill.delete` will find.
    """

    TYPE: ClassVar[str] = "skill.list"
    profile: str
    scope: SkillScope = "own"


class SkillDraft(Command):
    """Ask the model for a first draft of a skill — answered by `skill.drafted`.

    `/skill-creator <what it should do>`. The form belongs to the front-end
    and the *draft* cannot: a draft is a model call, and only the core makes
    those (§4.2 rule 1). So the request crosses, one generation happens here,
    and three fields come back to be edited.

    Nothing is written by this command, which is the whole reason it is not a
    `skill.save`: the draft is a head start inside a form the user still edits
    and confirms, and what they confirm arrives as an ordinary `skill.save`.

    ``session_id`` names the conversation the request came out of. It is not
    decoration: "write a skill for what we just did" is the common case, so
    the transcript goes to the drafter with the request, and the session is
    also where the core reports the wait (`turn.activity`) while the model is
    writing. Null when the request came from no conversation.
    """

    TYPE: ClassVar[str] = "skill.draft"
    profile: str
    request: str
    session_id: str | None = None


class CommandList(Command):
    """How often each slash command has been run — answered by `command.counts`.

    The frequency sort behind a "/" menu. Asked for on connect because a menu
    is sorted the first time it is drawn; restated by the core afterwards, so
    a front-end asks once and never polls.
    """

    TYPE: ClassVar[str] = "command.list"


class SkillGet(Command):
    """One skill's file, verbatim — the body `skill.save` will overwrite.

    Separate from `skill.list` rather than folded into its rows, for two
    reasons. A listing is drawn for every skill and a body is opened for one,
    so shipping forty bodies to draw a menu is waste; and a body fetched when
    the menu was drawn is a body that can be minutes stale by the time the
    editor opens over it — which, with a verbatim save behind it, is how an
    edit made elsewhere gets silently reverted. Answered by `skill.body`.
    """

    TYPE: ClassVar[str] = "skill.get"
    profile: str
    name: str


class SettingsGet(Command):
    """The settings file as text — answered by `settings.body`.

    Verbatim, api keys and all, and that is deliberate rather than an
    oversight of the rule that keeps keys out of `llm.catalog`. This is the
    one screen whose *purpose* is editing that file; it saves what it shows,
    so a redacted body would delete every key the user has on the next save,
    and a catalog row is drawn for people who are not editing keys at all.
    """

    TYPE: ClassVar[str] = "settings.get"


class SettingsSave(Command):
    """Write the settings file, and make the change take effect.

    Validation is split, because the two halves answer different questions at
    different speeds. A front-end can tell whether the text is *JSON* with the
    standard library and no idea what a setting is, which is the check an
    editor needs synchronously to refuse to close. Whether it is a valid
    `Settings` — which fields exist, what they may hold — is the core's model
    to know, so the core re-validates and refuses the write, saying why; a
    front-end is not asked to carry a copy of the schema in order to be
    trusted with it.

    Applying is the half that has no other home: only the core can rebuild the
    clients it built at startup, which is why the old front-end could do
    nothing better than say "applies on next start". Answered by
    `settings.body` (what actually landed on disk, normalised) plus whatever
    the change set in motion.
    """

    TYPE: ClassVar[str] = "settings.save"
    text: str


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


class WatchMove(Command):
    """alt+↑ / alt+↓ on a watch box: shift it one place in the column.

    The right column's half of `session.move`, in the same shape and for the
    same reasons — see there. Answered with `panel.update`, the column whole,
    in the order the store now holds.

    Scoped by the core to the column the box is in, which is the session that
    registered it: a store-wide swap would put it next to a box belonging to a
    conversation the user is not even looking at.
    """

    TYPE: ClassVar[str] = "watch.move"
    watch_id: int
    delta: int


class Shutdown(Command):
    TYPE: ClassVar[str] = "shutdown"


# ---------------------------------------------------------------------- events


class Palette(_Model):
    """The colours, by the job each does. `config.PaletteSettings`' twin.

    Strings rather than parsed colours, and they cross unvalidated on purpose:
    the receiving side has to check them anyway — it is the side holding a
    terminal in raw mode, where a malformed escape sequence is not a bad value
    but a lost screen — so checking them here as well would only move the error
    message somewhere the user cannot see it.
    """

    agent: str = "255"
    user: str = "215"
    chrome: str = "73"
    ok: str = "71"
    warn: str = "172"
    danger: str = "167"
    muted: str = "250"
    faint: str = "245"
    spinner: list[str] = ["73", "66", "23", "236"]
    flash: str = "23"


class DisplaySettings(_Model):
    """The settings a front-end *renders with*, and nothing else.

    The counterpart to `Hello.settings_digest`, and the reason that field is
    phrased the way it is: a UI cannot read the settings file (§4.2 rule 2),
    but two or three of the keys in it are about nothing except what a frame
    looks like, and a digest cannot answer "draw this how". So those keys —
    and strictly those — cross as a payload of their own.

    Not the whole tree, which is the point the digest was making: the tree
    holds api keys and endpoint addresses, and a process that only needs to
    know whether to print a timestamp has no business being handed them. Nor
    is it `settings.body`, which is the *file as text* and exists for one
    thing, the config editor opening over it. That is a read path answering a
    keypress; this is state, delivered without being asked for, on the one
    frame that is guaranteed to arrive before anything is drawn.

    `config.DisplaySettings` is the twin, field for field. Two models rather
    than an import for the reason `Part` and `Entry` are copies: this module
    is the only thing both processes agree on and it imports nothing of
    hpca's own.
    """

    # `▸ you 22-08-2026 13:04:47` against a bare `▸ you` (`ui.state._label`).
    chat_stamps: bool = True
    # Whether the cleared screen behind "Really quit?" rains (`ui.rain`).
    quit_rain: bool = True
    # And how many frames a second it falls at. The receiving side clamps
    # rather than trusts: a repaint loop is not a place to divide by zero.
    quit_rain_fps: int = 60
    # One breath of the decision prompt's answer line, in seconds
    # (`ui.ansi.pulse`). The receiving side treats a value it cannot divide by
    # as "use the built-in", because a repaint loop is not a place to raise.
    decision_pulse_seconds: float = 1.0
    # What the front-end draws with (`ui.theme`).
    palette: Palette = Palette()
    # How long the pane that just took focus is washed in `palette.flash`.
    # Zero is off. The receiving side clamps, for the reason the fps above
    # does: a frame is booked from this, and a negative delay books nothing.
    focus_flash_seconds: float = 0.1
    # Blank rows under each chat row (`ui.pane.SPACER_LINES`). The receiving
    # side clamps rather than trusts, for the reason the fps above does: this
    # is a count a repaint loop builds a list from, and a negative one would
    # be a `range` that quietly draws nothing while a very large one is a pane
    # made of gap.
    spacer_lines: int = 1


class Hello(Event):
    """First frame on connect: who the core is and what it is configured as."""

    TYPE: ClassVar[str] = "hello"
    version: int = PROTOCOL_VERSION
    profile: str = ""
    # Lets the UI notice that settings changed under a reconnect without
    # shipping the whole settings tree to a process that cannot use it.
    settings_digest: str = ""
    # The exception the digest carves out: the handful of keys that decide
    # what a frame looks like, which a front-end cannot render without and
    # cannot read for itself. Here rather than in a frame of its own because
    # the first frame is the one that arrives before anything is drawn — a UI
    # that had to ask would draw one frame in whatever it assumed, and a
    # timestamp appearing on the second frame is a redraw the user sees.
    display: DisplaySettings = DisplaySettings()


class DisplayChanged(Event):
    """The display settings again, because they have just been edited.

    `hello` alone would have made these restart-only, which is the very toast
    `settings.save` exists to stop printing: the config editor is *in* the
    app, and a key whose whole subject is what the screen looks like must
    take effect on the frame after the save.

    Its own event and not a re-sent `hello`: that one is a handshake, sent to
    one subscriber at the moment it attaches (`Core.subscribe`), and sending
    it again would have every attached front-end re-run its version check and
    re-ask for the sidebar, the catalog and the profiles. This is news, so it
    fans out to everyone — a second front-end on the same core is looking at
    the same settings file.

    Sent only when the section actually changed, for the same reason the
    clients are only rebuilt when the llm section did: a repaint of every
    conversation's chat rows is not the price of an edited log level.
    """

    TYPE: ClassVar[str] = "display.settings"
    display: DisplaySettings = DisplaySettings()


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


class LLMCatalog(Event):
    """Every LLM the core knows about — the answer to `llm.list`.

    The event this protocol was missing: nothing carried the configured
    backends across, so the new-session picker and the manage-LLMs screen
    could only be populated by a demo. Rule 2 of §4.2 puts the settings file
    out of a front-end's reach the same way it puts the database there, so a
    catalog has to be an event or it does not exist.

    Whole, never incremental, for the reason `session.rows` and `panel.update`
    are: a catalog is a handful of rows, and a diff would need the core to
    model what the front-end drew. It is restated when it changes — a default
    switched, an entry added — and again when the probes land.
    """

    TYPE: ClassVar[str] = "llm.catalog"
    entries: list[LLMEntry] = Field(default_factory=list)
    # Whether the ``reachable`` fields in this frame are answers or "not asked
    # yet". Carried because both are legitimate states of the same field and a
    # client redrawing ● / ○ has to know whether a None means "still
    # scanning" (leave the last marks up) or "nobody asked" (draw neither).
    probed: bool = False


class BackendProbed(Event):
    """What one endpoint answered — the reply to `backend.probe`.

    Three outcomes, told apart without a status enum because the two fields
    already say it: rows and nothing else means it answered and these are the
    models it serves (one auto-fills the form, several are a picker); no rows
    with ``needs_key`` means it is there but the key was missing or refused;
    no rows and no ``needs_key`` means nothing OpenAI-shaped answered at all.

    `LLMEntry` rather than a shape of its own, because a probed model is drawn
    by the same row renderer as a catalogued one and carries the same four
    facts. ``label`` is the model id here — the entry is not in a catalog yet,
    so there is nothing for a label to disambiguate it from.
    """

    TYPE: ClassVar[str] = "backend.probed"
    base_url: str
    models: list[LLMEntry] = Field(default_factory=list)
    needs_key: bool = False


class BackendScanned(Event):
    """A scan is over, and what its emptiness meant (`backend.scan`).

    The rows themselves have already arrived — each hit restates
    `llm.catalog` — so this frame exists for the verdict, which is the part a
    front-end cannot reach: "nothing found" means something different
    depending on whether the cluster's manifests declared an endpoint and
    whether anything already configured still answers, and both of those are
    the core's probes.

    Two texts rather than one, because they are rendered differently and the
    difference is the point. ``notice`` is a passing remark — nothing new
    turned up — and belongs in a toast. ``help`` is the tunnel recipe for the
    off-cluster case: several lines that have to be retyped into a shell, so
    it needs a window that holds a selection and waits to be dismissed. Both
    empty means the scan found things and there is nothing to explain.
    """

    TYPE: ClassVar[str] = "backend.scanned"
    # How many endpoints the two searches turned up, so a status line can say
    # so without counting rows it may have chosen not to draw.
    found: int = 0
    cluster: int = 0
    notice: str = ""
    help: str = ""


class ProfileRows(Event):
    """The profiles, whole — the answer to `profile.list`."""

    TYPE: ClassVar[str] = "profile.rows"
    rows: list[ProfileRow] = Field(default_factory=list)


class ProfileBody(Event):
    """One editable body, verbatim — the answer to `profile.get`.

    ``text`` is the file exactly as it is on disk, not a re-rendered version
    of what the core parsed out of it: a memory file the parser choked on is
    precisely the one a user opens the editor to fix, and handing back the
    parsed subset would quietly delete the lines it could not read.

    ``error`` is what separates "this file is empty" from "this file could not
    be read", which look identical in ``text`` and must not: an editor may
    open over the first and must refuse the second, because the save behind it
    writes verbatim. Empty ``error`` is the only permission to edit.
    """

    TYPE: ClassVar[str] = "profile.body"
    name: str
    kind: Literal["memories", "archive"]
    text: str = ""
    error: str = ""


class SkillRows(Event):
    """A profile's skills, as a menu draws them — answered to `skill.list`.

    Name, description and level. The body is a separate fetch (`skill.get`)
    and deliberately not here: see that command for why a body carried by a
    listing is a body that goes stale before it is edited.

    ``scope`` echoes the question, and it has to: the two scopes answer two
    screens — a menu asked for `visible`, an editor asked for `own` — and a
    front-end holding both would otherwise fill one list with the other's
    answer, offering `delete` on a skill that belongs to the package.
    """

    TYPE: ClassVar[str] = "skill.rows"
    profile: str
    scope: SkillScope = "own"
    skills: list[SkillRow] = Field(default_factory=list)


class SkillDrafted(Event):
    """The model's first draft of a skill — answered to `skill.draft`.

    The three fields of the form, for the form to open pre-filled. Not a file
    and not a `skill.save`: nothing has been written, and nothing will be
    until the user confirms what they see.

    ``error`` is why there is no draft — the model said nothing usable, or
    the backend is down. The event is sent either way, with the fields empty,
    because a failed draft must still open the empty form: the alternative is
    a command that silently swallows what the user typed.

    ``request`` is echoed so a front-end can tell which request this answers
    and put it in front of the user again if the draft is not what they meant.
    """

    TYPE: ClassVar[str] = "skill.drafted"
    profile: str
    request: str = ""
    name: str = ""
    description: str = ""
    body: str = ""
    error: str = ""


class CommandCounts(Event):
    """How often each slash command has been run — answered to `command.list`.

    Whole, never incremental, for the reason `llm.catalog` is: it is a handful
    of numbers, and a delta would need the core to model what the front-end
    holds.

    **Why not a field on `hello`.** That was the obvious place — the counts
    are read once when the menu is first drawn, and `hello` is what a client
    is handed on connect. But `hello` is a handshake: it is stated once and
    never restated, and these numbers change every time the user runs a
    command. A count carried by the greeting would be right for one frame and
    quietly wrong for the rest of the session, and the menu would keep its
    order until the front-end was restarted. So the core restates this after
    every command it counts, and a front-end that asked once stays current
    without polling a table it is not allowed to read (§4.2 rule 2).
    """

    TYPE: ClassVar[str] = "command.counts"
    counts: dict[str, int] = Field(default_factory=dict)


class SkillBody(Event):
    """One skill file, verbatim — the answer to `skill.get`.

    ``error`` carries the same meaning as `ProfileBody.error`, and for the
    same reason: `skill.save` writes what it is given.
    """

    TYPE: ClassVar[str] = "skill.body"
    profile: str
    name: str
    text: str = ""
    error: str = ""


class SettingsBody(Event):
    """The settings file as text — answered to `settings.get`, and restated
    after a `settings.save`.

    Restated on save because what lands on disk is not what was sent: the core
    writes the validated model back out, so keys get their canonical order and
    every default the file omitted becomes explicit. A front-end that kept its
    own copy of the text it sent would show something the file no longer says.

    ``error`` is why a save was refused — the pydantic verdict, one line, on
    the text that did not get written. The body then still describes the file
    that is still there.
    """

    TYPE: ClassVar[str] = "settings.body"
    text: str = ""
    error: str = ""


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
    """A turn began, and when.

    ``started_at`` is the same stamp `turn.activity` carries, sent from the
    turn's first frame rather than from its first activity report: the elapsed
    clock the user reads is "how long since I sent it", and a turn is silent
    for as long as the backend takes to answer the first time. The UI runs its
    own clock off this; the core never ticks one.
    """

    TYPE: ClassVar[str] = "turn.started"
    session_id: str
    started_at: str = ""  # ISO 8601, as everything stored in hpca is


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
    """What the backend said the last decision cost, and how long it took.

    ``prompt_tokens`` is what occupies the window; the completion is spent the
    moment it is generated, which is why the two are different claims and only
    the first moves the meter's fill.

    The other two are the speed. They are carried as the pair rather than as a
    ready-made rate because they are two different measurements: the token
    count is the backend's, and the wall clock is ours — an OpenAI-style body
    carries no timing at all, so `llm.LLMClient` times the request itself and
    puts ``request_seconds`` in the same usage dict (see `llm.py`). Dividing
    them is a rendering decision (the meter shows one decimal only where it
    carries information), and a client that wants to show "3.4s for 210
    tokens" instead cannot get that back out of a rate.

    Both default to nothing, and a client must treat them that way: a session
    restated after a backend switch has a prompt size and no fresh generation
    behind it, and a backend that reports no completion count leaves the
    speed unknown rather than zero.
    """

    TYPE: ClassVar[str] = "turn.usage"
    session_id: str
    prompt_tokens: int
    max_model_len: int | None = None
    completion_tokens: int = 0
    # Seconds of wall clock around the request that produced them. None when
    # nothing has been generated for this session yet.
    request_seconds: float | None = None


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


class TurnInterrupted(Event):
    """A stopped turn left nothing behind: here is the message back.

    Sent for the *exception*, not the rule. Stopping a turn keeps its work —
    the message and everything the agent got through stay in the conversation
    (`TurnScheduler.interrupt`), so handing the text back as well would have
    the user send it twice, and no such event is sent. The one case that still
    needs it is a stop that lands before the message reached the thread: there
    is then no exchange to keep and the sentence would simply be lost.

    The sibling of `turn.unqueued`, and deliberately not the same event. Both
    hand a message back to be edited and re-sent, and both are addressed so it
    can return to *its own* session as a draft rather than to whatever is on
    screen when the answer lands — but they say different things about the
    chat, and a UI that treated them alike would get one of them wrong:

    * `turn.unqueued` names a row and means "that row is gone"; everything
      else stands, the turn ahead is still running and its spinner with it.
      This one names no row. A `chat.reset` — an open of what the session
      holds now — precedes it and is what settles the screen, on this path and
      on the ordinary one where no text comes back at all (`AgentService`'s
      `turn.interrupt`).
    * A `turn.finished` says the turn is over, on both stop paths and on the
      ordinary end of a turn. This event never means it: it is about a
      message, and a ``seq`` field here would be a number every client had to
      know to ignore.

    The shape is otherwise the queue's on purpose, so parking the text as that
    session's draft is one routine on the UI side rather than two.
    """

    TYPE: ClassVar[str] = "turn.interrupted"
    session_id: str
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
    # Already trimmed by the core to what the user asked a peek to be worth
    # (`config.WatchSettings.peek_chars`): the whole point of peeking is not
    # to ship a gigabyte of progress bars. How much is a setting and not a
    # constant because the two ends of the range are both real — a tail read
    # over a tunnel wants to stay small, and a traceback wants to arrive
    # whole.
    text: str = ""


class MemoryProposals(Event):
    TYPE: ClassVar[str] = "memory.proposals"
    session_id: str
    proposals: list[Proposal] = Field(default_factory=list)


class CompactProposed(Event):
    """A summary `/compact` wrote, before it is anybody's history.

    A fold cannot be undone from the front-end — the summary becomes what the
    model sees — and a summary is exactly the kind of thing that comes back
    wrong: cut off, or missing the one path the next step needs. So the
    user-driven path offers it first and waits (`compact.resolve`), and this
    is the offer.

    ``summary`` is the message content as it would be stored, prefix and all,
    because that is what the user is being asked to accept. ``folded`` is how
    many messages it stands in for and ``guidance`` what they typed after the
    command, both so the screen can say what is being decided. ``attempt``
    counts from 1 and rises with every retry.

    ``truncated`` is the one thing the text cannot say for itself: the backend
    stopped at its length budget, or the cap cut the end off. It is a hint for
    the screen to warn with, not an error — a cut summary is still a summary,
    and whether it is good enough is the user's call.
    """

    TYPE: ClassVar[str] = "compact.proposed"
    session_id: str
    summary: str
    folded: int = 0
    guidance: str = ""
    attempt: int = 1
    truncated: bool = False


class ConfirmRequested(Event):
    """A yes/no question about one conversation that is not a tool approval.

    One case today: triage proposing a log signature it just learned (§5.5
    tier 3). It is deliberately not `decision.requested` — that one parks a
    graph thread and its answer resumes a turn, whereas this one comes from a
    poll and nothing is waiting on it. Conflating them would let an unanswered
    triage offer look like a stalled session.

    ``session_id`` is the conversation whose work raised it — the session the
    failed job was started from — and it is not decoration: this arrives from
    a poll rather than from a keypress, so without it the question reaches a
    user who is somewhere else entirely, about work they cannot see, with
    nothing on screen saying which of their conversations it came out of. It
    is what lets a client hold the question *in* that session instead
    (§3.2 property 1).
    """

    TYPE: ClassVar[str] = "confirm.requested"
    id: str
    session_id: str
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
    # A headline for the body, or empty. Carried because a few of the core's
    # answers are a heading plus a block — the skills a profile can see, a
    # summary a `/compact` just wrote, the xhigh warning whose first clause has
    # to land even if the paragraph under it is skimmed. Those are the toasts
    # `tui/app.py` passed `title=` for, and without the field the heading would
    # have to be glued onto the front of `text`, where a renderer can no longer
    # tell it apart from the body it is meant to introduce.
    #
    # A hint like `timeout`: a front-end with nowhere to put a heading is free
    # to ignore it, and none of these notifies is unreadable without one.
    title: str = ""
    # Seconds to keep it up, or None for the renderer's default. Carried
    # because a few warnings — a matched past struggle, the memory budget, a
    # curation run — are ones the user is meant to actually read, and the core
    # is what knows which those are. How long a second is remains the
    # renderer's business; this is a hint, not a layout instruction.
    timeout: float | None = None
