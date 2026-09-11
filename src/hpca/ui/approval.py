"""The inline decision prompt: a gated call, and the box that refuses it.

Two kinds of gate share this rendering, told apart by ``payload["kind"]``:

* ``destructive`` — the always-on gate for destructive operations.
* ``execution`` — manual mode showing a script or command before it
  runs; the user decides on the actual script text, not on a JSON blob.

The prompt is rendered inline at the foot of the chat column, **not as a
modal**, and that is a design decision rather than an implementation
convenience: a decision waiting in one session must not cover the other
columns or block a session the user switched to. Approvals are per session
while the user may be reading something else, so a decision that arrives in a
background session lights the sidebar's "!" and nothing more — opening that
session is what reveals its prompt (specs/specs-ui-acceptance.md, "Inline approval
prompts"). This is the rationale written down at ``tui/approval_screen.py``
lines 9-15, kept because it is the reason the shape is what it is.

Saying no has a second step, the box asking why (:func:`approval_reason_hint`);
the graph resumes with a verdict either way. Nothing crosses the wire until
that box is answered, so the refusal and the reason reach the model together
and the model is never told "no" twice.

And there is a third key that is not an answer: ``a`` opens a box for a
question to the agent about the call (`protocol.DecisionAsk`). The answer is
drawn under the call, the question stays open, and ``y``/``n``/``a`` work again
— so the user can approve, refuse, or keep asking. None of it reaches the
conversation: it is a side dialog the session log keeps and the model resuming
the turn never sees, which is what lets a user ask "why?" without paying for
the answer in the main context.

The text helpers below are pure functions over the interrupt payload — an
opaque dict as far as the protocol is concerned (`protocol.DecisionRequested`
passes the graph's interrupt value through untouched), so the UI is the side
that has to understand its shape. They were moved here whole from
``hpca/tui/approval_screen.py``; `tui/` is going away and they are the part of
it that has to survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hpca.transcript import call_arguments
from hpca.ui import theme
from hpca.ui.ansi import BOLD, PULSE_PERIOD, RESET, fold, pad, pulse, rule
from hpca.ui.editor import Editor

# The stages of the prompt. "ask" is the y/n question; "reason" keeps the
# call on screen — the reason is written *about* it — and adds the box for why.
# "question" is the box for asking the agent about the call, and unlike
# "reason" it leads back to "ask": a question is not an answer.
ASK, REASON, QUESTION = "ask", "reason", "question"

# What the three blocks are allowed to cost, in rows. The same figures the
# Textual bar's CSS carried (`max-height: 12` on the script, `6` on the box):
# the prompt shares the chat column, and a four-hundred-line diff that pushed
# the conversation off the screen would be answering a question nobody could
# still see the context for.
DETAIL_LINES = 8
SCRIPT_LINES = 12
REASON_LINES = 6
# What the call keeps of its rows once there is a dialog under it and not room
# for both. The dialog is what is being read by then — the call was read before
# the question was asked — but a question about "line 3" still wants line 3 in
# sight, so the call is squeezed rather than dropped.
CALL_LINES_ASKED = 3


# What a clipped block says instead of the lines it dropped. Counted rather
# than a bare ellipsis: "there is more" and "there are four hundred more" are
# different facts, and the second one is the one that changes the answer.
def _more(dropped: int, which: str = "more") -> str:
    return f"… {dropped} {which} line{'' if dropped == 1 else 's'}"


@dataclass
class Decision:
    """One parked approval, as the UI holds it while it is unanswered.

    ``payload`` is the core's (session state, re-emitted on subscribe);
    ``stage`` and ``reason`` are the UI's, and they are why this is an object
    rather than the bare dict `decision.requested` carries. specs-core-process
    §4.4 splits it exactly there: the decision moves to the core, the
    half-typed refusal reason stays in the UI, "a draft, same class as
    ``_drafts``". Living on the `SessionState` is what parks it — switching
    session swaps the state and nothing else, so the words typed about one
    session's script are still there on the way back.
    """

    payload: dict = field(default_factory=dict)
    stage: str = ASK
    reason: Editor = field(default_factory=lambda: Editor(wrap=True))
    # The question being typed to the agent — a draft, like the reason.
    question: Editor = field(default_factory=lambda: Editor(wrap=True))
    # The side dialog so far: `protocol.DialogTurn` dicts, the core's (it
    # restates the whole thread on every change), with the one exception
    # :meth:`asked` makes for the moment between sending and hearing back.
    turns: list[dict] = field(default_factory=list)

    @property
    def asking(self) -> bool:
        return self.stage == ASK

    @property
    def questioning(self) -> bool:
        return self.stage == QUESTION

    @property
    def waiting(self) -> bool:
        """A question is out and its answer is still being written."""
        return bool(self.turns) and self.turns[-1].get("answer") is None

    def decline(self) -> None:
        """Move to the second half: the answer is "no", and the box now
        opening collects what should be different. Nothing is sent yet."""
        self.stage = REASON

    def reason_text(self) -> str:
        return self.reason.text().strip()

    def ask(self) -> None:
        """Open the box for a question to the agent. Nothing is decided."""
        self.stage = QUESTION

    def back(self) -> None:
        """Out of the question box, unsent. What was typed stays in it, so
        ``a`` brings it back — the same courtesy the reason box extends
        across a session switch."""
        self.stage = ASK

    def question_text(self) -> str:
        return self.question.text().strip()

    def asked(self, text: str) -> None:
        """A question has been sent: back to the y/n, with the question drawn
        as waiting for its answer at once.

        Drawn here rather than when the core's `decision.dialog` comes back,
        because the gap between the two is when a second ``a`` would send a
        question over the top of the first. The core's thread replaces this
        one the moment it arrives, so the guess never outlives the fact.
        """
        self.turns = [*self.turns, {"question": text, "answer": None, "failed": False}]
        self.question.clear()
        self.stage = ASK

    def editor(self) -> Editor | None:
        """The box under the prompt, if this stage has one."""
        if self.stage == REASON:
            return self.reason
        if self.stage == QUESTION:
            return self.question
        return None


# ------------------------------------------------------------- what it says


def approval_kind(payload: dict) -> str:
    return payload.get("kind", "destructive")


def approval_title(payload: dict) -> str:
    """The question, naming the tool it is about.

    The destructive heading names it too, which it did not have to while the
    details below opened with a ``Tool: <name>`` line. That line is gone — it
    said nothing the heading could not — so the heading is now the only place
    the name appears, and a destructive tool that cannot describe its own call
    would otherwise put bare arguments on screen with nothing saying what they
    were arguments to.
    """
    if approval_kind(payload) == "execution":
        return f"Run this — {payload.get('tool')}?"
    return f"Destructive operation — {payload.get('tool')}?"


def approval_hint(payload: dict, *, asked: bool = False, waiting: bool = False) -> str:
    """The keys the question takes. ``a`` is left off while an answer is being
    written — the core takes one question at a time, and a key that does
    nothing is not one to offer — and says "again" once there is a dialog."""
    if approval_kind(payload) == "execution":
        keys = "(y) run script · (n) skip script"
    else:
        keys = "(y) approve · (n) deny"
    if waiting:
        return keys
    return keys + (" · (a) ask again" if asked else " · (a) ask the agent")


def approval_question_hint() -> str:
    """The keys the question box takes. Escape goes back rather than
    refusing: nothing has been decided, and the question was not an answer."""
    return "(enter) ask · (esc) back · (shift+enter) new line"


def approval_reason_title(payload: dict) -> str:
    """Replaces the question once it has been answered with "no".

    Asks about the *next* attempt rather than for a justification: what goes
    in the box is read by the model, and "what should be different" is the one
    thing it can act on.
    """
    if approval_kind(payload) == "execution":
        return "Skipped — what should be different about the script?"
    return "Denied — what should be different?"


def approval_reason_hint() -> str:
    """The keys the box takes. The same for either gate — what differs is the
    title above it."""
    return "(enter) send · (esc) no reason · (shift+enter) new line"


def approval_details(payload: dict) -> str:
    """What this particular call does — and nothing that only the agent cares
    about.

    Pointedly *not* shown is ``payload["description"]``: that is the tool's
    schema blurb, written for the model to pick the tool by ("Run a registered
    script as a tracked background process for work that outlives this
    turn..."). It describes the tool in general and never this call, so on a
    prompt asking about one concrete action it is prompt content leaking onto
    the screen. The tool's name is left out for the same reason it is not
    repeated twice — ``approval_title`` already says it in the heading.

    When the tool gave ``details`` that is the whole answer: it is the tool's
    own account of *this* call with the real paths already resolved ("edit
    /home/me/run.sh"), and the arguments it was built from would only say the
    same thing again, less clearly. Without ``details`` the arguments are all
    there is to judge by, so they are rendered instead.

    Returning "" is a real outcome, not a failure: for ``run_bash`` the command
    in the block below is the entire call, and there is nothing left to say
    above it.

    The arguments are rendered by :func:`hpca.transcript.call_arguments`, the
    same function the chat's own call row uses. The prompt and the record of
    what ran have to describe the same call in the same words — the prompt is
    gone the moment it is answered, and the row is where the user goes back to
    read it.
    """
    details = payload.get("details")
    if details:  # resolved real paths (§5.3)
        return str(details).strip()
    return call_arguments(
        payload.get("arguments") or {}, has_script=bool(payload.get("script"))
    )


def approval_script(payload: dict) -> str | None:
    """The script/command preview, if this gate has one to show."""
    return payload.get("script")


# ---------------------------------------------------------- what it looks like


def _wrapped(text: str, width: int, cap: int) -> list[str]:
    """A block folded to the width it has, and cut to the rows it may have."""
    if not text:
        return []
    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines += fold(paragraph, max(8, width)) or [""]
    if len(lines) <= cap:
        return lines
    return lines[: max(1, cap - 1)] + [_more(len(lines) - max(1, cap - 1))]


def _said(style: str, who: str, text: str, width: int) -> list[tuple[str, str]]:
    """One side of one exchange, under a hanging label so a wrapped answer
    still reads as the agent's."""
    label = f"  {who:<5} › "
    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines += fold(paragraph, max(8, width - len(label))) or [""]
    indent = " " * len(label)
    return [
        (style, (label if index == 0 else indent) + line)
        for index, line in enumerate(lines)
    ]


def _exchanges(
    decision: Decision, width: int, now: float | None, period: float
) -> list[list[tuple[str, str]]]:
    """The side dialog, one block of rows per question and its answer.

    Blocks rather than rows because that is the unit it is clipped in: an
    older exchange is dropped whole, and the newest one is never dropped at
    all (:func:`_fit_dialog`).

    The answer still being written is the line that breathes, and while it
    does the key hint does not (see :func:`prompt_rows`): what the user is
    waiting on is the answer, and two lines asking for the eye is neither.
    """
    blocks = []
    for turn in decision.turns:
        rows = _said(theme.user, "you", str(turn.get("question", "")), width)
        answer = turn.get("answer")
        if answer is None:
            style = pulse(now, period) if now is not None else theme.faint
            rows.append((style, f"  {'agent':<5} › answering…"))
        else:
            style = theme.warn if turn.get("failed") else theme.agent
            rows += _said(style, "agent", str(answer), width)
        blocks.append(rows)
    return blocks


def _sections(
    decision: Decision,
    width: int,
    now: float | None,
    *,
    focused: bool,
    period: float,
) -> tuple[list, list, list, list]:
    """The prompt in four parts: the heading, the call, the dialog about it,
    and the key hint. :func:`prompt_rows` is them end to end;
    :func:`render_decision` is them fitted to a height, which is why they are
    kept apart until then."""
    payload = decision.payload
    accent = theme.warn if approval_kind(payload) == "execution" else theme.danger
    head = [(BOLD + theme.chrome if focused else theme.faint, rule("decision", width))]
    title = (
        approval_reason_title(payload)
        if decision.stage == REASON
        else approval_title(payload)
    )
    head.append((accent + BOLD, f"  {title}"))
    call = [
        ("", f"  {line}")
        for line in _wrapped(approval_details(payload), width - 4, DETAIL_LINES)
    ]
    script = approval_script(payload)
    if script:
        # Set off by a gutter rather than a box: what is in here is the thing
        # being decided on, and it has to be told apart from the sentence
        # above it at a glance.
        for line in _wrapped(script, width - 6, SCRIPT_LINES):
            call.append((theme.faint, f"  │ {line}"))
    dialog = _exchanges(decision, width, now, period)
    if decision.stage == REASON:
        hint = approval_reason_hint()
    elif decision.stage == QUESTION:
        hint = approval_question_hint()
    else:
        hint = approval_hint(payload, asked=bool(decision.turns), waiting=decision.waiting)
    breathes = decision.asking and not decision.waiting and now is not None
    return head, call, dialog, [(pulse(now, period) if breathes else theme.faint, f"  {hint}")]


def prompt_rows(
    decision: Decision,
    width: int,
    now: float | None = None,
    *,
    focused: bool = False,
    period: float = PULSE_PERIOD,
) -> list[tuple[str, str]]:
    """The prompt above the box, as ``(style, text)`` rows.

    Styled rather than plain because two different facts are being said at
    once, and this is where they were fighting over one row.

    **The rule says focus.** Teal-and-bold when the keys are aimed here, dim
    when they are not — the same sentence `Pane.render` writes over the
    sessions, the chat and the watchers, and this prompt stands in the message
    box's slot in the focus ring (`app.SESSIONS…DECISION`). It carried the
    *severity* until now, which read well right up until you asked what an
    unfocused decision looks like: the answer was "identical", so the one
    region whose keys silently do nothing was the one region that could not
    say so. Focus is a four-region convention or it is not a convention.

    **The heading says severity.** An execution gate is a script about to run
    and a destructive one is something about to be lost, and those are not the
    same warning — the Textual bar said it with `border: heavy $error` /
    `$warning`, and dropping it to buy the rule back would be trading a real
    signal for a cosmetic one. It moves down one line instead, onto the bold
    line that already carried the same colour: `Run this — run_bash?` in
    yellow, `Destructive operation — delete_file?` in red, directly under the
    rule and in larger type than the rule ever was. Nothing is lost but the
    dashes it was painted on.

    ``now`` is the clock the answer line pulses on, and None is "do not" —
    which is what `decision_height` passes, because how tall this is must not
    depend on what time it is. Only the y/n line breathes: at the reason stage
    the hint is describing a box that already has the cursor in it, and two
    things asking for the eye at once is neither of them getting it. ``period``
    is how long one breath takes (`Display.decision_pulse_seconds`).
    """
    head, call, dialog, tail = _sections(
        decision, width, now, focused=focused, period=period
    )
    return head + call + [row for block in dialog for row in block] + tail


def box_height(decision: Decision, width: int) -> int:
    """Rows the box under the prompt wants — the refusal's reason or the
    question to the agent — after wrapping, capped. None at the y/n."""
    editor = decision.editor()
    if editor is None:
        return 0
    return max(1, min(REASON_LINES, editor.height(max(4, width - 2))))


def decision_height(decision: Decision, width: int, cap: int) -> int:
    """How tall the whole prompt would like to be, within what it may have."""
    wants = len(prompt_rows(decision, width)) + box_height(decision, width)
    return max(3, min(cap, wants))


def _cut(rows: list[tuple[str, str]], room: int) -> list[tuple[str, str]]:
    """The first rows of a block and a count of the rest, in ``room`` rows."""
    if len(rows) <= room:
        return rows
    if room <= 0:
        return []
    return rows[: room - 1] + [(theme.faint, f"  {_more(len(rows) - room + 1)}")]


def _fit_dialog(blocks: list[list], room: int) -> list[tuple[str, str]]:
    """The dialog in ``room`` rows: the newest exchange from its first line,
    then as many before it as fit whole, and a count of what did not.

    Newest first because it is the one being read — the question just asked
    and the answer just given — and from its *first* line because an answer
    read from the middle is not an answer. The older ones were read already.
    """
    if room <= 0 or not blocks:
        return []
    rows = list(blocks[-1])
    if len(rows) >= room:
        return _cut(rows, room)
    index = len(blocks) - 2
    # A row is held back for the count while there is still an older block
    # that might not fit; the oldest one needs none — nothing is left behind it.
    while index >= 0 and len(rows) + len(blocks[index]) + (index > 0) <= room:
        rows = blocks[index] + rows
        index -= 1
    if index >= 0:
        dropped = sum(len(block) for block in blocks[: index + 1])
        rows = [(theme.faint, f"  {_more(dropped, 'earlier')}")] + rows
    return rows


def render_decision(
    decision: Decision,
    width: int,
    height: int,
    *,
    focused: bool,
    now: float | None = None,
    period: float = PULSE_PERIOD,
) -> list[str]:
    """The prompt, in exactly ``height`` rows of exactly ``width`` cells.

    When the room is short the middle goes first: the rule, the heading, the
    key hint and the box are what the prompt *is*, and the details and the
    script are what it is about — clipping those says "there is more here",
    while clipping the hint would leave a question with no visible way to
    answer it.

    With a dialog in the middle too, the call gives way to it first — down to
    ``CALL_LINES_ASKED`` — and then the dialog to the call: the answer is what
    the user is reading, and the call is what they already read before they
    asked about it (:func:`_fit_dialog` for how the dialog itself gives way).
    """
    head, call, dialog, tail = _sections(
        decision, width, now, focused=focused, period=period
    )
    box = min(box_height(decision, width), max(0, height - 3))
    room = max(0, height - box)
    middle = max(0, room - len(head) - len(tail))
    talked = sum(len(block) for block in dialog)
    if len(call) + talked > middle:
        if dialog:
            keep = min(len(call), max(CALL_LINES_ASKED, middle - talked), middle)
        else:
            keep = middle
        call = _cut(call, keep)
        rows = head + call + _fit_dialog(dialog, middle - len(call)) + tail
    else:
        rows = head + call + [row for block in dialog for row in block] + tail
    out = [f"{style}{pad(text, width)}{RESET}" for style, text in rows[:room]]
    while len(out) < room:
        out.append(" " * width)
    editor = decision.editor()
    if box and editor is not None:
        body = editor.render(max(4, width - 2), box, focused=focused)
        mark = theme.warn if decision.stage == REASON else theme.chrome
        for index, line in enumerate(body):
            marker = "› " if index == 0 else "  "
            out.append((mark if focused else theme.faint) + marker + RESET + line)
    return out[:height]
