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
session is what reveals its prompt (specs-ui-acceptance.md, "Inline approval
prompts"). This is the rationale written down at ``tui/approval_screen.py``
lines 9-15, kept because it is the reason the shape is what it is.

Saying no has a second step, the box asking why (:func:`approval_reason_hint`);
the graph resumes with a verdict either way. Nothing crosses the wire until
that box is answered, so the refusal and the reason reach the model together
and the model is never told "no" twice.

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
from hpca.ui.ansi import BOLD, DIM, RED, RESET, YELLOW, fold, pad, pulse, rule
from hpca.ui.editor import Editor

# The two halves of the prompt. "ask" is the y/n question; "reason" keeps the
# call on screen — the reason is written *about* it — and adds the box for why.
ASK, REASON = "ask", "reason"

# What the three blocks are allowed to cost, in rows. The same figures the
# Textual bar's CSS carried (`max-height: 12` on the script, `6` on the box):
# the prompt shares the chat column, and a four-hundred-line diff that pushed
# the conversation off the screen would be answering a question nobody could
# still see the context for.
DETAIL_LINES = 8
SCRIPT_LINES = 12
REASON_LINES = 6


# What a clipped block says instead of the lines it dropped. Counted rather
# than a bare ellipsis: "there is more" and "there are four hundred more" are
# different facts, and the second one is the one that changes the answer.
def _more(dropped: int) -> str:
    return f"… {dropped} more line{'' if dropped == 1 else 's'}"


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

    @property
    def asking(self) -> bool:
        return self.stage == ASK

    def decline(self) -> None:
        """Move to the second half: the answer is "no", and the box now
        opening collects what should be different. Nothing is sent yet."""
        self.stage = REASON

    def reason_text(self) -> str:
        return self.reason.text().strip()


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


def approval_hint(payload: dict) -> str:
    if approval_kind(payload) == "execution":
        return "(y) run script · (n) skip script"
    return "(y) approve · (n) deny"


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


def prompt_rows(
    decision: Decision, width: int, now: float | None = None
) -> list[tuple[str, str]]:
    """The prompt above the box, as ``(style, text)`` rows.

    Styled rather than plain because the border carries a fact: an execution
    gate is a script about to run and a destructive one is something about to
    be lost, and those are not the same warning. The Textual bar said it with
    `border: heavy $error` / `$warning`; this says it with the rule and the
    heading, which is the same two colours in the space a row UI has.

    ``now`` is the clock the answer line pulses on, and None is "do not" —
    which is what `decision_height` passes, because how tall this is must not
    depend on what time it is. Only the y/n line breathes: at the reason stage
    the hint is describing a box that already has the cursor in it, and two
    things asking for the eye at once is neither of them getting it.
    """
    payload = decision.payload
    accent = YELLOW if approval_kind(payload) == "execution" else RED
    rows = [(accent, rule("decision", width))]
    title = (
        approval_title(payload)
        if decision.asking
        else approval_reason_title(payload)
    )
    rows.append((accent + BOLD, f"  {title}"))
    for line in _wrapped(approval_details(payload), width - 4, DETAIL_LINES):
        rows.append(("", f"  {line}"))
    script = approval_script(payload)
    if script:
        # Set off by a gutter rather than a box: what is in here is the thing
        # being decided on, and it has to be told apart from the sentence
        # above it at a glance.
        for line in _wrapped(script, width - 6, SCRIPT_LINES):
            rows.append((DIM, f"  │ {line}"))
    asking = decision.asking
    hint = approval_hint(payload) if asking else approval_reason_hint()
    style = pulse(now) if asking and now is not None else DIM
    rows.append((style, f"  {hint}"))
    return rows


def reason_height(decision: Decision, width: int) -> int:
    """Rows the refusal box wants, after wrapping, capped."""
    if decision.asking:
        return 0
    return max(1, min(REASON_LINES, decision.reason.height(max(4, width - 2))))


def decision_height(decision: Decision, width: int, cap: int) -> int:
    """How tall the whole prompt would like to be, within what it may have."""
    wants = len(prompt_rows(decision, width)) + reason_height(decision, width)
    return max(3, min(cap, wants))


def render_decision(
    decision: Decision,
    width: int,
    height: int,
    *,
    focused: bool,
    now: float | None = None,
) -> list[str]:
    """The prompt, in exactly ``height`` rows of exactly ``width`` cells.

    When the room is short the middle goes first: the rule, the heading, the
    key hint and the box are what the prompt *is*, and the details and the
    script are what it is about — clipping those says "there is more here",
    while clipping the hint would leave a question with no visible way to
    answer it.
    """
    rows = prompt_rows(decision, width, now)
    box = min(reason_height(decision, width), max(0, height - 3))
    room = max(0, height - box)
    if len(rows) > room:
        keep_head, keep_tail = 2, 1
        middle = rows[keep_head : len(rows) - keep_tail]
        allowed = max(0, room - keep_head - keep_tail)
        if allowed and middle:
            middle = middle[: max(0, allowed - 1)] + [
                (DIM, f"  {_more(len(middle) - max(0, allowed - 1))}")
            ]
        else:
            middle = []
        rows = rows[:keep_head] + middle + rows[len(rows) - keep_tail :]
    out = [f"{style}{pad(text, width)}{RESET}" for style, text in rows[:room]]
    while len(out) < room:
        out.append(" " * width)
    if box:
        body = decision.reason.render(max(4, width - 2), box, focused=focused)
        for index, line in enumerate(body):
            marker = "› " if index == 0 else "  "
            out.append((YELLOW if focused else DIM) + marker + RESET + line)
    return out[:height]
