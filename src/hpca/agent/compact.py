"""Context compaction (redesign Phase 6).

The orchestrator sends the whole message history every round, so a long
session eventually overflows the model window — on a 27B with a 32k context
that is not a distant edge case. Compaction folds the older part of the
history into one summary message once the estimated prompt approaches the
backend's limit.

Two properties matter more than summary quality here:

* **The tail is never touched.** Recent turns stay verbatim, because that is
  where the current task lives. Only the older half is folded.
* **Nothing is silently lost.** Before the old messages are dropped, the
  self-review loop gets a look at them (the caller's ``on_evict`` hook), so
  a durable learning can be proposed while the evidence still exists. This is
  Hermes' ``on_pre_compress`` idea: the moment before context is discarded is
  the last chance to extract anything from it.

The user can also ask for a fold before the window forces one (``/compact``,
see :func:`hpca.agent.graph.propose_compaction`) and say what it has to carry —
the ``guidance`` argument below. That instruction does two jobs: it steers the
summarizer, and it stays in the folded view afterwards, so a next step
declared while compacting still frames the turns that follow it.

A summary the user asked for is also one they get to *refuse*. Nothing here
writes anything; :func:`summarize` produces a candidate, and the user-driven
path holds it up for review before it lands (``compact.proposed`` /
``compact.resolve``, `core.service._compact`). A refusal comes back as a
sentence about what the summary got wrong, and that sentence steers the next
attempt — ``comment`` below. The summary a fold cannot be taken back from is
worth one round trip.
"""

from __future__ import annotations

from dataclasses import dataclass

from hpca.agent.history import call_text, is_tool_call_message
from hpca.llm import Message

# Compact when the estimated prompt exceeds this share of the window. Leaves
# room for the system prompt, the reply, and the next few tool results.
COMPACT_AT = 0.7
# Keep at least this many recent messages verbatim, whatever the budget says.
# Counted in messages, and one tool round is two of them since §4.3 (the call
# as an assistant turn, then its result), so this is roughly twelve rounds.
KEEP_RECENT = 24
CHARS_PER_TOKEN = 4  # crude but model-independent; see profiles.estimate_tokens
SUMMARY_PREFIX = "[earlier in this session]"
# Marks the user's own instruction inside the summary message, so it survives
# the fold and keeps steering the turns that come after it.
FOCUS_PREFIX = "[the user asked for this summary, with these instructions]"
# How long a summary may be. Two of them, because a guided summary has to name
# the things it was told to keep and the automatic one only has to be usable —
# both bounded, or the fold frees nothing.
#
# These are also where the *generation* budget comes from (`max_tokens` below),
# and that is the point of the numbers: while the cap was 1500 characters and
# the budget 512 tokens, the model was invited to write twice what the cap
# would keep, so any summary over ~200 words was sliced mid-word by `[:cap]`
# with nothing on screen saying so. A cut summary is exactly what a user
# rejects, so the two bounds now agree — the model runs out of room roughly
# where the cap would have cut it — and whatever cutting is left is reported
# rather than silent (`Summary.truncated`).
MAX_SUMMARY_CHARS = 2400
MAX_GUIDED_SUMMARY_CHARS = 6000
MAX_GUIDANCE_CHARS = 1000
# Room on top of the cap, so an ordinary summary finishes its last sentence
# inside the budget rather than against it.
TOKEN_SLACK = 64
# What a clipped summary ends with: the cut is the one thing about it the
# reader cannot see for themselves.
ELLIPSIS = " …"

SYSTEM_PROMPT = (
    "You compress the earlier part of a working session between a scientist "
    "and an HPC assistant, so the assistant can keep working without the "
    "full history. Write a factual summary, at most 200 words, covering: "
    "what the user is trying to achieve, what has been established (paths, "
    "tools, parameters, findings), what has been tried and failed, and what "
    "remains open. Keep concrete identifiers - paths, job ids, tool names, "
    "error messages - verbatim; they are what makes the summary usable. "
    "Do not add advice or next steps."
)

# Appended when the user drove the compaction themselves and said what it is
# for. Their instruction is either "keep X" or "here is what I do next" — both
# mean the same thing to the summarizer: that material is load-bearing.
GUIDANCE_PROMPT = (
    "The user asked for this compaction and gave these instructions for it:\n\n"
    "{guidance}\n\n"
    "Follow them. They outrank the 200 word limit: whatever they name - and "
    "whatever a next step they describe will need - goes into the summary in "
    "full, concrete detail, verbatim for paths, commands, parameters, ids and "
    "error text. Compress everything else harder to make room. Do not invent "
    "anything the session did not establish: if the instructions ask for "
    "something that was never covered, say so in one line."
)

# Appended when the user turned a summary down and said why. The rejected text
# comes along: "again, but keep the sbatch flags" is an edit of something, and
# a summarizer that cannot see what it wrote rewrites from scratch and loses
# whatever the user did *not* complain about.
REVISION_PROMPT = (
    "You already wrote a summary of this session and the user turned it "
    "down. Here it is:\n\n{previous}\n\nWhat the user said about it:\n\n"
    "{comment}\n\nWrite the summary again, from the session below, doing "
    "what they asked. Keep what the earlier attempt got right - they objected "
    "to one thing, not to all of it - and change what they named. If they say "
    "it was cut off or incomplete, the fix is to finish it, not to start it "
    "differently."
)


def estimate_tokens(messages: list[Message]) -> int:
    """Rough size of what actually goes on the wire.

    Counts ``api_content`` where present: that sidecar (recalled memory) is
    what the backend receives, so measuring ``content`` would under-count
    exactly the messages carrying extra payload.
    """
    return (
        sum(
            len(str(m.get("api_content") or m.get("content", "")))
            for m in messages
        )
        // CHARS_PER_TOKEN
    )


def should_compact(messages: list[Message], *, max_model_len: int | None) -> bool:
    if not max_model_len or len(messages) <= KEEP_RECENT:
        return False
    return estimate_tokens(messages) > max_model_len * COMPACT_AT


def split(messages: list[Message]) -> tuple[list[Message], list[Message]]:
    """(to summarize, to keep verbatim). Keeps at least KEEP_RECENT.

    A call and its result are one unit (:func:`hpca.agent.history.tool_exchange`).
    A cut landing between them would open the verbatim tail on a result whose
    call was folded away — the orphaned-result shape §4.3 exists to remove — so
    the boundary steps back to take the call along. The tail only ever grows.
    """
    keep = max(KEEP_RECENT, len(messages) // 3)
    if len(messages) <= keep:
        return [], list(messages)
    cut = len(messages) - keep
    if is_tool_call_message(messages[cut - 1]):
        cut -= 1
    return list(messages[:cut]), list(messages[cut:])


def is_summary(message: Message) -> bool:
    return str(message.get("content", "")).startswith(SUMMARY_PREFIX)


# One ordinary message contributes at most this much to a summarize prompt —
# enough to carry what it was about without any single tool result dominating.
MAX_MESSAGE_CHARS = 400


def transcript(messages: list[Message]) -> str:
    lines = []
    for message in messages:
        if message["role"] == "system":
            continue
        # A native-protocol call carries nothing in its content — the call is
        # in tool_calls — so summarizing the content alone would hand the
        # summarizer a blank assistant line where an action was. What the
        # session did is exactly what a summary must not lose.
        content = " ".join((call_text(message) or str(message["content"])).split())
        # An earlier summary already *is* the compressed form of a long stretch
        # of session. Cutting it at the per-message limit would throw away most
        # of what that fold decided to keep, so the second compaction of a
        # session would quietly lose its beginning.
        limit = MAX_GUIDED_SUMMARY_CHARS if is_summary(message) else MAX_MESSAGE_CHARS
        lines.append(f"{message['role']}: {content[:limit]}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Summary:
    """A summary and whether anything was cut off the end of it.

    ``truncated`` is the one thing a reader cannot tell from the text: a
    summary that stops mid-sentence looks the same as one the model chose to
    end there. It is set when the backend stopped at ``max_tokens`` or when
    :func:`clip` had to cut, and it is what the review screen warns with —
    "this is cut, ask for it again" is a decision only the user can make.
    """

    message: Message
    truncated: bool = False

    @property
    def content(self) -> str:
        return str(self.message.get("content", ""))


def budget(cap: int) -> int:
    """The generation budget for a summary capped at ``cap`` characters.

    Derived rather than chosen, because the two disagreeing is what cut
    summaries in half: a budget larger than the cap invites text the cap then
    slices, and one smaller makes the cap decorative.
    """
    return cap // CHARS_PER_TOKEN + TOKEN_SLACK


def clip(text: str, cap: int) -> tuple[str, bool]:
    """``text`` within ``cap`` characters, cut where a reader would cut it.

    ``[:cap]`` lands mid-word, and a summary ending "the job failed with a seg"
    reads as a broken tool rather than as a long summary. So the cut steps back
    to the last sentence end, or failing that the last space, and says it
    happened with an ellipsis. Returns the text and whether anything was lost.
    """
    if len(text) <= cap:
        return text, False
    head = text[:cap]
    end = max(head.rfind(x) for x in (".", "!", "?", "\n"))
    if end < cap // 2:
        end = head.rfind(" ")
    if end < cap // 2:
        return head.rstrip() + ELLIPSIS, True
    return head[: end + 1].rstrip() + ELLIPSIS, True


def summary_body(message: Message) -> str:
    """What the model actually wrote, without the two markers around it.

    The stored message carries the prefix that makes it recognisable
    (:func:`is_summary`) and, when there was one, the user's instruction under
    `FOCUS_PREFIX`. Neither is the summarizer's own text, so neither is handed
    back to it as "your previous attempt" — the instruction reaches the retry
    as an instruction, in its own place in the prompt.
    """
    content = str(message.get("content", ""))
    if content.startswith(SUMMARY_PREFIX):
        content = content[len(SUMMARY_PREFIX) :]
    return content.split(f"\n\n{FOCUS_PREFIX}")[0].strip()


async def summarize(
    llm,
    messages: list[Message],
    *,
    guidance: str | None = None,
    previous: str | None = None,
    comment: str | None = None,
) -> Summary:
    """One summary message standing in for ``messages``.

    Rides the user role like every other machine-generated message in this
    conversation (tool results included) — a small model follows the shape it
    has already seen far more reliably than a new one.

    ``guidance`` is the user's own instruction for this compaction (the text
    after ``/compact``): what to preserve, or what they are about to do next.
    It steers the summarizer *and* is appended to the resulting message, so the
    stated intent keeps framing the session once the history behind it is gone.

    ``previous`` and ``comment`` are the retry (``compact.resolve`` with
    ``retry``): the summary the user turned down and what they said about it.
    They steer this attempt and nothing else — the comment is about *this*
    summary ("you cut it off", "keep the sbatch flags") rather than about the
    session, so it is not appended to the message the way ``guidance`` is, and
    "make it shorter" does not come back as a standing instruction to the turns
    that follow.

    Never thinks. The titler learned this first: reasoning is billed against
    the same ``max_tokens`` as the answer, so on a reasoning backend a budget
    sized for the summary is spent working out what to write and the summary
    itself arrives cut off — or empty.
    """
    focus = (guidance or "").strip()[:MAX_GUIDANCE_CHARS]
    note = (comment or "").strip()[:MAX_GUIDANCE_CHARS]
    system = SYSTEM_PROMPT
    if focus:
        system = f"{SYSTEM_PROMPT}\n\n{GUIDANCE_PROMPT.format(guidance=focus)}"
    if note:
        system = "{}\n\n{}".format(
            system,
            REVISION_PROMPT.format(
                previous=(previous or "").strip() or "(nothing usable)",
                comment=note,
            ),
        )
    # A guided summary is the roomier one, and a rejected summary is guided by
    # definition: the user has just said what it has to do differently, and the
    # commonest thing they say is that it stopped too early.
    cap = MAX_GUIDED_SUMMARY_CHARS if (focus or note) else MAX_SUMMARY_CHARS
    response = await llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": transcript(messages)},
        ],
        max_tokens=budget(cap),
        enable_thinking=False,
    )
    summary, cut = clip((response.content or "").strip(), cap)
    if not summary:
        raise ValueError("empty summary")
    content = f"{SUMMARY_PREFIX}\n{summary}"
    if focus:
        content = f"{content}\n\n{FOCUS_PREFIX}\n{focus}"
    return Summary(
        {"role": "user", "content": content},
        truncated=cut or response.finish_reason == "length",
    )
