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
see :func:`hpca.agent.graph.compact_now`) and say what it has to carry — the
``guidance`` argument below. That instruction does two jobs: it steers the
summarizer, and it stays in the folded view afterwards, so a next step
declared while compacting still frames the turns that follow it.
"""

from __future__ import annotations

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
MAX_SUMMARY_CHARS = 1500
# A guided summary has to name the things it was told to keep, so it gets more
# room than the automatic one — still bounded, or the fold frees nothing.
MAX_GUIDED_SUMMARY_CHARS = 3000
MAX_GUIDANCE_CHARS = 1000

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


async def summarize(
    llm, messages: list[Message], *, guidance: str | None = None
) -> Message:
    """One summary message standing in for ``messages``.

    Rides the user role like every other machine-generated message in this
    conversation (tool results included) — a small model follows the shape it
    has already seen far more reliably than a new one.

    ``guidance`` is the user's own instruction for this compaction (the text
    after ``/compact``): what to preserve, or what they are about to do next.
    It steers the summarizer *and* is appended to the resulting message, so the
    stated intent keeps framing the session once the history behind it is gone.
    """
    focus = (guidance or "").strip()[:MAX_GUIDANCE_CHARS]
    system = SYSTEM_PROMPT
    if focus:
        system = f"{SYSTEM_PROMPT}\n\n{GUIDANCE_PROMPT.format(guidance=focus)}"
    response = await llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": transcript(messages)},
        ],
        max_tokens=1024 if focus else 512,
    )
    cap = MAX_GUIDED_SUMMARY_CHARS if focus else MAX_SUMMARY_CHARS
    summary = (response.content or "").strip()[:cap]
    if not summary:
        raise ValueError("empty summary")
    content = f"{SUMMARY_PREFIX}\n{summary}"
    if focus:
        content = f"{content}\n\n{FOCUS_PREFIX}\n{focus}"
    return {"role": "user", "content": content}
