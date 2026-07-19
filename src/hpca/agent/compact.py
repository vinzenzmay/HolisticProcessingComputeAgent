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
"""

from __future__ import annotations

from hpca.llm import Message

# Compact when the estimated prompt exceeds this share of the window. Leaves
# room for the system prompt, the reply, and the next few tool results.
COMPACT_AT = 0.7
# Keep at least this many recent messages verbatim, whatever the budget says.
KEEP_RECENT = 12
CHARS_PER_TOKEN = 4  # crude but model-independent; see profiles.estimate_tokens
SUMMARY_PREFIX = "[earlier in this session]"
MAX_SUMMARY_CHARS = 1500

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
    """(to summarize, to keep verbatim). Keeps at least KEEP_RECENT."""
    keep = max(KEEP_RECENT, len(messages) // 3)
    if len(messages) <= keep:
        return [], list(messages)
    return list(messages[:-keep]), list(messages[-keep:])


def is_summary(message: Message) -> bool:
    return str(message.get("content", "")).startswith(SUMMARY_PREFIX)


def transcript(messages: list[Message]) -> str:
    lines = []
    for message in messages:
        if message["role"] == "system":
            continue
        content = " ".join(str(message["content"]).split())[:400]
        lines.append(f"{message['role']}: {content}")
    return "\n".join(lines)


async def summarize(llm, messages: list[Message]) -> Message:
    """One summary message standing in for ``messages``.

    Rides the user role like every other machine-generated message in this
    conversation (tool results included) — a small model follows the shape it
    has already seen far more reliably than a new one.
    """
    response = await llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript(messages)},
        ],
        max_tokens=512,
    )
    summary = (response.content or "").strip()[:MAX_SUMMARY_CHARS]
    if not summary:
        raise ValueError("empty summary")
    return {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary}"}
