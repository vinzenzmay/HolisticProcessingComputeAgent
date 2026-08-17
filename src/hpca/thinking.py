"""Thinking effort: the per-session dial on how hard the model reasons.

A reasoning backend can be asked *how much* to think, not just whether to.
vLLM exposes that as OpenAI's ``reasoning_effort``, and Qwen3.8 accepts
exactly three levels — measured against the cluster's
``Qwen3.8-27B-FP8`` (vLLM, ``--reasoning-parser qwen3``): sending anything
else comes back a 400 saying *"Supported types are xhigh (default), medium,
and low"*. There is no ``high``, however much the name suggests one. With
thinking on and no level given the server picks ``xhigh``, which is why "on"
is never a safe default here — the unqualified switch this replaces meant the
slowest setting the model has.

``off`` is the fourth choice HPCA offers, and the only one that is not a value
of that enum: it turns thinking off outright (``enable_thinking: False``) and
sends no ``reasoning_effort`` at all, so a backend that has never heard of the
parameter is left exactly as it was.

**Why this is per session and not per request.** Mechanically the level is a
prompt injection at position 0, prepended *before* the app's own system
prompt — xhigh contributes "Reasoning effort is set to xhigh. Please think
carefully through the task, validate key assumptions, …", low contributes a
two-line "keep it brief", and medium injects nothing at all. It shows up as a
per-level prompt size for one otherwise identical agent decision: 2621 tokens
at medium, 2647 at low, 2659 at xhigh (2623 with thinking off).

Two levels therefore differ from the very first token of the prompt, and what
they share is only the chat template's opening — measured on a ~100-token
prompt, xhigh and low have **9 tokens in common** before diverging, and that
number does not grow with the conversation, because the divergence is at the
front. Changing the level mid-conversation therefore throws away the prefix KV
cache for the whole session — on a prefix-caching server
(``--enable-prefix-caching``) that is the difference between a cached prefill
and a full one, every round, for the rest of the turn. So it is a stable
choice a session is set to, like its mode (§3.5) and its backend, not
something a caller varies per call.

``thinking_budget``, which the Qwen chat template also accepts, is silently a
no-op on this server; the level is the only working control.
"""

from __future__ import annotations

from typing import Literal

ThinkingEffort = Literal["off", "low", "medium", "xhigh"]

# Ordered least to most work, which is the order the chooser lists them in.
# "off" leads because it is the shipped default.
EFFORTS: tuple[ThinkingEffort, ...] = ("off", "low", "medium", "xhigh")

# One line each, for the chooser and the settings screen. The point of the
# hints is the *cost*: the levels are indistinguishable by name, and the only
# thing a user can act on is how long each makes a turn take.
EFFORT_HINTS: dict[str, str] = {
    "off": "no thinking channel — fastest, and what every turn did before",
    "low": "a short deliberation before answering",
    "medium": "the model's own default amount of thinking",
    "xhigh": "loses any turn that writes a file on Qwen3.8 — details on selecting",
}

# The flag beside xhigh's *name* in the chooser, where the hint above is its
# explanation. It lives here rather than in the screen so the list and the
# confirmation toast cannot drift apart, and it says "unusable" rather than
# "slow" because that is what was measured: not a caution about patience, a
# level that does not currently work.
XHIGH_INLINE_WARNING = "⚠ NOT USABLE"

# Shown when xhigh is picked. This is a "does not work" notice, not a caution,
# and it says so first because a user who reads only the opening clause should
# still come away with the right decision.
#
# Measured on Qwen3.8-27B-FP8, a 200-line create_file — a write that costs
# ~3400-3700 completion tokens with thinking off, i.e. ~85% of the base cap
# before a single thinking token. At xhigh the turn was lost at a 4096 cap
# (473s) and lost again at 8192 (947s): doubling the budget bought a failure
# that cost twice as much, because thinking at this level expands into
# whatever room it is given rather than being sized by the task. There is no
# way to reserve the payload's room from it — ``thinking_budget`` is a no-op
# on this server (see the module docstring).
#
# Deliberately not overstated: a short decision *does* complete at xhigh
# (measured: a diagnostic question answered with a tool call in 1267 tokens).
# What cannot be relied on is any turn that writes a file, which is most of
# what this agent does, so the level is not usable as a session setting.
XHIGH_WARNING = (
    "xhigh does not currently work on Qwen3.8 — it is offered because the "
    "model advertises it, not because it is usable. Any turn that writes a "
    "file is likely to be LOST: thinking is spent from the same token budget "
    "as the answer, and at this level it expands to fill whatever budget it "
    "is given. Measured on a 200-line file write, the turn was lost after 8 "
    "minutes at a 4096-token cap and lost again after 16 minutes at 8192. "
    "Short question-answering turns do complete, but every round thinks for "
    "minutes. Use low or medium; thinking is not generally better anyway, and "
    "small models often route tools worse with it on."
)


def normalize_effort(effort: str | None) -> ThinkingEffort:
    """The stored value as one of the four levels; anything unknown is ``off``.

    Sessions store the level as a plain string (an empty one means "use the
    configured default"), and a settings file is hand-edited, so the level
    reaching the wire has to be validated somewhere. Here rather than at each
    reader: an unrecognised level must never become a 400 on a real turn.
    """
    if effort in EFFORTS:
        return effort  # type: ignore[return-value]
    return "off"


def wire_thinking(effort: str | None) -> tuple[bool, str | None]:
    """``(enable_thinking, reasoning_effort)`` for one level.

    ``off`` is the asymmetric one: thinking is disabled *and* no level is sent,
    so the request looks exactly like a pre-effort one and a backend without
    the parameter is unaffected. The other three send both, because thinking
    has to be on for a level to mean anything.
    """
    level = normalize_effort(effort)
    if level == "off":
        return False, None
    return True, level
