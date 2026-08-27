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
# hints is what a level *does to a turn*: the levels are indistinguishable by
# name, the ordering least-to-most work says nothing about which is quickest,
# and the one thing a user can act on is how each behaves in practice.
#
# These are field notes, not a benchmark. Measured cost is in project.md §3.6;
# what is recorded here is what each level makes turns feel like on the
# cluster's Qwen3.8 — including that "more thinking" is not monotonically
# slower, because low spends its deliberation re-deciding steps it has already
# taken and medium injects no preamble at all.
EFFORT_HINTS: dict[str, str] = {
    "off": "no thinking channel — good for simple tasks",
    "low": "spends much effort on correction steps",
    "medium": "the model's own default amount of thinking — fastest",
    "xhigh": "thinks hardest, and overthinks everything",
}


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
