"""Curated-memory edit operations (redesign Phase 3).

The model proposes memory edits as a batch; the user approves them. This
module is the pure, testable half: resolving substring addresses, applying a
batch atomically against a copy, and the two guards that keep a small model
from corrupting the store.

Two ideas are taken from Hermes' memory tool:

* **Substring addressing.** ``replace``/``remove`` name their target by a
  short unique substring rather than an ID, because an ID is something a 27B
  will happily invent. An ambiguous substring is an error listing candidates,
  never a guess.
* **Final-state budgeting.** The character budget is checked once, against
  the result of the whole batch — so one call can free room and use it.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date

from hpca.profiles import Memory, MemoryScope, Profile

# Text that must never enter a system prompt unreviewed. Deliberately short:
# every write is human-approved anyway, so this exists to make an injection
# attempt *visible* in the approval dialog, not to be a security boundary.
THREAT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (all |any )?(previous|prior|above) instructions",
        r"disregard (all |any )?(previous|prior|above)",
        r"you are now\b",
        r"^\s*(system|assistant)\s*:",
        r"<\|im_(start|end)\|>",
        r"</?(system|memory-context)>",
    )
]


# How much of an entry is offered back as a *copyable address* when an
# operation has to be corrected. The old answer here was 60 characters with an
# ellipsis on the end, and that ellipsis was a bug: the tool told the model to
# correct its `match` from the listing, and the listing was not something that
# could be matched. Models copied the elided string — U+2026 and all — or
# spliced two entries across the "; " that joined them, and every such attempt
# missed. So an address is never elided: it is a prefix long enough to be
# unique among the scope's entries (they share openings — "IGV-like Godot
# alignment viewer: …" twice over) and short enough to be reissued verbatim.
ADDRESS_MIN = 48
ADDRESS_MAX = 160

# How many entries a refusal lists, nearest-first, so the one that was meant
# is at the top even when the scope holds dozens.
ADDRESS_LISTED = 8


class MemoryOpError(Exception):
    """An operation that cannot be applied as written."""


@dataclass
class MemoryOp:
    op: str  # add | replace | remove | demote
    scope: MemoryScope = MemoryScope.SYSTEM_PROMPT
    match: str = ""  # replace/remove/demote: a unique substring of the target
    text: str = ""  # add/replace: the new text

    def describe(self) -> str:
        if self.op == "add":
            return f"add to {self.scope.value}: {self.text}"
        if self.op == "remove":
            return f"remove from {self.scope.value}: “{self.match}”"
        if self.op == "demote":
            return f"move “{self.match}” from system-prompt to rag"
        return f"replace “{self.match}” ({self.scope.value}) with: {self.text}"


@dataclass
class BatchResult:
    profile: Profile  # a copy with the batch applied
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # no-ops, e.g. exact dupes
    flagged: list[str] = field(default_factory=list)  # threat-pattern hits


def scan_threats(text: str) -> list[str]:
    """Threat patterns present in a proposed memory, for the approval dialog."""
    return [pattern.pattern for pattern in THREAT_PATTERNS if pattern.search(text)]


def resolve(profile: Profile, scope: MemoryScope, match: str) -> Memory:
    """The one memory in ``scope`` containing ``match``.

    Raises with usable addresses when the substring is ambiguous or absent —
    guessing on the model's behalf is how the wrong memory gets deleted, and
    an unusable listing is how a model ends up bisecting its way to shorter
    and shorter substrings that can never hit.
    """
    if not match.strip():
        raise MemoryOpError("Give a short substring of the memory to address.")
    needle = match.strip().lower()
    candidates = [
        memory
        for memory in profile.memories
        if memory.scope == scope and needle in memory.text.lower()
    ]
    if not candidates:
        raise MemoryOpError(
            f"No {scope.value} memory contains “{match}”. "
            f"{address_help(profile, scope, match)}"
        )
    if len(candidates) > 1:
        raise MemoryOpError(
            f"“{match}” matches {len(candidates)} {scope.value} memories and "
            "must address exactly one — use a longer, unique substring by "
            f"copying one of these:\n{_address_lines(profile, candidates)}"
        )
    return candidates[0]


def _prefixes(text: str) -> list[str]:
    """Word-boundary prefixes of ``text``, shortest usable first, then all of
    it — so a scope whose entries share a long opening still gets an address
    that separates them."""
    ends = [
        m.end()
        for m in re.finditer(r"\S+", text)
        if ADDRESS_MIN <= m.end() <= ADDRESS_MAX
    ]
    return [text[:end] for end in ends] + [text]


def address_for(profile: Profile, memory: Memory) -> str:
    """A substring of ``memory`` that :func:`resolve` maps back to it alone.

    Built from the entry's first line rather than a whitespace-collapsed copy,
    because `resolve` matches against the raw text: a collapsed prefix of a
    multi-line memory would be quoted back as an address and then not be found.
    """
    others = [
        other
        for other in profile.memories
        if other.scope == memory.scope and other is not memory
    ]
    head = memory.text.split("\n", 1)[0].strip()
    for candidate in _prefixes(head):
        low = candidate.lower()
        if low and not any(low in other.text.lower() for other in others):
            return candidate
    # Entries identical for their whole first line: only the rest of the text
    # can separate them, so hand back all of it rather than a prefix that is
    # certain to come back ambiguous.
    return memory.text.strip()


def _address_lines(profile: Profile, memories: list[Memory]) -> str:
    """One address per line, indented and unquoted.

    Both of those are deliberate. Indentation is stripped by `resolve`, so a
    model that copies the whole line still hits; a quote character or a bullet
    would not be, and the failure mode being fixed here is exactly a model
    copying a decoration back into `match`.
    """
    return "\n".join(f"    {address_for(profile, memory)}" for memory in memories)


def _nearest_first(memories: list[Memory], match: str) -> list[Memory]:
    """The scope's entries, closest to the failed address first."""
    needle = match.strip().lower()
    if not needle:
        return memories
    return sorted(
        memories,
        key=lambda memory: difflib.SequenceMatcher(
            None, needle, memory.text[: ADDRESS_MAX * 2].lower()
        ).ratio(),
        reverse=True,
    )


def address_help(profile: Profile, scope: MemoryScope, match: str = "") -> str:
    """What a scope's entries are, in a form that can be reissued as ``match``."""
    entries = [memory for memory in profile.memories if memory.scope == scope]
    if not entries:
        return f"There are no {scope.value} memories to address."
    ordered = _nearest_first(entries, match)
    shown, rest = ordered[:ADDRESS_LISTED], len(entries) - ADDRESS_LISTED
    lines = [
        f"Address a {scope.value} memory by copying ONE of the indented lines "
        "below into `match`, exactly as written. Each line is the *opening* "
        "of one entry, closest first, and is enough to address it. Leading "
        "spaces are ignored; nothing else is — a shortened, retyped or "
        "truncated version will not be found, and shortening it again counts "
        "as the same failed attempt.",
        _address_lines(profile, shown),
    ]
    if rest > 0:
        lines.append(f"    … and {rest} more not listed.")
    return "\n".join(lines)


def _summarize(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def inventory_report(profile: Profile, cap: int) -> str:
    """What the model is shown when a batch does not fit: the full system-prompt
    scope plus its usage, so it can reissue one batch that frees room and adds.

    Demotion is offered before removal — a situational memory moved to RAG stops
    costing context on every turn but stays retrievable, so nothing has to be
    thrown away to make room. The entries come back as addresses, not as an
    elided inventory: the next call has to name one of them in `match`.
    """
    return (
        f"The system-prompt memory is full ({profile.usage_meter(cap)}). "
        "Reissue ONE batch that frees room and adds: prefer 'demote' on "
        "situational entries (they move to rag and stay retrievable) over "
        "removing them outright. "
        f"{address_help(profile, MemoryScope.SYSTEM_PROMPT)}"
    )


def apply_batch(
    profile: Profile,
    operations: list[MemoryOp],
    *,
    backend: str = "",
    system_prompt_cap: int | None = None,
) -> BatchResult:
    """Apply operations to a copy of ``profile``; budget checked at the end.

    Raises ``MemoryOpError`` if any operation cannot be applied or the final
    state breaks the system-prompt budget — the batch is all-or-nothing, so a
    partly applied edit can never leave memory in a state nobody approved.
    """
    if not operations:
        raise MemoryOpError("No operations given.")
    working = Profile(
        name=profile.name,
        created=profile.created,
        default_backend=profile.default_backend,
        memories=[Memory(**vars(memory)) for memory in profile.memories],
        problems=list(profile.problems),
    )
    result = BatchResult(profile=working)
    today = date.today().isoformat()
    for operation in operations:
        if operation.op == "add":
            text = operation.text.strip()
            if not text:
                raise MemoryOpError("An add needs text.")
            if any(
                memory.scope == operation.scope and memory.text.strip() == text
                for memory in working.memories
            ):
                result.skipped.append(f"already present: {_summarize(text)}")
                continue
            working.memories.append(
                Memory(
                    text=text,
                    scope=operation.scope,
                    backend=backend,
                    created=today,
                    kind=_kind_for(operation),
                )
            )
            result.applied.append(operation.describe())
            result.flagged += scan_threats(text)
        elif operation.op == "remove":
            target = resolve(working, operation.scope, operation.match)
            working.memories.remove(target)
            result.applied.append(operation.describe())
        elif operation.op == "demote":
            # RAG is retrieved, not injected: the entry stops costing context
            # every turn but is still there when it matches. Demotion always
            # addresses a system-prompt memory (RAG has nowhere lower to go).
            target = resolve(working, MemoryScope.SYSTEM_PROMPT, operation.match)
            target.scope = MemoryScope.RAG
            result.applied.append(operation.describe())
        elif operation.op == "replace":
            text = operation.text.strip()
            if not text:
                raise MemoryOpError("A replace needs text.")
            target = resolve(working, operation.scope, operation.match)
            target.text = text
            target.backend = backend or target.backend
            target.created = today
            result.applied.append(operation.describe())
            result.flagged += scan_threats(text)
        else:
            raise MemoryOpError(f"Unknown operation {operation.op!r}.")
    if system_prompt_cap is not None and working.over_budget(system_prompt_cap):
        raise MemoryOpError(inventory_report(working, system_prompt_cap))
    return result


def _kind_for(operation: MemoryOp) -> str:
    return "fact" if operation.scope is MemoryScope.SYSTEM_PROMPT else "learning"


def drift_detected(profile: Profile, path_text: str) -> bool:
    """Whether the file on disk no longer matches the loaded profile.

    Users edit these files by hand mid-session (§6.4). Rewriting the file
    from a stale in-memory copy would silently discard those edits, so a
    mismatch means: back up, refuse, reload.
    """
    on_disk = Profile.parse(path_text, name=profile.name)
    return [(m.scope, m.text.strip()) for m in on_disk.memories] != [
        (m.scope, m.text.strip()) for m in profile.memories
    ]
