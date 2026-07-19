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

import re
from dataclasses import dataclass, field
from datetime import date

from hpca.profiles import Memory, Profile

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


class MemoryOpError(Exception):
    """An operation that cannot be applied as written."""


@dataclass
class MemoryOp:
    op: str  # add | replace | remove | demote
    tier: int = 2
    match: str = ""  # replace/remove/demote: a unique substring of the target
    text: str = ""  # add/replace: the new text

    def describe(self) -> str:
        if self.op == "add":
            return f"add to tier {self.tier}: {self.text}"
        if self.op == "remove":
            return f"remove from tier {self.tier}: “{self.match}”"
        if self.op == "demote":
            return f"move “{self.match}” from tier {self.tier} to tier 3"
        return f"replace “{self.match}” (tier {self.tier}) with: {self.text}"


@dataclass
class BatchResult:
    profile: Profile  # a copy with the batch applied
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # no-ops, e.g. exact dupes
    flagged: list[str] = field(default_factory=list)  # threat-pattern hits


def scan_threats(text: str) -> list[str]:
    """Threat patterns present in a proposed memory, for the approval dialog."""
    return [pattern.pattern for pattern in THREAT_PATTERNS if pattern.search(text)]


def resolve(profile: Profile, tier: int, match: str) -> Memory:
    """The one memory in ``tier`` containing ``match``.

    Raises with the candidates when the substring is ambiguous or absent —
    guessing on the model's behalf is how the wrong memory gets deleted.
    """
    if not match.strip():
        raise MemoryOpError("Give a short substring of the memory to address.")
    needle = match.strip().lower()
    candidates = [
        memory
        for memory in profile.memories
        if memory.tier == tier and needle in memory.text.lower()
    ]
    if not candidates:
        raise MemoryOpError(
            f"No tier-{tier} memory contains “{match}”. "
            f"Current tier-{tier} memories: {_inventory(profile, tier)}"
        )
    if len(candidates) > 1:
        shown = "; ".join(_summarize(m.text) for m in candidates[:4])
        raise MemoryOpError(
            f"“{match}” matches {len(candidates)} tier-{tier} memories "
            f"({shown}) — use a longer, unique substring."
        )
    return candidates[0]


def _summarize(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _inventory(profile: Profile, tier: int) -> str:
    texts = [_summarize(m.text) for m in profile.memories if m.tier == tier]
    return "; ".join(texts) if texts else "(none)"


def inventory_report(profile: Profile, tier: int, cap: int) -> str:
    """What the model is shown when a batch does not fit: the full tier plus
    its usage, so it can reissue one batch that frees room and adds.

    Demotion is offered before removal — a situational memory moved to tier 3
    stops costing context on every turn but stays retrievable, so nothing has
    to be thrown away to make room.
    """
    return (
        f"Tier {tier} is full ({profile.usage_meter(tier, cap)}). "
        f"Current tier-{tier} memories: {_inventory(profile, tier)}. "
        "Reissue ONE batch that frees room and adds: prefer 'demote' on "
        "situational entries (they move to tier 3 and stay retrievable) over "
        "removing them outright."
    )


def apply_batch(
    profile: Profile,
    operations: list[MemoryOp],
    *,
    backend: str = "",
    caps: dict[int, int] | None = None,
) -> BatchResult:
    """Apply operations to a copy of ``profile``; budget checked at the end.

    Raises ``MemoryOpError`` if any operation cannot be applied or the final
    state breaks a tier budget — the batch is all-or-nothing, so a partly
    applied edit can never leave memory in a state nobody approved.
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
                memory.tier == operation.tier and memory.text.strip() == text
                for memory in working.memories
            ):
                result.skipped.append(f"already present: {_summarize(text)}")
                continue
            working.memories.append(
                Memory(
                    text=text,
                    tier=operation.tier,
                    backend=backend,
                    created=today,
                    kind=_kind_for(operation),
                )
            )
            result.applied.append(operation.describe())
            result.flagged += scan_threats(text)
        elif operation.op == "remove":
            target = resolve(working, operation.tier, operation.match)
            working.memories.remove(target)
            result.applied.append(operation.describe())
        elif operation.op == "demote":
            # Tier 3 is retrieved, not injected: the entry stops costing
            # context every turn but is still there when it matches.
            target = resolve(working, operation.tier, operation.match)
            target.tier = 3
            result.applied.append(operation.describe())
        elif operation.op == "replace":
            text = operation.text.strip()
            if not text:
                raise MemoryOpError("A replace needs text.")
            target = resolve(working, operation.tier, operation.match)
            target.text = text
            target.backend = backend or target.backend
            target.created = today
            result.applied.append(operation.describe())
            result.flagged += scan_threats(text)
        else:
            raise MemoryOpError(f"Unknown operation {operation.op!r}.")
    for tier, cap in (caps or {}).items():
        if working.tier_chars(tier) > cap:
            raise MemoryOpError(inventory_report(working, tier, cap))
    return result


def _kind_for(operation: MemoryOp) -> str:
    return "fact" if operation.tier == 1 else "learning"


def drift_detected(profile: Profile, path_text: str) -> bool:
    """Whether the file on disk no longer matches the loaded profile.

    Users edit these files by hand mid-session (§6.4). Rewriting the file
    from a stale in-memory copy would silently discard those edits, so a
    mismatch means: back up, refuse, reload.
    """
    on_disk = Profile.parse(path_text, name=profile.name)
    return [(m.tier, m.text.strip()) for m in on_disk.memories] != [
        (m.tier, m.text.strip()) for m in profile.memories
    ]
