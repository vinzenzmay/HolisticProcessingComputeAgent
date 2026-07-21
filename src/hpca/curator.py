"""The curator: slow consolidation of what /conclude accumulated.

Self-review adds; nothing removes. Left alone, the RAG scope fills with
near-duplicate struggle notes from the same rough problem and retrieval quality
decays. The curator is the counterweight — a periodic, idle-triggered pass that
ages entries out and proposes merges.

Two rules, both from Hermes' curator, both about being safe enough to run
unattended:

* **Archive, never delete.** The worst outcome of a bad archival decision is
  a file the user can move back; the worst outcome of a bad deletion is gone.
* **Deterministic first, model second.** Ageing is arithmetic on dates and
  needs no model. Only merge *proposals* involve the LLM, and they are
  written to a file for the user to approve — never applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from hpca.config import app_dir
from hpca.profiles import Memory, MemoryScope, Profile, profiles_dir

STALE_DAYS = 30
ARCHIVE_DAYS = 90
PINNED_KIND = "pinned"


def state_path() -> Path:
    return profiles_dir() / ".curator_state.json"


def archive_path(profile: str) -> Path:
    return profiles_dir() / f"{profile}.archive.md"


@dataclass
class CuratorReport:
    archived: list[Memory]
    stale: list[Memory]

    @property
    def changed(self) -> bool:
        return bool(self.archived)

    def summary(self) -> str:
        parts = []
        if self.archived:
            parts.append(f"archived {len(self.archived)}")
        if self.stale:
            parts.append(f"{len(self.stale)} going stale")
        return ", ".join(parts)


def _age_days(created: str) -> int | None:
    try:
        return (date.today() - date.fromisoformat(created)).days
    except (ValueError, TypeError):
        return None


def load_state() -> dict:
    try:
        return json.loads(state_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    profiles_dir().mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(state, indent=2) + "\n")


def due(
    *,
    now: datetime | None = None,
    interval_days: int = 7,
    state: dict | None = None,
) -> bool:
    """Whether a curator pass is due. Failing open would mean running on
    every idle tick, so an unreadable timestamp counts as 'just ran'."""
    state = load_state() if state is None else state
    last = state.get("last_run")
    if not last:
        return True
    now = now or datetime.now(timezone.utc)
    try:
        previous = datetime.fromisoformat(last)
    except ValueError:
        return False
    return now - previous >= timedelta(days=interval_days)


def curate(
    profile: Profile,
    *,
    stale_days: int = STALE_DAYS,
    archive_days: int = ARCHIVE_DAYS,
) -> CuratorReport:
    """Age RAG entries out of the profile. Mutates ``profile`` in place.

    Only RAG is aged: the system-prompt scope is small, curated, and injected
    every turn, so the user is already looking at it. RAG is the scope that
    grows unattended, which is the one that needs a gardener.
    """
    archived, stale = [], []
    for memory in list(profile.memories):
        if memory.scope is not MemoryScope.RAG or memory.kind == PINNED_KIND:
            continue
        age = _age_days(memory.created)
        if age is None:
            continue  # hand-written entry with no date: leave it alone
        if age >= archive_days:
            profile.memories.remove(memory)
            archived.append(memory)
        elif age >= stale_days:
            stale.append(memory)
    return CuratorReport(archived=archived, stale=stale)


def append_to_archive(profile_name: str, memories: list[Memory]) -> Path:
    """Archived entries land in a plain markdown file next to the profile —
    recoverable by moving a block back, which is the whole point."""
    path = archive_path(profile_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    lines = (
        []
        if path.exists()
        else [
            f"# Archived memories for {profile_name}",
            "",
            "To restore one, move its block back under the matching "
            "`## [system-prompt|rag]` heading in the profile file.",
            "",
        ]
    )
    for memory in memories:
        # The scope is recorded because restoring means putting the block back
        # under the right heading, and the parser accepts it anywhere — a RAG
        # note restored into system-prompt would start costing context on
        # every turn.
        meta = ", ".join(
            f"{key}: {value}"
            for key, value in (
                ("scope", memory.scope.value),
                ("backend", memory.backend),
                ("created", memory.created),
                ("kind", memory.kind),
                ("archived", stamp),
            )
            if value
        )
        lines += ["", f"<!-- {meta} -->", memory.text]
    with path.open("a") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def run(
    profile_names: list[str],
    *,
    stale_days: int = STALE_DAYS,
    archive_days: int = ARCHIVE_DAYS,
) -> dict[str, CuratorReport]:
    """One pass over every profile. Returns what changed, per profile."""
    reports: dict[str, CuratorReport] = {}
    for name in profile_names:
        profile = Profile.load(name)
        report = curate(profile, stale_days=stale_days, archive_days=archive_days)
        if report.archived:
            append_to_archive(name, report.archived)
            profile.save()
        if report.archived or report.stale:
            reports[name] = report
    save_state({"last_run": datetime.now(timezone.utc).isoformat()})
    return reports
