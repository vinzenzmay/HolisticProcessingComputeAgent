"""Profile memory: what is frozen, what is recalled, and what may be written.

Lifted out of `HpcaApp` (specs-core-process.md §7). Five properties in here are
load-bearing and cheap to break by accident, so they are stated once, at the
top, rather than left to be rediscovered from the code:

**The snapshot is frozen per profile.** :meth:`MemoryService.snapshot` loads a
profile once and hands the same object to every turn until something
invalidates it. That is not caching for its own sake: the memories go into the
*system prompt*, and the backend's prefix cache only helps while that prefix
stays byte-identical. Reloading per turn would mean re-reading a file that
almost never changed and paying a full prompt re-ingest for it. So the refresh
points are explicit and few — an approved write, a profile edit, a profile
switch, a curator pass — and each of them is a place where invalidating the
prefix cache is worth what it buys.

**RAG is retrieved, never injected.** The system-prompt scope is held to a hard
token budget; the RAG scope is unbounded because it costs nothing until a
request matches it. That asymmetry is why :meth:`write_blocked` only ever
refuses system-prompt writes, and why recall is a search rather than a dump.

**Nothing is written until the user says so.** The `memory` tool does not write;
it queues (:meth:`queue_edits`), and the queue is reviewed at `/conclude`. A
small model with write access to its own memory is the failure mode this whole
design avoids, so every path here ends at a human.

**Approval is a round trip, not a call.** `HpcaApp._review_proposals` pushed a
modal and awaited the user inside the write path. Across a socket that is not
available: the proposals are *emitted* (`memory.proposals`) and the answer
arrives later as its own command. So production and application are separate
methods, and the unanswered set is held here, keyed by session — the same shape
as the parked approval interrupt in §4.4, and for the same reason: the decision
is core state, and a front-end that dies mid-review must not take it with it.

**Emitting is not deciding.** A warning about a past struggle is a toast; a
background turn must not toast. So the matching stays here and the *caller*
says whether a warning is wanted — see :meth:`warn_about_struggles`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Callable, Literal

from hpca import curator
from hpca.agent.conclude import MemoryProposal, propose_memories
from hpca.agent.memory_context import note_line, retrieved_line
from hpca.agent.reflect import Reflection, propose_reflections
from hpca.agent.skill_tools import add_skill_tools
from hpca.agent.struggle import STRUGGLE_KIND, matching_struggles
from hpca.core.deps import CoreDeps
from hpca.memory_index import MemoryIndex
from hpca.memory_ops import (
    BatchResult,
    MemoryOp,
    MemoryOpError,
    apply_batch,
    drift_detected,
)
from hpca.profiles import DEFAULT_PROFILE, Memory, MemoryScope, Profile
from hpca.protocol import MemoryProposals, Notify, Proposal, TurnActivity
from hpca.runner import running_session_ids
from hpca.sessions import Session, SessionStore
from hpca.skills import (
    Skill,
    copy_profile_skills,
    delete_own_skill,
    delete_profile_skills,
    load_own_skills,
    load_skills,
    patched_body,
    skill_path,
    summarize_skills,
    write_skill,
)

if TYPE_CHECKING:
    from hpca.agent.tools import ToolRegistry
    from hpca.curator import CuratorReport

Severity = Literal["information", "warning", "error"]

# What a held proposal set came from. The three producers write to memory in
# three different ways, and the applier has to know which — a reflection may
# rewrite a skill file, a `/memorize` proposal never does.
KIND_MEMORIZE = "memorize"
KIND_REFLECTION = "reflection"
KIND_FLAGGED = "flagged"


@dataclass
class ProposalSet:
    """Proposals emitted and waiting for an answer that arrives separately.

    Held rather than passed to a callback because the answer crosses a socket:
    between the emit and the reply the front-end may repaint, switch session or
    die. The authoritative objects stay here and the wire carries a *rendering*
    of them (see :meth:`MemoryService._wire`), so nothing has to round-trip
    through a process that only draws them.
    """

    session_id: str
    profile: str
    kind: str  # one of the KIND_* strings above
    # Which of the three the list holds follows from `kind`; they are kept in
    # one field because the round trip is identical for all three and only the
    # applier cares what it is writing.
    items: list[MemoryProposal | Reflection | MemoryOp]
    # The model these were formed under, captured at proposal time. Memories
    # are tagged with the backend that learned them, and by the time the answer
    # arrives the user may be looking at a session on another one.
    backend: str = ""
    # A flagged batch is approved whole: it is often a trade (free two entries
    # to fit one), and half of a trade is a state nobody chose.
    all_or_nothing: bool = False
    # Flagged batches only: the pre-computed result and the copy it was
    # computed from, kept so the drift check at apply time compares against
    # what the user actually reviewed.
    batch: BatchResult | None = None
    loaded: Profile | None = None
    flagged: list[str] = field(default_factory=list)


class MemoryService:
    """Profile memory, recall and the write path, with no front-end in it.

    Everything it wants to say goes through ``deps.emit``; everything it needs
    from the rest of the runtime arrives as a callable, so this class can be
    built in a test with a fake LLM and an in-memory database and nothing else.

    ``llm_for`` and ``backend_name`` are the seam onto `core.backends`: the
    per-session client and the model label are that service's business, and
    resolving them lazily (rather than being handed a client) is what lets a
    session's backend change between two `/conclude`s.
    """

    def __init__(
        self,
        deps: CoreDeps,
        *,
        index: MemoryIndex | None = None,
        llm_for: Callable[[str, "Session | None"], Any] | None = None,
        backend_name: Callable[["Session | None"], str] | None = None,
        busy_profiles: Callable[[], set[str]] | None = None,
        tools: "ToolRegistry | None" = None,
        project_root: Path | None = None,
    ) -> None:
        self._deps = deps
        self._index = index
        self._llm_for = llm_for
        self._backend_name = backend_name or (lambda session: "")
        # Which profiles have a turn in flight. The turn table belongs to the
        # scheduler, so it is read through a callable rather than copied here
        # — a stale copy would refuse a deletion that is now fine, or allow one
        # that is not.
        self._busy_profiles = busy_profiles or (lambda: set())
        self._tools = tools
        # Pinned so project-level skills stay testable; the app resolves it
        # from the working directory it was started in.
        self._project_root = project_root
        # Frozen per-profile memory views. See the module docstring for why
        # this is not reloaded per turn.
        self._snapshots: dict[str, Profile] = {}
        # Memory the agent flagged mid-conversation with the `memory` tool, per
        # session. Nothing is written until the user runs /conclude, which
        # reviews these together with the self-review proposals.
        self._pending_edits: dict[str, list[MemoryOp]] = {}
        # Emitted proposal sets awaiting an answer, per session.
        self._pending_proposals: dict[str, ProposalSet] = {}
        self.skills: list[Skill] = load_skills(
            deps.profile, project_root=self._project_root
        )

    # ------------------------------------------------------------ plumbing

    def _notify(self, text: str, severity: Severity = "information") -> None:
        self._deps.emit(Notify(severity=severity, text=text))

    def _activity(self, session: "Session | None", label: str) -> None:
        """The one event that replaced show_working/hide_working (§7).

        An empty label means "no longer working"; whether that draws a spinner
        in the chat or a glyph in the sidebar is the renderer's decision, which
        is exactly the decision this service must not make.
        """
        if session is None:
            return
        self._deps.emit(
            TurnActivity(
                session_id=session.session_id,
                activity=label,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    def _settings(self):
        return self._deps.settings.memory

    def _cap(self) -> int:
        return self._settings().system_prompt_token_cap

    @property
    def project_root(self) -> Path | None:
        """Where project-level skills are read from, or None for the working
        directory. Public because the slash commands that list and remove
        skills have to look in the same place this service writes them, and
        two answers to "which project" would silently disagree in a test."""
        return self._project_root

    # ------------------------------------------------------------ snapshots

    def snapshot(self, profile: str) -> Profile:
        """The frozen memory view for a profile — loaded once, reused across
        turns. See the module docstring for why it is not reloaded per turn.

        Loading is also when the RAG index is rebuilt: the markdown file is the
        source of truth and may have been edited by hand since.
        """
        snapshot = self._snapshots.get(profile)
        if snapshot is None:
            snapshot = Profile.load(profile)
            self._snapshots[profile] = snapshot
            if self._index is not None:
                try:
                    self._index.reindex(snapshot)
                except Exception:
                    pass  # retrieval is an optimization; never block a turn
        return snapshot

    def invalidate(self, profile: str) -> None:
        """Drop a profile's frozen view; the next read reloads from disk and
        rebuilds the RAG index.

        Called after approved writes and profile edits — the deliberate refresh
        points where invalidating the backend's prefix cache is worth it."""
        self._snapshots.pop(profile, None)

    def invalidate_all(self) -> None:
        """Drop every frozen view, for an edit that may have touched any
        profile (the profiles manager writes whichever the user picked)."""
        self._snapshots.clear()

    @property
    def working_memory(self) -> Profile:
        """The working profile's memory.

        `HpcaApp` kept this as `profile_memory`, a second reference alongside
        the snapshot for the same profile, and had to assign both in four
        places to keep them equal. One object cannot drift from itself.
        """
        return self.snapshot(self._deps.profile)

    def set_working_profile(self, profile: str) -> None:
        """The working profile is the open session's: its memories, its file
        under the editor, its name in the top bar.

        A switch is one of the explicit refresh points — the user may have
        edited that profile's file since it was last read — so the view is
        reloaded rather than taken from the cache.
        """
        if profile == self._deps.profile:
            return
        self._deps.profile = profile
        self.invalidate(profile)
        memory = self.snapshot(profile)
        self.skills = load_skills(profile, project_root=self._project_root)
        if memory.problems:
            self._notify(
                "Profile file has problems: " + "; ".join(memory.problems[:3]),
                "warning",
            )

    def refresh_skills(self) -> None:
        """Make a skill change visible without a graph rebuild: the working
        profile's list feeds the next turn's prompt. read_skill is registered
        from the start (HPCA ships skills), but a caller may have passed in its
        own registry, so top it up rather than assume."""
        self.skills = load_skills(
            self._deps.profile, project_root=self._project_root
        )
        self._register_skill_tool()

    def _register_skill_tool(self) -> None:
        if self._tools is not None and "read_skill" not in self._tools.names():
            add_skill_tools(self._tools)

    # --------------------------------------------------------------- recall

    def recall_lines(
        self,
        user_text: str,
        memory: Profile,
        profile: str,
        *,
        backend: str = "",
    ) -> list[str]:
        """What this request recalls: matching struggle notes plus retrieved
        RAG memories, in that order and without duplicates.

        RAG is retrieved rather than injected wholesale, which is what lets it
        grow: situational memories cost context only on the turns they actually
        match.

        ``backend`` is the model the *asking* session runs on; it breaks ties
        in favour of memories learned on the same backend. `HpcaApp` read it
        off whatever session was on screen, which was wrong for a background
        turn — here the caller passes the turn's own.
        """
        lines = [note_line(m) for m in matching_struggles(memory.memories, user_text)]
        seen = set(lines)
        if self._index is not None:
            try:
                hits = self._index.search(
                    user_text,
                    profile=profile,
                    active_backend=backend,
                    limit=self._settings().rag_prefetch_count,
                )
            except Exception:
                hits = []
            for hit in hits:
                line = retrieved_line(hit)
                if line not in seen:
                    seen.add(line)
                    lines.append(line)
        return lines

    def warn_about_struggles(
        self, text: str, *, memory: Profile | None = None, warn: bool = True
    ) -> list[Memory]:
        """§4.4: the past struggles this request resembles.

        The model gets its own copy as a fenced memory-context block when the
        turn starts (see :meth:`recall_lines`); the user gets a toast. Whether
        the toast is wanted is the caller's call, not this service's: a turn
        the user is watching should warn, a background completion reporting in
        should not interrupt whatever they are reading instead.
        """
        target = memory if memory is not None else self.working_memory
        matches = matching_struggles(target.memories, text)
        if warn:
            for match in matches:
                first_line = match.text.splitlines()[0]
                self._notify(
                    f"I have struggled with this before: {first_line}", "warning"
                )
        return matches

    # ---------------------------------------------------------- write guard

    def write_blocked(
        self, scope: MemoryScope, text: str, memory: Profile | None = None
    ) -> bool:
        """Hard token budget on writes: a full system-prompt scope rejects new
        memories until the user condenses it. Injection never truncates; only
        growth is stopped. RAG is retrieved, not injected, so it has no budget.

        ``memory`` is the profile the caller is about to write to — pass the
        same object, or the budget gets checked against one state and the write
        lands in another.

        Reported, not raised: the write is refused and the user is told how to
        make room, because the alternative is a memory silently vanishing.
        """
        if scope is not MemoryScope.SYSTEM_PROMPT:
            return False  # RAG is retrieved, not injected: no budget
        target = memory if memory is not None else self.working_memory
        cap = self._cap()
        if not target.would_exceed(text, cap=cap):
            return False
        self._notify(
            f"System-prompt memory is full ({target.usage_meter(cap)}) — "
            "memory NOT saved. Condense the profile, then retry.",
            "warning",
        )
        return True

    def check_memory_caps(self) -> bool:
        """Size warning; returns whether the system-prompt scope is over budget.

        It can only get over cap through hand edits (in-app writes are rejected
        at the cap), so the fix offered is the external editor."""
        cap = self._cap()
        memory = self.working_memory
        if not memory.over_budget(cap):
            return False
        self._notify(
            f"System-prompt memory is over its budget ({memory.usage_meter(cap)})"
            " — edit the profile externally to condense it.",
            "warning",
        )
        return True

    # ------------------------------------------------------- flagged edits

    def queue_edits(self, session_id: str, operations: list[MemoryOp]) -> str:
        """The `memory` tool's path: queue a flagged batch for review at the
        next /conclude. Nothing is written now — the agent flags, the user
        decides. Returns the tool result so the model knows it was noted."""
        if not operations:
            return "Nothing to flag."
        self._pending_edits.setdefault(session_id, []).extend(operations)
        return (
            f"Noted {len(operations)} memory change(s) — they will be reviewed "
            "together when the user runs /conclude. Nothing is saved yet."
        )

    def pending_edits(self, session_id: str) -> list[MemoryOp]:
        """What this session has flagged so far. Read-only: `/conclude` asks
        before doing anything, and "nothing to conclude" is a legitimate
        answer that must not consume the queue on its way to being given."""
        return list(self._pending_edits.get(session_id, ()))

    def propose_flagged_edits(self, session: Session) -> ProposalSet | None:
        """Offer the facts flagged this session for review, at /conclude.

        The queue is *not* cleared here. An emitted set may never be answered —
        the front-end can die between the two — and losing a session's flagged
        memory to a repaint would be silent. :meth:`apply_answered` clears it,
        whichever way the answer goes.
        """
        operations = self.pending_edits(session.session_id)
        if not operations:
            return None
        profile = session.profile
        loaded = Profile.load(profile)
        backend = self._backend_name(session)
        try:
            result = apply_batch(
                loaded,
                operations,
                backend=backend,
                system_prompt_cap=self._cap(),
            )
        except MemoryOpError as e:
            # Unapplicable as written, and it will be just as unapplicable next
            # time: drop the queue rather than re-offering a batch that cannot
            # land (this is what popping-before-the-modal used to do).
            self._pending_edits.pop(session.session_id, None)
            self._notify(f"Flagged memory not applied: {e}", "warning")
            return None
        if not result.applied:
            self._pending_edits.pop(session.session_id, None)
            return None
        return self._emit_proposals(
            ProposalSet(
                session_id=session.session_id,
                profile=profile,
                kind=KIND_FLAGGED,
                items=operations,
                backend=backend,
                all_or_nothing=True,
                batch=result,
                loaded=loaded,
                flagged=result.flagged,
            )
        )

    # ------------------------------------------------------- proposal round

    async def propose_from_note(
        self, session: Session | None, messages: list[dict], note: str
    ) -> ProposalSet | None:
        """`/memorize <note>`: the model turns the note plus the conversation
        so far into durable memory proposals.

        Each still needs the user's approval (§5.3 — a small model writes
        these, so review is essential), which arrives as its own command.
        """
        profile = session.profile if session is not None else self._deps.profile
        # Loaded fresh rather than taken from the snapshot: the proposals are
        # de-duplicated against what is already known, and a stale view would
        # invite the model to re-propose what the user just wrote by hand.
        memory = Profile.load(profile)
        backend = self._backend_name(session)
        try:
            self._activity(session, "forming memories")
            proposals = await propose_memories(
                self._llm("memorize", session),
                messages,
                system_prompt_memories=memory.scope_text(MemoryScope.SYSTEM_PROMPT),
                guidance=note,
            )
        except Exception as e:
            self._notify(f"/memorize failed: {e}", "error")
            return None
        finally:
            self._activity(session, "")
        if not proposals:
            self._notify("The model proposed no memories for that note.")
            return None
        return self._emit_proposals(
            ProposalSet(
                session_id=session.session_id if session is not None else "",
                profile=profile,
                kind=KIND_MEMORIZE,
                items=list(proposals),
                backend=backend,
            )
        )

    async def review_conversation(
        self, session: Session, messages: list[dict], *, span: str = "whole"
    ) -> ProposalSet | None:
        """The /conclude self-review: propose what this conversation is worth
        keeping — memories, struggle notes, skill changes.

        Both scopes are handed to the reviewer as already-known, which is what
        stops it re-proposing a RAG note it has just been shown in-context (and
        proposing it into the always-injected budget RAG exists to keep it out
        of — see `agent.reflect`).
        """
        memory = self.snapshot(session.profile)
        skills = load_skills(session.profile, project_root=self._project_root)
        backend = self._backend_name(session)
        try:
            self._activity(session, "reviewing conversation")
            proposals = await propose_reflections(
                self._llm("conclude", session),
                messages,
                system_prompt_memories=memory.scope_text(MemoryScope.SYSTEM_PROMPT),
                rag_memories=memory.scope_text(MemoryScope.RAG),
                skills=summarize_skills(skills),
                allow_new_skills=self._settings().propose_new_skills,
                span=span,
            )
        except Exception as e:
            self._notify(f"/conclude failed: {e}", "error")
            return None
        finally:
            self._activity(session, "")
        if not proposals:
            return None
        return self._emit_proposals(
            ProposalSet(
                session_id=session.session_id,
                profile=session.profile,
                kind=KIND_REFLECTION,
                items=list(proposals),
                backend=backend,
            )
        )

    def pending_proposals(self, session_id: str) -> ProposalSet | None:
        """The set this session is waiting on, if any.

        Exposed for the same reason a parked decision is re-emitted on
        subscribe (§4.4): a review that outlives the front-end showing it must
        still be answerable afterwards.
        """
        return self._pending_proposals.get(session_id)

    def apply_answered(
        self, session_id: str, approved: Sequence[bool]
    ) -> int:
        """Write exactly the approved proposals of the set this session holds.

        ``approved`` is positional against ``ProposalSet.items``; a short
        answer leaves the rest rejected, because an answer that never arrived
        is not an approval. Returns how many proposals were kept (0 or 1 for an
        all-or-nothing batch — it is one decision, not many).
        """
        pending = self._pending_proposals.pop(session_id, None)
        if pending is None:
            return 0
        flags = list(approved) + [False] * (len(pending.items) - len(approved))
        if pending.kind == KIND_FLAGGED:
            return self._apply_flagged(pending, flags)
        if pending.kind == KIND_REFLECTION:
            kept = sum(
                1
                for item, keep in zip(pending.items, flags)
                if keep and self._apply_reflection(item, pending)
            )
            if kept:
                self.check_memory_caps()
            return kept
        return self._apply_memorize(pending, flags)

    # ------------------------------------------------------------ appliers

    def _apply_memorize(self, pending: ProposalSet, flags: list[bool]) -> int:
        """Persist the approved `/memorize` proposals, as one write.

        Loaded from disk rather than written into the frozen snapshot: the
        profile file is the source of truth and the user may have edited it
        while the proposals were on screen. Merge, don't clobber.
        """
        target = Profile.load(pending.profile)
        kept = 0
        for proposal, keep in zip(pending.items, flags):
            if not keep:
                continue
            if self.write_blocked(proposal.scope, proposal.text, target):
                continue
            target.add_memory(
                proposal.text,
                scope=proposal.scope,
                backend=pending.backend,
                kind=proposal.kind,
            )
            kept += 1
        if kept:
            target.save()
            self.invalidate(pending.profile)
        self._notify(f"Kept {kept} of {len(pending.items)} proposed memories.")
        self.check_memory_caps()
        return kept

    def _apply_reflection(
        self, proposal: Reflection, pending: ProposalSet
    ) -> bool:
        """Persist one approved proposal; returns whether anything was written."""
        profile = pending.profile
        if proposal.kind in ("memory", "struggle"):
            text = proposal.memory_text()
            scope = proposal.target_scope()
            target = Profile.load(profile)  # merge, don't clobber
            if self.write_blocked(scope, text, target):
                return False
            target.add_memory(
                text,
                scope=scope,
                backend=pending.backend,
                kind=STRUGGLE_KIND if proposal.kind == "struggle" else "learning",
            )
            target.save()
            self.invalidate(profile)
            return True
        return self._apply_skill_reflection(proposal, profile)

    def _apply_skill_reflection(self, proposal: Reflection, profile: str) -> bool:
        existing = {
            s.name: s
            for s in load_skills(profile, project_root=self._project_root)
        }
        if proposal.kind == "skill_patch":
            skill = existing.get(proposal.skill_name)
            if skill is None:
                self._notify(
                    f"No skill named “{proposal.skill_name}” to patch.", "warning"
                )
                return False
            # Patches land in the profile's own copy, never in _shared/: one
            # profile's correction must not change another's procedure.
            skill.body = patched_body(skill, proposal.text)
            write_skill(skill, profile)
        else:
            if proposal.skill_name in existing:
                self._notify(
                    f"A skill named “{proposal.skill_name}” already exists.",
                    "warning",
                )
                return False
            write_skill(
                Skill(
                    name=proposal.skill_name,
                    description=" ".join(proposal.text.split())[:60],
                    triggers=proposal.keywords,
                    body=proposal.text,
                ),
                profile,
            )
        if profile == self._deps.profile:
            self.skills = load_skills(profile, project_root=self._project_root)
        self._register_skill_tool()  # the first skill enables the tool
        return True

    def _apply_flagged(self, pending: ProposalSet, flags: list[bool]) -> int:
        """Commit (or discard) the batch the agent flagged this session.

        All-or-nothing: a batch is often a trade — free two entries to fit one
        — and applying half of that leaves memory in a state nobody approved.
        """
        self._pending_edits.pop(pending.session_id, None)
        if pending.batch is None or not flags or not all(flags):
            self._notify("Discarded the flagged memory changes.")
            return 0
        # The user may have edited this file by hand since it was loaded;
        # rewriting from a stale copy would silently discard those edits.
        path = Profile.path_for(pending.profile)
        if (
            pending.loaded is not None
            and path.exists()
            and drift_detected(pending.loaded, path.read_text())
        ):
            backup = path.with_suffix(f".bak.{int(time())}")
            backup.write_text(path.read_text())
            self.invalidate(pending.profile)
            self._notify(
                "Flagged memory not applied: the profile file changed on disk "
                f"since this session read it (backed up to {backup.name}).",
                "warning",
            )
            return 0
        pending.batch.profile.save()
        self.invalidate(pending.profile)
        self.check_memory_caps()
        return 1

    # -------------------------------------------------------------- curator

    def run_curator_if_due(self) -> dict[str, "CuratorReport"]:
        """Age old RAG entries out, at most once every few days (P6).

        Run at startup rather than on a timer: the core is idle then by
        definition, and this touches the same profile files a turn reads.
        """
        interval = self._settings().curator_interval_days
        if interval <= 0 or not curator.due(interval_days=interval):
            return {}
        try:
            reports = curator.run(
                Profile.list_profiles(),
                stale_days=self._settings().curator_stale_days,
                archive_days=self._settings().curator_archive_days,
            )
        except Exception as e:
            self._notify(f"Memory curation skipped: {e}", "warning")
            return {}
        for name, report in reports.items():
            if report.changed:
                self.invalidate(name)
                self._notify(
                    f"Memory curation ({name}): {report.summary()} — "
                    f"archived entries are in {name}.archive.md"
                )
        return reports

    # ----------------------------------------------------- profile / skills

    def save_profile_memories(self, name: str, text: str) -> None:
        """Persist the raw memory text a user edited; report parse trouble but
        never lose their edits — the file is theirs to fix by hand (§6.4)."""
        profile = Profile.parse(text, name=name)
        profile.save()
        if profile.problems:
            self._notify(
                "Saved with problems: " + "; ".join(profile.problems[:3]),
                "warning",
            )
        else:
            self._notify(f"Saved memories for “{name}”.")
        self.invalidate(name)

    def save_profile_archive(self, name: str, text: str) -> None:
        """Persist a hand-edited archive file. Emptying it removes the file —
        the archive is a plain appendix, not part of the loaded profile."""
        path = curator.archive_path(name)
        if text.strip():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text if text.endswith("\n") else text + "\n")
            self._notify(f"Saved archive for “{name}”.")
        elif path.exists():
            path.unlink()
            self._notify(f"Cleared archive for “{name}”.")

    def save_skill_file(self, profile: str, name: str, text: str) -> None:
        """Persist a hand-edited skill file verbatim (front matter and body).
        The user owns the file; a parse problem is reported, never fatal."""
        path = skill_path(name, profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text.endswith("\n") else text + "\n")
        if profile == self._deps.profile:
            self.skills = load_skills(profile, project_root=self._project_root)
        self._notify(f"Saved skill “{name}”.")

    def delete_profile_skill(self, profile: str, name: str) -> None:
        """Delete one of a profile's own skills (never shared/project)."""
        skill = next(
            (s for s in load_own_skills(profile) if s.name == name), None
        )
        if skill is None or not delete_own_skill(skill, profile):
            self._notify(f"No skill “{name}” to delete.", "warning")
            return
        if profile == self._deps.profile:
            self.skills = load_skills(profile, project_root=self._project_root)
        self._notify(f"Deleted skill “{name}”.")

    def create_profile(self, name: str) -> str | None:
        """Make a blank profile; returns an error message, or None on success."""
        try:
            cleaned = Profile.validate_name(name)
        except ValueError as e:
            return str(e)
        Profile.create(cleaned)
        return None

    def duplicate_profile(self, source: str, name: str) -> str | None:
        """Fork a profile: same learnings, its own future.

        Copies the memories and the source's own skills, then the two are
        independent — which is the point, a shared base that specialises in
        different directions. Returns an error message, or None on success.
        """
        try:
            cleaned = Profile.validate_name(name)
        except ValueError as e:
            return str(e)
        if source not in Profile.list_profiles():
            return f"There is no profile called “{source}”."
        Profile.duplicate(source, cleaned)
        skills = copy_profile_skills(source, cleaned)
        self.invalidate(cleaned)
        memories = len(Profile.load(cleaned).memories)
        self._notify(
            f"Copied “{source}” to “{cleaned}” "
            f"({memories} memor{'y' if memories == 1 else 'ies'}"
            + (f", {skills} skill{'' if skills == 1 else 's'}" if skills else "")
            + "). They diverge from here."
        )
        return None

    async def profile_delete_blocker(self, name: str) -> str | None:
        """Why this profile cannot be deleted right now, or None if it can.

        A profile is in use while a turn on one of its sessions is in flight,
        or while any of its sessions has a live sub-process — deleting it then
        would strand running work under a gone profile.
        """
        if name in self._busy_profiles():
            return f"“{name}” has a reply in progress — wait for it to finish."

        def busy(conn) -> bool:
            ids = {s.session_id for s in SessionStore(conn).list(profile=name)}
            return bool(ids & running_session_ids(conn))

        if await self._deps.db(busy):
            return (
                f"“{name}” has running sub-process(es) — "
                "stop them before deleting it."
            )
        return None

    async def delete_profile(self, name: str) -> None:
        """Delete a profile and everything it learned.

        Its memories, retrieval index and accumulated skills go with it:
        leaving orphaned skills behind would silently resurrect them under a
        profile created with the same name later. Its sessions fall back to the
        default rather than pointing at a file that no longer exists.
        """
        moved = await self._deps.db(
            lambda conn: SessionStore(conn).reassign_profile(name, DEFAULT_PROFILE)
        )
        Profile.delete(name)
        delete_profile_skills(name)
        self.invalidate(name)
        if self._index is not None:
            self._index.forget_profile(name)
        if self._deps.profile == name:  # unlikely, but keep the core coherent
            self.set_working_profile(DEFAULT_PROFILE)
        self._notify(
            f"Deleted “{name}”"
            + (f"; {moved} session(s) moved to default" if moved else "")
        )

    # -------------------------------------------------------------- helpers

    def _llm(self, label: str, session: "Session | None"):
        """The client for one of the core's own sub-agent calls.

        Resolved per call through `core.backends`, so a review runs on the
        session's *own* backend — the same live client its chat uses — rather
        than on whichever one happened to be current when this service was
        built (§ per-session LLM).
        """
        if self._llm_for is None:
            raise RuntimeError("no LLM is wired up for memory sub-agent calls")
        return self._llm_for(label, session)

    def _emit_proposals(self, pending: ProposalSet) -> ProposalSet:
        self._pending_proposals[pending.session_id] = pending
        self._deps.emit(
            MemoryProposals(
                session_id=pending.session_id,
                proposals=[self._wire(pending.kind, item) for item in pending.items],
            )
        )
        return pending

    @staticmethod
    def _wire(kind: str, item: Any) -> Proposal:
        """One proposal as the far side will *render* it.

        `protocol.Proposal` carries three strings, which is enough to draw the
        review and deliberately not enough to write anything: the objects that
        get applied never leave this process, so a front-end cannot smuggle an
        edited memory back in an approval. For a reflection that means
        ``kind`` carries `describe()`'s wording — it is the only field left
        that can name the skill a patch belongs to.
        """
        if kind == KIND_FLAGGED:
            return Proposal(
                scope=item.scope.value, kind=item.op, text=item.describe()
            )
        if kind == KIND_REFLECTION:
            return Proposal(
                scope=item.target_scope().value,
                kind=item.describe(),
                text=item.memory_text(),
            )
        return Proposal(scope=item.scope.value, kind=item.kind, text=item.text)
