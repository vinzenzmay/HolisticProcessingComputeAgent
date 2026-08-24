"""Profile memory: what is frozen, what is recalled, and what may be written.

Lifted out of `HpcaApp` (specs/specs-core-process.md §7). Five properties in here are
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

**A deferred write still has to answer.** Because nothing is applied, the model
never sees the effect of what it just did, so the *result string* is the only
feedback there is — and for a while it was a constant. A `demote` addressing
text that existed nowhere read exactly like a successful `add`, and the model,
having no way to tell, kept shortening its substring and trying again. So
:meth:`queue_edits` resolves every address at queue time and reports the state
it leaves behind: success and failure must be distinguishable, or a tool call
has no terminating condition.

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

import logging
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
from hpca.protocol import (
    MemoryProposals,
    Notify,
    ProfileRow,
    ProfileRows,
    Proposal,
    TurnActivity,
)
from hpca.runner import running_session_ids
from hpca.sessions import Session, SessionStore
from hpca.skills import (
    Skill,
    SkillLevel,
    copy_profile_skills,
    delete_own_skill,
    delete_profile_skills,
    load_own_skills,
    load_project_skills,
    load_skills,
    patched_body,
    skill_path,
    summarize_skills,
    write_skill,
)

if TYPE_CHECKING:
    from hpca.agent.tools import ToolRegistry
    from hpca.curator import CuratorReport

logger = logging.getLogger("hpca.core.memory_service")

Severity = Literal["information", "warning", "error"]

# What a held proposal set came from. The three producers write to memory in
# three different ways, and the applier has to know which — a reflection may
# rewrite a skill file, a `/memorize` proposal never does.
KIND_MEMORIZE = "memorize"
KIND_REFLECTION = "reflection"
KIND_FLAGGED = "flagged"

# How many changes one session may leave waiting for review. A refusal in
# :meth:`MemoryService.queue_edits` is normally something the model can act on
# and reissue, which is exactly what a runaway needs; this one it cannot retry
# its way out of, so it is the last stop before an unbounded queue. It is also
# a review limit: the user approves the batch whole, in one dialog, and past a
# couple of dozen entries that dialog stops being read.
MEMORY_QUEUE_LIMIT = 25

# How many times one operation may be refused, unchanged, before the tool
# stops for the rest of the session. Three, because the ladder needs a rung
# between "here is what is wrong" and "stop": the first refusal names the
# entries that do exist, the second says the model has already been told, and
# a model that reissues the same bytes a third time is not reading the answer
# at all — which is the runaway this whole path exists to end.
MEMORY_REPEAT_LIMIT = 3

# How many operations may be refused *since the last one that made it into the
# queue* before the tool stops for the session. Twelve: four operations run to
# the end of their three-strike ladder, or twelve distinct misses in a row.
# Since every refusal now comes back with the scope's actual entries, one miss
# should be enough for a model that reads it; twelve in a row with nothing at
# all landing in between is not a model that is going to arrive. The count is
# deliberately of *wasted* calls — ones that leave nothing for the user to
# review — which is why it is much tighter than the queue limit above.
MEMORY_REJECT_LIMIT = 12

# Said the second time an operation is refused unchanged. The point is less
# the content than that the string *moved*: an answer that never changes is
# the only thing a looping model is actually reacting to, so the second
# refusal must not read like the first, and must say what happens next.
REPEATED = (
    "You were told this once already in this session; reissuing it unchanged "
    "will not change the answer. Correct it from the entries listed above or "
    "drop it — one more identical attempt stops this tool for the session."
)


def _brief(text: str, limit: int = 70) -> str:
    """One change on one line. The full text is in the review dialog; here it
    only has to be recognizable enough to not be flagged twice."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _listing(operations: list[MemoryOp], shown: int = 6) -> str:
    items = "; ".join(_brief(operation.describe()) for operation in operations[:shown])
    rest = len(operations) - shown
    return items + (f"; … and {rest} more" if rest > 0 else "")


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
        # The other half of that ledger: what `memory` was *refused*, per
        # session. Refusals leave no trace anywhere else — that is the whole
        # problem with them — so a repeat is invisible unless it is counted
        # here. Keyed by operation identity, and never cleared, so an
        # operation refused twice an hour apart is still on its third strike.
        self._rejected_ops: dict[str, dict[tuple[str, ...], int]] = {}
        # Refusals since the last change that actually made it into the queue,
        # per session. Reset by progress, on purpose: see `queue_edits`.
        self._rejected_run: dict[str, int] = {}
        # Sessions where `memory` has stopped answering, and the answer it
        # gives instead. Set only by a terminal refusal.
        self._edits_closed: dict[str, str] = {}
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

    def queue_edits(
        self,
        session_id: str,
        operations: list[MemoryOp],
        *,
        profile: str | None = None,
    ) -> str:
        """The `memory` tool's path: queue a flagged batch for review at the
        next /conclude. Nothing is written now — the agent flags, the user
        decides. Returns the tool result the model reads.

        That result used to be one constant string, and the constant was the
        bug. `match` was resolved only at /conclude, so a `demote` addressing
        a substring that existed nowhere got the same "Noted 1 memory
        change(s)" as a successful `add`. A model that cannot tell a hit from
        a miss has no basis to stop, and one did not: it loosened the
        substring and reissued the call until the user killed the turn. So the
        addresses are resolved *here*, and the answer says which operations
        landed, which did not and why, and what is now waiting.

        They are resolved against the profile as this queue would leave it,
        not as it is on disk. Nothing is applied until /conclude, so an entry
        the model added two calls ago exists in no file to be matched — and
        adding a fact and then correcting it in the same conversation is the
        commonest shape this tool is used in.

        ``profile`` says which profile the addresses resolve against and
        defaults to the working one, which is the session's own except for a
        background turn running under another. Wrong there only in what this
        message *says*: the apply path loads the session's profile itself and
        the batch is all-or-nothing, so nothing can land in the wrong file.

        **Refusals are counted too, or the loop survives the fix.** Naming the
        miss makes a model correct itself; it does not make an incorrigible
        one stop. Neither guard above bites on a refused operation: it never
        enters the queue, so the queue limit is never approached and it is
        never a duplicate *of* anything queued. A model can therefore reissue
        one invalid `demote` forever and get a byte-identical answer every
        time — the same "tool result that does not move" that caused the
        original runaway, only with a better sentence in it. So refusals get
        their own ledger, and the answer to a repeat escalates: the second
        says it has been said before, the third stops the tool for the
        session. Every shape of the loop now ends somewhere — the same
        operation at :data:`MEMORY_REPEAT_LIMIT`, different bad operations at
        :data:`MEMORY_REJECT_LIMIT`, and good ones at
        :data:`MEMORY_QUEUE_LIMIT`.
        """
        if not operations:
            return "Nothing to flag."
        queued = self._pending_edits.setdefault(session_id, [])
        closed = self._edits_closed.get(session_id)
        if closed:
            # The queue line is rebuilt rather than replayed: /conclude can
            # empty the queue after the tool has stopped, and a stored answer
            # would go on naming entries the user has already reviewed.
            return f"{closed}\n{self._waiting_line(queued)}"
        if len(queued) >= MEMORY_QUEUE_LIMIT:
            return (
                f"Refused: {len(queued)} memory changes are already waiting "
                f"for review, which is the limit ({MEMORY_QUEUE_LIMIT}). Do "
                "not call this tool again in this session — tell the user to "
                "run /conclude to review what is queued."
            )
        projected = self._projected(
            Profile.load(profile or self._deps.profile), queued
        )
        accepted: list[MemoryOp] = []
        refused: list[str] = []
        for operation in operations:
            problem, projected = self._admit(
                operation, projected, queued + accepted
            )
            if not problem:
                accepted.append(operation)
                continue
            strikes, run = self._record_rejection(session_id, operation)
            if strikes >= MEMORY_REPEAT_LIMIT or run >= MEMORY_REJECT_LIMIT:
                queued.extend(accepted)
                return self._close_edits(
                    session_id, operation, strikes, run, queued
                )
            refused.append(problem if strikes == 1 else f"{problem} {REPEATED}")
        queued.extend(accepted)
        if accepted:
            # Progress resets the run: a session that is landing changes is
            # using the tool, not looping in it, and should not be shut down
            # hours later for misses it recovered from. Safe to reset because
            # it is not the only bound — the repeat ledger below is never
            # reset, so alternating a good change with the *same* bad one
            # still hits the third strike, and alternating it with distinct
            # good ones fills the queue to its own limit instead.
            self._rejected_run[session_id] = 0
        return self._queue_report(len(operations), accepted, refused, queued)

    def _record_rejection(
        self, session_id: str, operation: MemoryOp
    ) -> tuple[int, int]:
        """Book one refusal: how often *this* operation has been refused this
        session, and how many refusals have gone by without one landing.

        Identity is the operation's four fields, compared exactly. The loop
        this bounds reissues the same bytes, and an exact key cannot punish a
        model for a genuinely new attempt — a shortened substring is a
        different operation and starts its own count, which is what the run
        limit is for.
        """
        key = (
            operation.op,
            operation.scope.value,
            operation.match,
            operation.text,
        )
        seen = self._rejected_ops.setdefault(session_id, {})
        seen[key] = seen.get(key, 0) + 1
        run = self._rejected_run.get(session_id, 0) + 1
        self._rejected_run[session_id] = run
        return seen[key], run

    def _close_edits(
        self,
        session_id: str,
        operation: MemoryOp,
        strikes: int,
        run: int,
        queued: list[MemoryOp],
    ) -> str:
        """Stop answering `memory` for this session, and say so once and for
        all.

        Deliberately not undone by /conclude: what ran out is the model's
        ability to address memory correctly, and reviewing the queue does not
        change that. The message is careful, for the same reason, not to
        promise that /conclude reopens the tool — it asks the model to hand
        the remaining work to the user, which is the one move that still
        works. Whatever *was* queued is listed, so nothing looks lost.
        """
        if strikes >= MEMORY_REPEAT_LIMIT:
            why = (
                f"Refused: the same operation has now been refused {strikes} "
                f"times in this session, unchanged — {_brief(operation.describe())}. "
                "Reissuing it cannot make it work."
            )
        else:
            why = (
                f"Refused: {run} memory operations have been refused in this "
                f"session without one being queued, which is the limit "
                f"({MEMORY_REJECT_LIMIT})."
            )
        message = (
            f"{why} Do not call this tool again in this session — tell the "
            "user in plain words what you wanted to save and leave it to them "
            "to run /conclude."
        )
        self._edits_closed[session_id] = message
        return f"{message}\n{self._waiting_line(queued)}"

    def _admit(
        self, operation: MemoryOp, projected: Profile, queued: list[MemoryOp]
    ) -> tuple[str, Profile]:
        """Why this operation cannot join the queue, or "" and the profile it
        would leave behind.

        The duplicate check comes first and is exact rather than semantic: a
        model reissuing byte-identical operations is the tightest form of the
        loop, and it is the one case where "you already did this" is both true
        and enough to stop. Everything else is delegated to `apply_batch`
        against the projection, so queue-time addressing and /conclude-time
        addressing cannot drift apart — and `resolve`'s refusal already names
        the scope's current entries, which is what lets the model correct
        itself in one step instead of bisecting its way to a shorter
        substring.
        """
        if len(queued) >= MEMORY_QUEUE_LIMIT:
            return (
                f"{_brief(operation.describe())} — the queue is full "
                f"({MEMORY_QUEUE_LIMIT}); ask the user to run /conclude."
            ), projected
        if operation in queued:
            return (
                f"{_brief(operation.describe())} — identical to a change "
                "already queued this session."
            ), projected
        try:
            # No cap here: the budget is a property of the final batch, and
            # `propose_flagged_edits` checks it against the profile the write
            # will actually land in. Refusing an add now because the batch
            # that frees room has not been issued yet would be a wall the
            # model could only retry against.
            result = apply_batch(projected, [operation])
        except MemoryOpError as e:
            return f"{_brief(operation.describe())} — {e}", projected
        if not result.applied:
            reason = result.skipped[0] if result.skipped else "no change"
            return f"{_brief(operation.describe())} — {reason}", projected
        return "", result.profile

    @staticmethod
    def _projected(loaded: Profile, operations: list[MemoryOp]) -> Profile:
        """The profile as the queue standing so far would leave it.

        Every queued operation resolved cleanly when it was queued, so this
        normally cannot raise; it still can if the user hand-edited the file
        in between, and then the file as it is now is the honest answer — the
        queue is re-checked whole at /conclude anyway, where a batch that no
        longer applies is dropped with a warning the user can see.
        """
        if not operations:
            return loaded
        try:
            return apply_batch(loaded, operations).profile
        except MemoryOpError:
            return loaded

    @staticmethod
    def _queue_report(
        asked: int,
        accepted: list[MemoryOp],
        refused: list[str],
        queued: list[MemoryOp],
    ) -> str:
        """What the model is told: what landed, what did not and why, and what
        is now waiting.

        The last part is what makes the call observable at all. Without it two
        calls that leave entirely different queues can read identically, and a
        tool whose result does not move when the state does is a tool the
        model can only guess about. It is kept to one summarized line per
        change, and truncated, because this text is paid for on every call.
        """
        if accepted and refused:
            lines = [f"Queued {len(accepted)} of {asked} memory change(s)."]
        elif accepted:
            lines = [f"Queued {len(accepted)} memory change(s)."]
        else:
            lines = [f"Queued none of the {asked} memory change(s) asked for."]
        lines += [f"NOT queued — {reason}" for reason in refused[:3]]
        if len(refused) > 3:
            lines.append(f"… and {len(refused) - 3} more not queued.")
        lines.append(MemoryService._waiting_line(queued))
        return "\n".join(lines)

    @staticmethod
    def _waiting_line(queued: list[MemoryOp]) -> str:
        """The one line that makes the call observable: what the queue holds
        now. Shared with the terminal refusals so that stopping the tool never
        also hides what the session already banked."""
        if not queued:
            return "Nothing is waiting for review."
        return (
            f"Waiting for the user's review at /conclude ({len(queued)}, "
            f"nothing saved yet): {_listing(queued)}"
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
        kept = self._apply(pending, flags)
        if kept:
            # What was just written is a number on the profiles screen, and
            # nothing else would have said so: `profile.rows` is sent when it
            # is asked for, and the only thing that asks is a client opening
            # (`emit_profiles`). Once per answer rather than once per approved
            # item — three memories are one change to one count.
            self.emit_profiles()
        return kept

    def _apply(self, pending: ProposalSet, flags: list[bool]) -> int:
        """The applier for the kind of set this is — see `KIND_*`."""
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
                self.emit_profiles()  # entries left the file for the archive
                self._notify(
                    f"Memory curation ({name}): {report.summary()} — "
                    f"archived entries are in {name}.archive.md"
                )
        return reports

    # ----------------------------------------------------- profile / skills

    def emit_profiles(self) -> None:
        """The profiles, whole — the answer to `profile.list`.

        Read here rather than derived by a front-end, which is what was
        happening: a picker was assembled out of `hello`'s profile plus
        whatever profiles the sidebar rows named, which misses every profile
        that has no session, and can carry neither the memory count nor the
        provenance because those live in files only the core reads.

        It lives on this service rather than on `CoreService` because it is
        not only an answer to a question. The row carries a *count*, and the
        count changes in here — an approved review, a hand-edited file, a
        curator pass — so each of those restates the listing. Before that, a
        profile that had just gained two memories went on saying "1 memory"
        until the next start: nothing but `profile.list` ever sent these rows,
        and nothing asked for them again.

        A broken profile file counts nothing rather than taking the listing
        down with it: the screen exists partly so that such a profile can be
        opened and fixed, and it cannot be opened from a screen that failed to
        draw.
        """
        counts = self._session_counts()
        rows = []
        for name in Profile.list_profiles():
            memories, copied_from = 0, ""
            try:
                profile = Profile.load(name)
            except Exception:
                logger.exception("could not read profile %s", name)
            else:
                memories = len(profile.memories)
                copied_from = profile.copied_from
            rows.append(
                ProfileRow(
                    name=name,
                    memories=memories,
                    sessions=counts.get(name, 0),
                    copied_from=copied_from,
                    # Two different questions: which profile a deleted one's
                    # sessions fall back to, and which one the core is running
                    # under right now (`protocol.ProfileRow`).
                    is_default=name == DEFAULT_PROFILE,
                    working=name == self._deps.profile,
                )
            )
        self._deps.emit(ProfileRows(rows=rows))

    def _session_counts(self) -> dict[str, int]:
        """How many conversations each profile has, or nothing at all.

        Read through the direct connection rather than the async seam, the way
        `core.backends` reads the same table: this is a synchronous restatement
        called from write paths that are themselves synchronous, and one
        `COUNT(*) GROUP BY` on the sessions table is not what `deps.db` exists
        to keep off the loop.

        A core built without a connection (a test with nothing but a fake LLM)
        counts nothing, and the row then says nothing about sessions rather
        than refusing to draw — the same bargain the memory count makes with a
        profile file it cannot read.
        """
        if self._deps.conn is None:
            return {}
        try:
            return SessionStore(self._deps.conn).counts_by_profile()
        except Exception:
            logger.exception("could not count sessions per profile")
            return {}

    def profile_body(self, name: str, kind: str) -> tuple[str, str]:
        """One editable profile file as text, and why it could not be read.

        The read half of :meth:`save_profile_memories` /
        :meth:`save_profile_archive`, and it belongs beside them because they
        are the pair that has to agree about *which bytes*: the save writes
        verbatim, so a read that returned anything but the file itself would
        turn every edit into a silent rewrite of what it failed to show.

        Hence the raw file rather than ``Profile.load(name).render()``, which
        is what a front-end was doing. A memory file the parser choked on is
        exactly the one someone opens an editor to fix, and rendering the
        parsed subset back would delete the lines it could not read — with the
        user believing they had just saved them.

        Returns ``(text, error)``. A file that is not there is "", not an
        error: a fresh profile has no archive yet, and refusing to open an
        editor over it would leave no way to write the first line. An error is
        for a name nothing answers to, or a file that exists and would not be
        read — both cases where an editor must refuse rather than edit blind.
        """
        if name not in Profile.list_profiles():
            return "", f"There is no profile called “{name}”."
        if kind == "archive":
            path = curator.archive_path(name)
        elif kind == "memories":
            path = Profile.path_for(name)
        else:  # pragma: no cover - the protocol only admits the two kinds
            return "", f"There is no “{kind}” to edit."
        return self._read(path)

    def skill_body(self, profile: str, name: str) -> tuple[str, str]:
        """One skill file the user owns, verbatim, and why not.

        The same two levels :meth:`delete_profile_skill` removes — this
        profile's and this project's — because they are the two a front-end
        draws as removable and offers to open (`SkillInfo.removable`). A
        shared or shipped skill is still refused: it is not this profile's to
        edit, and handing its body to an editor whose save would land in the
        profile's own directory would silently fork it.

        It was own-only for a milestone while the screens had already widened,
        so a skill created at the project level appeared on the profile's
        skills screen and answered Enter with "has no skill of its own"
        (specs/specs-ui-coverage.md §9.9).
        """
        # Project before profile on a name clash, matching load precedence and
        # `delete_profile_skill`: the file the user can see is the one they
        # mean to open.
        for skill in load_project_skills(project_root=self._project_root):
            if skill.name == name:
                return self._read(
                    skill_path(
                        name,
                        profile,
                        level="project",
                        project_root=self._project_root,
                    )
                )
        for skill in load_own_skills(profile):
            if skill.name == name:
                return self._read(skill_path(name, profile))
        return "", f"Profile “{profile}” has no skill “{name}” of its own."

    def own_skills(self, profile: str) -> list[Skill]:
        """The skills a profile may edit and delete — its own, never shared."""
        return load_own_skills(profile)

    @staticmethod
    def _read(path: Path) -> tuple[str, str]:
        """``(text, error)`` for one file. Missing is empty, not an error."""
        try:
            return path.read_text(), ""
        except FileNotFoundError:
            return "", ""
        except OSError as e:
            return "", f"Could not read {path.name}: {e.strerror or e}"
        except UnicodeDecodeError:
            # A file an editor cannot show is a file a verbatim save would
            # destroy; better to refuse than to hand back a lossy decode.
            return "", f"{path.name} is not text."

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
        # A hand edit is the other way the count moves, and whoever made it is
        # usually standing on the profiles screen that shows it.
        self.emit_profiles()

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

    def save_skill_file(
        self, profile: str, name: str, text: str, *, level: SkillLevel = "profile"
    ) -> None:
        """Persist a hand-edited skill file verbatim (front matter and body).
        The user owns the file; a parse problem is reported, never fatal.

        ``level`` is where it lands — the profile's own directory, the shared
        one every profile sees, or this project's. It defaults to the
        profile's, which is where an edited file came from and where a
        self-review patch goes, so the editor paths need no opinion about it.
        """
        path = skill_path(
            name, profile, level=level, project_root=self._project_root
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text.endswith("\n") else text + "\n")
        if profile == self._deps.profile:
            self.skills = load_skills(profile, project_root=self._project_root)
        self._notify(f"Saved skill “{name}”.")

    def delete_profile_skill(self, profile: str, name: str) -> None:
        """Delete one skill the user owns: this profile's, or this project's.

        The two removable levels, and only those — `delete_own_skill` will not
        touch `_shared/` or the shipped files, because removing one would
        silently change every other profile that sees it. The project's are
        here because a project skill is written from the creator like any
        other and would otherwise be listed as removable and refuse to go.
        """
        by_name = {s.name: s for s in load_own_skills(profile)}
        # Project shadows profile on a name clash, matching load precedence:
        # the file the user can see is the one they mean to remove.
        by_name.update(
            {
                s.name: s
                for s in load_project_skills(project_root=self._project_root)
            }
        )
        skill = by_name.get(name)
        if skill is None or not delete_own_skill(
            skill, profile, project_root=self._project_root
        ):
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
