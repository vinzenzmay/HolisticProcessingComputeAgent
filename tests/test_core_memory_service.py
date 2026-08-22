"""Tests for hpca.core.memory_service: the memory domain without a front-end.

Everything the service says comes out of ``deps.emit``, so the assertions are
on a plain list of protocol events rather than on a running app — which is the
point of the extraction, and the reason these tests need neither Textual nor a
pilot to make a memory get written.

No LLM: the two sub-agent calls (`/memorize`, `/conclude`) go through a fake
that hands back the JSON the real prompts are schema-constrained to produce.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

import pytest

from hpca import curator
from hpca.agent.struggle import STRUGGLE_KIND
from hpca.config import Settings
from hpca.core.deps import CoreDeps
from hpca.core.memory_service import MemoryService
from hpca.db import connect, init_db
from hpca.llm import ChatResponse
from hpca.memory_index import MemoryIndex
from hpca.memory_ops import MemoryOp
from hpca.profiles import MemoryScope, Profile
from hpca.protocol import MemoryProposals, Notify, TurnActivity
from hpca.sessions import Session, SessionStore
from hpca.skills import Skill, load_skills, write_skill

SP = MemoryScope.SYSTEM_PROMPT
RAG = MemoryScope.RAG
BACKEND = "qwen3-6b"


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


class FakeLLM:
    """The sub-agent backend. Replies are queued by the test; an unqueued call
    fails loudly, because a service that reaches for the model when it should
    not is exactly the bug these tests are here to catch."""

    def __init__(self) -> None:
        self.replies: list[str] = []
        self.calls: list[list[dict]] = []

    def reply(self, payload: str) -> None:
        self.replies.append(payload)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("unexpected LLM call")
        return ChatResponse(content=self.replies.pop(0))


@dataclass
class Harness:
    service: MemoryService
    deps: CoreDeps
    events: list
    conn: sqlite3.Connection
    llm: FakeLLM
    busy: set[str] = field(default_factory=set)

    def notices(self) -> list[str]:
        return [e.text for e in self.events if isinstance(e, Notify)]

    def said(self, fragment: str) -> bool:
        return any(fragment in text for text in self.notices())

    def proposals(self) -> list[MemoryProposals]:
        return [e for e in self.events if isinstance(e, MemoryProposals)]


@pytest.fixture
def harness(hpca_home, tmp_path):
    conn = connect(tmp_path / "core.db")
    init_db(conn)
    events: list = []
    busy: set[str] = set()
    llm = FakeLLM()

    async def db(fn):
        # The real runner hands sqlite to its own thread; a test needs the
        # seam, not the thread.
        return fn(conn)

    deps = CoreDeps(
        settings=Settings(),
        app_dir=hpca_home,
        db=db,
        emit=events.append,
        conn=conn,
    )
    service = MemoryService(
        deps,
        index=MemoryIndex(conn),
        llm_for=lambda label, session: llm,
        backend_name=lambda session: BACKEND,
        busy_profiles=lambda: busy,
    )
    return Harness(
        service=service, deps=deps, events=events, conn=conn, llm=llm, busy=busy
    )


@pytest.fixture
def session(harness) -> Session:
    return SessionStore(harness.conn).create(profile="default", title="a session")


def stored(profile: str = "default") -> list[str]:
    return [m.text for m in Profile.load(profile).memories]


def write_memory(text: str, *, scope=SP, kind: str = "", profile: str = "default"):
    loaded = Profile.load(profile)
    memory = loaded.add_memory(text, scope=scope, backend=BACKEND, kind=kind)
    loaded.save()
    return memory


def reflections(*items: dict) -> str:
    return json.dumps({"proposals": list(items)})


class TestSnapshotFreeze:
    """The system-prompt prefix has to stay byte-stable for the backend's
    prefix cache, so a profile is read once and reused until something
    deliberately invalidates it."""

    def test_the_same_view_serves_every_turn(self, harness):
        first = harness.service.snapshot("default")
        assert harness.service.snapshot("default") is first

    def test_a_write_on_disk_is_not_picked_up_silently(self, harness):
        harness.service.snapshot("default")
        write_memory("STAR needs 40G on this cluster.")
        assert stored() == ["STAR needs 40G on this cluster."]
        assert [m.text for m in harness.service.snapshot("default").memories] == []

    def test_invalidation_is_what_reloads_it(self, harness):
        harness.service.snapshot("default")
        write_memory("STAR needs 40G on this cluster.")
        harness.service.invalidate("default")
        assert [m.text for m in harness.service.snapshot("default").memories] == [
            "STAR needs 40G on this cluster."
        ]

    def test_reloading_rebuilds_the_retrieval_index(self, harness):
        """The markdown file is the source of truth and may have been edited by
        hand, so the index is derived again whenever the view is."""
        harness.service.snapshot("default")
        write_memory("mamba resolves faster than conda", scope=RAG)
        harness.service.invalidate("default")
        harness.service.snapshot("default")
        hits = MemoryIndex(harness.conn).search("mamba", profile="default")
        assert [hit.text for hit in hits] == ["mamba resolves faster than conda"]

    def test_profiles_are_frozen_apart(self, harness):
        harness.service.snapshot("default")
        harness.service.snapshot("other")
        write_memory("only in other", profile="other")
        harness.service.invalidate("other")
        assert [m.text for m in harness.service.snapshot("other").memories] == [
            "only in other"
        ]
        assert harness.service.snapshot("default").memories == []


class TestRecall:
    def test_struggle_notes_and_rag_hits_both_come_back(self, harness):
        write_memory(
            "STAR alignment ran out of memory\nkeywords: star, alignment",
            scope=RAG,
            kind=STRUGGLE_KIND,
        )
        write_memory("mamba resolves faster than conda", scope=RAG)
        memory = harness.service.snapshot("default")
        lines = harness.service.recall_lines(
            "the star alignment failed and mamba was slow", memory, "default"
        )
        assert any("STAR alignment ran out of memory" in line for line in lines)
        assert any("mamba resolves faster" in line for line in lines)

    def test_a_note_matched_twice_is_recalled_once(self, harness):
        """A struggle note lives in the RAG scope, so the keyword matcher and
        the index can both return it; the model must not be told twice."""
        write_memory(
            "STAR alignment ran out of memory\nkeywords: star, alignment",
            scope=RAG,
            kind=STRUGGLE_KIND,
        )
        memory = harness.service.snapshot("default")
        lines = harness.service.recall_lines(
            "the star alignment failed", memory, "default"
        )
        assert sum("STAR alignment ran out" in line for line in lines) == 1

    def test_the_keywords_line_is_for_the_matcher_not_the_model(self, harness):
        write_memory(
            "STAR alignment ran out of memory\nkeywords: star, alignment",
            scope=RAG,
            kind=STRUGGLE_KIND,
        )
        memory = harness.service.snapshot("default")
        lines = harness.service.recall_lines("star", memory, "default")
        assert "keywords:" not in lines[0]

    def test_nothing_matching_recalls_nothing(self, harness):
        write_memory("mamba resolves faster than conda", scope=RAG)
        memory = harness.service.snapshot("default")
        assert harness.service.recall_lines("hello", memory, "default") == []


class TestStruggleWarning:
    def test_a_matching_request_warns_the_user(self, harness):
        write_memory(
            "STAR ran out of memory\nkeywords: star", scope=RAG, kind=STRUGGLE_KIND
        )
        matches = harness.service.warn_about_struggles("run star again")
        assert len(matches) == 1
        assert harness.said("I have struggled with this before")

    def test_the_caller_can_take_the_matches_without_the_toast(self, harness):
        """A background turn wants its recall, not an interruption on a screen
        showing something else."""
        write_memory(
            "STAR ran out of memory\nkeywords: star", scope=RAG, kind=STRUGGLE_KIND
        )
        matches = harness.service.warn_about_struggles("run star again", warn=False)
        assert len(matches) == 1
        assert harness.notices() == []


class TestWriteBudget:
    """A full system-prompt scope refuses new writes until the user condenses
    it; injection never truncates, and RAG has no budget at all."""

    def fill(self):
        write_memory("x " * 6000)  # over the 2400-token cap

    def test_a_blocked_write_is_reported_not_performed(self, harness):
        self.fill()
        assert harness.service.write_blocked(SP, "a new learning") is True
        assert harness.said("memory NOT saved")

    def test_room_available_allows_the_write(self, harness):
        assert harness.service.write_blocked(SP, "a short site note") is False
        assert harness.notices() == []

    def test_rag_is_retrieved_so_it_has_no_budget(self, harness):
        self.fill()
        assert harness.service.write_blocked(RAG, "x" * 10000) is False

    def test_over_budget_is_only_reachable_by_hand_and_is_reported(self, harness):
        self.fill()
        assert harness.service.check_memory_caps() is True
        assert harness.said("over its budget")

    def test_a_profile_within_budget_stays_quiet(self, harness):
        assert harness.service.check_memory_caps() is False
        assert harness.notices() == []


class TestFlaggedEdits:
    """The `memory` tool flags; nothing is written until /conclude."""

    def add_op(self, text="STAR needs 40G here."):
        return MemoryOp(op="add", scope=SP, text=text)

    def test_flagging_writes_nothing_and_says_so(self, harness, session):
        answer = harness.service.queue_edits(session.session_id, [self.add_op()])
        assert "Nothing is saved yet" in answer
        assert stored() == []

    def test_an_empty_batch_is_not_queued(self, harness, session):
        assert harness.service.queue_edits(session.session_id, []) == "Nothing to flag."
        assert harness.service.pending_edits(session.session_id) == []

    def test_the_queue_survives_being_offered_for_review(self, harness, session):
        """An emitted set may never be answered — the front-end can die between
        the two — so the queue is only cleared by the answer."""
        harness.service.queue_edits(session.session_id, [self.add_op()])
        harness.service.propose_flagged_edits(session)
        assert len(harness.service.pending_edits(session.session_id)) == 1

    def test_applying_writes_the_batch_and_clears_the_queue(self, harness, session):
        harness.service.queue_edits(session.session_id, [self.add_op()])
        harness.service.propose_flagged_edits(session)
        assert harness.service.apply_answered(session.session_id, [True]) == 1
        assert stored() == ["STAR needs 40G here."]
        assert harness.service.pending_edits(session.session_id) == []

    def test_rejecting_discards_them(self, harness, session):
        harness.service.queue_edits(session.session_id, [self.add_op()])
        harness.service.propose_flagged_edits(session)
        assert harness.service.apply_answered(session.session_id, [False]) == 0
        assert stored() == []
        assert harness.service.pending_edits(session.session_id) == []
        assert harness.said("Discarded the flagged memory changes")

    def test_a_batch_is_approved_whole(self, harness, session):
        """Half of a trade — free an entry to fit another — is a state nobody
        chose, so a partial answer applies nothing."""
        harness.service.queue_edits(
            session.session_id, [self.add_op("one"), self.add_op("two")]
        )
        harness.service.propose_flagged_edits(session)
        assert harness.service.apply_answered(session.session_id, [True, False]) == 0
        assert stored() == []

    def test_nothing_flagged_proposes_nothing(self, harness, session):
        assert harness.service.propose_flagged_edits(session) is None
        assert harness.proposals() == []

    def test_an_unapplicable_batch_is_reported_and_dropped(self, harness, session):
        harness.service.queue_edits(
            session.session_id, [MemoryOp(op="remove", scope=SP, match="absent")]
        )
        assert harness.service.propose_flagged_edits(session) is None
        assert harness.said("Flagged memory not applied")
        assert harness.service.pending_edits(session.session_id) == []

    def test_a_hand_edit_since_the_batch_was_read_wins(self, harness, session):
        """Rewriting from a stale copy would silently discard what the user
        typed into the file, so the write is refused and backed up instead."""
        harness.service.queue_edits(session.session_id, [self.add_op()])
        harness.service.propose_flagged_edits(session)
        write_memory("added by hand while the review was open")
        assert harness.service.apply_answered(session.session_id, [True]) == 0
        assert stored() == ["added by hand while the review was open"]
        assert harness.said("changed on disk")


class TestReviewRoundTrip:
    """Producing proposals and applying an answered set are two calls, because
    between them the answer crosses a socket (§4.4)."""

    TWO = reflections(
        {"kind": "memory", "scope": "system-prompt", "text": "STAR needs 40G."},
        {"kind": "memory", "scope": "rag", "text": "The cluster is called cubi."},
    )

    async def test_the_proposals_are_emitted_and_written_by_nobody_yet(
        self, harness, session
    ):
        harness.llm.reply(self.TWO)
        pending = await harness.service.review_conversation(session, [])
        assert pending is not None
        event = harness.proposals()[-1]
        assert event.session_id == session.session_id
        assert [p.text for p in event.proposals] == [
            "STAR needs 40G.",
            "The cluster is called cubi.",
        ]
        assert stored() == []

    async def test_applying_writes_exactly_the_approved_ones(self, harness, session):
        harness.llm.reply(self.TWO)
        await harness.service.review_conversation(session, [])
        assert harness.service.apply_answered(session.session_id, [True, False]) == 1
        assert stored() == ["STAR needs 40G."]

    async def test_an_answer_that_never_came_is_not_an_approval(
        self, harness, session
    ):
        harness.llm.reply(self.TWO)
        await harness.service.review_conversation(session, [])
        assert harness.service.apply_answered(session.session_id, []) == 0
        assert stored() == []

    async def test_the_set_is_answered_once(self, harness, session):
        harness.llm.reply(self.TWO)
        await harness.service.review_conversation(session, [])
        harness.service.apply_answered(session.session_id, [True, True])
        assert harness.service.apply_answered(session.session_id, [True, True]) == 0
        assert len(stored()) == 2

    async def test_the_set_is_held_until_it_is_answered(self, harness, session):
        """The same reason a parked decision is: a review that outlives the
        front-end showing it must still be answerable afterwards."""
        harness.llm.reply(self.TWO)
        await harness.service.review_conversation(session, [])
        assert harness.service.pending_proposals(session.session_id) is not None
        harness.service.apply_answered(session.session_id, [False, False])
        assert harness.service.pending_proposals(session.session_id) is None

    async def test_a_struggle_note_is_stored_as_rag_with_its_keywords(
        self, harness, session
    ):
        harness.llm.reply(
            reflections(
                {
                    "kind": "struggle",
                    "scope": "system-prompt",  # overridden: struggles are RAG
                    "text": "STAR needs a lot of memory.",
                    "keywords": ["star", "alignment"],
                }
            )
        )
        await harness.service.review_conversation(session, [])
        harness.service.apply_answered(session.session_id, [True])
        memory = Profile.load("default").memories[0]
        assert memory.scope is RAG
        assert memory.kind == STRUGGLE_KIND
        assert "keywords: star, alignment" in memory.text

    async def test_an_approved_write_invalidates_the_frozen_view(
        self, harness, session
    ):
        harness.service.snapshot("default")
        harness.llm.reply(self.TWO)
        await harness.service.review_conversation(session, [])
        harness.service.apply_answered(session.session_id, [True, False])
        assert [m.text for m in harness.service.snapshot("default").memories] == [
            "STAR needs 40G."
        ]

    async def test_a_full_scope_refuses_an_approved_memory(self, harness, session):
        write_memory("x " * 6000)
        harness.llm.reply(self.TWO)
        await harness.service.review_conversation(session, [])
        assert harness.service.apply_answered(session.session_id, [True, False]) == 0
        assert harness.said("memory NOT saved")

    async def test_the_review_reports_a_failing_model_rather_than_raising(
        self, harness, session
    ):
        harness.llm.reply("not json at all")
        harness.llm.reply("still not json")
        assert await harness.service.review_conversation(session, []) is None
        assert harness.said("/conclude failed")

    async def test_the_session_is_told_what_the_core_is_doing(
        self, harness, session
    ):
        """One activity event in and one out — the spinner the app used to draw
        from `_backend_working`, without asking whether anything is on screen."""
        harness.llm.reply(self.TWO)
        await harness.service.review_conversation(session, [])
        activity = [e.activity for e in harness.events if isinstance(e, TurnActivity)]
        assert activity == ["reviewing conversation", ""]

    async def test_a_skill_patch_lands_in_the_profiles_own_copy(
        self, harness, session
    ):
        write_skill(
            Skill(name="align", description="d", triggers=[], body="step one"),
            "default",
        )
        harness.llm.reply(
            reflections(
                {
                    "kind": "skill_patch",
                    "text": "pass --outSAMtype BAM",
                    "skill_name": "align",
                }
            )
        )
        await harness.service.review_conversation(session, [])
        assert harness.service.apply_answered(session.session_id, [True]) == 1
        body = next(s for s in load_skills("default") if s.name == "align").body
        assert "pass --outSAMtype BAM" in body

    async def test_a_patch_to_a_skill_that_is_gone_is_reported(
        self, harness, session
    ):
        harness.llm.reply(
            reflections(
                {"kind": "skill_patch", "text": "fix", "skill_name": "missing"}
            )
        )
        await harness.service.review_conversation(session, [])
        assert harness.service.apply_answered(session.session_id, [True]) == 0
        assert harness.said("No skill named")


class TestMemorize:
    NOTE = json.dumps(
        {
            "proposals": [
                {"scope": "system-prompt", "kind": "fact", "text": "STAR needs 40G."}
            ]
        }
    )

    async def test_the_note_and_the_conversation_reach_the_model(
        self, harness, session
    ):
        harness.llm.reply(self.NOTE)
        await harness.service.propose_from_note(
            session,
            [{"role": "user", "content": "my STAR job was killed"}],
            "that was a memory limit",
        )
        prompt = harness.llm.calls[-1][-1]["content"]
        assert "my STAR job was killed" in prompt
        assert "that was a memory limit" in prompt

    async def test_approving_keeps_it_and_says_how_many(self, harness, session):
        harness.llm.reply(self.NOTE)
        await harness.service.propose_from_note(session, [], "keep this")
        assert harness.service.apply_answered(session.session_id, [True]) == 1
        assert stored() == ["STAR needs 40G."]
        assert harness.said("Kept 1 of 1 proposed memories")

    async def test_rejecting_saves_nothing(self, harness, session):
        harness.llm.reply(self.NOTE)
        await harness.service.propose_from_note(session, [], "forget this")
        assert harness.service.apply_answered(session.session_id, [False]) == 0
        assert stored() == []

    async def test_the_model_proposing_nothing_is_said_plainly(
        self, harness, session
    ):
        harness.llm.reply(json.dumps({"proposals": []}))
        assert await harness.service.propose_from_note(session, [], "nothing") is None
        assert harness.said("proposed no memories")

    async def test_a_hand_edit_is_merged_not_clobbered(self, harness, session):
        """The proposals were formed against one state of the file and land in
        whatever it holds when the answer arrives."""
        harness.llm.reply(self.NOTE)
        await harness.service.propose_from_note(session, [], "keep this")
        write_memory("added by hand meanwhile")
        harness.service.apply_answered(session.session_id, [True])
        assert stored() == ["added by hand meanwhile", "STAR needs 40G."]


class TestCurator:
    """Ageing RAG entries out runs at startup, at most once every few days."""

    def old_note(self):
        memory = write_memory("an old struggle note", scope=RAG)
        loaded = Profile.load("default")
        loaded.memories[-1].created = (date.today() - timedelta(days=200)).isoformat()
        loaded.save()
        return memory

    def test_it_runs_and_archives_when_due(self, harness):
        self.old_note()
        assert harness.service.run_curator_if_due()
        assert stored() == []
        assert curator.archive_path("default").exists()

    def test_it_does_not_run_again_within_the_interval(self, harness):
        self.old_note()
        harness.service.run_curator_if_due()
        write_memory("another old note", scope=RAG)
        assert harness.service.run_curator_if_due() == {}
        assert stored() == ["another old note"]

    def test_a_zero_interval_disables_it(self, harness):
        self.old_note()
        harness.deps.settings.memory.curator_interval_days = 0
        assert harness.service.run_curator_if_due() == {}
        assert not curator.archive_path("default").exists()

    def test_an_archived_profile_is_not_left_frozen(self, harness):
        self.old_note()
        harness.service.snapshot("default")
        harness.service.run_curator_if_due()
        assert harness.service.snapshot("default").memories == []


class TestProfileLifecycle:
    async def test_deletion_is_refused_while_a_turn_is_in_flight(self, harness):
        harness.busy.add("lab")
        blocker = await harness.service.profile_delete_blocker("lab")
        assert blocker is not None and "reply in progress" in blocker

    async def test_deletion_is_refused_while_a_sub_process_runs(self, harness):
        session = SessionStore(harness.conn).create(profile="lab")
        harness.conn.execute(
            "INSERT INTO processes (pid, session_id, name, state) "
            "VALUES (?, ?, ?, 'running')",
            (os.getpid(), session.session_id, "a script"),
        )
        harness.conn.commit()
        blocker = await harness.service.profile_delete_blocker("lab")
        assert blocker is not None and "sub-process" in blocker

    async def test_an_idle_profile_can_go(self, harness):
        SessionStore(harness.conn).create(profile="lab")
        assert await harness.service.profile_delete_blocker("lab") is None

    async def test_deleting_takes_its_learnings_with_it(self, harness):
        Profile.create("lab")
        write_memory("only the lab knows this", scope=RAG, profile="lab")
        write_skill(
            Skill(name="labskill", description="d", triggers=[], body="b"), "lab"
        )
        harness.service.snapshot("lab")
        session = SessionStore(harness.conn).create(profile="lab")
        await harness.service.delete_profile("lab")
        assert not Profile.path_for("lab").exists()
        assert load_skills("lab") == load_skills("nobody")
        assert MemoryIndex(harness.conn).search("lab knows", profile="lab") == []
        moved = SessionStore(harness.conn).get(session.session_id)
        assert moved.profile == "default"

    def test_creating_rejects_a_name_that_would_escape_the_directory(self, harness):
        assert harness.service.create_profile("../etc") is not None
        assert harness.service.create_profile("lab") is None
        assert "lab" in Profile.list_profiles()

    def test_duplicating_carries_the_learnings_over(self, harness):
        Profile.create("lab")
        write_memory("STAR needs 40G here.", profile="lab")
        assert harness.service.duplicate_profile("lab", "lab2") is None
        assert stored("lab2") == ["STAR needs 40G here."]
        assert harness.said("They diverge from here")

    def test_duplicating_an_absent_profile_is_refused(self, harness):
        assert harness.service.duplicate_profile("ghost", "copy") is not None

    def test_saving_hand_edited_memories_reports_problems_but_keeps_them(
        self, harness
    ):
        harness.service.save_profile_memories("default", "no front matter here")
        assert harness.said("Saved with problems")
        assert Profile.path_for("default").read_text()

    def test_saving_memories_unfreezes_the_profile(self, harness):
        harness.service.snapshot("default")
        harness.service.save_profile_memories(
            "default", "---\nname: default\n---\n\n## [system-prompt]\n\nedited\n"
        )
        assert [m.text for m in harness.service.snapshot("default").memories] == [
            "edited"
        ]

    def test_an_emptied_archive_file_is_removed(self, harness):
        harness.service.save_profile_archive("default", "some archived block")
        assert curator.archive_path("default").exists()
        harness.service.save_profile_archive("default", "   ")
        assert not curator.archive_path("default").exists()

    def test_a_skill_file_is_saved_verbatim(self, harness):
        harness.service.save_skill_file("default", "align", "---\nname: align\n---\nb")
        assert any(s.name == "align" for s in harness.service.skills)

    def test_deleting_an_absent_skill_is_reported(self, harness):
        harness.service.delete_profile_skill("default", "ghost")
        assert harness.said("No skill “ghost” to delete")

    def test_deleting_a_skill_refreshes_the_working_list(self, harness):
        write_skill(
            Skill(name="align", description="d", triggers=[], body="b"), "default"
        )
        harness.service.refresh_skills()
        harness.service.delete_profile_skill("default", "align")
        assert not any(s.name == "align" for s in harness.service.skills)


class TestWorkingProfile:
    def test_switching_makes_the_new_profile_the_default_one(self, harness):
        Profile.create("lab")
        harness.service.set_working_profile("lab")
        assert harness.deps.profile == "lab"
        assert harness.service.working_memory.name == "lab"

    def test_switching_rereads_the_file(self, harness):
        """A switch is an explicit refresh point: the user may have edited that
        profile since it was last read."""
        Profile.create("lab")
        harness.service.snapshot("lab")
        write_memory("edited between switches", profile="lab")
        harness.service.set_working_profile("lab")
        assert [m.text for m in harness.service.working_memory.memories] == [
            "edited between switches"
        ]

    def test_a_broken_profile_file_is_reported_on_the_switch(self, harness):
        Profile.path_for("lab").parent.mkdir(parents=True, exist_ok=True)
        Profile.path_for("lab").write_text("## nonsense heading\n\ntext\n")
        harness.service.set_working_profile("lab")
        assert harness.said("Profile file has problems")

    def test_switching_to_the_current_profile_is_a_no_op(self, harness):
        first = harness.service.working_memory
        harness.service.set_working_profile("default")
        assert harness.service.working_memory is first


class TestHeadless:
    def test_the_service_does_not_import_the_front_end(self):
        """The rule `hpca.core` exists to enforce (see its __init__): a runtime
        that can reach for a widget is a runtime that only runs in the UI.

        `textual` is checked alongside `hpca.ui` because the dependency is gone
        as of M9 and this is one of the places that would notice it returning.
        """
        code = (
            "import sys, hpca.core.memory_service; "
            "leaked = sorted(m for m in sys.modules "
            "if m.startswith('textual') or m.startswith('hpca.ui')); "
            "assert not leaked, leaked"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
