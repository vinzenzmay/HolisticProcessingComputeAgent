"""TUI wiring for skills and struggle notes (milestone 12).

Reflection (memories, struggle notes and skill patches/new skills) is now
generated ONLY when the user runs ``/conclude``. There is no automatic
turn-counter review, no struggle-triggered review, and no pre-eviction review.
So every test that used to rely on a review firing on its own drives it by
submitting ``/conclude``.
"""

import json

import pytest

from hpca.agent.struggle import STRUGGLE_KIND
from hpca.llm import ChatResponse
from hpca.profiles import MemoryScope, Profile
from hpca.skills import load_own_skills, load_skills, skills_dir
from hpca.tui.app import ChatInput, HpcaApp
from hpca.tui.memory_screens import ReflectionScreen


def is_title_request(json_schema):
    """The app names a session by asking the model (§3 sessions column); that
    call is not one of the queued decisions."""
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})

class RecordingLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        self.calls.append(list(messages))
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


def tool_json(tool, **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def write_skill_file(content: str, name: str = "bam.md") -> None:
    skills_dir().mkdir(parents=True, exist_ok=True)
    (skills_dir() / name).write_text(content)


SKILL = """\
---
name: bam-subset
description: Subset a BAM file by region
triggers: [bam]
---

1. Index the BAM.
2. Run samtools view.
"""


async def submit(app, pilot, text, *, expect_modal=False):
    """Submit a chat message.

    With expect_modal, the agent worker parks on push_screen_wait until the
    user answers, so waiting for workers to finish would deadlock — pump the
    event loop instead and let the caller press a key.
    """
    if app.active_session is None:
        await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    if expect_modal:
        for _ in range(20):
            await pilot.pause()
    else:
        await app.workers.wait_for_complete()
        await pilot.pause()


class TestSkills:
    async def test_skills_not_listed_in_prompt_but_tool_registered(self, hpca_home):
        write_skill_file(SKILL)
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            assert "read_skill" in app._tools.names()
            await submit(app, pilot, "hello")
            system = llm.calls[0][0]["content"]
            # The list is kept out of the prompt: neither the skill's name nor
            # its description nor its body appears. Only a note that skills
            # exist and how to fetch one on demand (read_skill).
            assert "bam-subset" not in system
            assert "Subset a BAM file by region" not in system
            assert "samtools view" not in system
            assert "read_skill" in system

    async def test_read_skill_returns_body(self, hpca_home):
        write_skill_file(SKILL)
        llm = RecordingLLM(
            [tool_json("read_skill", name="bam-subset"), respond_json("following it")]
        )
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "subset my bam")
            texts = app.chat_log_texts()
            assert any("samtools view" in t for t in texts)

    async def test_shipped_skills_keep_the_tool_and_note_present(self, hpca_home):
        """The user has written no skills, but HPCA ships some — so read_skill
        is registered and the prompt still says skills exist."""
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            assert "read_skill" in app._tools.names()
            await submit(app, pilot, "hello")
            assert "read_skill" in llm.calls[0][0]["content"]


REVIEW_JSON = json.dumps(
    {
        "proposals": [
            {
                "kind": "struggle",
                "text": "Snakemake dry-runs fail when site profiles are involved.",
                "keywords": ["snakemake", "dry-run"],
            }
        ]
    }
)

MEMORY_JSON = json.dumps(
    {
        "proposals": [
            {
                "kind": "memory",
                "scope": "system-prompt",
                "text": "User pins conda environments by hash.",
            }
        ]
    }
)

NOTHING_JSON = json.dumps({"proposals": []})


class TestSelfReview:
    """Reflection now fires only on /conclude — there is no automatic review.
    Both a struggling and a clean conversation surface their learnings, but
    only once the user asks for them."""

    async def test_conclude_saves_struggle_on_approval(self, hpca_home):
        # turn 1 consumes respond_json; /conclude consumes REVIEW_JSON
        llm = RecordingLLM([respond_json("ok"), REVIEW_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run my snakemake workflow")
            await submit(app, pilot, "/conclude", expect_modal=True)
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            memories = Profile.load("default").memories
            assert len(memories) == 1
            assert memories[0].kind == STRUGGLE_KIND
            assert "keywords: snakemake, dry-run" in memories[0].text

    async def test_rejected_proposal_not_saved(self, hpca_home):
        llm = RecordingLLM([respond_json("ok"), REVIEW_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run my snakemake workflow")
            await submit(app, pilot, "/conclude", expect_modal=True)
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert Profile.load("default").memories == []

    async def test_clean_turn_does_not_review(self, hpca_home):
        """A normal turn triggers no review at all — nothing is proposed and
        nothing is saved until the user runs /conclude."""
        llm = RecordingLLM([respond_json("all good")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "hello")
            assert not isinstance(app.screen, ReflectionScreen)
            assert Profile.load("default").memories == []

    async def test_conclude_on_a_clean_session_proposes_and_saves(self, hpca_home):
        """A conversation that never tripped a failure still yields memories —
        the old struggle-only heuristic missed a user simply stating a
        preference; /conclude reviews the whole thing regardless."""
        llm = RecordingLLM([respond_json("noted"), MEMORY_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "I always pin conda envs by hash")
            assert not isinstance(app.screen, ReflectionScreen)
            await submit(app, pilot, "/conclude", expect_modal=True)
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            memories = Profile.load("default").memories
            assert len(memories) == 1
            assert memories[0].scope is MemoryScope.SYSTEM_PROMPT

    async def test_nothing_to_save_is_silent(self, hpca_home):
        # /conclude with an empty proposal list shows no dialog and saves nothing
        llm = RecordingLLM([respond_json("ok"), NOTHING_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "hello")
            await submit(app, pilot, "/conclude")
            assert not isinstance(app.screen, ReflectionScreen)
            assert Profile.load("default").memories == []


SKILL_PATCH_JSON = json.dumps(
    {
        "proposals": [
            {
                "kind": "skill_patch",
                "text": "Index the BAM before subsetting.",
                "skill_name": "bam-subset",
            }
        ]
    }
)

NEW_SKILL_JSON = json.dumps(
    {
        "proposals": [
            {
                "kind": "skill_new",
                "text": "Run fastqc then multiqc over the run directory.",
                "skill_name": "read-qc",
                "keywords": ["fastqc", "qc"],
            }
        ]
    }
)


class TestSkillLearning:
    async def test_patch_appends_to_the_profile_copy(self, hpca_home):
        write_skill_file(SKILL)
        llm = RecordingLLM([respond_json("ok"), SKILL_PATCH_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "subset a bam")
            await submit(app, pilot, "/conclude", expect_modal=True)
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            patched = [s for s in load_skills("default") if s.name == "bam-subset"][0]
            assert "Index the BAM before subsetting." in patched.body
            assert "samtools view" in patched.body  # original steps survive
            # written to the profile's own copy, not the shared original
            assert (skills_dir() / "default" / "bam-subset.md").exists()
            assert "Corrections" not in (skills_dir() / "bam.md").read_text()

    async def test_new_skill_created_when_enabled(self, hpca_home):
        llm = RecordingLLM([respond_json("ok"), NEW_SKILL_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run qc")
            await submit(app, pilot, "/conclude", expect_modal=True)
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            names = [s.name for s in load_skills("default")]
            assert "read-qc" in names
            assert "read_skill" in app._tools.names()  # tool now enabled

    async def test_new_skills_can_be_disabled(self, hpca_home):
        # with new skills disabled the reflection drops the skill_new proposal,
        # so /conclude finds nothing to keep and writes no skill
        llm = RecordingLLM([respond_json("ok"), NEW_SKILL_JSON])
        app = HpcaApp(llm=llm)
        app.settings.memory.propose_new_skills = False
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run qc")
            await submit(app, pilot, "/conclude")
            assert not isinstance(app.screen, ReflectionScreen)
            assert load_own_skills("default") == []


class TestFencedRecall:
    async def test_matching_struggle_is_fenced_into_the_model_message(
        self, hpca_home
    ):
        """Redesign Phase 1: the model gets recalled struggle notes as a
        fenced block on the API copy of the user message; the stored
        transcript keeps the clean text."""
        profile = Profile.load("default")
        profile.add_memory(
            "Snakemake dry-runs fail here.\nkeywords: snakemake, dry-run",
            scope=MemoryScope.RAG,
            kind=STRUGGLE_KIND,
        )
        profile.save()
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "please run my snakemake workflow")
            user = llm.calls[0][-1]
            assert user["role"] == "user"
            assert user["content"] == "please run my snakemake workflow"
            assert "<memory-context>" in user["api_content"]
            assert "Snakemake dry-runs fail here." in user["api_content"]
            assert "keywords:" not in user["api_content"]
            # the chat transcript shows only the clean message
            assert not any("<memory-context>" in t for t in app.chat_log_texts())

    async def test_non_matching_request_carries_no_memory_block(self, hpca_home):
        # The sidecar still rides (it carries the volatile environment facts),
        # but with nothing recalled it must hold no memory-context block.
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "hello")
            user = llm.calls[0][-1]
            assert "<memory-context>" not in user.get("api_content", "")
            assert user["content"] == "hello"
            # the volatile facts ride the tail, not the cacheable system prefix
            assert "<environment>" in user["api_content"]
            assert "Current date and time" not in llm.calls[0][0]["content"]


class TestRagRecall:
    """Redesign Phase 5: RAG memories are retrieved per request, not injected."""

    def write_rag(self, text, **kwargs):
        profile = Profile.load("default")
        profile.add_memory(text, scope=MemoryScope.RAG, **kwargs)
        profile.save()

    async def test_rag_is_not_in_the_system_prompt(self, hpca_home):
        self.write_rag("Deepvariant needs a GPU partition here.")
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "hello")
            assert "Deepvariant" not in llm.calls[0][0]["content"]

    async def test_matching_request_retrieves_it(self, hpca_home):
        self.write_rag("Deepvariant needs a GPU partition here.")
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run deepvariant on this sample")
            user = llm.calls[0][-1]
            assert "Deepvariant needs a GPU partition here." in user["api_content"]
            assert "<memory-context>" in user["api_content"]
            # the stored message stays the user's own words
            assert user["content"] == "run deepvariant on this sample"

    async def test_unrelated_request_retrieves_nothing(self, hpca_home):
        self.write_rag("Deepvariant needs a GPU partition here.")
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "what time is it")
            # sidecar carries env facts, but no RAG memory was retrieved
            user = llm.calls[0][-1]
            assert "Deepvariant" not in user.get("api_content", "")
            assert "<memory-context>" not in user.get("api_content", "")

    async def test_recall_is_visible_in_the_transcript(self, hpca_home):
        self.write_rag("Deepvariant needs a GPU partition here.")
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run deepvariant on this sample")
            texts = app.chat_log_texts()
            assert any("Deepvariant needs a GPU partition" in t for t in texts)
            # shown as recall, not as something the user said
            assert any(e.kind == "recall" for e in app._chat_entries)

    async def test_prefetch_respects_the_char_budget(self, hpca_home):
        for i in range(5):
            self.write_rag(f"bam handling note {i}: " + "x" * 400)
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "help me with this bam handling problem")
            block = llm.calls[0][-1]["api_content"]
            assert len(block) < 1200  # message + fence, budget is 800

    async def test_struggle_notes_from_review_land_in_rag(self, hpca_home):
        llm = RecordingLLM([respond_json("ok"), REVIEW_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run my snakemake workflow")
            await submit(app, pilot, "/conclude", expect_modal=True)
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            memories = Profile.load("default").memories
            assert memories[0].scope is MemoryScope.RAG  # situational: retrieved
