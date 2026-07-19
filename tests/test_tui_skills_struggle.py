"""TUI wiring for skills and struggle notes (milestone 12)."""

import json

import pytest
from textual.widgets import ListView

from hpca.agent.struggle import STRUGGLE_KIND
from hpca.llm import ChatResponse
from hpca.profiles import Profile
from hpca.skills import load_skills, skills_dir
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
    async def test_skills_listed_in_prompt_and_tool_registered(self, hpca_home):
        write_skill_file(SKILL)
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            assert "read_skill" in app._tools.names()
            await submit(app, pilot, "hello")
            system = llm.calls[0][0]["content"]
            assert "bam-subset: Subset a BAM file by region" in system
            # only the summary, never the body (§ prompt budget)
            assert "samtools view" not in system

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

    async def test_no_skills_no_tool_no_section(self, hpca_home):
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            assert "read_skill" not in app._tools.names()
            await submit(app, pilot, "hello")
            assert "skills" not in llm.calls[0][0]["content"].lower()


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

NOTHING_JSON = json.dumps({"proposals": []})


class TestSelfReview:
    """Redesign Phase 4: reviews fire on a struggling turn AND on a counter,
    so learnings from conversations that went fine are captured too."""

    async def test_failed_turn_reviews_and_saves_on_approval(self, hpca_home):
        # decision fails -> graph reports failure -> review kicks in at once
        llm = RecordingLLM(["garbage"] * 4 + [REVIEW_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run my snakemake workflow", expect_modal=True)
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            memories = Profile.load("default").memories
            assert len(memories) == 1
            assert memories[0].kind == STRUGGLE_KIND
            assert "keywords: snakemake, dry-run" in memories[0].text

    async def test_rejected_proposal_not_saved(self, hpca_home):
        llm = RecordingLLM(["garbage"] * 4 + [REVIEW_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run my snakemake workflow", expect_modal=True)
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert Profile.load("default").memories == []

    async def test_clean_turn_does_not_review_before_the_interval(self, hpca_home):
        llm = RecordingLLM([respond_json("all good")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "hello")
            assert not isinstance(app.screen, ReflectionScreen)
            assert Profile.load("default").memories == []

    async def test_counter_triggers_review_on_a_clean_session(self, hpca_home):
        """The gap the old struggle-only heuristic left: a user stating a
        preference never trips a failure marker."""
        llm = RecordingLLM([respond_json("noted")] * 2 + [REVIEW_JSON])
        app = HpcaApp(llm=llm)
        app.settings.memory.review_interval = 2
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "first message")
            assert not isinstance(app.screen, ReflectionScreen)
            await submit(app, pilot, "second message", expect_modal=True)
            assert isinstance(app.screen, ReflectionScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(Profile.load("default").memories) == 1

    async def test_nothing_to_save_is_silent(self, hpca_home):
        llm = RecordingLLM([respond_json("ok")] + [NOTHING_JSON])
        app = HpcaApp(llm=llm)
        app.settings.memory.review_interval = 1
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "hello")
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
        llm = RecordingLLM([respond_json("ok")] + [SKILL_PATCH_JSON])
        app = HpcaApp(llm=llm)
        app.settings.memory.review_interval = 1
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "subset a bam", expect_modal=True)
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
        llm = RecordingLLM([respond_json("ok")] + [NEW_SKILL_JSON])
        app = HpcaApp(llm=llm)
        app.settings.memory.review_interval = 1
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run qc", expect_modal=True)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            names = [s.name for s in load_skills("default")]
            assert "read-qc" in names
            assert "read_skill" in app._tools.names()  # tool now enabled

    async def test_new_skills_can_be_disabled(self, hpca_home):
        llm = RecordingLLM([respond_json("ok")] + [NEW_SKILL_JSON])
        app = HpcaApp(llm=llm)
        app.settings.memory.review_interval = 1
        app.settings.memory.propose_new_skills = False
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run qc")
            assert load_skills("default") == []


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
            tier=2,
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

    async def test_non_matching_request_has_no_sidecar(self, hpca_home):
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "hello")
            assert "api_content" not in llm.calls[0][-1]
