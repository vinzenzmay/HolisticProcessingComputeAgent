"""TUI wiring for skills and struggle notes (milestone 12)."""

import json

import pytest
from textual.widgets import Input

from hpca.agent.struggle import STRUGGLE_KIND
from hpca.llm import ChatResponse
from hpca.profiles import Profile
from hpca.skills import skills_dir
from hpca.tui.app import HpcaApp
from hpca.tui.memory_screens import MemoryProposalScreen


class RecordingLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
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


def write_skill(content: str, name: str = "bam.md") -> None:
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
    chat_input = app.query_one("#chat-input", Input)
    chat_input.focus()
    chat_input.value = text
    await pilot.press("enter")
    if expect_modal:
        for _ in range(20):
            await pilot.pause()
    else:
        await app.workers.wait_for_complete()
        await pilot.pause()


class TestSkills:
    async def test_skills_listed_in_prompt_and_tool_registered(self, hpca_home):
        write_skill(SKILL)
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
        write_skill(SKILL)
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


STRUGGLE_JSON = json.dumps(
    {
        "note": "Snakemake dry-runs fail when site profiles are involved.",
        "keywords": ["snakemake", "dry-run"],
    }
)


class TestStruggleNotes:
    async def test_failed_turn_proposes_note_and_saves_on_approval(self, hpca_home):
        # decision fails -> graph reports failure -> reflection kicks in
        llm = RecordingLLM(["garbage"] * 4 + [STRUGGLE_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run my snakemake workflow", expect_modal=True)
            assert isinstance(app.screen, MemoryProposalScreen)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            memories = Profile.load("default").memories
            assert len(memories) == 1
            assert memories[0].kind == STRUGGLE_KIND
            assert "keywords: snakemake, dry-run" in memories[0].text

    async def test_rejected_note_not_saved(self, hpca_home):
        llm = RecordingLLM(["garbage"] * 4 + [STRUGGLE_JSON])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "run my snakemake workflow", expect_modal=True)
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert Profile.load("default").memories == []

    async def test_clean_turn_proposes_nothing(self, hpca_home):
        llm = RecordingLLM([respond_json("all good")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "hello")
            assert not isinstance(app.screen, MemoryProposalScreen)
            assert Profile.load("default").memories == []

    async def test_warns_up_front_on_matching_request(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory(
            "Snakemake dry-runs fail here.\nkeywords: snakemake, dry-run",
            tier=2,
            kind=STRUGGLE_KIND,
        )
        profile.save()
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)):
            matches = app.warn_about_struggles("please run my snakemake workflow")
            assert len(matches) == 1
            assert app.warn_about_struggles("align a bam file") == []
