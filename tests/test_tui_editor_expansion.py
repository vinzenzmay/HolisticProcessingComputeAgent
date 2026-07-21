"""The memory editor reaches everything /conclude produces: the two memory
scopes (via the profile file), the RAG archive, and the profile's skills."""

import json

import pytest
from textual.widgets import ListView, TextArea

from hpca import curator
from hpca.llm import ChatResponse
from hpca.profiles import Memory, MemoryScope
from hpca.skills import Skill, load_own_skills, write_skill
from hpca.tui.app import HpcaApp
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.profiles_screen import (
    MemoryEditorScreen,
    ProfileSkillsScreen,
    ProfilesScreen,
)


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


class FakeLLM:
    def __init__(self, outputs=()):
        self._outputs = list(outputs)

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=json.dumps({"title": "a test session"}))
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def open_profiles_on(app, pilot, name="default"):
    await pilot.press("a")
    await pilot.pause()
    assert isinstance(app.screen, ProfilesScreen)
    listview = app.screen.query_one("#profiles-list", ListView)
    for i, item in enumerate(listview.children):
        if getattr(item, "data_profile", None) == name:
            listview.index = i
            break
    await pilot.pause()
    return listview


def a_skill(name="bam-subset"):
    return Skill(
        name=name,
        description="Subset a BAM file by region",
        triggers=["bam"],
        body="1. Index the BAM.\n2. Run samtools view.",
    )


class TestArchiveEditing:
    async def test_r_opens_the_archive_and_keeps_edits(self, hpca_home):
        curator.append_to_archive(
            "default",
            [Memory(text="an aged-out note", scope=MemoryScope.RAG, created="2020-01-01")],
        )
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await open_profiles_on(app, pilot)
            await pilot.press("r")
            await pilot.pause()
            assert isinstance(app.screen, MemoryEditorScreen)
            editor = app.screen.query_one("#memory-editor", TextArea)
            assert "an aged-out note" in editor.text
            editor.text = editor.text + "\nhand-added line\n"
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
            assert "hand-added line" in curator.archive_path("default").read_text()

    async def test_r_is_inert_on_the_new_profile_row(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            listview = await open_profiles_on(app, pilot)
            listview.index = 0  # the "(new profile)" row
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            assert isinstance(app.screen, ProfilesScreen)


class TestSkillEditing:
    async def test_s_lists_and_edits_a_skill(self, hpca_home):
        write_skill(a_skill(), "default")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await open_profiles_on(app, pilot)
            await pilot.press("s")
            for _ in range(6):
                await pilot.pause()
            assert isinstance(app.screen, ProfileSkillsScreen)
            skills_list = app.screen.query_one("#pskills-list", ListView)
            assert any(
                getattr(item, "data_skill", None) == "bam-subset"
                for item in skills_list.children
            )
            skills_list.focus()
            skills_list.index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, MemoryEditorScreen)
            editor = app.screen.query_one("#memory-editor", TextArea)
            assert "samtools view" in editor.text
            editor.text = editor.text + "\n3. Index the output.\n"
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            body = [s for s in load_own_skills("default") if s.name == "bam-subset"][0].body
            assert "Index the output." in body

    async def test_d_deletes_a_skill(self, hpca_home):
        write_skill(a_skill(), "default")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await open_profiles_on(app, pilot)
            await pilot.press("s")
            for _ in range(6):
                await pilot.pause()
            assert isinstance(app.screen, ProfileSkillsScreen)
            skills_list = app.screen.query_one("#pskills-list", ListView)
            assert any(
                getattr(item, "data_skill", None) == "bam-subset"
                for item in skills_list.children
            )
            skills_list.focus()
            skills_list.index = 0
            await pilot.pause()
            await pilot.press("d")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
            assert load_own_skills("default") == []
