"""TUI slash commands for skills (§5.1): /skill-creator, /skills-list,
/skill-remove — creating, listing, and removing per-profile skills."""

import json

import pytest
from textual.widgets import Input, ListView, TextArea

from hpca.llm import ChatResponse
from hpca.skills import load_own_skills, load_skills
from hpca.tui.app import ChatInput, HpcaApp
from hpca.tui.inspect_screen import InspectScreen
from hpca.tui.skill_screens import SkillCreatorScreen, SkillPickerScreen


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})


class FakeLLM:
    def __init__(self, outputs=()):
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


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


async def submit(app, pilot, text):
    """Type a slash command into the chat entry and send it."""
    if app.active_session is None:
        await app.start_new_session()
        await pilot.pause()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await pilot.pause()


async def create_skill(app, pilot, name, description, body):
    await submit(app, pilot, "/skill-creator")
    assert isinstance(app.screen, SkillCreatorScreen)
    app.screen.query_one("#skill-name", Input).value = name
    app.screen.query_one("#skill-description", Input).value = description
    app.screen.query_one("#skill-body", TextArea).text = body
    await pilot.press("escape")  # save is resolved on escape
    await pilot.pause()
    from hpca.tui.confirm_screen import ConfirmScreen

    if isinstance(app.screen, ConfirmScreen):
        await pilot.press("y")  # "Save skill?" -> yes
    await app.workers.wait_for_complete()
    await pilot.pause()


async def create_skill_at_level(app, pilot, name, description, body, level):
    """Like ``create_skill`` but pick a level (global / profile / project)."""
    from textual.widgets import RadioButton

    await submit(app, pilot, "/skill-creator")
    assert isinstance(app.screen, SkillCreatorScreen)
    app.screen.query_one("#skill-name", Input).value = name
    app.screen.query_one("#skill-description", Input).value = description
    app.screen.query_one("#skill-body", TextArea).text = body
    app.screen.query_one(f"#level-{level}", RadioButton).value = True
    await pilot.pause()
    await pilot.press("escape")  # save is resolved on escape
    await pilot.pause()
    from hpca.tui.confirm_screen import ConfirmScreen

    if isinstance(app.screen, ConfirmScreen):
        await pilot.press("y")  # "Save skill?" -> yes
    await app.workers.wait_for_complete()
    await pilot.pause()


GRILL_BODY = "Interview me relentlessly about every aspect of this plan."


class TestSkillCreator:
    async def test_creates_a_skill_tied_to_the_profile(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(
                app, pilot, "grilling", "Stress-test a plan", GRILL_BODY
            )
            own = load_own_skills(app.profile)
            assert [s.name for s in own] == ["grilling"]
            assert own[0].description == "Stress-test a plan"
            assert "Interview me" in own[0].body

    async def test_first_skill_enables_the_read_skill_tool(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            assert "read_skill" not in app._tools.names()
            await create_skill(app, pilot, "grilling", "d", GRILL_BODY)
            assert "read_skill" in app._tools.names()

    async def test_new_skill_is_in_the_next_system_prompt(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(app, pilot, "grilling", "Stress-test a plan", GRILL_BODY)
            # what the model is told about — surfaced in every mode, planning
            # included (the mode suffix is appended to this base prompt)
            assert "grilling" in app._render_system_prompt()

    async def test_empty_name_is_rejected(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/skill-creator")
            app.screen.query_one("#skill-body", TextArea).text = "some body"
            await pilot.press("escape")  # body but no name
            await pilot.pause()
            assert isinstance(app.screen, SkillCreatorScreen)  # not dismissed
            assert load_own_skills(app.profile) == []

    async def test_duplicate_name_is_refused(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(app, pilot, "grilling", "d", GRILL_BODY)
            await create_skill(app, pilot, "grilling", "again", "other body")
            assert len(load_own_skills(app.profile)) == 1


class TestSkillLevelSelection:
    """The creator lets the user pick where a new skill lives."""

    async def test_global_skill_is_visible_but_not_own(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill_at_level(
                app, pilot, "shared-one", "For everyone", GRILL_BODY, "global"
            )
            # Global skills are not the profile's "own" (not removable per-profile)
            assert load_own_skills(app.profile) == []
            # …but every profile can see them.
            assert "shared-one" in [s.name for s in load_skills(app.profile)]

    async def test_project_skill_lands_in_cwd(self, hpca_home, monkeypatch):
        monkeypatch.chdir(hpca_home)  # keep the project dir inside tmp_path
        from hpca.skills import load_project_skills

        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill_at_level(
                app, pilot, "proj-one", "Here only", GRILL_BODY, "project"
            )
            assert [
                s.name for s in load_project_skills(project_root=hpca_home)
            ] == ["proj-one"]
            assert load_own_skills(app.profile) == []

    async def test_project_skill_is_removable(self, hpca_home, monkeypatch):
        monkeypatch.chdir(hpca_home)
        from hpca.skills import load_project_skills

        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill_at_level(
                app, pilot, "proj-one", "d", GRILL_BODY, "project"
            )
            await submit(app, pilot, "/skill-remove")
            assert isinstance(app.screen, SkillPickerScreen)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("y")  # confirm removal
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert load_project_skills(project_root=hpca_home) == []


class TestSkillsList:
    async def test_lists_the_profiles_skills(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(app, pilot, "grilling", "Stress-test a plan", GRILL_BODY)
            await submit(app, pilot, "/skills-list")
            assert isinstance(app.screen, InspectScreen)
            body = app.screen.body_text()
            assert "grilling" in body
            assert "Stress-test a plan" in body

    async def test_no_skills_notifies_instead_of_opening(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/skills-list")
            assert not isinstance(app.screen, InspectScreen)


class TestSkillRemove:
    async def test_removes_the_chosen_skill(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(app, pilot, "grilling", "d", GRILL_BODY)
            await submit(app, pilot, "/skill-remove")
            assert isinstance(app.screen, SkillPickerScreen)
            await pilot.press("enter")  # pick the highlighted skill
            await pilot.pause()
            await pilot.press("y")  # confirm removal
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert load_own_skills(app.profile) == []

    async def test_cancel_keeps_the_skill(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(app, pilot, "grilling", "d", GRILL_BODY)
            await submit(app, pilot, "/skill-remove")
            assert isinstance(app.screen, SkillPickerScreen)
            await pilot.press("escape")  # back out
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert [s.name for s in load_own_skills(app.profile)] == ["grilling"]

    async def test_no_own_skills_notifies(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/skill-remove")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not isinstance(app.screen, SkillPickerScreen)
