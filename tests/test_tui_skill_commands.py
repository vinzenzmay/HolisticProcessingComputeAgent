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
from tests.conftest import wait_for_screen


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


def is_draft_request(json_schema):
    """The /skill-creator draft call — the only schema asking for a body."""
    return bool(json_schema) and "body" in (json_schema.get("properties") or {})


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


class DraftingLLM(FakeLLM):
    """A backend that also answers the /skill-creator draft call — with a
    draft, or by falling over, depending on what the test is about."""

    def __init__(self, draft=None, outputs=(), fail=False):
        super().__init__(outputs)
        self._draft = draft or {}
        self._fail = fail
        self.draft_prompts = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_draft_request(json_schema):
            self.draft_prompts.append(list(messages))
            if self._fail:
                raise RuntimeError("backend down")
            return ChatResponse(content=json.dumps(self._draft))
        return await super().chat(messages, json_schema=json_schema, **kwargs)


NOTEBOOK_DRAFT = {
    "name": "Jupyter Runs",
    "description": "Start a notebook run and watch it to completion",
    "body": "1. Submit with papermill.\n2. Poll squeue until it clears.",
}


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

    async def test_the_read_skill_tool_is_there_from_the_start(self, hpca_home):
        """HPCA ships skills, so the tool is registered before the user writes
        one — and a new skill does not disturb it."""
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            assert "read_skill" in app._tools.names()
            await create_skill(app, pilot, "grilling", "d", GRILL_BODY)
            assert "read_skill" in app._tools.names()

    async def test_skill_list_stays_out_of_the_system_prompt(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(app, pilot, "grilling", "Stress-test a plan", GRILL_BODY)
            prompt = app._render_system_prompt()
            # The skill exists but is never enumerated in the prompt — neither
            # its name nor its description. Only a note that skills exist and
            # how to fetch one (read_skill).
            assert "grilling" not in prompt
            assert "Stress-test a plan" not in prompt
            assert "read_skill" in prompt

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


class TestSkillCreatorDraft:
    """"/skill-creator <what it should do>" drafts first: the model fills the
    same form, the user still edits it and still confirms the save."""

    async def test_the_form_opens_pre_filled(self, hpca_home):
        llm = DraftingLLM(NOTEBOOK_DRAFT)
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(
                app,
                pilot,
                "/skill-creator a skill that starts and monitors jupyter runs",
            )
            screen = await wait_for_screen(app, pilot, SkillCreatorScreen)
            # The name is normalised into something invocable as /<skill>.
            assert screen.query_one("#skill-name", Input).value == "jupyter-runs"
            assert (
                screen.query_one("#skill-description", Input).value
                == "Start a notebook run and watch it to completion"
            )
            assert "papermill" in screen.query_one("#skill-body", TextArea).text

    async def test_the_pre_filled_draft_saves_like_any_other_skill(self, hpca_home):
        app = HpcaApp(llm=DraftingLLM(NOTEBOOK_DRAFT))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/skill-creator monitor jupyter runs")
            await wait_for_screen(app, pilot, SkillCreatorScreen)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("y")  # it still asks before writing
            await app.workers.wait_for_complete()
            await pilot.pause()
            own = load_own_skills(app.profile)
            assert [s.name for s in own] == ["jupyter-runs"]
            assert "papermill" in own[0].body

    async def test_the_request_and_the_conversation_reach_the_drafter(self, hpca_home):
        llm = DraftingLLM(NOTEBOOK_DRAFT, outputs=[respond_json("use papermill")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "how do I run a notebook headless?")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await submit(app, pilot, "/skill-creator turn that into a skill")
            await wait_for_screen(app, pilot, SkillCreatorScreen)
            prompt = _user_message(llm.draft_prompts[0])["content"]
            assert "turn that into a skill" in prompt
            assert "run a notebook headless" in prompt  # the conversation

    async def test_the_spinner_names_the_wait(self, hpca_home):
        """Drafting is a real generation; the user is told what it is for."""
        import asyncio

        from hpca.tui.app import WorkingIndicator

        release = asyncio.Event()

        class SlowLLM(DraftingLLM):
            async def chat(self, messages, *, json_schema=None, **kwargs):
                if is_draft_request(json_schema):
                    await release.wait()
                return await super().chat(messages, json_schema=json_schema, **kwargs)

        app = HpcaApp(llm=SlowLLM(NOTEBOOK_DRAFT))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/skill-creator monitor jupyter runs")
            spinner = app.query_one(WorkingIndicator)
            assert "drafting a skill" in spinner._frame_text()
            release.set()
            await wait_for_screen(app, pilot, SkillCreatorScreen)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            # …and it is gone once the form has been dealt with.
            assert not app.query(WorkingIndicator)

    async def test_a_bare_command_drafts_nothing(self, hpca_home):
        """The old behaviour is untouched: no argument, no backend call."""
        llm = DraftingLLM(NOTEBOOK_DRAFT)
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/skill-creator")
            assert isinstance(app.screen, SkillCreatorScreen)
            assert llm.draft_prompts == []
            assert app.screen.query_one("#skill-name", Input).value == ""
            assert app.screen.query_one("#skill-body", TextArea).text == ""

    async def test_a_failed_draft_still_opens_the_form(self, hpca_home):
        """A backend that will not draft must not eat the command."""
        app = HpcaApp(llm=DraftingLLM(fail=True))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/skill-creator monitor jupyter runs")
            screen = await wait_for_screen(app, pilot, SkillCreatorScreen)
            assert screen.query_one("#skill-name", Input).value == ""

    async def test_an_empty_pre_filled_form_can_still_be_abandoned(self, hpca_home):
        """Clearing every field and pressing escape cancels, draft or not."""
        app = HpcaApp(llm=DraftingLLM(NOTEBOOK_DRAFT))
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/skill-creator monitor jupyter runs")
            screen = await wait_for_screen(app, pilot, SkillCreatorScreen)
            screen.query_one("#skill-name", Input).value = ""
            screen.query_one("#skill-description", Input).value = ""
            screen.query_one("#skill-body", TextArea).text = ""
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not isinstance(app.screen, SkillCreatorScreen)
            assert load_own_skills(app.profile) == []


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


def _user_message(messages):
    """The user message the model was handed, sidecar and all."""
    return next(m for m in reversed(messages) if m["role"] == "user")


class TestSkillSlashInvocation:
    """Typing "/<skill> …" runs a normal turn with that skill's procedure
    forced into the model's copy of the message (not the stored transcript)."""

    async def test_skill_shows_in_the_command_menu(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(app, pilot, "grilling", "Stress-test a plan", GRILL_BODY)
            names = [name for name, _ in app._matching_commands("grill")]
            assert "grilling" in names
            # empty needle lists everything, skills included
            assert "grilling" in [n for n, _ in app._matching_commands("")]

    async def test_builtin_command_wins_a_name_clash(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(app, pilot, "skills-list", "shadowed", GRILL_BODY)
            # The built-in /skills-list is not treated as a skill invocation.
            assert app._slash_skill("/skills-list") is None
            # …and it appears once in the menu, not twice.
            assert [n for n, _ in app._matching_commands("skills-list")] == [
                "skills-list"
            ]

    async def test_invoking_a_skill_forces_its_procedure(self, hpca_home):
        llm = FakeLLM(outputs=[respond_json("grilled")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(app, pilot, "grilling", "Stress-test a plan", GRILL_BODY)
            await submit(app, pilot, "/grilling review my plan")
            await app.workers.wait_for_complete()
            await pilot.pause()
            user = _user_message(llm.calls[-1])
            # Transcript keeps the raw command; the model sees the request plus
            # the skill body on the sidecar, without the "/grilling" prefix.
            assert user["content"] == "/grilling review my plan"
            assert "Interview me relentlessly" in user["api_content"]
            assert "review my plan" in user["api_content"]
            assert not user["api_content"].startswith("/grilling")

    async def test_plan_is_invocable_out_of_the_box(self, hpca_home):
        """A shipped skill needs no setup: "/plan …" on a fresh install hands
        the model HPCA's own planning procedure."""
        llm = FakeLLM(outputs=[respond_json("planning")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/plan the trash browser")
            await app.workers.wait_for_complete()
            await pilot.pause()
            user = _user_message(llm.calls[-1])
            assert user["content"] == "/plan the trash browser"
            assert "specs.md" in user["api_content"]
            assert "the trash browser" in user["api_content"]

    async def test_unknown_slash_is_still_an_error(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/nope do a thing")
            await pilot.pause()
            # Not a skill and not a built-in: no turn started, no crash.
            assert app._slash_skill("/nope do a thing") is None


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

    async def test_lists_the_shipped_skills_when_the_user_has_none(self, hpca_home):
        """A fresh install is never empty: HPCA's own skills are there, tagged
        so they are not mistaken for something the user wrote."""
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(app, pilot, "/skills-list")
            assert isinstance(app.screen, InspectScreen)
            body = app.screen.body_text()
            assert "plan  (built-in)" in body
            assert "grillme  (built-in)" in body

    async def test_own_skills_are_untagged(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await create_skill(app, pilot, "grilling", "Stress-test a plan", GRILL_BODY)
            await submit(app, pilot, "/skills-list")
            assert "• grilling\n" in app.screen.body_text()


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
