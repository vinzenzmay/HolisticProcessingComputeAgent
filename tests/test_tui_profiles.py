"""Profiles & learnings screen (a): list, edit memories, add and delete."""

import json

import pytest
from textual.widgets import Input, ListView, TextArea

from hpca.llm import ChatResponse
from hpca.profiles import MemoryScope, Profile
from hpca.tui.app import HpcaApp
from hpca.tui.confirm_screen import ConfirmScreen
from hpca.tui.profiles_screen import MemoryEditorScreen, ProfilesScreen
from hpca.tui.rename_screen import RenameScreen


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


async def open_profiles(app, pilot):
    await pilot.press("a")
    await pilot.pause()
    assert isinstance(app.screen, ProfilesScreen)
    return app.screen.query_one("#profiles-list", ListView)


def profile_rows(screen):
    return [
        str(item.query_one("Label").content)
        for item in screen.query_one("#profiles-list", ListView).children
    ]


class TestOpening:
    async def test_a_opens_only_from_the_sessions_column(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.check_action("manage_profiles", ()) is True
            app._focus_column("chat")
            await pilot.pause()
            assert app.check_action("manage_profiles", ()) is False
            app._focus_column("sessions")
            await pilot.pause()
            await pilot.press("a")
            assert isinstance(app.screen, ProfilesScreen)

    async def test_lists_profiles_with_the_new_row_and_default_marked(self, hpca_home):
        Profile.create("alpha")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await open_profiles(app, pilot)
            rows = profile_rows(app.screen)
            assert rows[0] == "(new profile)"
            assert any("default" in r and "★" in r for r in rows)
            assert any(r.startswith("alpha") for r in rows)

    async def test_escape_returns_to_the_main_screen(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await open_profiles(app, pilot)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ProfilesScreen)


class TestEditMemories:
    async def test_enter_opens_the_memories_and_keeps_edits(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("The cluster is cubi.", scope=MemoryScope.SYSTEM_PROMPT)
        profile.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            profiles = await open_profiles(app, pilot)
            profiles.index = 1  # the default profile
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, MemoryEditorScreen)
            editor = app.screen.query_one("#memory-editor", TextArea)
            assert "The cluster is cubi." in editor.text
            editor.text = editor.text.replace(
                "The cluster is cubi.", "The cluster is cubi (Slurm 25)."
            )
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)  # keep changes?
            await pilot.press("y")
            await pilot.pause()
            saved = Profile.load("default")
            assert any("Slurm 25" in m.text for m in saved.memories)

    async def test_declining_keep_discards_edits(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("original memory", scope=MemoryScope.SYSTEM_PROMPT)
        profile.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            profiles = await open_profiles(app, pilot)
            profiles.index = 1
            await pilot.press("enter")
            await pilot.pause()
            editor = app.screen.query_one("#memory-editor", TextArea)
            editor.text = editor.text.replace("original memory", "tampered")
            await pilot.press("escape")
            await pilot.press("n")  # do not keep
            await pilot.pause()
            saved = Profile.load("default")
            assert any("original memory" in m.text for m in saved.memories)
            assert not any("tampered" in m.text for m in saved.memories)

    async def test_no_edits_does_not_ask_to_keep(self, hpca_home):
        Profile.load("default").save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            profiles = await open_profiles(app, pilot)
            profiles.index = 1
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("escape")  # nothing changed
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert isinstance(app.screen, ProfilesScreen)

    async def test_editing_the_active_profile_reaches_the_next_turn(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            profiles = await open_profiles(app, pilot)
            profiles.index = 1  # default is the active profile
            await pilot.press("enter")
            await pilot.pause()
            editor = app.screen.query_one("#memory-editor", TextArea)
            editor.text = (
                editor.text.rstrip() + "\n\n## [system-prompt]\n\nSTAR needs 40G.\n"
            )
            await pilot.press("escape")
            await pilot.press("y")
            await pilot.pause()
            assert "STAR needs 40G." in app.profile_memory.scope_text(
                MemoryScope.SYSTEM_PROMPT
            )


class TestAddProfile:
    async def test_enter_on_the_new_profile_row_prompts_and_creates(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            profiles = await open_profiles(app, pilot)
            profiles.index = 0  # "(new profile)"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, RenameScreen)
            app.screen.query_one(Input).value = "bam-work"
            await pilot.press("enter")
            await pilot.pause()
            assert "bam-work" in Profile.list_profiles()
            assert any(r.startswith("bam-work") for r in profile_rows(app.screen))

    async def test_a_bad_name_is_reported_and_nothing_is_created(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            profiles = await open_profiles(app, pilot)
            profiles.index = 0  # "(new profile)"
            await pilot.press("enter")
            await pilot.pause()
            app.screen.query_one(Input).value = "../escape"
            await pilot.press("enter")
            await pilot.pause()
            assert Profile.list_profiles() == ["default"]


class TestDeleteProfile:
    async def test_delete_reassigns_sessions_to_default(self, hpca_home):
        Profile.create("alpha")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            moved = app.session_store.create(profile="alpha", title="on alpha")
            profiles = await open_profiles(app, pilot)
            profiles.index = next(
                i
                for i, item in enumerate(profiles.children)
                if getattr(item, "data_profile", None) == "alpha"
            )
            await pilot.press("d")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
            assert "alpha" not in Profile.list_profiles()
            assert app.session_store.get(moved.session_id).profile == "default"

    async def test_default_cannot_be_deleted(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            profiles = await open_profiles(app, pilot)
            profiles.index = 1  # the default row
            assert profiles.check_action("delete_profile", ()) is False
            await pilot.press("d")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmScreen)
            assert "default" in Profile.list_profiles()

    async def test_delete_is_inert_on_the_new_profile_row(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            profiles = await open_profiles(app, pilot)
            profiles.index = 0  # "(new profile)"
            assert profiles.check_action("delete_profile", ()) is False


class TestDeleteGuards:
    async def test_a_profile_with_a_reply_in_progress_is_blocked(self, hpca_home):
        Profile.create("alpha")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            busy = app.session_store.create(profile="alpha", title="busy")
            from hpca.tui.app import TurnState

            # a turn on an alpha session is in flight
            app._turns[busy.session_id] = TurnState(session=busy)
            assert app.profile_delete_blocker("alpha") is not None
            assert "reply in progress" in app.profile_delete_blocker("alpha")

    async def test_a_profile_with_a_running_process_is_blocked(
        self, hpca_home, monkeypatch
    ):
        import hpca.tui.app as app_module

        Profile.create("alpha")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            session = app.session_store.create(profile="alpha", title="with proc")
            monkeypatch.setattr(
                app_module, "running_session_ids", lambda conn: {session.session_id}
            )
            blocker = app.profile_delete_blocker("alpha")
            assert blocker is not None
            assert "running sub-process" in blocker

    async def test_an_idle_profile_is_deletable(self, hpca_home, monkeypatch):
        import hpca.tui.app as app_module

        Profile.create("alpha")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            app.session_store.create(profile="alpha", title="idle")
            monkeypatch.setattr(app_module, "running_session_ids", lambda conn: set())
            assert app.profile_delete_blocker("alpha") is None


class RecordingLLM(FakeLLM):
    def __init__(self, outputs=()):
        super().__init__(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if not is_title_request(json_schema):
            self.calls.append(list(messages))
        return await super().chat(messages, json_schema=json_schema, **kwargs)


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


async def pick_new_session(app, pilot, profile):
    """(new session) -> picker -> choose the given profile."""
    from hpca.tui.profiles_screen import ProfilePickerScreen

    sessions_list = app.query_one("#sessions-list", ListView)
    sessions_list.focus()
    sessions_list.index = 0
    await pilot.press("enter")
    await pilot.pause()
    assert isinstance(app.screen, ProfilePickerScreen)
    picker = app.screen.query_one("#picker-list", ListView)
    picker.index = next(
        i
        for i, item in enumerate(picker.children)
        if getattr(item, "data_profile", None) == profile
    )
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


class TestSessionProfilePicker:
    async def test_the_session_runs_under_the_chosen_profile(self, hpca_home):
        Profile.create("alpha")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pick_new_session(app, pilot, "alpha")
            assert app.active_session.profile == "alpha"
            assert app.profile == "alpha"  # working profile follows the session
            from hpca.tui.app import TopBar

            assert "alpha" in app.query_one(TopBar).render_text()

    async def test_the_chosen_profiles_memories_reach_the_prompt(self, hpca_home):
        alpha = Profile.create("alpha")
        alpha.add_memory(
            "Alpha-only fact: use scratch volume B.", scope=MemoryScope.SYSTEM_PROMPT
        )
        alpha.save()
        llm = RecordingLLM([respond_json("ok")])
        app = HpcaApp(llm=llm)
        async with app.run_test(size=(120, 40)) as pilot:
            await pick_new_session(app, pilot, "alpha")
            from hpca.tui.app import ChatInput

            chat_input = app.query_one("#chat-input", ChatInput)
            chat_input.focus()
            chat_input.text = "hello"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            system = llm.calls[0][0]
            assert system["role"] == "system"
            assert "Alpha-only fact" in system["content"]

    async def test_sessions_of_all_profiles_stay_listed_with_tags(self, hpca_home):
        Profile.create("alpha")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            app.session_store.create(profile="default", title="on default")
            app.session_store.create(profile="alpha", title="on alpha")
            await app._reload_sessions()
            await pilot.pause()
            rows = [
                str(item.query_one("Label").content)
                for item in app.query_one("#sessions-list", ListView).children
            ]
            assert any(r == "on default" for r in rows)  # default: no tag
            assert any("on alpha" in r and "alpha" in r for r in rows)

    async def test_opening_a_session_switches_to_its_profile(self, hpca_home):
        alpha = Profile.create("alpha")
        alpha.add_memory("Alpha-only fact.", scope=MemoryScope.SYSTEM_PROMPT)
        alpha.save()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            session = app.session_store.create(profile="alpha", title="on alpha")
            await app.open_session(session)
            await pilot.pause()
            assert app.profile == "alpha"
            assert "Alpha-only fact." in app.profile_memory.scope_text(
                MemoryScope.SYSTEM_PROMPT
            )

    async def test_creating_a_profile_inside_the_picker_uses_it(self, hpca_home):
        from hpca.tui.profiles_screen import ProfilePickerScreen

        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            sessions_list = app.query_one("#sessions-list", ListView)
            sessions_list.focus()
            sessions_list.index = 0
            await pilot.press("enter")
            await pilot.pause()
            picker = app.screen.query_one("#picker-list", ListView)
            picker.index = len(picker.children) - 1  # "(new profile)"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, RenameScreen)
            app.screen.query_one(Input).value = "fresh"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert "fresh" in Profile.list_profiles()
            assert app.active_session.profile == "fresh"

    async def test_reused_untouched_session_is_retagged(self, hpca_home):
        Profile.create("alpha")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await pick_new_session(app, pilot, "default")
            first = app.active_session
            await pick_new_session(app, pilot, "alpha")
            # same empty session, now under the newly chosen profile
            assert app.active_session.session_id == first.session_id
            assert app.session_store.get(first.session_id).profile == "alpha"
            assert len(app.session_store.list_all()) == 1


class TestCopyProfile:
    """A base profile forked per specialism: the copy inherits everything the
    original learned, then the two accumulate separately."""

    def base_with_memories(self, name="base"):
        profile = Profile.create(name)
        profile.add_memory("Cluster is cubi.", scope=MemoryScope.SYSTEM_PROMPT)
        profile.add_memory("The user prefers R.", scope=MemoryScope.SYSTEM_PROMPT)
        profile.save()
        return profile

    def row_index(self, profiles, name):
        return next(
            i
            for i, item in enumerate(profiles.children)
            if getattr(item, "data_profile", None) == name
        )

    async def copy_via_ui(self, app, pilot, source, new_name):
        profiles = await open_profiles(app, pilot)
        profiles.index = self.row_index(profiles, source)
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, RenameScreen)
        app.screen.query_one(Input).value = new_name
        await pilot.press("enter")
        await pilot.pause()

    async def test_copy_inherits_the_memories(self, hpca_home):
        self.base_with_memories()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await self.copy_via_ui(app, pilot, "base", "variants")
            assert "variants" in Profile.list_profiles()
            texts = [m.text for m in Profile.load("variants").memories]
            assert texts == ["Cluster is cubi.", "The user prefers R."]

    async def test_provenance_shown_in_the_list(self, hpca_home):
        self.base_with_memories()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await self.copy_via_ui(app, pilot, "base", "variants")
            rows = profile_rows(app.screen)
            assert any("copied from base" in row for row in rows)

    async def test_the_two_diverge(self, hpca_home):
        self.base_with_memories()
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await self.copy_via_ui(app, pilot, "base", "variants")

            copy = Profile.load("variants")
            copy.add_memory("Deepvariant needs a GPU.", scope=MemoryScope.SYSTEM_PROMPT)
            copy.save()

            base_texts = [m.text for m in Profile.load("base").memories]
            assert "Deepvariant needs a GPU." not in base_texts
            assert "The user prefers R." in base_texts  # the shared base

    async def test_the_default_is_copyable_though_not_deletable(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            profiles = await open_profiles(app, pilot)
            profiles.index = self.row_index(profiles, "default")
            assert profiles.check_action("copy_profile", ()) is True
            assert profiles.check_action("delete_profile", ()) is False

    async def test_copy_is_inert_on_the_new_profile_row(self, hpca_home):
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            profiles = await open_profiles(app, pilot)
            profiles.index = 0  # "(new profile)"
            assert profiles.check_action("copy_profile", ()) is False

    async def test_a_duplicate_name_is_refused(self, hpca_home):
        self.base_with_memories()
        Profile.create("taken")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await self.copy_via_ui(app, pilot, "base", "taken")
            # the existing profile is untouched, not overwritten
            assert Profile.load("taken").memories == []

    async def test_copying_carries_the_profiles_own_skills(self, hpca_home):
        from hpca.skills import load_own_skills, skills_dir

        self.base_with_memories()
        (skills_dir() / "base").mkdir(parents=True, exist_ok=True)
        (skills_dir() / "base" / "a.md").write_text(
            "---\nname: align\n---\nthe procedure"
        )
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)) as pilot:
            await self.copy_via_ui(app, pilot, "base", "variants")
            assert [s.name for s in load_own_skills("variants")] == ["align"]

    async def test_deleting_a_profile_removes_its_skills(self, hpca_home):
        from hpca.skills import skills_dir

        Profile.create("doomed")
        (skills_dir() / "doomed").mkdir(parents=True, exist_ok=True)
        (skills_dir() / "doomed" / "a.md").write_text("---\nname: x\n---\nbody")
        app = HpcaApp(llm=FakeLLM())
        async with app.run_test(size=(120, 40)):
            app.delete_profile("doomed")
            assert not (skills_dir() / "doomed").exists()
