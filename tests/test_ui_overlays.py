"""Tests for hpca.ui.overlays: the screens that draw over the rows.

Each overlay is an independent render/handle pair, so most of these construct
the app and press the key that opens it — which is also the only thing that
proves the key is wired.

The claims come from specs-ui-acceptance.md: "Profiles", "Backends", "Thinking
effort", "Memory", "Skills and self-review", and the config-editor bullets
under "Navigating the entry". The M7 screens are the settings editor, the
profiles list and its three children, thinking, switch-LLM, inspect, the memory
review, manage-LLMs and the backend form.
"""

import pytest

from hpca.ui.app import CHAT, SESSIONS, RowUI
from hpca.ui.demo import build, sample_catalog
from hpca.ui.overlays import (
    KEEP_CHANGES,
    PREVIEW_CHARS,
    BackendFormOverlay,
    ConfigOverlay,
    HelpOverlay,
    InspectOverlay,
    LlmOverlay,
    MemoryReviewOverlay,
    ProfilesOverlay,
    PromptOverlay,
    RewindOverlay,
    SkillsOverlay,
    SwitchLlmOverlay,
    TextEditOverlay,
    ThinkingOverlay,
)
from hpca.ui.state import (
    BackendInfo,
    CopyProfile,
    CreateProfile,
    DeleteProfile,
    DeleteSkill,
    ProfileInfo,
    Proposal,
    ResolveMemory,
    SaveProfile,
    SaveSettings,
    SaveSkill,
    SetBackend,
    SetThinking,
    SkillInfo,
)
from tests.ui_harness import frame, on_own_message, plain, recorded, widths

HELP_MENTIONS = ["m", "a", "c", "→", "←", "^u", "^← ^→", "shift-← →", "^del"]
WIDTHS = [80, 100, 137]


def opened(key: str, width: int = 120, height: int = 40, *, focus=SESSIONS):
    """The app with one screen open, from the row the key belongs to."""
    ui = recorded(build())
    ui.focus = focus
    ui.handle(key, width, height)
    return ui


def screen(ui: RowUI, width: int = 120, height: int = 40) -> str:
    return "\n".join(frame(ui, width, height))


def press(ui: RowUI, *keys: str, width: int = 120, height: int = 40) -> RowUI:
    for key in keys:
        ui.handle(key, width, height)
    return ui


def type_text(ui: RowUI, text: str) -> RowUI:
    return press(ui, *text)


def sent(ui: RowUI, kind: type) -> list:
    return [x for x in ui.intents if isinstance(x, kind)]


def profiles() -> list[ProfileInfo]:
    return [
        ProfileInfo(
            name="hpc",
            memories=12,
            loaded=True,
            working=True,
            text="scratch is /scratch/proj\n",
            archive="## [rag] old\n",
            skills=[SkillInfo("merge-vcfs", "how to merge shards", "body\n")],
        ),
        ProfileInfo(name="default", memories=0, default=True, loaded=True),
        ProfileInfo(
            name="writing", memories=5, copied_from="hpc", loaded=True, text="x\n"
        ),
    ]


def catalog() -> list[BackendInfo]:
    return [
        BackendInfo(
            label="qwen3",
            model="qwen3-27b-fp8",
            base_url="http://a/v1",
            context=112000,
            active=True,
            reachable=True,
        ),
        BackendInfo(
            label="llama", model="llama-3.3-70b", base_url="http://b/v1",
            reachable=False,
        ),
        BackendInfo(
            label="locked", model="secret", base_url="http://c/v1", needs_key=True
        ),
    ]


# --------------------------------------------------------------- the key list


class TestHelpOverlay:
    def test_question_mark_opens_help(self):
        assert isinstance(opened("?").overlay, HelpOverlay)

    @pytest.mark.parametrize("key", HELP_MENTIONS)
    def test_help_mentions_the_key(self, key: str):
        # The list is longer than a 40-row terminal, so both ends are read.
        ui = opened("?")
        seen = screen(ui)
        ui.handle("end", 120, 40)
        seen += screen(ui)
        assert key in seen

    def test_help_scrolls_back_to_the_top(self):
        ui = press(opened("?"), "end", "home")
        assert ui.overlay.offset == 0

    def test_any_other_key_closes_help(self):
        assert press(opened("?"), "x").overlay is None

    def test_the_arrows_scroll_rather_than_close(self):
        ui = press(opened("?"), "down")
        assert isinstance(ui.overlay, HelpOverlay)
        assert ui.overlay.offset == 1


# ------------------------------------------------------------- the rewind


class TestRewindOverlay:
    def test_enter_opens_the_rewind(self):
        ui = build()
        on_own_message(ui)
        press(ui, "enter")
        assert isinstance(ui.overlay, RewindOverlay)

    def test_it_offers_all_three(self):
        ui = build()
        on_own_message(ui)
        seen = screen(press(ui, "enter"))
        assert "fork the session from here" in seen
        assert "roll this conversation back to here" in seen
        assert "copy it into the message box" in seen

    def test_the_footer_names_all_three(self):
        ui = build()
        on_own_message(ui)
        press(ui, "enter")
        foot = plain(ui.render(160, 40)[-1])
        for pair in ("f fork", "r roll back", "c / enter copy", "esc cancel"):
            assert pair in foot

    def test_a_key_it_has_no_answer_for_leaves_it_open(self):
        ui = build()
        on_own_message(ui)
        assert isinstance(press(ui, "enter", "z").overlay, RewindOverlay)

    def test_esc_cancels_and_changes_nothing(self):
        ui = build()
        index = on_own_message(ui)
        press(ui, "enter", "esc")
        assert ui.overlay is None
        assert ui.chat.items[index].kind == "user"
        assert ui.input.text() == ""

    def test_long_messages_are_cut_to_a_preview_and_say_so(self):
        shown = "\n".join(plain(x) for x in RewindOverlay("x" * 900, 0).render(80, 20))
        assert shown.count("x") <= PREVIEW_CHARS + 10
        assert "…" in shown


# ------------------------------------------------- the config editor (item 26)


class TestConfigEditor:
    """specs-ui-acceptance.md, "Navigating the entry", the `c` bullets."""

    def test_c_opens_the_config_editor(self):
        assert isinstance(opened("c").overlay, ConfigOverlay)

    def test_it_is_not_offered_in_the_chat_column(self):
        assert opened("c", focus=CHAT).overlay is None

    def test_nor_is_it_in_the_footer_there(self):
        ui = build()
        ui.focus = CHAT
        assert " c config" not in plain(ui.render(160, 40)[-1])

    def test_it_is_prefilled_with_the_current_settings_json(self):
        assert "local_cache" in screen(opened("c"))

    def test_and_numbers_the_lines_because_a_json_error_names_one(self):
        assert "  1 {" in screen(opened("c"))

    def test_escape_with_no_changes_closes_without_asking(self):
        ui = press(opened("c"), "esc")
        assert ui.overlay is None
        assert sent(ui, SaveSettings) == []

    def test_escape_with_changes_asks(self):
        ui = opened("c")
        ui.overlay.editor.set_text('{"a": 1}')
        press(ui, "esc")
        assert isinstance(ui.overlay, ConfigOverlay), "it stays up to ask"
        assert KEEP_CHANGES in screen(ui)

    def test_and_saves_on_yes(self):
        ui = opened("c")
        ui.overlay.editor.set_text('{"a": 1}')
        press(ui, "esc", "y")
        assert ui.overlay is None
        assert sent(ui, SaveSettings)[-1].text == '{"a": 1}'

    def test_and_discards_on_no(self):
        ui = opened("c")
        ui.overlay.editor.set_text('{"a": 1}')
        press(ui, "esc", "n")
        assert ui.overlay is None
        assert sent(ui, SaveSettings) == []

    def test_invalid_json_shows_an_error_and_keeps_it_open(self):
        ui = opened("c")
        ui.overlay.editor.set_text("{oh no")
        press(ui, "esc")
        assert isinstance(ui.overlay, ConfigOverlay)
        assert "invalid JSON" in screen(ui)
        assert sent(ui, SaveSettings) == []

    def test_and_does_not_ask_the_keep_question_over_broken_json(self):
        # Asking would offer a "yes" that cannot be honoured.
        ui = opened("c")
        ui.overlay.editor.set_text("{oh no")
        press(ui, "esc")
        assert KEEP_CHANGES not in screen(ui)

    def test_invalid_values_show_an_error_and_keep_it_open(self):
        # Syntax is this module's check; whether the object is usable settings
        # is a validator handed in from outside (§3.1).
        ui = opened("c")
        ui.overlay._validate = lambda text: "invalid: llm.model — not accepted"
        ui.overlay.editor.set_text('{"llm": {"model": 3}}')
        press(ui, "esc")
        assert isinstance(ui.overlay, ConfigOverlay)
        assert "llm.model" in screen(ui)

    def test_a_fixed_file_can_then_be_kept(self):
        ui = opened("c")
        ui.overlay.editor.set_text("{oh no")
        press(ui, "esc")
        ui.overlay.editor.set_text('{"ok": true}')
        press(ui, "esc", "y")
        assert sent(ui, SaveSettings)[-1].text == '{"ok": true}'


# ------------------------------------------------------- profiles (item 27)


class TestProfilesList:
    """specs-ui-acceptance.md, "Profiles"."""

    def test_a_opens_the_profiles_screen(self):
        assert isinstance(opened("a").overlay, ProfilesOverlay)

    def test_only_from_the_sessions_column(self):
        assert opened("a", focus=CHAT).overlay is None

    def test_escape_returns_to_the_main_screen(self):
        assert press(opened("a"), "esc").overlay is None

    def test_the_list_shows_the_profiles(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        seen = screen(press(ui, "a"))
        assert "hpc" in seen and "writing" in seen

    def test_and_a_new_profile_row(self):
        assert "(new profile)" in screen(opened("a"))

    def test_the_default_is_marked(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        assert "default ★" in screen(press(ui, "a"))

    def test_the_row_counts_memories(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        assert "12 memories" in screen(press(ui, "a"))

    def test_and_shows_provenance(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        assert "copied from hpc" in screen(press(ui, "a"))


class TestProfileMemories:
    def _open(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        return press(ui, "a", "enter")

    def test_enter_opens_a_profiles_memories_in_the_editor(self):
        ui = self._open()
        assert isinstance(ui.overlay, TextEditOverlay)
        assert "scratch is /scratch/proj" in ui.overlay.editor.text()

    def test_no_edits_means_no_question(self):
        ui = press(self._open(), "esc")
        assert isinstance(ui.overlay, ProfilesOverlay), "back to the list"
        assert sent(ui, SaveProfile) == []

    def test_edits_are_kept(self):
        ui = type_text(self._open(), "X")
        press(ui, "esc", "y")
        saved = sent(ui, SaveProfile)[-1]
        assert (saved.name, saved.kind) == ("hpc", "memories")
        assert "X" in saved.text

    def test_declining_keep_discards_them(self):
        ui = type_text(self._open(), "X")
        press(ui, "esc", "n")
        assert sent(ui, SaveProfile) == []

    def test_escaping_the_editor_lands_back_on_the_list(self):
        ui = press(self._open(), "esc")
        assert isinstance(ui.overlay, ProfilesOverlay)
        assert press(ui, "esc").overlay is None

    def test_r_opens_the_rag_archive(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        press(ui, "a", "r")
        assert isinstance(ui.overlay, TextEditOverlay)
        assert ui.overlay.kind == "archive"

    def test_and_keeps_its_edits(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        press(ui, "a", "r")
        type_text(ui, "Z")
        press(ui, "esc", "y")
        assert sent(ui, SaveProfile)[-1].kind == "archive"

    def test_r_is_inert_on_the_new_profile_row(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        press(ui, "a", "end", "r")
        assert isinstance(ui.overlay, ProfilesOverlay)

    def test_an_unreadable_profile_is_not_opened_at_all(self):
        # `profile.save` writes verbatim, so an editor over text nobody could
        # fetch would truncate the file it failed to show.
        ui = RowUI(profiles=[ProfileInfo(name="hpc", loaded=False)])
        ui.focus = SESSIONS
        press(ui, "a", "enter")
        assert isinstance(ui.overlay, ProfilesOverlay)
        assert "could not be read" in screen(ui)


class TestProfileLifecycle:
    def _list(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        return press(ui, "a")

    def test_enter_on_new_profile_prompts(self):
        ui = press(self._list(), "end", "enter")
        assert isinstance(ui.overlay, PromptOverlay)

    def test_and_creates(self):
        ui = press(self._list(), "end", "enter")
        type_text(ui, "bench")
        press(ui, "enter")
        assert sent(ui, CreateProfile)[-1].name == "bench"

    def test_an_empty_name_creates_nothing(self):
        ui = press(self._list(), "end", "enter", "enter")
        assert sent(ui, CreateProfile) == []
        assert isinstance(ui.overlay, PromptOverlay), "and says so, on screen"

    def test_c_copies_under_a_new_name(self):
        ui = press(self._list(), "c")
        assert isinstance(ui.overlay, PromptOverlay)
        press(ui, "ctrl-u")
        type_text(ui, "hpc-gpu")
        press(ui, "enter")
        copied = sent(ui, CopyProfile)[-1]
        assert (copied.source, copied.name) == ("hpc", "hpc-gpu")

    def test_the_copy_prompt_is_prefilled(self):
        ui = press(self._list(), "c")
        assert ui.overlay.editor.text() == "hpc copy"

    def test_the_default_is_copyable(self):
        ui = press(self._list(), "down", "c")
        assert isinstance(ui.overlay, PromptOverlay)

    def test_copy_is_inert_on_the_new_row(self):
        ui = press(self._list(), "end", "c")
        assert isinstance(ui.overlay, ProfilesOverlay)

    def test_d_asks_first(self):
        ui = press(self._list(), "d")
        assert sent(ui, DeleteProfile) == []
        assert "Delete profile" in screen(ui)

    def test_and_says_the_sessions_move_to_the_default(self):
        assert "sessions move to the default" in screen(press(self._list(), "d"))

    def test_yes_deletes(self):
        ui = press(self._list(), "d", "y")
        assert sent(ui, DeleteProfile)[-1].name == "hpc"

    def test_and_its_skills_go_with_it(self):
        assert "skills go with it" in screen(press(self._list(), "d"))

    def test_no_keeps_it(self):
        ui = press(self._list(), "d", "n")
        assert sent(ui, DeleteProfile) == []
        assert isinstance(ui.overlay, ProfilesOverlay)

    def test_the_default_profile_cannot_be_deleted(self):
        ui = press(self._list(), "down", "d")
        assert sent(ui, DeleteProfile) == []
        assert "cannot be deleted" in screen(ui)

    def test_delete_is_inert_on_the_new_row(self):
        ui = press(self._list(), "end", "d")
        assert sent(ui, DeleteProfile) == []
        assert "Delete profile" not in screen(ui)

    def test_the_footer_drops_the_profile_keys_on_the_new_row(self):
        ui = press(self._list(), "end")
        foot = plain(ui.render(160, 40)[-1])
        assert "create a profile" in foot
        assert "d delete" not in foot


class TestProfileSkills:
    def _skills(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        return press(ui, "a", "s")

    def test_s_lists_a_profiles_skills(self):
        ui = self._skills()
        assert isinstance(ui.overlay, SkillsOverlay)
        assert "merge-vcfs" in screen(ui)

    def test_s_is_inert_on_the_new_profile_row(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        press(ui, "a", "end", "s")
        assert isinstance(ui.overlay, ProfilesOverlay)

    def test_enter_edits_one(self):
        ui = press(self._skills(), "enter")
        assert isinstance(ui.overlay, TextEditOverlay)
        assert ui.overlay.name == "merge-vcfs"

    def test_and_keeps_the_edit(self):
        ui = press(self._skills(), "enter")
        type_text(ui, "Q")
        press(ui, "esc", "y")
        saved = sent(ui, SaveSkill)[-1]
        assert (saved.profile, saved.name) == ("hpc", "merge-vcfs")
        assert "Q" in saved.text

    def test_escaping_the_skill_file_lands_back_on_the_skills(self):
        ui = press(self._skills(), "enter", "esc")
        assert isinstance(ui.overlay, SkillsOverlay)

    def test_and_the_one_after_that_on_the_profiles(self):
        ui = press(self._skills(), "enter", "esc", "esc")
        assert isinstance(ui.overlay, ProfilesOverlay)

    def test_d_deletes_one_after_asking(self):
        ui = press(self._skills(), "d")
        assert sent(ui, DeleteSkill) == []
        press(ui, "y")
        deleted = sent(ui, DeleteSkill)[-1]
        assert (deleted.profile, deleted.name) == ("hpc", "merge-vcfs")

    def test_declining_keeps_it(self):
        ui = press(self._skills(), "d", "n")
        assert sent(ui, DeleteSkill) == []

    def test_a_profile_with_no_skills_says_so(self):
        ui = RowUI(profiles=[ProfileInfo(name="bare", loaded=True)])
        ui.focus = SESSIONS
        assert "(no skills for this profile)" in screen(press(ui, "a", "s"))


# ------------------------------------------------------ thinking (item 30)


class TestThinking:
    """specs-ui-acceptance.md, "Thinking effort"."""

    def _open(self):
        ui = recorded(build())
        ui.session.thinking = "low"
        ui.thinking()
        return ui

    def test_the_command_opens_the_chooser(self):
        ui = recorded(build())
        ui.focus = CHAT
        press(ui, "i")
        type_text(ui, "/thinking")
        press(ui, "enter")
        assert isinstance(ui.overlay, ThinkingOverlay)

    def test_and_sends_no_turn(self):
        ui = recorded(build())
        ui.focus = CHAT
        press(ui, "i")
        type_text(ui, "/thinking")
        before = len(ui.intents)
        press(ui, "enter")
        assert len(ui.intents) == before

    def test_it_lists_all_four_levels(self):
        seen = screen(self._open())
        for level in ("off", "low", "medium", "xhigh"):
            assert level in seen

    def test_xhigh_is_flagged_unusable(self):
        assert "NOT USABLE" in screen(self._open())

    def test_the_current_level_is_starred(self):
        assert "low ★" in screen(self._open())

    def test_and_preselected(self):
        ui = self._open()
        assert ui.overlay.item(118).text == "low"

    def test_enter_stores_it_on_the_session(self):
        ui = press(self._open(), "down", "enter")
        assert sent(ui, SetThinking)[-1].effort == "medium"
        assert ui.session.thinking == "medium"

    def test_escape_leaves_the_level_alone(self):
        ui = press(self._open(), "down", "esc")
        assert sent(ui, SetThinking) == []
        assert ui.session.thinking == "low"

    def test_the_context_bar_shows_the_level(self):
        ui = press(self._open(), "down", "enter")
        assert "think medium" in screen(ui)

    def test_without_a_session_it_says_so(self):
        ui = RowUI()
        ui.thinking()
        assert ui.overlay is None
        assert "no session open" in ui.note

    def test_an_unknown_stored_level_falls_back_to_off(self):
        assert ThinkingOverlay("nonsense").current == "off"


# ------------------------------------------------------ switch llm (item 29)


class TestSwitchLlm:
    """specs-ui-acceptance.md, "Backends", the ctrl+L bullets."""

    def _open(self):
        ui = recorded(build())
        ui.catalog = catalog()
        ui.focus = CHAT
        return press(ui, "ctrl-l")

    def test_ctrl_l_opens_the_switcher(self):
        assert isinstance(self._open().overlay, SwitchLlmOverlay)

    def test_it_is_not_offered_from_the_sessions_column(self):
        ui = recorded(build())
        ui.catalog = catalog()
        ui.focus = SESSIONS
        press(ui, "ctrl-l")
        assert ui.overlay is None

    def test_without_backends_it_warns(self):
        ui = recorded(build())
        ui.catalog = []
        ui.focus = CHAT
        press(ui, "ctrl-l")
        assert ui.overlay is None
        assert "no backends configured" in ui.note

    def test_it_lists_the_catalog(self):
        seen = screen(self._open())
        assert "qwen3" in seen and "llama" in seen

    def test_the_session_s_own_backend_is_starred(self):
        ui = recorded(build())
        ui.catalog = catalog()
        ui.session.model = "qwen3-27b-fp8"
        ui.focus = CHAT
        press(ui, "ctrl-l")
        assert "★ ● qwen3" in screen(ui)

    def test_enter_sets_the_sessions_backend(self):
        ui = press(self._open(), "enter")
        picked = sent(ui, SetBackend)[-1]
        assert picked.session_id == ui.active_id
        assert picked.backend["model"] == "qwen3-27b-fp8"

    def test_and_updates_the_model_line(self):
        ui = press(self._open(), "enter")
        assert "qwen3-27b-fp8" in screen(ui)

    def test_escape_changes_nothing(self):
        ui = press(self._open(), "down", "esc")
        assert sent(ui, SetBackend) == []

    def test_a_key_locked_entry_is_refused_rather_than_half_sent(self):
        # The catalog never carries an api key, so a blob built from it would
        # make a client the endpoint answers 401 to.
        ui = press(self._open(), "end", "enter")
        assert sent(ui, SetBackend) == []
        assert isinstance(ui.overlay, SwitchLlmOverlay)
        assert "needs an api key" in screen(ui)

    def test_an_unprobed_entry_is_not_drawn_as_disconnected(self):
        info = BackendInfo(label="a", model="m", base_url="u", reachable=None)
        drawn = plain(SwitchLlmOverlay([info]).render(100, 12)[2])
        assert "○" not in drawn

    def test_an_empty_catalog_says_where_to_add_one(self):
        assert "manage-llms" in "\n".join(
            plain(x) for x in SwitchLlmOverlay([]).render(100, 12)
        )


# ----------------------------------------------------- manage llms (item 28)


class TestManageLlms:
    def test_m_opens_manage_llms(self):
        assert isinstance(opened("m").overlay, LlmOverlay)

    def test_only_from_the_sessions_column(self):
        assert opened("m", focus=CHAT).overlay is None

    def test_two_stacked_panels(self):
        body = screen(opened("m"))
        assert "── discovered " in body
        assert "── configured " in body

    def test_the_catalog_is_in_the_configured_panel(self):
        assert "qwen3-27b-fp8" in screen(opened("m"))

    def test_the_marks_are_all_three(self):
        # ● answered, ○ did not, · nobody asked — the demo catalog has one of
        # each precisely so this can be asserted.
        body = screen(opened("m"))
        assert "●" in body and "○" in body and "★" in body

    def test_nothing_is_discovered_and_it_says_why(self):
        assert "nothing discovered" in screen(opened("m"))

    def test_the_footer_offers_add_only_on_the_discovered_panel(self):
        ui = opened("m")
        assert "enter add to catalog" in plain(ui.render(160, 40)[-1])
        press(ui, "ctrl-down")
        assert "add to catalog" not in plain(ui.render(160, 40)[-1])

    def test_enter_on_a_configured_entry_sets_no_global_default(self):
        ui = press(opened("m"), "ctrl-down", "enter")
        assert sent(ui, SetBackend) == []
        assert "already in the catalog" in screen(ui)

    def test_a_opens_the_manual_form(self):
        assert isinstance(press(opened("m"), "a").overlay, BackendFormOverlay)

    def test_and_a_saved_form_configures_and_persists_it(self):
        ui = press(opened("m"), "a")
        type_text(ui, "http://new/v1")
        press(ui, "tab")
        type_text(ui, "mistral")
        press(ui, "enter")
        saved = sent(ui, SetBackend)[-1]
        assert saved.backend["base_url"] == "http://new/v1"
        assert saved.session_id == "", "the global half: no session named"

    def test_and_the_catalog_shows_it_before_the_core_answers(self):
        ui = press(opened("m"), "a")
        type_text(ui, "http://new/v1")
        press(ui, "tab")
        type_text(ui, "mistral")
        press(ui, "enter")
        assert isinstance(ui.overlay, LlmOverlay)
        assert "mistral" in screen(ui)

    def test_escape_closes(self):
        assert press(opened("m"), "esc").overlay is None

    def test_the_arrows_do_not_switch_panels(self):
        # ←/→ open and close a row, like every other list here; ^↑/^↓ move
        # between the panels, which is what the main view does.
        ui = opened("m")
        press(ui, "right")
        assert ui.overlay.side == 0
        press(ui, "ctrl-down")
        assert ui.overlay.side == 1

    def test_a_second_catalog_restates_the_open_screen(self):
        ui = opened("m")
        ui.overlay.catalog_changed(
            [BackendInfo(label="brand-new", model="m", base_url="u")]
        )
        assert "brand-new" in screen(ui)


class TestBackendForm:
    def test_it_offers_the_four_fields(self):
        body = "\n".join(plain(x) for x in BackendFormOverlay().render(100, 20))
        for field in ("endpoint url", "model", "api key", "context length"):
            assert field in body

    def test_a_discovered_endpoint_locks_the_url(self):
        form = BackendFormOverlay(base_url="http://x/v1", locked_url=True)
        form.handle("Z", 100, 20)
        assert form.value("base_url") == "http://x/v1"

    def test_and_opens_on_the_model_rather_than_the_url(self):
        form = BackendFormOverlay(base_url="http://x/v1", locked_url=True)
        assert form.field() == "model"

    def test_a_blank_url_is_refused(self):
        form = BackendFormOverlay()
        assert form.handle("enter", 100, 20) is True
        assert form.backend is None
        assert "endpoint url is needed" in form.note

    def test_a_non_numeric_context_is_refused(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.at = 3
        for ch in "many":
            form.handle(ch, 100, 20)
        assert form.handle("enter", 100, 20) is True
        assert "whole number" in form.note

    def test_the_key_is_masked(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.at = 2
        for ch in "s3cret":
            form.handle(ch, 100, 20)
        body = "\n".join(plain(x) for x in form.render(100, 20))
        assert "s3cret" not in body
        assert "••••••" in body

    def test_and_still_reaches_the_command(self):
        form = BackendFormOverlay(base_url="http://x/v1", model="m")
        form.at = 2
        for ch in "s3cret":
            form.handle(ch, 100, 20)
        form.handle("enter", 100, 20)
        assert form.sent[-1].backend["api_key"] == "s3cret"

    def test_escape_saves_nothing(self):
        form = BackendFormOverlay(base_url="http://x/v1", model="m")
        assert form.handle("esc", 100, 20) is False
        assert form.sent == []


# --------------------------------------------------- the inspect window (31)


class TestInspect:
    def test_it_shows_the_body(self):
        ui = build()
        ui.inspect("skills", "merge-vcfs — how to merge shards")
        assert "merge-vcfs" in screen(ui)

    def test_and_the_title(self):
        ui = build()
        ui.inspect("skills for hpc", "body")
        assert "skills for hpc" in screen(ui)

    def test_it_scrolls(self):
        window = InspectOverlay("\n".join(f"line {i}" for i in range(200)))
        window.handle("pgdn", 100, 20)
        assert window.offset > 0

    def test_and_says_how_far_down(self):
        window = InspectOverlay("\n".join(f"line {i}" for i in range(200)))
        assert "/200" in "\n".join(plain(x) for x in window.render(100, 20))

    def test_an_ordinary_key_does_not_close_it(self):
        # Unlike the key list: this one usually holds something being copied
        # out of it, and a stray keystroke would cost the whole errand.
        ui = build()
        ui.inspect("t", "body")
        assert isinstance(press(ui, "x").overlay, InspectOverlay)

    def test_escape_closes_it(self):
        ui = build()
        ui.inspect("t", "body")
        assert press(ui, "esc").overlay is None

    def test_long_lines_are_wrapped_rather_than_cut(self):
        recipe = "ssh -L 20001:node042:20001 " + "x" * 200
        window = InspectOverlay(recipe)
        body = "\n".join(plain(x) for x in window.render(60, 20))
        assert body.count("x") == 200


# --------------------------------------------------- the memory review (32)


def proposals() -> list[Proposal]:
    return [
        Proposal(scope="profile", kind="preference", text="prefers sniffles"),
        Proposal(scope="profile", kind="fact", text="scratch is /scratch/proj"),
        Proposal(scope="rag", kind="struggle", text="the merge needs -Oz"),
    ]


class TestMemoryReview:
    def _open(self):
        ui = recorded(build())
        ui.session.proposals = proposals()
        ui.review_memories()
        return ui

    def test_it_shows_one_proposal_at_a_time(self):
        seen = screen(self._open())
        assert "prefers sniffles" in seen
        assert "scratch is /scratch/proj" not in seen

    def test_and_says_where_it_is_in_the_batch(self):
        assert "1/3" in screen(self._open())

    def test_and_what_the_proposal_is(self):
        assert "profile · preference" in screen(self._open())

    def test_y_and_n_are_answered_positionally(self):
        ui = press(self._open(), "y", "n", "y")
        assert sent(ui, ResolveMemory)[-1].approved == (True, False, True)

    def test_and_named_by_session(self):
        ui = press(self._open(), "y", "y", "y")
        assert sent(ui, ResolveMemory)[-1].session_id == ui.active_id

    def test_the_screen_closes_when_the_batch_is_done(self):
        ui = press(self._open(), "y", "y", "y")
        assert ui.overlay is None

    def test_escaping_early_answers_short(self):
        # A short list rejects the rest (`protocol.MemoryResolve`), so leaving
        # early is a decision rather than a way out of one.
        ui = press(self._open(), "y", "esc")
        assert sent(ui, ResolveMemory)[-1].approved == (True,)

    def test_rejecting_everything_saves_nothing(self):
        ui = press(self._open(), "n", "n", "n")
        assert sent(ui, ResolveMemory)[-1].approved == (False, False, False)

    def test_nothing_to_review_opens_nothing(self):
        ui = build()
        ui.session.proposals = []
        ui.review_memories()
        assert ui.overlay is None

    def test_a_screen_with_no_proposals_still_draws(self):
        drawn = MemoryReviewOverlay().render(80, 12)
        assert widths(drawn) == {80}


# ---------------------------------------------------------- the frame itself

# Overlays are where the width discipline breaks, because they draw frames
# inside frames. Every screen, at the three widths the port is measured at.


def all_screens() -> dict:
    catalog_rows = catalog()
    return {
        "help": HelpOverlay(),
        "config": ConfigOverlay('{"a": 1}\n'),
        "profiles": ProfilesOverlay(profiles()),
        "skills": SkillsOverlay(profiles()[0]),
        "textedit": TextEditOverlay("some memories\n", profile="hpc"),
        "thinking": ThinkingOverlay("medium", session_id="s1"),
        "switch": SwitchLlmOverlay(catalog_rows, session_id="s1"),
        "llms": LlmOverlay(catalog_rows),
        "form": BackendFormOverlay(base_url="http://x/v1"),
        "inspect": InspectOverlay("a line\n" * 40, title="skills"),
        "memory": MemoryReviewOverlay(proposals(), "s1"),
        "prompt": PromptOverlay("a name", title="new profile", hint="short"),
    }


@pytest.mark.parametrize("name", sorted(all_screens()))
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("height", [14, 24, 40])
def test_every_row_is_exactly_the_width(name, width, height):
    drawn = all_screens()[name].render(width, height)
    assert len(drawn) == height
    assert widths(drawn) == {width}


@pytest.mark.parametrize("name", sorted(all_screens()))
@pytest.mark.parametrize("width", WIDTHS)
def test_and_stays_so_while_a_question_is_up(name, width):
    overlay = all_screens()[name]
    overlay.ask("Really?")
    drawn = overlay.render(width, 24)
    assert len(drawn) == 24
    assert widths(drawn) == {width}


@pytest.mark.parametrize("width,height", [(80, 24), (120, 40), (60, 14)])
@pytest.mark.parametrize("key", ["?", "m", "a", "c"])
def test_an_open_screen_fills_the_terminal(key, width, height):
    ui = build()
    ui.focus = SESSIONS
    ui.handle(key, width, height)
    drawn = ui.render(width, height)
    assert len(drawn) == height
    assert widths(drawn) == {width}


def test_an_overlay_covers_the_rows_it_is_drawn_over():
    ui = build()
    ui.focus = CHAT
    assert "── chat " in screen(ui)
    press(ui, "?")
    assert "── chat " not in screen(ui)


def test_a_child_screen_covers_its_parent():
    ui = RowUI(profiles=profiles())
    ui.focus = SESSIONS
    press(ui, "a")
    assert "── profiles " in screen(ui)
    press(ui, "enter")
    assert "── profiles " not in screen(ui)


def test_and_the_stack_unwinds_one_at_a_time():
    ui = RowUI(profiles=profiles())
    ui.focus = SESSIONS
    press(ui, "a", "s", "enter")
    assert len(ui.overlays) == 3
    press(ui, "esc")
    assert len(ui.overlays) == 2
    press(ui, "esc", "esc")
    assert ui.overlays == []


def test_the_demo_still_builds_every_screen():
    # `python -m hpca.ui.run --demo` has to keep working, and the cheapest
    # proof is that each key opens something that draws.
    ui = build(chat=20, sessions=4, watchers=2)
    for key in ("?", "m", "a", "c"):
        ui.focus = SESSIONS
        ui.handle(key, 100, 30)
        assert ui.overlay is not None, key
        assert widths(ui.render(100, 30)) == {100}
        ui.overlay = None


def test_the_catalog_reaches_the_new_session_picker():
    ui = build()
    ui.focus = SESSIONS
    ui.session_pane.cursor = 0
    press(ui, "home", "enter")
    assert "which profile" in screen(ui)
    press(ui, "enter")
    # The second stage exists because the demo core answered `llm.list`.
    assert "which llm" in screen(ui)
    assert sample_catalog()[0].label in screen(ui)
