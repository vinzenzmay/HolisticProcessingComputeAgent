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

from hpca.ui.app import CHAT, SESSIONS, WATCHERS, RowUI
from hpca.ui.demo import build, sample_catalog
from hpca.ui.overlays import (
    CUT_WARNING,
    KEEP_CHANGES,
    PREVIEW_CHARS,
    BackendFormOverlay,
    CompactReviewOverlay,
    ConfigOverlay,
    ModelPickerOverlay,
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
from hpca.ui.overlays.backends import KEY_REQUIRED, backend_head
from hpca.ui.state import (
    BackendInfo,
    ChatEntry,
    CompactProposal,
    CopyProfile,
    ProbeBackend,
    RemoveBackend,
    ScanBackends,
    CreateProfile,
    DeleteProfile,
    DeleteSkill,
    EditProfile,
    Fetch,
    FetchSkills,
    ProfileInfo,
    Proposal,
    ResolveCompact,
    ResolveMemory,
    SaveProfile,
    SaveSettings,
    SaveSkill,
    SetBackend,
    SetThinking,
    SkillInfo,
)
from tests.ui_harness import (
    frame,
    on_own_message,
    plain,
    recorded,
    served,
    widths,
)

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


# The bodies the editors fetch when they open (`profile.get`, `skill.get`),
# as the local `served()` core answers them. A key that is missing is answered
# with an error, which is how the core says a file could not be read.
BODIES = {
    ("profile", ("hpc", "memories")): "scratch is /scratch/proj\n",
    ("profile", ("hpc", "archive")): "## [rag] old\n",
    ("profile", ("default", "memories")): "",
    ("profile", ("writing", "memories")): "x\n",
    ("skill", ("hpc", "merge-vcfs")): "body\n",
}
OWN_SKILLS = {"hpc": [SkillInfo("merge-vcfs", "how to merge shards")]}


def with_profiles(rows=None, bodies=None, skills=None) -> RowUI:
    """A bare `RowUI` on the sessions column, with the read paths answered."""
    ui = RowUI(profiles=profiles() if rows is None else rows)
    ui.focus = SESSIONS
    return served(
        ui,
        BODIES if bodies is None else bodies,
        OWN_SKILLS if skills is None else skills,
    )


def profiles() -> list[ProfileInfo]:
    """What `profile.rows` fills in — counts and flags, and no bodies.

    The bodies are fetched when an editor opens (`profile.get`, `skill.get`),
    so a `ProfileInfo` no longer carries any: what a screen shows before its
    answer arrives is "fetching…", which is the point of it.
    """
    return [
        ProfileInfo(
            name="hpc",
            memories=12,
            sessions=3,
            working=True,
            skills=[SkillInfo("merge-vcfs", "how to merge shards")],
        ),
        ProfileInfo(name="default", memories=0, default=True),
        ProfileInfo(name="writing", memories=5, copied_from="hpc"),
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

    def test_it_offers_both_cuts(self):
        ui = build()
        on_own_message(ui)
        seen = screen(press(ui, "enter"))
        assert "fork the session from here" in seen
        assert "roll this conversation back to here" in seen

    def test_it_says_where_the_message_goes(self):
        # Both cuts hand it back to the box, and a message reappearing where
        # the user is about to type is alarming if the dialog never said so.
        ui = build()
        on_own_message(ui)
        assert "comes back to the box" in screen(press(ui, "enter"))

    def test_and_no_longer_the_copy(self):
        # It is `c` on the chat row now, which needs no dialog in front of it.
        ui = build()
        on_own_message(ui)
        assert "copy it into the message box" not in screen(press(ui, "enter"))

    def test_the_footer_names_them_both(self):
        ui = build()
        on_own_message(ui)
        press(ui, "enter")
        foot = plain(ui.render(160, 40)[-1])
        for pair in ("f / enter fork", "r roll back", "esc cancel"):
            assert pair in foot

    def test_and_offers_no_key_it_will_not_answer(self):
        ui = build()
        on_own_message(ui)
        press(ui, "enter")
        assert "copy" not in plain(ui.render(160, 40)[-1])

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

    def test_and_counts_the_conversations_filed_under_it(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        assert "3 sessions" in screen(press(ui, "a"))

    def test_a_profile_nobody_has_talked_under_says_nothing_about_sessions(self):
        # Zero is also what an uncountable answer looks like, so the row omits
        # the phrase rather than claiming "0 sessions".
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        assert "0 sessions" not in screen(press(ui, "a"))

    def test_and_shows_provenance(self):
        ui = RowUI(profiles=profiles())
        ui.focus = SESSIONS
        assert "copied from hpc" in screen(press(ui, "a"))


class TestProfileMemories:
    def _open(self):
        return press(with_profiles(), "a", "enter")

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
        ui = press(with_profiles(), "a", "r")
        assert isinstance(ui.overlay, TextEditOverlay)
        assert ui.overlay.kind == "archive"

    def test_and_keeps_its_edits(self):
        ui = press(with_profiles(), "a", "r")
        type_text(ui, "Z")
        press(ui, "esc", "y")
        assert sent(ui, SaveProfile)[-1].kind == "archive"

    def test_r_is_inert_on_the_new_profile_row(self):
        ui = press(with_profiles(), "a", "end", "r")
        assert isinstance(ui.overlay, ProfilesOverlay)

    def test_the_body_is_fetched_when_the_editor_opens(self):
        # Never carried by the row that offered it: `profile.save` writes
        # verbatim, and a body shipped with a listing is already stale by the
        # time the editor is over it (`protocol.ProfileGet`).
        ui = press(with_profiles(), "a", "enter")
        asked = [x for x in ui.intents if isinstance(x, Fetch)]
        assert asked[-1] == Fetch("profile", ("hpc", "memories"))

    def test_an_unreadable_profile_cannot_be_edited(self):
        # `profile.save` writes verbatim, so a box that filled with nothing
        # would truncate the file it failed to show. `ProfileBody.error` is the
        # only thing that separates that from an empty file.
        ui = press(with_profiles(bodies={}), "a", "enter")
        assert isinstance(ui.overlay, TextEditOverlay)
        assert "could not be read" in screen(ui)
        type_text(ui, "X")
        assert ui.overlay.editor.text() == ""
        press(ui, "esc")
        assert sent(ui, SaveProfile) == [], "nothing may be written back"


class TestProfilesInTheUsersEditor:
    """With a terminal to hand over, the profile under the cursor opens in
    `$EDITOR` — asked for in those words.

    The screen stays up behind it, because the point of editing from a list is
    that the next one is one keypress away. Nothing comes back through
    `child_closed`: there is no child, and the save is sent by the side that
    ran the editor (`UIClient._run_editor`).
    """

    def _external(self, **kw) -> RowUI:
        ui = with_profiles(**kw)
        ui.suspend = lambda run: run()
        ui.edit_text = lambda text, done: None
        return ui

    def test_enter_sends_the_row_the_cursor_is_on(self):
        ui = press(self._external(), "a", "down", "enter")
        assert sent(ui, EditProfile)[-1] == EditProfile("default", "memories")

    def test_and_not_the_profile_the_core_is_working_under(self):
        # The whole of the complaint: the old key edited `ui.profile`, from a
        # screen where a *different* row was selected.
        ui = self._external()
        ui.core_profile = "hpc"
        press(ui, "a", "down", "enter")
        assert ui.profile == "hpc"
        assert sent(ui, EditProfile)[-1].name == "default"

    def test_no_in_app_editor_opens_over_it(self):
        ui = press(self._external(), "a", "enter")
        assert isinstance(ui.overlay, ProfilesOverlay), "the list stays up"
        assert "your editor" in ui.overlay.note

    def test_r_sends_the_archive_of_that_row(self):
        ui = press(self._external(), "a", "down", "r")
        assert sent(ui, EditProfile)[-1] == EditProfile("default", "archive")

    def test_the_new_profile_row_still_names_one_instead(self):
        ui = press(self._external(), "a", "end", "enter")
        assert isinstance(ui.overlay, PromptOverlay)
        assert sent(ui, EditProfile) == []

    def test_without_a_terminal_the_in_app_editor_is_what_opens(self):
        # A UI with nothing to hand the terminal over to keeps the screen it
        # has always had, rather than being told there is no editor and left
        # with no way to edit a profile at all.
        ui = press(with_profiles(), "a", "enter")
        assert isinstance(ui.overlay, TextEditOverlay)
        assert sent(ui, EditProfile) == []


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
        return press(with_profiles(), "a", "s")

    def test_s_lists_a_profiles_skills(self):
        ui = self._skills()
        assert isinstance(ui.overlay, SkillsOverlay)
        assert "merge-vcfs" in screen(ui)

    def test_s_is_inert_on_the_new_profile_row(self):
        ui = press(with_profiles(), "a", "end", "s")
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
        ui = with_profiles(rows=[ProfileInfo(name="bare")], skills={})
        assert "(no skills for this profile)" in screen(press(ui, "a", "s"))

    def test_the_list_is_asked_for_when_the_screen_opens(self):
        # `skill.list`, not a copy carried on the profile row: the screen is
        # also how a skill is deleted, so what it lists has to be current.
        ui = press(with_profiles(), "a", "s")
        assert FetchSkills("hpc") in ui.intents


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
        press(ui, "ctrl-down")
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
        # By label, not by a rebuilt blob: the catalog carries no api key, and
        # naming the entry the core already holds is what lets a key-locked
        # backend be switched to at all (`protocol.BackendSet`).
        ui = press(self._open(), "enter")
        picked = sent(ui, SetBackend)[-1]
        assert picked.session_id == ui.active_id
        assert picked.label == "qwen3"

    def test_and_updates_the_model_line(self):
        ui = press(self._open(), "enter")
        assert "qwen3-27b-fp8" in screen(ui)

    def test_escape_changes_nothing(self):
        ui = press(self._open(), "down", "esc")
        assert sent(ui, SetBackend) == []

    def test_a_key_locked_entry_is_reachable_now_that_labels_are(self):
        # It used to be refused: the catalog carries no api key, so a blob
        # built from it made a client the endpoint answered 401 to. The label
        # form leaves the key where the core already has it.
        ui = press(self._open(), "end", "enter")
        assert sent(ui, SetBackend)[-1].label == "locked"
        assert sent(ui, SetBackend)[-1].backend == {}

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

    def test_the_scan_goes_out_when_the_screen_opens(self):
        # §3.1: the core has implemented and tested the scan since `efb9b3f`
        # and nothing in the UI ever sent it — while this screen's docstring
        # said discovery "is not implemented anywhere".
        assert sent(opened("m"), ScanBackends)

    def test_and_the_panel_fills_from_the_frames_it_answers_with(self):
        # The demo core answers the way a real one does: a catalog per hit.
        assert "mistral-small-3.1" in screen(opened("m"))

    def test_a_scan_that_finds_nothing_says_so_on_the_panel(self):
        overlay = LlmOverlay(catalog())
        overlay.scanned(0, 0)
        assert "nothing discovered" in "\n".join(
            plain(x) for x in overlay.render(120, 30)
        )

    def test_and_says_what_it_found_when_it_finds_something(self):
        assert "2 endpoint(s) found" in screen(opened("m"))

    def test_the_screen_takes_keys_while_the_scan_is_still_out(self):
        # "The UI stays responsive and closable during a slow scan": nothing
        # here waits on the answer, so a screen whose scan never comes back
        # still moves, and still closes.
        overlay = LlmOverlay(catalog())
        overlay.scanning = True
        assert overlay.handle("ctrl-down", 120, 30) is True
        assert overlay.handle("esc", 120, 30) is False

    def test_a_rescan_asks_again(self):
        ui = opened("m")
        press(ui, "s")
        assert len(sent(ui, ScanBackends)) == 2

    def test_a_discovered_row_already_configured_is_not_drawn_twice(self):
        # `core/backends.py` emits discovered rows without excluding the
        # configured ones, so the dedup the Textual screen did at save time
        # has to happen here or the endpoint appears under two labels.
        rows = catalog() + [
            BackendInfo(
                label="qwen3-27b-fp8 @ a",
                model="qwen3-27b-fp8",
                base_url="http://a/v1",
                discovered=True,
            )
        ]
        assert LlmOverlay(rows).discovered == []

    def test_the_tunnel_recipe_opens_over_the_screen_that_asked(self):
        ui = opened("m")
        ui.scanned(found=0, cluster=0, help_text="ssh -L 20001:node042:20001 …")
        assert isinstance(ui.overlay, InspectOverlay)
        assert "ssh -L" in screen(ui)

    def test_and_escaping_it_lands_back_on_manage_llms(self):
        ui = opened("m")
        ui.scanned(found=0, cluster=0, help_text="ssh -L 20001:node042:20001 …")
        assert isinstance(press(ui, "esc").overlay, LlmOverlay)

    def test_it_waits_for_a_form_the_user_already_opened(self):
        ui = press(opened("m"), "a")
        ui.scanned(found=0, cluster=0, help_text="ssh -L 20001:node042:20001 …")
        assert isinstance(ui.overlay, BackendFormOverlay)

    def test_and_arrives_when_the_form_closes(self):
        ui = press(opened("m"), "a")
        ui.scanned(found=0, cluster=0, help_text="ssh -L 20001:node042:20001 …")
        press(ui, "esc")
        assert isinstance(ui.overlay, InspectOverlay)

    def test_a_notice_is_a_toast_and_not_a_window(self):
        # "An empty scan with a backend up says only 'nothing new'."
        ui = opened("m")
        ui.scanned(found=0, cluster=0, notice="Nothing new on localhost")
        assert isinstance(ui.overlay, LlmOverlay)
        assert "Nothing new on localhost" in ui.note

    def test_r_removes_a_configured_backend_after_asking(self):
        ui = opened("m")
        press(ui, "ctrl-down", "r")
        assert "Remove" in screen(ui)
        assert sent(ui, RemoveBackend) == []
        press(ui, "y")
        assert sent(ui, RemoveBackend)[-1].label.startswith("qwen3-27b-fp8")

    def test_and_denying_keeps_it(self):
        ui = opened("m")
        press(ui, "ctrl-down", "r", "n")
        assert sent(ui, RemoveBackend) == []
        assert "qwen3-27b-fp8" in screen(ui)

    def test_and_the_row_goes_before_the_core_answers(self):
        ui = opened("m")
        label = ui.overlay.configured[0].label
        press(ui, "ctrl-down", "r", "y")
        assert label not in [x.label for x in ui.overlay.configured]

    def test_enter_on_a_locked_row_opens_the_key_form(self):
        overlay = LlmOverlay(
            [
                BackendInfo(
                    label="locked",
                    model=KEY_REQUIRED,
                    base_url="http://c/v1",
                    needs_key=True,
                    discovered=True,
                )
            ]
        )
        assert overlay.handle("enter", 120, 30) is True
        form = overlay.child
        assert form.value("base_url") == "http://c/v1"
        assert form.value("model") == "", "the sentinel is not a model name"
        assert form.field() == "api_key" or form.locked_url

    def test_and_a_cluster_row_keeps_the_model_it_was_told(self):
        # The manifest names the model of an endpoint the sweep could only get
        # a 401 out of; retyping it would be work we were spared.
        overlay = LlmOverlay(
            [
                BackendInfo(
                    label="qwen @ node042",
                    model="qwen3-27b-fp8",
                    base_url="http://node042:20001/v1",
                    needs_key=True,
                    discovered=True,
                )
            ]
        )
        overlay.handle("enter", 120, 30)
        assert overlay.child.value("model") == "qwen3-27b-fp8"

    def test_the_sentinel_is_the_one_the_scanner_mints(self):
        from hpca.discover import KEY_REQUIRED as MINTED

        assert KEY_REQUIRED == MINTED

    def test_the_key_hint_is_only_for_a_row_still_locked(self):
        # "api key required" beside a backend the user has already keyed reads
        # as the key not having been accepted. The flag is minted to mean
        # "still locked" (`core.backends.BackendRegistry.catalog`); the row
        # just draws it.
        locked = BackendInfo(
            label="locked", model=KEY_REQUIRED, base_url="http://c/v1",
            needs_key=True, discovered=True,
        )
        keyed = BackendInfo(
            label="qwen3", model="qwen3-27b-fp8", base_url="http://a/v1",
            reachable=True,
        )
        assert "api key required" in backend_head(locked)
        assert "api key required" not in backend_head(keyed)

    def test_remove_is_inert_on_the_discovered_panel(self):
        ui = opened("m")
        press(ui, "r")
        assert sent(ui, RemoveBackend) == []
        assert "not in the catalog" in screen(ui)

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
        press(ui, "ctrl-s")
        saved = sent(ui, SetBackend)[-1]
        assert saved.backend["base_url"] == "http://new/v1"
        assert saved.session_id == "", "the global half: no session named"

    def test_and_the_catalog_shows_it_before_the_core_answers(self):
        ui = press(opened("m"), "a")
        type_text(ui, "http://new/v1")
        press(ui, "tab")
        type_text(ui, "mistral")
        press(ui, "ctrl-s")
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
        assert form.handle("ctrl-s", 100, 20) is True
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
        form.handle("ctrl-s", 100, 20)
        assert form.sent[-1].backend["api_key"] == "s3cret"

    # ------------------------------------------------- the probe (§3.1, M8a)

    def test_enter_probes_rather_than_saving(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        assert form.handle("enter", 100, 20) is True
        assert form.sent == [ProbeBackend("http://x/v1", "")]
        assert form.backend is None, "nothing is saved until it answers"

    def test_the_typed_key_goes_with_it(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.at = 2
        for ch in "s3cret":
            form.handle(ch, 100, 20)
        form.handle("enter", 100, 20)
        assert form.sent[-1].api_key == "s3cret"

    def test_one_model_fills_the_form_in(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.handle("enter", 100, 20)
        form.probed(
            "http://x/v1",
            [BackendInfo(label="qwen3", model="qwen3", context=112000)],
        )
        assert form.value("model") == "qwen3"
        assert form.value("max_model_len") == "112000"

    def test_and_the_next_enter_saves_it(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.handle("enter", 100, 20)
        form.probed("http://x/v1", [BackendInfo(label="q", model="q")])
        assert form.handle("enter", 100, 20) is False
        assert form.sent[-1].backend["model"] == "q"

    def test_editing_after_a_probe_makes_enter_check_again(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.probed("http://x/v1", [BackendInfo(label="q", model="q")])
        form.at = 2
        form.handle("k", 100, 20)  # a key nothing has checked
        assert form.handle("enter", 100, 20) is True
        assert isinstance(form.sent[-1], ProbeBackend)

    def test_a_rejected_key_warns_and_keeps_the_input(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.at = 2
        for ch in "wrong":
            form.handle(ch, 100, 20)
        form.probed("http://x/v1", [], needs_key=True)
        assert "refused that key" in form.note
        assert form.value("api_key") == "wrong"
        assert form.field() == "api_key", "the cursor lands where the fix is"

    def test_and_ctrl_s_saves_it_anyway(self):
        form = BackendFormOverlay(base_url="http://x/v1", model="m")
        form.probed("http://x/v1", [], needs_key=True)
        assert "ctrl+s" in form.note
        assert form.handle("ctrl-s", 100, 20) is False
        assert form.sent[-1].backend["base_url"] == "http://x/v1"

    def test_nothing_answering_is_told_apart_from_a_refused_key(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.probed("http://x/v1", [], needs_key=False)
        assert "nothing answered" in form.note

    def test_an_answer_for_another_endpoint_is_not_ours(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.probed("http://elsewhere/v1", [BackendInfo(label="q", model="q")])
        assert form.value("model") == ""

    def test_a_trailing_slash_is_still_the_same_endpoint(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.probed("http://x/v1/", [BackendInfo(label="q", model="q")])
        assert form.value("model") == "q"

    def test_several_models_open_a_picker(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.probed(
            "http://x/v1",
            [
                BackendInfo(label="qwen3", model="qwen3", context=112000),
                BackendInfo(label="llama", model="llama-3.3-70b"),
            ],
        )
        assert isinstance(form.child, ModelPickerOverlay)

    def test_and_picking_one_fills_the_form(self):
        form = BackendFormOverlay(base_url="http://x/v1")
        form.probed(
            "http://x/v1",
            [
                BackendInfo(label="qwen3", model="qwen3", context=112000),
                BackendInfo(label="llama", model="llama-3.3-70b"),
            ],
        )
        picker = form.child
        picker.handle("down", 100, 20)
        assert picker.handle("enter", 100, 20) is False
        form.child_closed(picker)
        assert form.value("model") == "llama-3.3-70b"

    def test_the_picker_reaches_the_form_through_the_app(self):
        ui = press(opened("m"), "a")
        type_text(ui, "http://many/v1")
        press(ui, "enter")
        assert isinstance(ui.overlay, ModelPickerOverlay)

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


# ------------------------------------------- enter on a watch box (31 again)


def peeked(width: int = 120, height: int = 40) -> RowUI:
    """The demo, with Enter pressed on the first watch box.

    The demo loopback is synchronous, so the answer is already on screen by
    the time `handle` returns — which is the whole reason these can press a
    key and read the frame on the next line.
    """
    ui = recorded(build())
    ui.focus = WATCHERS
    return press(ui, "enter", width=width, height=height)


class TestThePeekWindow:
    """Enter on a watcher opens the read-only window, not a toast.

    The toast was time-bound and drawn over the top rows of the frame, so a
    log had to be read in the seconds it was up and could not be selected out
    of the terminal at all. Everything below is that complaint, one claim at a
    time.
    """

    def test_it_opens_a_screen(self):
        assert isinstance(peeked().overlay, InspectOverlay)

    def test_and_names_the_watch_it_is_the_tail_of(self):
        assert peeked().overlay.title.startswith("job ")

    def test_a_peeked_log_keeps_its_lines(self):
        ui = peeked()
        rows = [x.strip() for x in frame(ui, 120, 40)]
        assert "[12:41:07] merging shard 3 of 8" in rows
        assert "[12:41:44] merging shard 4 of 8" in rows

    def test_escape_gives_the_rows_back(self):
        ui = peeked()
        assert press(ui, "esc").overlay is None
        assert "job 4821000" in screen(ui)
        assert "unwatch" in screen(ui)

    def test_and_leaves_the_cursor_on_the_box_it_was_pressed_from(self):
        # Nothing about opening a screen moves the focus, and `_settle_focus`
        # only ever adjudicates between the message box and a decision prompt
        # — so escaping the peek lands back where Enter was pressed rather
        # than stranding the cursor on a row that is not drawn.
        ui = press(peeked(), "esc")
        assert ui.focus == WATCHERS

    def test_an_ordinary_key_does_not_close_it(self):
        assert isinstance(press(peeked(), "j").overlay, InspectOverlay)

    def test_a_long_log_scrolls(self):
        ui = recorded(build())
        ui.window("job 4821000", "\n".join(f"[12:41:07] line {i}" for i in range(4000)))
        assert "line 0" in screen(ui)
        press(ui, "pgdn", "pgdn")
        seen = screen(ui)
        assert "line 0" not in seen
        assert "/4000" in seen
        press(ui, "end")
        assert "line 3999" in screen(ui)

    def test_a_line_wider_than_the_terminal_is_folded_and_not_cut(self):
        # A `srun` line with forty flags on it is one line in the file and has
        # to be readable here, since the terminal's own selection is what
        # copies it back out.
        ui = recorded(build())
        ui.window("job 4821000", "srun " + "--exclusive " * 30)
        assert screen(ui, 80, 40).count("--exclusive") == 30
        assert widths(ui.render(80, 40)) == {80}

    def test_a_log_full_of_control_codes_does_not_corrupt_the_frame(self):
        # A batch script's progress bar is escape sequences and carriage
        # returns, and this is the first caller whose text is arbitrary bytes.
        ui = recorded(build())
        ui.window("job 4821000", "\x1b[2J\x1b[HDone.\r\nnext\tline")
        assert widths(ui.render(100, 30)) == {100}
        assert "\x1b[2J" not in screen(ui, 100, 30)
        assert "Done." in screen(ui, 100, 30)

    def test_what_arrives_while_it_is_open_still_lands(self):
        # "The main UI should not be impeded": the window is drawn over the
        # frame and holds nothing of the session's, so a turn that streams
        # while a log is being read streams into the state behind it and is on
        # screen the moment escape gives the rows back.
        ui = peeked()
        rows = len(ui.session.entries)
        ui.session.append(
            ChatEntry(kind="assistant", text="arrived while peeking", seq=9999)
        )
        assert isinstance(ui.overlay, InspectOverlay)
        assert len(ui.session.entries) == rows + 1
        assert "arrived while peeking" in screen(press(ui, "esc"), 120, 400)

    def test_and_the_conversation_underneath_is_not_disturbed(self):
        ui = recorded(build())
        ui.focus = WATCHERS
        before = (ui.chat.cursor, len(ui.session.entries))
        press(ui, "enter", "esc")
        assert (ui.chat.cursor, len(ui.session.entries)) == before


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

    # A `/conclude` proposal is a paragraph, not a line, and this screen used
    # to show the first 300 characters of it with a "…" on the end — which
    # reads as a memory the model cut off rather than a screen that did.

    def test_a_long_proposal_is_shown_whole(self):
        ui = recorded(build())
        text = " ".join(f"sentence {n} about the cluster." for n in range(20))
        ui.session.proposals = [Proposal(scope="rag", kind="learning", text=text)]
        ui.review_memories()
        seen = screen(ui)
        assert "sentence 0" in seen and "sentence 19" in seen
        assert "…" not in seen

    def test_and_scrolls_when_it_does_not_fit(self):
        ui = recorded(build())
        ui.session.proposals = [
            Proposal(
                scope="rag",
                kind="learning",
                text="\n".join(f"line {n}" for n in range(80)),
            )
        ]
        ui.review_memories()
        assert "line 0" in screen(ui, height=24)
        press(ui, "pgdn", height=24)
        assert "line 0" not in screen(ui, height=24)

    def test_and_the_next_proposal_starts_at_its_top(self):
        # Landing half-way down would hide the beginning of a memory nobody
        # has read a word of yet.
        ui = self._open()
        press(ui, "down", "down")
        assert ui.overlay.offset == 2
        press(ui, "y")
        assert ui.overlay.offset == 0


SUMMARY = (
    "[earlier in this session]\n"
    "The user is aligning a 40-sample cohort with STAR on /scratch/proj. "
    "Job 8813 died with an OOM at 32G; 40G was the fix. The QC report is "
    "still unread."
)


def proposal(**kw) -> CompactProposal:
    return CompactProposal(
        **{"summary": SUMMARY, "folded": 46, **kw}
    )


class TestTheCompactionReview:
    """specs-ui-acceptance.md, "Compaction": the summary is read before it is
    anybody's history, and a bad one is sent back with a sentence about it.

    The screen exists because a fold cannot be undone: `/compact` used to write
    the summary and *then* show it, so "it stops mid-sentence" was something
    the user could only ever say after the fact.
    """

    def _open(self, **kw) -> RowUI:
        ui = recorded(build())
        ui.session.compaction = proposal(**kw)
        ui.review_compaction()
        return ui

    def test_it_shows_what_is_being_decided(self):
        seen = screen(self._open())
        assert "46 messages fold into this summary" in seen
        assert "died with an OOM at 32G" in seen

    def test_and_the_instruction_the_summary_was_written_for(self):
        assert "asked to keep: the STAR flags" in screen(
            self._open(guidance="the STAR flags")
        )

    def test_a_cut_summary_is_flagged_as_cut(self):
        # The one thing the text cannot say about itself, and the reason the
        # core carries `truncated` at all.
        assert CUT_WARNING in screen(self._open(truncated=True))

    def test_an_untruncated_one_is_not(self):
        assert CUT_WARNING not in screen(self._open())

    def test_a_retry_says_which_attempt_this_is(self):
        assert "attempt 3" in screen(self._open(attempt=3))

    def test_enter_accepts_it(self):
        ui = press(self._open(), "enter")
        assert sent(ui, ResolveCompact)[-1].action == "accept"
        assert ui.overlay is None

    def test_and_names_the_session_it_was_offered_for(self):
        ui = self._open()
        session_id = ui.active_id
        press(ui, "enter")
        assert sent(ui, ResolveCompact)[-1].session_id == session_id

    def test_d_discards_it(self):
        ui = press(self._open(), "d")
        assert sent(ui, ResolveCompact)[-1].action == "discard"

    def test_escape_answers_nothing(self):
        # The core goes on holding the offer, so escape is not a verdict —
        # which is what makes it safe on a screen that cost a generation.
        ui = press(self._open(), "esc")
        assert sent(ui, ResolveCompact) == []
        assert ui.overlay is None

    def test_r_asks_again_with_what_the_user_typed(self):
        ui = press(self._open(), "r", *"finish the last sentence", "enter")
        answer = sent(ui, ResolveCompact)[-1]
        assert answer.action == "retry"
        assert answer.comment == "finish the last sentence"

    def test_an_empty_comment_is_refused_rather_than_sent(self):
        # The same generation again, with the model none the wiser.
        ui = press(self._open(), "r", "enter")
        assert sent(ui, ResolveCompact) == []
        assert isinstance(ui.overlay, CompactReviewOverlay)

    def test_escaping_the_comment_box_keeps_the_summary_on_screen(self):
        ui = press(self._open(), "r", *"never mind", "esc")
        assert isinstance(ui.overlay, CompactReviewOverlay)
        assert not ui.overlay.commenting
        assert sent(ui, ResolveCompact) == []
        # and it can still be accepted
        press(ui, "enter")
        assert sent(ui, ResolveCompact)[-1].action == "accept"

    def test_the_summary_scrolls(self):
        ui = self._open(summary="\n".join(f"line {n}" for n in range(80)))
        assert "line 0" in screen(ui, height=24)
        press(ui, "pgdn", width=120, height=24)
        assert "line 0" not in screen(ui, height=24)

    def test_nothing_to_review_opens_nothing(self):
        ui = build()
        ui.session.compaction = None
        ui.review_compaction()
        assert ui.overlay is None

    def test_a_screen_with_no_summary_still_draws(self):
        drawn = CompactReviewOverlay().render(80, 12)
        assert widths(drawn) == {80}

    def test_it_waits_rather_than_landing_on_an_open_screen(self):
        # It arrives a model call after the keystroke, and by then the user may
        # be halfway through a settings file (`RowUI.land`).
        ui = recorded(build())
        ui.focus = SESSIONS
        ui.handle("c", 120, 40)  # the config editor
        assert isinstance(ui.overlay, ConfigOverlay)
        ui.session.compaction = proposal()
        ui.review_compaction()
        assert isinstance(ui.overlay, ConfigOverlay)
        ui.handle("esc", 120, 40)
        assert isinstance(ui.overlay, CompactReviewOverlay)


# ---------------------------------------------------------- the frame itself

# Overlays are where the width discipline breaks, because they draw frames
# inside frames. Every screen, at the three widths the port is measured at.


def scanning_llms(rows) -> LlmOverlay:
    """Manage-LLMs mid-sweep: an empty panel with a sentence in it, which is a
    different row from a panel with endpoints in it."""
    overlay = LlmOverlay([x for x in rows if not x.discovered])
    overlay.scanning = True
    overlay.note = "scanning…"
    overlay.refresh()
    return overlay


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
        "scanning": scanning_llms(catalog_rows),
        "form": BackendFormOverlay(base_url="http://x/v1"),
        "models": ModelPickerOverlay(catalog_rows),
        "inspect": InspectOverlay("a line\n" * 40, title="skills"),
        "memory": MemoryReviewOverlay(proposals(), "s1"),
        "compact": CompactReviewOverlay(proposal(truncated=True), "s1"),
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


# ------------------------------------------- a screen's footer wraps as well


class TestAScreenFooterWraps:
    """A screen has its own keys and the same narrow terminal to draw them in,
    so the wrapping is the frame's, not the rows'."""

    def test_no_hint_falls_off_the_end_of_a_narrow_screen_footer(self):
        ui = build()
        ui.overlay = LlmOverlay()
        rows = ui._screen_footer(ui.overlay, 60, 40)
        assert len(rows) > 1, "the case is only interesting once it wraps"
        shown = "  ".join(plain(row) for row in rows)
        for key, label in ui.overlay.footer():
            assert f"{key} {label}" in shown

    def test_the_body_gives_up_what_the_footer_takes(self):
        ui = build()
        ui.overlay = LlmOverlay()
        for width in (200, 120, 90, 60, 40, 24):
            rows = len(ui._screen_footer(ui.overlay, width, 40))
            assert ui._screen_h(ui.overlay, width, 40) == 40 - 1 - rows, width

    def test_the_frame_is_exact_at_every_width_the_footer_wraps_at(self):
        ui = build()
        ui.overlay = LlmOverlay()
        for width in range(200, 19, -1):
            drawn = ui.render(width, 40)
            assert len(drawn) == 40
            assert widths(drawn) == {width}, width

    def test_what_a_page_key_scrolls_is_what_was_drawn(self):
        # `handle` and `render` ask the same question, or page-down moves by a
        # different amount than the screen showed.
        ui = build()
        ui.overlay = LlmOverlay()
        drawn = len(ui._screen_footer(ui.overlay, 50, 40))
        assert ui._screen_h(ui.overlay, 50, 40) == 40 - 1 - drawn
