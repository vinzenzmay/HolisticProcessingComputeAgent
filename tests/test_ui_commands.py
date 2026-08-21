"""Tests for the slash commands and the menu that offers them (M8).

The claims come from specs-ui-acceptance.md: "Slash-command menu" whole, the
`/memorize` and `/conclude` claims under "Memory", "Compaction", and the
`/skill-*` claims under "Skills and self-review".

Three claims on that list were once "not satisfied here" and are now closed,
tested for what they promise: frequency ordering (the counts arrive as
`command.counts`), `/plan` on a fresh install (`skill.list` has a scope, and
the menu asks for the wide one), and `/skill-creator <what it should do>` (the
draft is a model call, so it crosses as `skill.draft` and comes back into the
same form).
"""

import subprocess
import sys

import pytest

from hpca.ui import commands
from hpca.ui.app import INPUT, RowUI
from hpca.ui.commands import BUILTINS, Command
from hpca.ui.demo import build
from hpca.ui.overlays import (
    InspectOverlay,
    SkillCreatorOverlay,
    SkillRemoveOverlay,
    ThinkingOverlay,
    skill_file,
)
from hpca.ui.state import (
    DeleteSkill,
    DraftSkill,
    ProfileInfo,
    RunCommand,
    SaveSkill,
    SessionState,
    SkillInfo,
    Submit,
)
from tests.ui_harness import frame, plain, served, widths

WIDTHS = [80, 100, 137]

# What a profile's own skills look like once `skill.rows` has answered. The
# description carries brackets and a dollar on purpose: it is user-written text
# from a file on disk, and the Textual menu had to assemble styled spans to
# stop a markup parser eating exactly these characters.
SKILLS = {
    "hpc": [
        SkillInfo("merge-vcfs", "merge shard VCFs [see $MANIFEST] (bcftools)"),
        SkillInfo("submit-gpu", "the partition and the flags that work"),
        # What HPCA ships. Callable by every profile with no setup, and not
        # this profile's to edit or remove — which is what `level` is for.
        SkillInfo("plan", "break work into steps first", level="builtin"),
    ]
}


def app(skills=None, profile: str = "hpc") -> RowUI:
    """A UI with one session, its profile's skills answered, box focused.

    No `recorded()`: a bare `RowUI` already records into ``ui.intents``, and
    wrapping it would put every intent in there twice.
    """
    ui = RowUI(profiles=[ProfileInfo(name=profile)])
    session = SessionState("s1", profile=profile)
    ui.sessions = [session]
    ui._states = {"s1": session}
    ui.refresh_sidebar()
    ui.active = 0
    ui.focus = INPUT
    return served(ui, {}, SKILLS if skills is None else skills)


def press(ui: RowUI, *keys: str, width: int = 120, height: int = 40) -> RowUI:
    for key in keys:
        ui.handle(key, width, height)
    return ui


def screen(ui: RowUI, width: int = 120, height: int = 40) -> str:
    return "\n".join(frame(ui, width, height))


def names(ui: RowUI) -> list[str]:
    return [x.name for x in ui.menu()]


def sent(ui: RowUI, kind: type) -> list:
    return [x for x in ui.intents if isinstance(x, kind)]


# ------------------------------------------------------------------ the menu


class TestTheMenu:
    """specs-ui-acceptance.md, "Slash-command menu"."""

    def test_a_slash_lists_every_command(self):
        ui = press(app(), "/")
        assert set(names(ui)) >= {x.name for x in BUILTINS}
        assert "merge-vcfs" in names(ui), "the profile's skills are offered too"

    def test_and_the_frame_shows_them(self):
        assert "/memorize" in screen(press(app(), "/"))

    def test_matching_is_substring_not_only_prefix(self):
        # "I remember half the name" is the case the menu exists for: `oncl`
        # is nowhere near the start of `conclude` and finds it anyway.
        assert names(press(app(), *"/oncl")) == ["conclude"]
        assert names(press(app(), *"/vcfs")) == ["merge-vcfs"]

    def test_skill_narrows_to_the_skill_commands(self):
        assert names(press(app(), *"/skill")) == [
            "skill-creator",
            "skills-list",
            "skill-remove",
        ]

    def test_a_space_after_the_name_closes_it(self):
        # The command is settled; what follows is its arguments.
        assert names(press(app(), *"/compact ")) == []

    def test_so_the_arrows_go_back_to_the_draft(self):
        # With the menu closed, ↑ is the editor's again — it moves within the
        # draft rather than picking a command that is already settled.
        ui = press(app(), *"/compact keep it", "alt-enter", *"and this")
        press(ui, "up")
        assert ui.focus == INPUT
        assert ui.input.row == 0, "back up into the first line of the draft"

    def test_no_match_hides_the_menu(self):
        assert names(press(app(), *"/zzz")) == []
        assert "commands (" not in screen(press(app(), *"/zzz"))

    def test_a_backslash_lists_them_too(self):
        assert "memorize" in names(press(app(), "\\"))

    def test_an_ordinary_draft_has_no_menu(self):
        assert names(press(app(), *"hello /not a command")) == []

    def test_down_and_up_move_the_selection(self):
        ui = press(app(), "/")
        assert ui.session.menu_at == 0
        press(ui, "down", "down")
        assert ui.session.menu_at == 2
        press(ui, "up")
        assert ui.session.menu_at == 1

    def test_and_the_selection_wraps(self):
        ui = press(app(), "/")
        press(ui, "up")
        assert ui.session.menu_at == len(ui.menu()) - 1

    def test_enter_fills_a_partial_command(self):
        ui = press(app(), *"/comp", "enter")
        assert ui.input.text() == "/compact "
        assert sent(ui, RunCommand) == [], "filling is not running"

    def test_tab_fills_it_too(self):
        assert press(app(), *"/comp", "tab").input.text() == "/compact "

    def test_enter_on_a_complete_command_runs_it(self):
        ui = press(app(), *"/compact", "enter")
        assert [x.name for x in sent(ui, RunCommand)] == ["compact"]

    def test_the_highlighted_row_is_what_is_filled(self):
        ui = press(app(), *"/skill", "down", "enter")
        assert ui.input.text() == "/skills-list "

    def test_a_skill_description_is_not_parsed_as_markup(self):
        # It is a sentence out of a file on disk. The Textual menu had to
        # assemble styled spans to keep a markup parser off exactly this.
        seen = screen(press(app(), *"/merge"))
        assert "[see $MANIFEST]" in seen

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_menu_keeps_every_row_exact(self, width: int):
        ui = press(app(), "/")
        assert widths(ui.render(width, 24)) == {width}

    @pytest.mark.parametrize("width", WIDTHS)
    def test_even_when_a_description_is_far_too_long(self, width: int):
        ui = app({"hpc": [SkillInfo("wide", "x" * 400)]})
        assert widths(press(ui, *"/wide").render(width, 24)) == {width}

    def test_the_footer_offers_the_menu_keys(self):
        foot = plain(press(app(), "/").render(160, 40)[-1])
        assert "⇥ complete" in foot and "↑↓ pick" in foot


class TestFrequencyOrdering:
    """"Frequency sorts the most-used first."

    The counts are the core's — rule 2 of §4.2 forbids this side reading the
    table — and they arrive as `command.counts`, asked for on connect and
    restated whenever one changes. Until one arrives the menu is in definition
    order, which is what a table nobody has run anything from would give.
    """

    def test_the_order_is_definition_order_until_the_counts_arrive(self):
        assert names(press(app(), "/"))[: len(BUILTINS)] == [
            x.name for x in BUILTINS
        ]

    def test_and_most_used_first_once_they_do(self):
        ui = app()
        ui.command_counts = {"thinking": 9, "conclude": 4}
        offered = names(press(ui, "/"))
        assert offered[:2] == ["thinking", "conclude"]
        # Everything uncounted keeps definition order behind them.
        rest = [x.name for x in BUILTINS if x.name not in ("thinking", "conclude")]
        assert [x for x in offered if x in rest] == rest

    def test_a_narrowed_menu_is_sorted_too(self):
        ui = app()
        ui.command_counts = {"skill-remove": 3}
        assert names(press(ui, *"/skill"))[0] == "skill-remove"

    def test_the_ui_does_not_read_the_usage_table(self):
        # The rule that makes a socket-separated core possible, checked in a
        # clean interpreter the way the protocol boundary is.
        code = (
            "import sys, hpca.ui.app; "
            "assert 'hpca.db' not in sys.modules, sorted(sys.modules)"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr

    def test_the_sort_itself_is_a_pure_function_of_the_counts(self):
        table = [Command("a", "/a"), Command("b", "/b"), Command("c", "/c")]
        assert [x.name for x in commands.matching("", table)] == ["a", "b", "c"]
        ranked = commands.matching("", table, {"c": 9, "b": 3})
        assert [x.name for x in ranked] == ["c", "b", "a"]


class TestAParkedDraft:
    """specs-ui-acceptance.md, "Drafts" — left undone in M6."""

    def _two(self) -> RowUI:
        from hpca.ui.state import SessionState

        ui = app()
        second = SessionState("s2", profile="hpc")
        ui.sessions.append(second)
        ui._states["s2"] = second
        ui.refresh_sidebar()
        return ui

    def test_a_parked_slash_draft_brings_its_menu_back(self):
        ui = self._two()
        press(ui, *"/comp")
        ui.active = 1  # switched away
        assert names(ui) == [], "the other session's draft is empty"
        ui.active = 0
        assert names(ui) == ["compact"]

    def test_a_session_with_no_draft_inherits_no_menu(self):
        ui = self._two()
        press(ui, *"/comp")
        ui.active = 1
        assert "commands (" not in screen(ui)


# -------------------------------------------------------------- the commands


class TestTheSevenBuiltins:
    """§4.3 item 24. Four go to the core; three draw a screen instead."""

    @pytest.mark.parametrize("name", ["memorize", "conclude", "compact"])
    def test_it_reaches_the_core_as_command_run(self, name: str):
        ui = press(app(), *f"/{name}", "enter")
        ran = sent(ui, RunCommand)[-1]
        assert (ran.name, ran.session_id) == (name, "s1")

    def test_and_carries_the_rest_of_the_line_unsplit(self):
        ui = press(app(), *"/memorize the queue is called gpu-a100", "enter")
        assert sent(ui, RunCommand)[-1].args == "the queue is called gpu-a100"

    def test_a_backslash_command_runs_the_same_way(self):
        ui = press(app(), *"\\memorize a note", "enter")
        assert sent(ui, RunCommand)[-1].name == "memorize"

    def test_the_draft_is_cleared_once_it_runs(self):
        assert press(app(), *"/conclude", "enter").input.text() == ""

    def test_skills_list_goes_to_the_core_with_the_session_on_it(self):
        # The session is how the core decides *which* profile is asking
        # (`core.service._list_skills`).
        ui = press(app(), *"/skills-list", "enter")
        ran = sent(ui, RunCommand)[-1]
        assert (ran.name, ran.session_id) == ("skills-list", "s1")

    def test_thinking_opens_the_chooser_and_sends_nothing(self):
        ui = press(app(), *"/thinking", "enter")
        assert isinstance(ui.overlay, ThinkingOverlay)
        assert sent(ui, RunCommand) == []

    def test_a_command_is_refused_while_this_session_is_working(self):
        ui = app()
        ui.session.turn.working = True
        press(ui, *"/compact", "enter")
        assert sent(ui, RunCommand) == []
        assert "cannot be queued" in ui.note

    def test_but_a_screen_command_is_not(self):
        # It draws something and asks the core for nothing, so there is no
        # worker for the running turn to collide with.
        ui = app()
        ui.session.turn.working = True
        press(ui, *"/thinking", "enter")
        assert isinstance(ui.overlay, ThinkingOverlay)

    def test_a_session_command_with_nothing_open_says_so(self):
        ui = RowUI()
        ui.focus = INPUT
        press(ui, *"/compact", "enter")
        assert sent(ui, RunCommand) == []
        assert "no session open" in ui.note


class TestWhatACompactDoesToTheMeter:
    """specs-ui-coverage.md §3.6: both halves were green and the seam was not.

    The core folds the thread and re-derives the fill (`forget_session` then
    `estimate_context`), and deliberately sends no `chat.reset` — a fold
    rewrites the conversation without taking a message out of it. So nothing
    ever cleared `Context.measured`, `Context.estimate` went on returning
    early, and a session compacted at 92% stayed at 92% indefinitely.
    """

    def measured(self) -> RowUI:
        ui = app()
        ui.session.context.measure(9200, 10000)
        return ui

    def test_the_measured_number_stops_claiming_to_be_measured(self):
        ui = self.measured()
        assert ui.session.context.measured
        press(ui, *"/compact", "enter")
        assert not ui.session.context.measured

    def test_and_the_bar_says_so_with_the_tilde(self):
        ui = self.measured()
        press(ui, *"/compact", "enter")
        assert "~9,200 / 10,000" in screen(ui)

    def test_and_the_fresh_estimate_is_no_longer_ignored(self):
        ui = self.measured()
        press(ui, *"/compact", "enter")
        ui.session.context.estimate(1200, 10000)
        assert "~1,200 / 10,000" in screen(ui)

    def test_the_fill_is_not_blanked_in_the_meantime(self):
        # Not `reset()`: the number is stale, not gone, and "no reply yet" on
        # a conversation with a hundred entries would be the worse lie.
        ui = self.measured()
        press(ui, *"/compact", "enter")
        assert ui.session.context.known
        assert "no reply yet" not in screen(ui)

    def test_a_command_that_does_not_fold_leaves_it_alone(self):
        ui = self.measured()
        press(ui, *"/conclude", "enter")
        assert ui.session.context.measured

    def test_a_refused_compact_leaves_it_alone_too(self):
        ui = self.measured()
        ui.session.turn.working = True
        press(ui, *"/compact", "enter")
        assert ui.session.context.measured


class TestALongAnswerGetsAWindow:
    """§3.7: `notify` carries a title, and a heading over a block is not a toast.

    `/skills-list` prints two lines per skill and `/compact` answers with the
    summary itself; both were being cut to three lines and "… more in the
    log", where they never were — the log records turns, not command answers.
    `RowUI.inspect` was built and tested for exactly this and had no caller.
    """

    def listing(self) -> str:
        return "\n".join(f"• skill-{i}\n    what it does" for i in range(6))

    def test_a_titled_block_opens_the_window(self):
        ui = app()
        ui.toast(self.listing(), title="Skills · profile “hpc”")
        assert isinstance(ui.overlay, InspectOverlay)
        assert "skill-5" in screen(ui)

    def test_and_the_heading_is_the_window_title(self):
        ui = app()
        ui.toast(self.listing(), title="Skills · profile “hpc”")
        assert "Skills · profile “hpc”" in screen(ui)

    def test_and_it_stays_until_escape(self):
        ui = app()
        ui.toast(self.listing(), title="Skills")
        assert isinstance(press(ui, "x").overlay, InspectOverlay)
        assert press(ui, "esc").overlay is None

    def test_a_one_liner_with_a_heading_is_still_a_toast(self):
        # The xhigh warning is a heading over one long paragraph, and it is
        # meant to be read and dismissed rather than opened.
        ui = app()
        ui.toast("x" * 400, title="Thinking: xhigh — NOT USABLE")
        assert ui.overlay is None

    def test_and_so_is_a_block_with_no_heading_at_all(self):
        ui = app()
        ui.toast("one\ntwo\nthree\nfour\nfive")
        assert ui.overlay is None

    def test_the_toast_is_raised_either_way(self):
        ui = app()
        ui.toast(self.listing(), title="Skills")
        assert ui.toasts, "the footer note and the block are not replaced by it"

    def test_it_waits_rather_than_covering_a_screen_the_user_opened(self):
        ui = app()
        press(ui, *"/thinking", "enter")
        ui.toast(self.listing(), title="Skills")
        assert isinstance(ui.overlay, ThinkingOverlay)
        press(ui, "esc")
        assert isinstance(ui.overlay, InspectOverlay)


class TestAnUnknownCommand:
    def test_it_is_reported_and_not_sent_to_the_model(self):
        ui = press(app(), *"/memorise a note", "enter")
        assert sent(ui, Submit) == []
        assert sent(ui, RunCommand) == []
        assert "unknown command: /memorise" in ui.note

    def test_the_draft_stays_to_be_corrected(self):
        ui = press(app(), *"/memorise a note", "enter")
        assert ui.input.text() == "/memorise a note"

    def test_and_correcting_it_then_runs(self):
        # The draft is still there to be fixed, which is the whole point of
        # not clearing it: the sentence attached to the typo survives.
        ui = press(app(), *"/memorise a note", "enter")
        for _ in range(len(" a note") + 1):
            press(ui, "left")
        press(ui, "backspace", "z")
        assert ui.input.text() == "/memorize a note"
        press(ui, "end", "enter")
        assert sent(ui, RunCommand)[-1].name == "memorize"


class TestASkillByName:
    """`/<skill> …` is an ordinary turn with the procedure forced in."""

    def test_it_runs_a_turn_carrying_the_skill(self):
        ui = press(app(), *"/merge-vcfs shard 4 and 5", "enter")
        turn = sent(ui, Submit)[-1]
        assert turn.forced_skill == "merge-vcfs"
        assert turn.text == "/merge-vcfs shard 4 and 5"
        assert sent(ui, RunCommand) == []

    def test_a_built_in_wins_a_name_clash(self):
        ui = app({"hpc": [SkillInfo("compact", "a skill with a built-in's name")]})
        press(ui, *"/compact", "enter")
        assert sent(ui, Submit) == []
        assert sent(ui, RunCommand)[-1].name == "compact"

    def test_a_skill_shows_in_the_menu(self):
        assert "merge-vcfs" in names(press(app(), "/"))

    def test_a_skill_queues_like_any_message(self):
        # It is a turn, not a command: the queue is where a turn waits.
        ui = app()
        ui.session.turn.working = True
        press(ui, *"/merge-vcfs go", "enter")
        assert sent(ui, Submit)[-1].forced_skill == "merge-vcfs"

    def test_a_shipped_skill_runs_out_of_the_box(self):
        """"`/plan …` works out of the box on a fresh install".

        The menu asks `skill.list` for the *visible* scope — everything the
        profile can call, shipped skills included — rather than the `own`
        scope an editor asks for. A profile with no skills of its own still
        has HPCA's, and that is what a fresh install is.
        """
        ui = press(app(), *"/plan the migration", "enter")
        turn = sent(ui, Submit)[-1]
        assert turn.forced_skill == "plan"
        assert turn.text == "/plan the migration"
        assert "unknown command" not in ui.note

    def test_and_a_profile_with_nothing_of_its_own_still_has_it(self):
        # The fresh install exactly: no profile skills at all.
        shipped = {"hpc": [SkillInfo("plan", "steps first", level="builtin")]}
        assert "plan" in names(press(app(shipped), "/"))
        ran = press(app(shipped), *"/plan it", "enter")
        assert sent(ran, Submit)[-1].forced_skill == "plan"

    def test_but_it_is_not_offered_for_removal(self):
        # Visible is not the same as the profile's own: a shipped skill
        # belongs to the package, and `skill.delete` will not take it.
        ui = press(app(), *"/skill-remove", "enter")
        assert "plan" not in screen(ui)
        assert "merge-vcfs" in screen(ui)


# ------------------------------------------------------------- the skill screens


class TestSkillCreator:
    """specs-ui-acceptance.md, "Skills and self-review", `/skill-creator`."""

    def _form(self, ui: RowUI | None = None) -> RowUI:
        return press(ui or app(), *"/skill-creator", "enter")

    def test_it_opens_the_form_and_makes_no_backend_call(self):
        ui = self._form()
        assert isinstance(ui.overlay, SkillCreatorOverlay)
        assert sent(ui, RunCommand) == []

    def test_it_is_tied_to_the_profile(self):
        assert self._form().overlay.profile == "hpc"

    def test_an_empty_form_is_abandoned_with_escape(self):
        ui = press(self._form(), "esc")
        assert ui.overlay is None

    def test_an_empty_name_is_rejected(self):
        ui = press(self._form(), "ctrl-s")
        assert isinstance(ui.overlay, SkillCreatorOverlay)
        assert "a name is needed" in screen(ui)

    def test_a_duplicate_name_is_refused(self):
        ui = press(self._form(), *"merge-vcfs", "ctrl-s")
        assert isinstance(ui.overlay, SkillCreatorOverlay)
        assert "already exists" in screen(ui)

    def test_a_name_with_a_space_is_refused(self):
        ui = press(self._form(), *"two words", "ctrl-s")
        assert "cannot contain spaces" in screen(ui)

    def test_it_saves_name_description_and_body(self):
        ui = self._form()
        press(ui, *"queue-check", "tab", *"how to read the queue", "tab")
        press(ui, *"run squeue -u $USER", "ctrl-s")
        saved = sent(ui, SaveSkill)[-1]
        assert (saved.profile, saved.name) == ("hpc", "queue-check")
        assert "how to read the queue" in saved.text
        assert "squeue -u $USER" in saved.text
        assert saved.text.startswith("---\n"), "front matter and all"

    def test_and_the_new_skill_is_in_the_menu_at_once(self):
        ui = self._form()
        press(ui, *"queue-check", "tab", *"how to read the queue", "ctrl-s")
        assert "queue-check" in names(press(ui, "/"))

    def test_a_half_written_form_asks_before_it_is_abandoned(self):
        ui = press(self._form(), *"queue-check", "esc")
        assert isinstance(ui.overlay, SkillCreatorOverlay)
        assert "Save this skill?" in screen(ui)

    def test_a_description_with_a_colon_still_parses_as_a_skill_file(self):
        import yaml

        text = skill_file("q", "note: this has a colon", "body")
        front = text.split("---")[1]
        assert yaml.safe_load(front)["description"] == "note: this has a colon"

    def test_an_argument_asks_the_core_for_a_draft(self):
        """"`/skill-creator <what it should do>` drafts via the model."

        A draft is a model call, so it happens on the core's side and crosses
        as `skill.draft` — with the open session on it, so the conversation
        goes to the drafter and the wait is reported where the user is
        looking. Nothing is drawn in the meantime.
        """
        ui = press(app(), *"/skill-creator something about queues", "enter")
        asked = sent(ui, DraftSkill)[-1]
        assert (asked.profile, asked.request) == ("hpc", "something about queues")
        assert asked.session_id == "s1"
        assert ui.overlay is None, "the form opens when the draft lands"
        assert sent(ui, RunCommand) == []
        assert "drafting" in ui.note

    def test_and_a_bare_command_asks_for_none(self):
        ui = self._form()
        assert sent(ui, DraftSkill) == []
        assert sent(ui, RunCommand) == []

    def test_the_draft_opens_the_same_form_pre_filled(self):
        ui = press(app(), *"/skill-creator watch a run", "enter")
        ui.skill_drafted(
            SkillInfo("watch-run", "when a run needs watching", "1. squeue"),
            "watch a run",
        )
        assert isinstance(ui.overlay, SkillCreatorOverlay)
        assert "watch-run" in screen(ui)
        assert "1. squeue" in screen(ui)

    def test_and_it_saves_like_any_other_skill(self):
        # Edited and confirmed exactly like a hand-typed one: the draft is a
        # head start, not an author.
        ui = press(app(), *"/skill-creator watch a run", "enter")
        ui.skill_drafted(
            SkillInfo("watch-run", "when a run needs watching", "1. squeue"),
            "watch a run",
        )
        press(ui, "ctrl-s")
        saved = sent(ui, SaveSkill)[-1]
        assert (saved.profile, saved.name) == ("hpc", "watch-run")
        assert "1. squeue" in saved.text and saved.text.startswith("---\n")

    def test_a_failed_draft_still_opens_the_empty_form(self):
        ui = press(app(), *"/skill-creator watch a run", "enter")
        ui.skill_drafted(None, "watch a run", error="no backend answered")
        assert isinstance(ui.overlay, SkillCreatorOverlay)
        assert ui.overlay.value("name") == ""
        assert "could not draft" in screen(ui)

    def test_and_an_empty_pre_filled_form_is_still_abandoned_with_escape(self):
        ui = press(app(), *"/skill-creator watch a run", "enter")
        ui.skill_drafted(None, "watch a run", error="no backend answered")
        press(ui, "esc")
        assert ui.overlay is None
        assert sent(ui, SaveSkill) == []

    def test_the_level_is_the_profiles_by_default(self):
        ui = self._form()
        press(ui, *"queue-check", "ctrl-s")
        assert sent(ui, SaveSkill)[-1].level == "profile"

    def test_and_a_global_skill_is_visible_but_not_own(self):
        ui = self._form()
        press(ui, *"queue-check", "tab", "tab", "right")  # name, then the level
        assert ui.overlay.level == "global"
        press(ui, "ctrl-s")
        assert sent(ui, SaveSkill)[-1].level == "global"
        # Offered by the menu, and not by the picker that deletes.
        assert "queue-check" in names(press(ui, "/"))
        assert "queue-check" not in screen(press(ui, *"/skill-remove", "enter"))

    def test_and_a_project_skill_is_removable(self):
        ui = self._form()
        press(ui, *"run-cohort", "tab", "tab", "right", "right")
        assert ui.overlay.level == "project"
        press(ui, "ctrl-s")
        assert sent(ui, SaveSkill)[-1].level == "project"
        assert "run-cohort" in screen(press(ui, *"/skill-remove", "enter"))

    def test_a_name_taken_at_another_level_is_not_a_duplicate(self):
        # The same name at two levels is how a project overrides a profile's
        # procedure; only the same name at the *same* level is a collision.
        ui = self._form()
        press(ui, *"merge-vcfs", "tab", "tab", "right", "right", "ctrl-s")
        assert sent(ui, SaveSkill)[-1].level == "project"

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_form_keeps_every_row_exact(self, width: int):
        assert widths(self._form().render(width, 24)) == {width}


class TestSkillRemove:
    def _picker(self, ui: RowUI | None = None) -> RowUI:
        return press(ui or app(), *"/skill-remove", "enter")

    def test_a_bare_command_opens_a_picker_of_the_profiles_own(self):
        ui = self._picker()
        assert isinstance(ui.overlay, SkillRemoveOverlay)
        assert "merge-vcfs" in screen(ui)

    def test_enter_asks_first(self):
        ui = press(self._picker(), "enter")
        assert sent(ui, DeleteSkill) == []
        assert "Remove skill" in screen(ui)

    def test_and_yes_removes_the_chosen_one(self):
        ui = press(self._picker(), "down", "enter", "y")
        gone = sent(ui, DeleteSkill)[-1]
        assert (gone.profile, gone.name) == ("hpc", "submit-gpu")

    def test_cancel_keeps_it(self):
        ui = press(self._picker(), "enter", "n")
        assert sent(ui, DeleteSkill) == []

    def test_escape_keeps_them_all(self):
        ui = press(self._picker(), "esc")
        assert ui.overlay is None
        assert sent(ui, DeleteSkill) == []

    def test_no_own_skills_notifies_instead_of_opening_anything(self):
        ui = self._picker(app({"hpc": []}))
        assert ui.overlay is None
        assert "no skills of its own" in ui.note

    def test_a_named_skill_goes_to_the_core_instead(self):
        # `/skill-remove <name>` is a command the core answers; only the bare
        # form is a picker.
        ui = press(app(), *"/skill-remove merge-vcfs", "enter")
        assert ui.overlay is None
        ran = sent(ui, RunCommand)[-1]
        assert (ran.name, ran.args) == ("skill-remove", "merge-vcfs")


# ---------------------------------------------------------- the pure functions


class TestParsing:
    @pytest.mark.parametrize("prefix", ["/", "\\"])
    def test_both_prefixes_split_the_same(self, prefix: str):
        assert commands.split(f"{prefix}compact keep it") == ("compact", "keep it")

    def test_a_plain_message_is_not_a_command(self):
        assert commands.split("compact the logs") is None

    def test_a_name_being_typed_opens_the_menu(self):
        assert commands.typed_name("/comp") == "comp"

    def test_a_settled_name_closes_it(self):
        assert commands.typed_name("/compact x") is None

    def test_a_multi_line_draft_that_opens_with_a_slash_closes_it(self):
        assert commands.typed_name("/notes\nand more") is None

    def test_a_skill_named_like_a_builtin_is_dropped(self):
        table = commands.all_commands([("compact", "mine"), ("ok", "fine")])
        assert [x.name for x in table if x.name == "compact"] == ["compact"]
        assert next(x for x in table if x.name == "compact").builtin

    def test_a_skill_whose_name_has_a_space_is_left_out(self):
        # `/<skill>` splits on the first space, so it could never be selected.
        table = commands.all_commands([("two words", "x")])
        assert all(" " not in x.name for x in table)


def test_the_demo_still_draws_the_menu():
    """`python -m hpca.ui.run --demo` is the one end-to-end check M0 left."""
    ui = build()
    ui.focus = INPUT
    press(ui, "/")
    assert "/compact" in screen(ui)
    assert widths(ui.render(100, 30)) == {100}
