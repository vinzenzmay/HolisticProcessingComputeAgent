"""Tests for M8's four edges: toasts, the clipboard, `$EDITOR` and quit.

§4.3 items 35-38, plus the claim carried over from the deleted
`test_tui_notify.py` — "a toast must never crash on arbitrary text, including
text containing square brackets from LLM output". That one is the reason this
file exists: the Textual app answered it with ``markup=False``, this renderer
has no markup to turn off, and what is left is width and control characters.
"""

import contextlib
import io
import json
import os

import pytest

from hpca import protocol
from hpca.ui import theme, toasts
from hpca.ui.ansi import RESET
from hpca.ui.app import CHAT, INPUT, SESSIONS, WATCHERS, RowUI
from hpca.ui.demo import build
from hpca.ui.overlays import ConfigOverlay, HelpOverlay
from hpca.ui.run import Loop, copier
from hpca.ui.screen import ENTER_MODES, EXIT_MODES, Screen
from hpca.ui.state import EditProfile, Toast
from tests.ui_harness import (
    clocked,
    connected,
    footer,
    frame,
    plain,
    settle,
    widths,
)

WIDTHS = [80, 100, 137]

# An escape sequence, a tab, a carriage return and a DEL — every one of them
# zero cells to `cell_width` and *something* to a terminal, which is the
# combination a differential repaint cannot survive.
NASTY = "\x1b[31mred\x1b[0m\ttabbed\r\x7f and [brackets] and $(dollars)"


def press(ui: RowUI, *keys: str, width: int = 120, height: int = 40) -> RowUI:
    for key in keys:
        ui.handle(key, width, height)
    return ui


def screen(ui: RowUI, width: int = 120, height: int = 40) -> str:
    return "\n".join(frame(ui, width, height))


# ------------------------------------------------------------------- toasts


class TestAToastNeverBreaksTheFrame:
    """The hard requirement, carried over from `test_tui_notify.py`."""

    @pytest.mark.parametrize("width", WIDTHS)
    def test_arbitrary_text_keeps_every_row_exact(self, width: int):
        ui = clocked(build())
        ui.toast(NASTY, "error")
        assert widths(ui.render(width, 24)) == {width}

    def test_and_the_control_characters_are_gone_from_the_frame(self):
        # Not merely "the widths still add up": an escape sequence that
        # reached the terminal would move the cursor, and the next row would
        # be painted somewhere else entirely.
        ui = clocked(build())
        ui.toast(NASTY)
        body = "".join(ui.render(120, 24))
        assert "\x1b[31m" not in body
        assert "\t" not in body and "\r" not in body and "\x7f" not in body

    def test_the_brackets_survive_as_characters(self):
        # They are text, not markup. The failure this replaces was the *other*
        # direction: a bracket parsed and taking the app down with it.
        ui = clocked(build())
        ui.toast("wrote [3] files to $(pwd)")
        assert "[3]" in screen(ui)

    @pytest.mark.parametrize("width", WIDTHS)
    def test_one_hundred_kilobytes_on_one_line(self, width: int):
        ui = clocked(build())
        ui.toast("x" * 100_000)
        rows = ui.render(width, 24)
        assert widths(rows) == {width}
        assert len(rows) == 24

    def test_and_it_costs_a_bounded_number_of_rows(self):
        block = toasts.one(Toast("y" * 100_000), 80, 20)
        assert len(block) <= 2 + toasts.MAX_BODY
        assert toasts.CLIPPED in plain("".join(block))

    def test_a_hundred_lines_are_clipped_rather_than_drawn(self):
        block = toasts.one(Toast("\n".join(str(i) for i in range(100))), 80, 20)
        assert len(block) <= 2 + toasts.MAX_BODY

    @pytest.mark.parametrize("width", WIDTHS)
    def test_a_toast_over_an_overlay_is_exact_too(self, width: int):
        ui = clocked(build())
        ui.overlay = HelpOverlay()
        ui.toast(NASTY, "warning")
        assert widths(ui.render(width, 24)) == {width}

    def test_a_terminal_too_short_for_one_still_renders(self):
        ui = clocked(build())
        ui.toast("something")
        assert widths(ui.render(80, 4)) == {80}


class TestWhatAToastSays:
    def test_the_title_the_core_sends_is_drawn(self):
        ui = clocked(build())
        ui.toast("• merge-vcfs\n• submit-gpu", title="Removable skills")
        seen = screen(ui)
        assert "Removable skills" in seen
        assert "merge-vcfs" in seen

    def test_the_severity_names_itself(self):
        ui = clocked(build())
        ui.toast("the backend is down", "error")
        assert "error" in screen(ui)

    def test_it_is_in_the_footer_too(self):
        # One line, whatever the block did with the rest: a message that
        # expired while the user was reading the chat is still answerable.
        ui = clocked(build())
        ui.toast("line one\nline two")
        assert "line one line two" in footer(ui, 120, 24)


class TestWhereAToastIsDrawn:
    """At the bottom, over the panes and never over the key hints.

    It used to sit directly under the header, over the sessions list — the
    controls a user reaches for first — so a notification hid the rows that
    mattered most for as long as it was up.
    """

    def rows(self, ui: RowUI, height: int = 40) -> tuple[list[str], int]:
        return ui.render(120, height), len(ui._footer(120, height))

    def test_it_ends_on_the_row_above_the_footer(self):
        ui = clocked(build())
        ui.toast("look down here")
        drawn, footer_h = self.rows(ui)
        text = [plain(x) for x in drawn]
        last = len(text) - footer_h - 1
        assert "look down here" in text[last]
        assert "── information" in "".join(text[last - 3 : last])

    def test_the_top_of_the_frame_is_left_alone(self):
        quiet = clocked(build())
        before = frame(quiet, 120, 40)
        ui = clocked(build())
        ui.toast("look down here")
        after = frame(ui, 120, 40)
        assert after[:20] == before[:20]

    def test_and_so_are_the_key_hints(self):
        ui = clocked(build())
        ui.toast("look down here")
        assert "── information" not in footer(ui, 120, 40)

    def test_it_always_has_a_ground_of_its_own(self):
        ui = clocked(build())
        ui.toast("look down here")
        drawn, footer_h = self.rows(ui)
        # Above the footer: the footer's note says the same thing in one line.
        toast_rows = [x for x in drawn[:-footer_h] if "look down here" in plain(x)]
        assert toast_rows and all(theme.overlay in x for x in toast_rows)
        # The ground is restated after every reset, or it would stop at the
        # first run that states its own style.
        for row in toast_rows:
            for piece in row.split(RESET)[1:-1]:
                assert piece.startswith(theme.overlay)

    def test_nothing_else_is_drawn_on_it(self):
        ui = clocked(build())
        ui.toast("look down here")
        drawn, footer_h = self.rows(ui)
        top = len(drawn) - footer_h - 4
        assert not any(theme.overlay in x for x in drawn[:top])
        assert not any(theme.overlay in x for x in drawn[-footer_h:])

    def test_every_row_of_it_does(self):
        ui = clocked(build())
        ui.toast("one\ntwo", "warning", title="Several rows")
        drawn, footer_h = self.rows(ui)
        bottom = len(drawn) - footer_h
        block = drawn[bottom - 4 : bottom]
        assert "Several rows" in plain(block[1])
        assert all(x.startswith(theme.overlay) for x in block)

    def test_over_a_screen_it_is_at_the_bottom_too(self):
        ui = clocked(build())
        ui.overlay = HelpOverlay()
        ui.toast("look down here")
        text = [plain(x) for x in ui.render(120, 40)]
        footer_h = len(ui._screen_footer(ui.overlay, 120, 40))
        assert "look down here" in text[len(text) - footer_h - 1]


class TestAToastGoesOnItsOwn:
    def test_it_is_on_screen_while_it_is_live(self):
        ui = clocked(build())
        ui.toast("still here")
        assert "still here" in screen(ui)

    def test_and_gone_afterwards(self):
        ui = clocked(build())
        ui.toast("not for long", timeout=2.0)
        ui._now += 2.5
        assert "── information" not in screen(ui)

    def test_the_frame_books_its_own_expiry(self):
        # Through `next_wake`, which is how everything that changes with no
        # keypress behind it gets its repaint — there is no timer of its own.
        ui = clocked(build())
        ui.toast("tick", timeout=2.0)
        assert 2.0 < toasts.next_wake(ui.toasts, ui.clock()) <= 2.1
        assert (ui.next_wake() or 0) > 0

    def test_an_idle_ui_with_no_toast_books_nothing_for_one(self):
        ui = clocked(build())
        ui.toasts.clear()
        assert toasts.next_wake(ui.toasts, ui.clock()) is None

    def test_three_at_once_is_the_most_that_is_shown(self):
        ui = clocked(build())
        for i in range(5):
            ui.toast(f"number {i}")
        assert len(toasts.live(ui.toasts, ui.clock())) == toasts.MAX_TOASTS

    def test_and_the_newest_is_the_one_kept(self):
        ui = clocked(build())
        for i in range(5):
            ui.toast(f"number {i}")
        assert "number 4" in screen(ui)
        assert "number 0" not in screen(ui)

    def test_an_expired_stack_is_dropped_rather_than_kept_forever(self):
        ui = clocked(build())
        ui.toast("gone", timeout=1.0)
        ui._now += 5
        ui.render(120, 24)
        assert ui.toasts == []


class TestNotifyReachesIt:
    async def test_the_title_crosses_the_wire(self):
        async with connected() as wire:
            await wire.tell(
                protocol.Notify(title="Reasoning effort", text="• off\n• low")
            )
            assert wire.ui.toasts[-1].title == "Reasoning effort"
            assert "Reasoning effort" in wire.screen()

    async def test_and_the_timeout_does(self):
        async with connected() as wire:
            await wire.tell(protocol.Notify(text="read this", timeout=30.0))
            assert wire.ui.toasts[-1].timeout == 30.0


# ---------------------------------------------------------------- clipboard


class TestTheClipboard:
    """§4.3 item 36: `c` on the chat row, through `ClipboardManager`."""

    def _ready(self):
        copied: list[str] = []
        ui = clocked(build())
        ui.clipboard = lambda text: copied.append(text) or "Copied via osc52"
        ui.focus = CHAT
        return ui, copied

    def test_c_copies_the_row_under_the_cursor(self):
        ui, copied = self._ready()
        press(ui, "home", "c")
        assert copied, "nothing was handed to the clipboard"

    def test_it_copies_the_message_and_not_the_row_it_is_drawn_as(self):
        # `you   ` and the tool markers are decoration; pasting them into a
        # shell is a paper cut every time.
        ui, copied = self._ready()
        press(ui, "home", "c")
        assert not copied[-1].startswith("you   ")

    def test_and_says_where_it_went(self):
        ui, _ = self._ready()
        press(ui, "home", "c")
        assert "Copied via osc52" in ui.note

    def test_a_tier_that_raises_is_reported_rather_than_fatal(self):
        ui = clocked(build())

        def boom(text: str) -> str:
            raise OSError("no terminal")

        ui.clipboard = boom
        ui.focus = CHAT
        press(ui, "home", "c")
        assert "copy failed" in ui.note

    def test_with_no_clipboard_wired_up_it_says_so(self):
        ui = clocked(build())
        ui.clipboard = None
        ui.focus = CHAT
        press(ui, "home", "c")
        assert "no clipboard" in ui.note

    def test_c_copies_from_the_chat_column_and_nowhere_else(self):
        # Everywhere else `c` is the config editor, which is the one screen
        # that had to give the key up for this — and did, on one row only.
        ui, copied = self._ready()
        ui.focus = SESSIONS
        press(ui, "c")
        ui.focus = WATCHERS
        press(ui, "c")
        assert copied == []

    def test_and_y_no_longer_copies_anywhere(self):
        ui, copied = self._ready()
        for row in (CHAT, SESSIONS, WATCHERS):
            ui.focus = row
            press(ui, "home", "y")
        assert copied == []

    def test_the_footer_offers_it_in_the_chat(self):
        ui = clocked(build())
        ui.focus = CHAT
        assert "c copy" in plain(ui.render(160, 40)[-1])

    def test_it_goes_through_the_tiered_manager_and_the_screen_emits(self):
        # `hpca.clipboard` is framework-free already: it takes an injected
        # `emit`, and `run.copier` gives it the terminal. Nothing about OSC 52
        # or the multiplexer wrapping is reimplemented in the UI.
        out = io.StringIO()
        read, write = os.pipe()
        try:
            copy = copier(Screen(fd=read, out=out))
            message = copy("scratch is /scratch/proj")
        finally:
            os.close(read)
            os.close(write)
        assert "osc52" in message or "file" in message
        if "osc52" in message:
            assert "\x1b]52;c;" in out.getvalue()


# ------------------------------------------------------------------ $EDITOR


class TestSuspendingTheTerminal:
    """§4.3 item 37, tested without an editor: what it does to the terminal.

    `Screen` is driven over a pipe and a `StringIO`, which is how every other
    terminal claim in this suite is checked — the mode strings and the diff
    baseline are the two things worth asserting, and neither needs a tty.
    """

    def _screen(self):
        read, write = os.pipe()
        out = io.StringIO()
        return Screen(fd=read, out=out), out, (read, write)

    def test_it_leaves_the_alternate_screen_and_comes_back(self):
        scr, out, fds = self._screen()
        try:
            with scr:
                out.truncate(0), out.seek(0)
                with scr.suspended():
                    left = out.getvalue()
                back = out.getvalue()
        finally:
            for fd in fds:
                os.close(fd)
        assert EXIT_MODES in left, "the editor gets its own screen back"
        assert ENTER_MODES in back[len(left):], "and the UI takes it back"

    def test_a_traceback_inside_still_restores_the_ui(self):
        scr, out, fds = self._screen()
        try:
            with scr:
                out.truncate(0), out.seek(0)
                with pytest.raises(RuntimeError):
                    with scr.suspended():
                        raise RuntimeError("the editor died badly")
                back = out.getvalue()
        finally:
            for fd in fds:
                os.close(fd)
        assert back.endswith(ENTER_MODES)

    def test_and_the_diff_baseline_is_dropped(self):
        # Whatever ran in there owned the screen, so the previous frame
        # describes something that is no longer on it.
        scr, out, fds = self._screen()
        try:
            with scr:
                scr.paint(["a", "b"])
                assert scr._prev == ["a", "b"]
                with scr.suspended():
                    pass
                assert scr._prev == []
        finally:
            for fd in fds:
                os.close(fd)


class TestTheLoopSideOfCtrlE:
    async def test_the_reader_is_off_the_loop_while_the_editor_runs(self):
        # A callback firing on the same fd would eat the user's keystrokes
        # into a frame nobody can see.
        import asyncio

        read, write = os.pipe()
        scr = Screen(fd=read, out=io.StringIO())
        loop = Loop(build(), scr, size=lambda: (80, 24))
        asyncio.get_running_loop().add_reader(read, lambda: None)
        seen = {}
        try:
            with scr:
                loop.suspend(lambda: seen.update(inside=True))
        finally:
            asyncio.get_running_loop().remove_reader(read)
            os.close(read)
            os.close(write)
        assert seen == {"inside": True}
        assert loop._full, "the frame after it is painted whole"

    async def test_the_ui_gets_a_suspend_wired_up_when_the_loop_runs(self):
        ui = build()
        assert ui.suspend is None
        read, write = os.pipe()
        scr = Screen(fd=read, out=io.StringIO())
        loop = Loop(ui, scr, size=lambda: (80, 24))
        loop.stop()
        try:
            await loop.run()
        finally:
            os.close(read)
            os.close(write)
        assert callable(ui.suspend)


class TestCtrlEInTheApp:
    """`^e` opens the *message* — which is what the footer has always said.

    It used to open the active profile's memories from both the box and the
    rows, which is the one hint in the footer that named the wrong thing: a
    key advertised beside "send" and "new line" that edited a file the user
    had not selected and was not looking at. The profile moved to where a
    profile is chosen (`a`, then the row) and the settings to `c`.
    """

    def _wired(self, ui: RowUI, done: str | None = None) -> list[str]:
        """A fake `edit_text`: it records the draft it was handed, and hands
        ``done`` back when there is one to hand back."""
        asked: list[str] = []

        def edit(text, apply):
            asked.append(text)
            if done is not None:
                apply(done)

        ui.suspend = lambda run: run()
        ui.edit_text = edit
        return asked

    def test_with_nothing_wired_up_the_key_says_so(self):
        # The honest answer, and the reason the guard is in `app.py`: a key
        # that silently did nothing looks exactly like an editor that opened
        # and closed again.
        ui = clocked(build())
        ui.focus = CHAT
        press(ui, "ctrl-e")
        assert "no editor" in ui.note

    def test_it_hands_the_draft_over_and_not_a_profile(self):
        ui = clocked(build())
        asked = self._wired(ui)
        profiles: list[str] = []
        ui.edit_profile = lambda name, kind="memories": profiles.append(name)
        ui.focus = INPUT
        press(ui, *"squeue is empty", "ctrl-e")
        assert asked == ["squeue is empty"]
        assert profiles == [], "the profile is edited from the profiles screen"

    def test_what_comes_back_is_the_draft(self):
        ui = clocked(build())
        self._wired(ui, done="the queue is drained")
        ui.focus = INPUT
        press(ui, *"the queue", "ctrl-e")
        assert ui.input.text() == "the queue is drained"

    def test_an_empty_buffer_clears_it_and_says_so(self):
        # Applied rather than refused — deleting the message and saving is a
        # thing a person does on purpose — but it is the one case that can
        # lose work, so it is the one case that is said out loud.
        ui = clocked(build())
        self._wired(ui, done="")
        ui.focus = INPUT
        press(ui, *"never mind", "ctrl-e")
        assert ui.input.text() == ""
        assert "empty" in ui.note

    def test_from_the_rows_it_edits_the_draft_and_brings_the_focus_back(self):
        # Like a paste from the rows (`_paste`): editing the message is an
        # unambiguous "I am writing", and coming back to a cursor parked on a
        # chat row would hide the text that was just edited.
        ui = clocked(build())
        asked = self._wired(ui, done="from the chat row")
        ui.focus = CHAT
        press(ui, "ctrl-e")
        assert asked == [""]
        assert ui.input.text() == "from the chat row"
        assert ui.focus == INPUT

    def test_ctrl_e_is_a_key_and_not_a_character(self):
        from hpca.ui.keys import decode

        assert decode("\x05") == (["ctrl-e"], "")


class TestTheEditorItself:
    """What `client.py` does once the body has arrived — with a fake `$EDITOR`.

    A shell script rather than a real editor: what is being tested is the
    resolution order, the round trip through a file, and what happens when it
    exits badly. `hpca.editor.resolve_editor` is not reimplemented, so the
    "settings, then $VISUAL, then $EDITOR, then nano" order is its own test
    (`tests/test_editor.py::TestResolveEditor` — the pointer here used to name
    a file that does not exist); this checks that this side uses it.
    """

    def _editor(self, tmp_path, body: str, code: int = 0) -> str:
        """A shell script standing in for `$EDITOR`: it writes ``body`` into
        the file it is handed and exits with ``code``."""
        wanted = tmp_path / "wanted.txt"
        wanted.write_text(body)
        script = tmp_path / "fake-editor"
        script.write_text(f'#!/bin/sh\ncat "{wanted}" > "$1"\nexit {code}\n')
        script.chmod(0o755)
        return str(script)

    async def _run(self, monkeypatch, tmp_path, editor: str):
        async with connected() as wire:
            monkeypatch.setenv("EDITOR", editor)
            monkeypatch.delenv("VISUAL", raising=False)
            wire.ui.suspend = lambda run: run()
            wire.ui.core_profile = "hpc"
            wire.client.edit_profile("hpc")
            await wire.client.flush()
            await settle()
            await wire.tell(
                protocol.ProfileBody(
                    name="hpc", kind="memories", text="scratch is /scratch\n"
                )
            )
            return wire

    async def test_it_asks_for_the_body_first(self, monkeypatch, tmp_path):
        # `profile.save` writes verbatim, so an editor opened over a stale copy
        # would revert whatever the agent learned in between.
        async with connected() as wire:
            wire.ui.suspend = lambda run: run()
            wire.client.edit_profile("hpc")
            await wire.client.flush()
            await settle()
            asked = wire.peer.last(protocol.ProfileGet)
            assert (asked.name, asked.kind) == ("hpc", "memories")

    async def test_what_the_editor_wrote_is_saved_back(self, monkeypatch, tmp_path):
        editor = self._editor(tmp_path, "the queue is gpu-a100\n")
        wire = await self._run(monkeypatch, tmp_path, editor)
        await wire.client.flush()
        await settle()
        saved = wire.peer.last(protocol.ProfileSave)
        assert (saved.name, saved.kind) == ("hpc", "memories")
        assert saved.text == "the queue is gpu-a100\n"
        assert "reloaded" in wire.ui.note

    async def test_an_unchanged_file_is_not_written_back(
        self, monkeypatch, tmp_path
    ):
        editor = self._editor(tmp_path, "scratch is /scratch\n")
        wire = await self._run(monkeypatch, tmp_path, editor)
        await wire.client.flush()
        await settle()
        assert wire.peer.took(protocol.ProfileSave) == []
        assert "unchanged" in wire.ui.note

    async def test_an_editor_that_exits_badly_saves_nothing(
        self, monkeypatch, tmp_path
    ):
        editor = self._editor(tmp_path, "half a file", code=3)
        wire = await self._run(monkeypatch, tmp_path, editor)
        await wire.client.flush()
        await settle()
        assert wire.peer.took(protocol.ProfileSave) == []
        assert "exited with 3" in wire.ui.note

    async def test_an_editor_that_does_not_exist_is_a_sentence(
        self, monkeypatch, tmp_path
    ):
        wire = await self._run(
            monkeypatch, tmp_path, str(tmp_path / "no-such-editor")
        )
        assert "could not run" in wire.ui.note

    async def test_a_body_that_could_not_be_read_is_not_edited(
        self, monkeypatch, tmp_path
    ):
        async with connected() as wire:
            suspended: list[int] = []
            wire.ui.suspend = lambda run: suspended.append(1)
            wire.client.edit_profile("hpc")
            await wire.client.flush()
            await settle()
            await wire.tell(
                protocol.ProfileBody(
                    name="hpc", kind="memories", error="permission denied"
                )
            )
            assert suspended == []
            assert "permission denied" in wire.ui.note

    async def test_the_settings_choose_the_editor_over_the_environment(
        self, monkeypatch, tmp_path
    ):
        # Resolution order is `hpca.editor.resolve_editor`'s, and the settings
        # half is read out of the JSON this UI already holds — no config
        # module is imported to read one field.
        chosen = self._editor(tmp_path, "from the settings\n")
        async with connected() as wire:
            monkeypatch.setenv("EDITOR", str(tmp_path / "never-run"))
            wire.ui.settings_json = json.dumps({"editor": chosen})
            wire.ui.suspend = lambda run: run()
            wire.client.edit_profile("hpc")
            await wire.client.flush()
            await settle()
            await wire.tell(
                protocol.ProfileBody(name="hpc", kind="memories", text="old\n")
            )
            await wire.client.flush()
            await settle()
            assert wire.peer.last(protocol.ProfileSave).text == "from the settings\n"


def fake_editor(tmp_path, body: str, code: int = 0, name: str = "fake-editor"):
    """A shell script standing in for `$EDITOR`, and what it saw.

    It records the *name* of the file it was handed and the text that was in
    it, writes ``body`` over it and exits with ``code``. The name matters: it
    is the only thing an editor has to go on when it decides whether it is
    holding prose or JSON, so it is worth asserting rather than assuming.
    """
    wanted = tmp_path / f"{name}.wanted"
    wanted.write_text(body)
    seen = tmp_path / f"{name}.seen"
    given = tmp_path / f"{name}.given"
    script = tmp_path / name
    script.write_text(
        "#!/bin/sh\n"
        f'basename "$1" > "{seen}"\n'
        f'cp "$1" "{given}"\n'
        f'cat "{wanted}" > "$1"\n'
        f"exit {code}\n"
    )
    script.chmod(0o755)
    return str(script), seen, given


class TestTheDraftInTheEditor:
    """`^e` end to end, with a fake `$EDITOR` and the demo core behind it.

    The draft is the one editor here with no wire in it — the message being
    written is the front-end's own state until it is sent — so this needs no
    scripted core at all: the round trip is the file.
    """

    def _ui(self, monkeypatch, tmp_path, body: str, code: int = 0):
        editor, seen, given = fake_editor(tmp_path, body, code)
        monkeypatch.setenv("EDITOR", editor)
        monkeypatch.delenv("VISUAL", raising=False)
        ui = clocked(build())
        ui.suspend = lambda run: run()
        ui.focus = INPUT
        return ui, seen, given

    def test_the_file_is_markdown(self, monkeypatch, tmp_path):
        # A message is prose, and `.md` is what turns on highlighting, spell
        # checking and soft wrap without the user configuring anything.
        ui, seen, _ = self._ui(monkeypatch, tmp_path, "edited\n")
        press(ui, *"draft", "ctrl-e")
        assert seen.read_text().strip() == "message.md"

    def test_the_draft_goes_in_with_a_final_newline(self, monkeypatch, tmp_path):
        ui, _, given = self._ui(monkeypatch, tmp_path, "edited\n")
        press(ui, *"draft", "ctrl-e")
        assert given.read_text() == "draft\n"

    def test_what_comes_back_is_the_draft(self, monkeypatch, tmp_path):
        ui, _, _ = self._ui(monkeypatch, tmp_path, "the queue is gpu-a100\n")
        press(ui, *"draft", "ctrl-e")
        assert ui.input.text() == "the queue is gpu-a100"

    def test_trailing_newlines_do_not_come_back_with_it(
        self, monkeypatch, tmp_path
    ):
        # The editor's own convention, which would otherwise arrive as blank
        # lines at the end of the box and trailing whitespace in what is sent.
        ui, _, _ = self._ui(monkeypatch, tmp_path, "one\ntwo\n\n\n")
        press(ui, *"draft", "ctrl-e")
        assert ui.input.text() == "one\ntwo"

    def test_an_editor_that_exits_badly_leaves_the_draft_alone(
        self, monkeypatch, tmp_path
    ):
        ui, _, _ = self._ui(monkeypatch, tmp_path, "half a message", code=3)
        press(ui, *"draft", "ctrl-e")
        assert ui.input.text() == "draft"
        assert "exited with 3" in ui.note

    def test_a_file_saved_unchanged_says_so(self, monkeypatch, tmp_path):
        ui, _, _ = self._ui(monkeypatch, tmp_path, "draft\n")
        press(ui, *"draft", "ctrl-e")
        assert ui.input.text() == "draft"
        assert "unchanged" in ui.note

    def test_an_emptied_buffer_empties_the_draft(self, monkeypatch, tmp_path):
        ui, _, _ = self._ui(monkeypatch, tmp_path, "")
        press(ui, *"draft", "ctrl-e")
        assert ui.input.text() == ""
        assert "empty" in ui.note


class TestTheProfileTheUserSelected:
    """`a`, the row, Enter — and the profile that opens is that row's.

    The old ctrl+e asked for `ui.profile`, the profile the core is working
    under, from a screen where nothing had been selected at all.
    """

    async def test_the_intent_asks_for_the_named_profile(self):
        async with connected() as wire:
            wire.ui.suspend = lambda run: run()
            wire.ui.send(EditProfile("writing", "archive"))
            await wire.client.flush()
            await settle()
            asked = wire.peer.last(protocol.ProfileGet)
            assert (asked.name, asked.kind) == ("writing", "archive")

    async def test_the_answer_to_a_different_one_is_not_edited(self):
        # Two profiles' bodies come down one event; the fetch this one made is
        # picked out by name *and* kind, or a memories fetch opens the archive.
        async with connected() as wire:
            suspended: list[int] = []
            wire.ui.suspend = lambda run: suspended.append(1)
            wire.ui.send(EditProfile("writing", "archive"))
            await wire.client.flush()
            await settle()
            await wire.tell(
                protocol.ProfileBody(name="hpc", kind="memories", text="x")
            )
            assert suspended == []

    async def test_what_the_editor_wrote_goes_back_to_that_profile(
        self, monkeypatch, tmp_path
    ):
        editor, _, _ = fake_editor(tmp_path, "the archive, edited\n")
        monkeypatch.setenv("EDITOR", editor)
        monkeypatch.delenv("VISUAL", raising=False)
        async with connected() as wire:
            wire.ui.suspend = lambda run: run()
            wire.ui.send(EditProfile("writing", "archive"))
            await wire.client.flush()
            await settle()
            await wire.tell(
                protocol.ProfileBody(
                    name="writing", kind="archive", text="old\n"
                )
            )
            await wire.client.flush()
            await settle()
            saved = wire.peer.last(protocol.ProfileSave)
            assert (saved.name, saved.kind) == ("writing", "archive")
            assert saved.text == "the archive, edited\n"


@contextlib.asynccontextmanager
async def edited_settings(monkeypatch, tmp_path, body: str, code: int = 0):
    """Press `c`, answer the fetch, and let a fake `$EDITOR` write ``body``.

    Everything after that is what the test is about: what was sent, what was
    refused, and where the text ends up when it was.
    """
    editor, seen, given = fake_editor(tmp_path, body, code)
    monkeypatch.setenv("EDITOR", editor)
    monkeypatch.delenv("VISUAL", raising=False)
    async with connected() as wire:
        wire.ui.suspend = lambda run: run()
        wire.ui.focus = SESSIONS
        await wire.press("c")
        assert wire.ui.overlay is None, "the file is in $EDITOR, not in a form"
        await wire.tell(protocol.SettingsBody(text='{"editor": null}\n'))
        yield wire, seen, given


class TestTheSettingsFileInTheEditor:
    """`c` opens the settings file in `$EDITOR`, and refuses to lose it.

    The file is the interface either way — `overlays/config.py` says why — so
    all that changes is which editor holds the text. What does not change is
    that nothing malformed reaches disk: this side checks the JSON, the core
    checks the model (`service._save_settings` refuses and writes nothing),
    and text that fails either check comes back on screen in the editor
    overlay rather than being dropped.
    """

    async def test_it_asks_for_the_file_first(self):
        # The core rewrites the file on every save, so the copy this UI holds
        # can already differ from the file in ways nobody typed.
        async with connected() as wire:
            wire.ui.suspend = lambda run: run()
            wire.ui.focus = SESSIONS
            await wire.press("c")
            assert wire.peer.took(protocol.SettingsGet), "settings.get first"
            assert wire.ui.overlay is None

    async def test_the_file_is_json(self, monkeypatch, tmp_path):
        async with edited_settings(
            monkeypatch, tmp_path, '{"editor": "vim"}\n'
        ) as (wire, seen, _):
            assert seen.read_text().strip() == "settings.json"

    async def test_what_the_editor_wrote_is_saved(self, monkeypatch, tmp_path):
        async with edited_settings(
            monkeypatch, tmp_path, '{"editor": "vim"}\n'
        ) as (wire, _, _):
            assert (
                wire.peer.last(protocol.SettingsSave).text
                == '{"editor": "vim"}\n'
            )

    async def test_broken_json_is_never_sent(self, monkeypatch, tmp_path):
        async with edited_settings(monkeypatch, tmp_path, '{"editor":') as (
            wire,
            _,
            _,
        ):
            assert wire.peer.took(protocol.SettingsSave) == []

    async def test_and_comes_back_on_screen_with_the_reason(
        self, monkeypatch, tmp_path
    ):
        # The user's minute of typing is not on disk, not on the wire, and not
        # thrown away either: it is in the one editor that refuses to close
        # over it.
        async with edited_settings(monkeypatch, tmp_path, '{"editor":') as (
            wire,
            _,
            _,
        ):
            assert isinstance(wire.ui.overlay, ConfigOverlay)
            assert '{"editor":' in wire.ui.overlay.editor.text()
            assert "invalid json" in wire.ui.overlay.note

    async def test_that_screen_will_not_close_while_it_is_broken(
        self, monkeypatch, tmp_path
    ):
        async with edited_settings(monkeypatch, tmp_path, '{"editor":') as (
            wire,
            _,
            _,
        ):
            await wire.press("esc")
            assert isinstance(wire.ui.overlay, ConfigOverlay), "refused"

    async def test_an_editor_that_exits_badly_saves_nothing(
        self, monkeypatch, tmp_path
    ):
        async with edited_settings(
            monkeypatch, tmp_path, '{"editor": "vim"}\n', code=3
        ) as (wire, _, _):
            assert wire.peer.took(protocol.SettingsSave) == []
            assert "exited with 3" in wire.ui.note

    async def test_a_refusal_from_the_core_lands_in_the_same_place(
        self, monkeypatch, tmp_path
    ):
        # Valid JSON and invalid settings: the core is the only side that can
        # tell, it refuses and writes nothing, and by then `$EDITOR` has
        # closed — so the rejected text has to come back here.
        async with edited_settings(
            monkeypatch, tmp_path, '{"llm": {"timeout": "soon"}}\n'
        ) as (wire, _, _):
            await wire.tell(
                protocol.SettingsBody(
                    text='{"editor": null}\n',
                    error="Invalid settings: llm.timeout is not a number",
                )
            )
            assert isinstance(wire.ui.overlay, ConfigOverlay)
            assert "timeout" in wire.ui.overlay.editor.text()
            assert "Invalid settings" in wire.ui.overlay.note

    def test_without_a_terminal_c_opens_the_form(self):
        # The in-app editor is kept as the way in when there is nothing to
        # hand the terminal over to — a `RowUI` driven headless, a test, the
        # demo. A key that answered "no editor is wired up here" and left no
        # way to edit the settings at all is the worse of the two.
        ui = clocked(build())
        ui.focus = SESSIONS
        press(ui, "c")
        assert isinstance(ui.overlay, ConfigOverlay)


# --------------------------------------------------------------------- quit


class TestQuit:
    """specs/specs-ui-acceptance.md, "Backends", the `q` bullets. §4.3 item 38."""

    def _sessions(self) -> RowUI:
        ui = clocked(build())
        ui.focus = SESSIONS
        return ui

    def test_q_asks_first(self):
        ui = self._sessions()
        assert ui.handle("q", 120, 40) is True, "not yet"
        assert "Really quit?" in screen(ui)

    def test_n_stays(self):
        ui = press(self._sessions(), "q")
        assert ui.handle("n", 120, 40) is True
        assert ui.confirm is None

    def test_escape_stays_too(self):
        ui = press(self._sessions(), "q")
        assert ui.handle("esc", 120, 40) is True

    def test_y_quits(self):
        ui = press(self._sessions(), "q")
        assert ui.handle("y", 120, 40) is False, "the loop ends"

    def test_ctrl_q_does_nothing_it_belongs_to_zellij(self):
        ui = self._sessions()
        assert ui.handle("\x11", 120, 40) is True
        assert ui.confirm is None
        assert screen(ui) == screen(self._sessions())

    def test_q_typed_in_the_message_box_is_a_letter(self):
        ui = clocked(build())
        ui.focus = INPUT
        press(ui, "q")
        assert ui.input.text() == "q"
        assert ui.confirm is None

    def test_q_is_inert_on_the_watchers_column(self):
        ui = clocked(build())
        ui.focus = WATCHERS
        assert ui.handle("q", 120, 40) is True
        assert ui.confirm is None

    def test_and_in_the_chat_column(self):
        ui = clocked(build())
        ui.focus = CHAT
        assert ui.handle("q", 120, 40) is True
        assert ui.confirm is None

    def test_and_on_another_screen(self):
        ui = self._sessions()
        ui.overlay = HelpOverlay()
        press(ui, "q")
        assert ui.confirm is None

    def test_ctrl_c_still_ends_it_without_a_question(self):
        # `quit` is the signal, not the letter: a terminal sending ^C means it.
        ui = self._sessions()
        assert ui.handle("quit", 120, 40) is False

    def test_the_footer_offers_quit_only_where_it_works(self):
        ui = clocked(build())
        ui.focus = SESSIONS
        assert "q quit" in plain(ui.render(200, 40)[-1])
        ui.focus = WATCHERS
        assert "q quit" not in plain(ui.render(200, 40)[-1])
