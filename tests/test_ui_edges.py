"""Tests for M8's four edges: toasts, the clipboard, `$EDITOR` and quit.

§4.3 items 35-38, plus the claim carried over from the deleted
`test_tui_notify.py` — "a toast must never crash on arbitrary text, including
text containing square brackets from LLM output". That one is the reason this
file exists: the Textual app answered it with ``markup=False``, this renderer
has no markup to turn off, and what is left is width and control characters.
"""

import io
import json
import os

import pytest

from hpca import protocol
from hpca.ui import toasts
from hpca.ui.app import CHAT, INPUT, SESSIONS, WATCHERS, RowUI
from hpca.ui.demo import build
from hpca.ui.overlays import HelpOverlay
from hpca.ui.run import Loop, copier
from hpca.ui.screen import ENTER_MODES, EXIT_MODES, Screen
from hpca.ui.state import Toast
from tests.ui_harness import clocked, connected, frame, plain, settle, widths

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
        assert "line one line two" in plain(ui.render(120, 24)[-1])


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
                protocol.Notify(title="Thinking effort", text="• off\n• low")
            )
            assert wire.ui.toasts[-1].title == "Thinking effort"
            assert "Thinking effort" in wire.screen()

    async def test_and_the_timeout_does(self):
        async with connected() as wire:
            await wire.tell(protocol.Notify(text="read this", timeout=30.0))
            assert wire.ui.toasts[-1].timeout == 30.0


# ---------------------------------------------------------------- clipboard


class TestTheClipboard:
    """§4.3 item 36: `y` on the chat row, through `ClipboardManager`."""

    def _ready(self):
        copied: list[str] = []
        ui = clocked(build())
        ui.clipboard = lambda text: copied.append(text) or "Copied via osc52"
        ui.focus = CHAT
        return ui, copied

    def test_y_copies_the_row_under_the_cursor(self):
        ui, copied = self._ready()
        press(ui, "home", "y")
        assert copied, "nothing was handed to the clipboard"

    def test_it_copies_the_message_and_not_the_row_it_is_drawn_as(self):
        # `you   ` and the tool markers are decoration; pasting them into a
        # shell is a paper cut every time.
        ui, copied = self._ready()
        press(ui, "home", "y")
        assert not copied[-1].startswith("you   ")

    def test_and_says_where_it_went(self):
        ui, _ = self._ready()
        press(ui, "home", "y")
        assert "Copied via osc52" in ui.note

    def test_a_tier_that_raises_is_reported_rather_than_fatal(self):
        ui = clocked(build())

        def boom(text: str) -> str:
            raise OSError("no terminal")

        ui.clipboard = boom
        ui.focus = CHAT
        press(ui, "home", "y")
        assert "copy failed" in ui.note

    def test_with_no_clipboard_wired_up_it_says_so(self):
        ui = clocked(build())
        ui.clipboard = None
        ui.focus = CHAT
        press(ui, "home", "y")
        assert "no clipboard" in ui.note

    def test_y_is_the_chat_column_s_key_and_no_one_else_s(self):
        ui, copied = self._ready()
        ui.focus = SESSIONS
        press(ui, "y")
        ui.focus = WATCHERS
        press(ui, "y")
        assert copied == []

    def test_the_footer_offers_it_in_the_chat(self):
        ui = clocked(build())
        ui.focus = CHAT
        assert "y copy" in plain(ui.render(160, 40)[-1])

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
    def test_with_nothing_wired_up_the_key_says_so(self):
        ui = clocked(build())
        ui.focus = CHAT
        press(ui, "ctrl-e")
        assert "no editor" in ui.note

    def test_it_asks_for_the_profile_and_then_suspends(self):
        ui = clocked(build())
        asked: list[str] = []
        ui.suspend = lambda run: run()
        ui.edit_profile = asked.append
        ui.focus = CHAT
        press(ui, "ctrl-e")
        assert asked == [ui.profile]

    def test_it_works_from_the_message_box_too(self):
        ui = clocked(build())
        asked: list[str] = []
        ui.suspend = lambda run: run()
        ui.edit_profile = asked.append
        ui.focus = INPUT
        press(ui, "ctrl-e")
        assert asked == [ui.profile]

    def test_ctrl_e_is_a_key_and_not_a_character(self):
        from hpca.ui.keys import decode

        assert decode("\x05") == (["ctrl-e"], "")


class TestTheEditorItself:
    """What `client.py` does once the body has arrived — with a fake `$EDITOR`.

    A shell script rather than a real editor: what is being tested is the
    resolution order, the round trip through a file, and what happens when it
    exits badly. `hpca.editor.resolve_editor` is not reimplemented, so the
    "settings, then $VISUAL, then $EDITOR, then nano" order is its own test
    (`tests/test_memory_ops.py`); this checks that this side uses it.
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


# --------------------------------------------------------------------- quit


class TestQuit:
    """specs-ui-acceptance.md, "Backends", the `q` bullets. §4.3 item 38."""

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
