"""Tests for hpca.ui.screen: the terminal modes and the differential paint.

`Screen` used to need a real terminal, so what could be checked here was only
the mode strings. M3 made its fd and its output stream injectable — a
descriptor that is not a tty simply keeps its line discipline — so the paint
itself is testable too, over a pipe and a `StringIO`.

The resize lives in `test_ui_run.py` now: it is `loop.add_signal_handler`, and
a signal handler on the loop is only meaningful with a loop under it.
"""

import io
import os

import pytest

from hpca.ui.ansi import ESC
from hpca.ui.screen import ENTER_MODES, EXIT_MODES, PASTE_OFF, PASTE_ON, Screen


@pytest.fixture
def pipe():
    read, write = os.pipe()
    yield read, write
    for fd in (read, write):
        try:
            os.close(fd)
        except OSError:
            pass


@pytest.fixture
def screen(pipe):
    """A real `Screen` with nothing terminal about it."""
    out = io.StringIO()
    return Screen(fd=pipe[0], out=out), out


class TestTerminalModes:
    def test_bracketed_paste_is_turned_on(self):
        assert PASTE_ON in ENTER_MODES

    def test_and_off_again_on_the_way_out(self):
        # Left on, it outlives the app: every shell prompt after quitting would
        # be handed ESC[200~ around anything pasted into it.
        assert PASTE_OFF in EXIT_MODES

    @pytest.mark.parametrize("mode", ["1049", "25", "7", "2004"])
    def test_every_mode_set_is_reset(self, mode: str):
        assert f"[?{mode}h" in ENTER_MODES or f"[?{mode}l" in ENTER_MODES
        assert f"[?{mode}h" in EXIT_MODES or f"[?{mode}l" in EXIT_MODES


class TestEnterAndLeave:
    def test_entering_writes_the_modes(self, screen):
        scr, out = screen
        with scr:
            assert ENTER_MODES in out.getvalue()

    def test_leaving_writes_them_back(self, screen):
        scr, out = screen
        with scr:
            pass
        assert out.getvalue().endswith(EXIT_MODES)

    def test_a_traceback_still_leaves_the_terminal_usable(self, screen):
        # The whole reason this is a context manager: an exception must not
        # leave the user in the alternate screen with no cursor and no echo.
        scr, out = screen
        with pytest.raises(ZeroDivisionError), scr:
            1 / 0
        assert EXIT_MODES in out.getvalue()

    def test_a_pipe_is_not_put_into_raw_mode(self, screen):
        # There is no line discipline on a pipe to save and restore, and
        # tcgetattr on one raises — which used to be the whole reason this
        # class needed a terminal to be tested at all.
        scr, _ = screen
        with scr:
            assert scr._saved is None


class TestPaint:
    def test_the_first_full_paint_clears_and_writes_every_row(self, screen):
        scr, out = screen
        scr.paint(["one", "two"], full=True)
        painted = out.getvalue()
        assert f"{ESC}[2J" in painted
        assert "one" in painted and "two" in painted

    def test_an_unchanged_row_is_not_written_again(self, screen):
        scr, out = screen
        scr.paint(["one", "two"], full=True)
        out.truncate(0), out.seek(0)
        scr.paint(["one", "different"])
        painted = out.getvalue()
        assert "different" in painted
        assert "one" not in painted

    def test_a_full_paint_writes_it_anyway(self, screen):
        # What a resize needs: the diff baseline was taken at the old geometry
        # and says nothing true about the new one.
        scr, out = screen
        scr.paint(["one", "two"], full=True)
        out.truncate(0), out.seek(0)
        scr.paint(["one", "two"], full=True)
        assert "one" in out.getvalue()

    def test_the_frame_is_one_synchronised_update(self, screen):
        # Without ?2026 a wide frame tears: the terminal draws what has arrived
        # so far and the rest lands on the next refresh.
        scr, out = screen
        scr.paint(["one"], full=True)
        painted = out.getvalue()
        assert painted.startswith(f"{ESC}[?2026h")
        assert painted.endswith(f"{ESC}[?2026l")
