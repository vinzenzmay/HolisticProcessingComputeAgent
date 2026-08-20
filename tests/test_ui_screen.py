"""Tests for hpca.ui.screen: the terminal modes and the resize wake-up.

`Screen` itself needs a real terminal, so what is checked here is the two
things that are decidable without one — the modes it turns on are the modes it
turns off again, and a SIGWINCH reaches the poll instead of waiting for it to
time out.
"""

import os
import signal

import pytest

from hpca.ui.screen import ENTER_MODES, EXIT_MODES, PASTE_OFF, PASTE_ON, Resizes


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


class TestResizes:
    def test_nothing_is_pending_before_a_signal(self):
        with Resizes() as resizes:
            resizes.taken()  # the first frame is always a full paint
            assert resizes.taken() is False

    def test_a_sigwinch_is_seen(self):
        with Resizes() as resizes:
            resizes.taken()
            os.kill(os.getpid(), signal.SIGWINCH)
            assert resizes.taken() is True

    def test_and_only_once(self):
        with Resizes() as resizes:
            resizes.taken()
            os.kill(os.getpid(), signal.SIGWINCH)
            resizes.taken()
            assert resizes.taken() is False

    def test_the_first_frame_is_a_full_paint(self):
        # There is no previous frame to diff against, so the loop must be told
        # to draw the whole screen once before it starts trusting its baseline.
        with Resizes() as resizes:
            assert resizes.taken() is True

    def test_the_signal_wakes_a_poll_that_would_otherwise_block(self):
        # The point of the self-pipe: a bare handler runs between bytecodes and
        # PEP 475 restarts the interrupted select, so the resize would not be
        # noticed until the poll timed out.
        import select

        with Resizes() as resizes:
            resizes.taken()
            os.kill(os.getpid(), signal.SIGWINCH)
            assert select.select([resizes.fd], [], [], 0)[0] == [resizes.fd]
            resizes.taken()
            assert select.select([resizes.fd], [], [], 0)[0] == []

    def test_the_handler_is_put_back_afterwards(self):
        before = signal.getsignal(signal.SIGWINCH)
        with Resizes():
            pass
        assert signal.getsignal(signal.SIGWINCH) is before

    def test_and_so_is_the_wakeup_fd(self):
        before = signal.set_wakeup_fd(-1)
        signal.set_wakeup_fd(before)
        with Resizes():
            pass
        after = signal.set_wakeup_fd(-1)
        signal.set_wakeup_fd(after)
        assert after == before
