"""The ESC-CR shift+enter workaround has to survive Textual's key parser."""

from textual._xterm_parser import XTermParser

from hpca.tui.termkeys import patch_alt_enter


def _keys(sequence: str) -> list[tuple[str, str | None]]:
    """Decode a byte sequence the way the input thread does."""
    parser = XTermParser()
    events = [*parser.feed(sequence), *parser.feed("")]
    return [(e.key, e.character) for e in events]


def test_esc_cr_is_alt_enter_not_a_bare_enter():
    # Terminals bound to send ESC CR for shift+enter (Alacritty's
    # chars = "\r") must not land on plain enter — that sends the draft.
    patch_alt_enter()
    assert _keys("\x1b\r") == [("alt+enter", None)]


def test_plain_return_and_the_kitty_sequences_are_untouched():
    patch_alt_enter()
    assert _keys("\r") == [("enter", "\r")]
    assert _keys("\x1b[13;2u") == [("shift+enter", None)]
    # ESC ESC stays two escapes; the patch must not turn it into alt+escape.
    assert _keys("\x1b\x1b") == [("escape", "\x1b"), ("escape", "\x1b")]


def test_patching_twice_does_not_stack_wrappers():
    patch_alt_enter()
    patch_alt_enter()
    assert _keys("\x1b\r") == [("alt+enter", None)]
