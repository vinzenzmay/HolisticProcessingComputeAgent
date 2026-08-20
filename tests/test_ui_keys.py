"""Tests for hpca.ui.keys: one read off the wire into key names.

The regression these guard is a real one — an escape sequence the table did not
know used to be read as the esc key followed by its letters, so reaching for
shift+arrow threw you out of the message box and into the chat.
"""

import pytest

from hpca.ui.keys import decode, escape_len

MAPPED = [
    (b"\x1b[1;2D", "shift-left"),
    (b"\x1b[1;2C", "shift-right"),
    (b"\x1b[1;5D", "ctrl-left"),
    (b"\x1b[1;5C", "ctrl-right"),
    (b"\x1b[1;3D", "ctrl-left"),
    (b"\x1b[3;5~", "ctrl-delete"),
    (b"\x1b[1;6D", "shift-ctrl-left"),
]


@pytest.mark.parametrize("data,name", MAPPED, ids=[n for _, n in MAPPED])
def test_the_sequence_decodes(data: bytes, name: str):
    assert decode(data) == [name]


def test_a_sequence_nobody_mapped_is_dropped_whole():
    assert decode(b"\x1b[1;9Z") == []


def test_bare_esc_is_still_esc():
    assert decode(b"\x1b") == ["esc"]


def test_esc_then_typing_is_not_two_keys():
    # ESC[200~ is the bracketed-paste marker: unknown here, and dropped whole
    # rather than arriving as "esc" plus the digits.
    assert decode(b"\x1b[200~") == []


def test_a_run_of_keys_decodes_in_order():
    assert decode(b"hi\r") == ["h", "i", "enter"]


def test_an_unknown_escape_does_not_swallow_what_follows_it():
    assert decode(b"\x1b[1;9Zab") == ["a", "b"]


def test_a_csi_is_measured_to_its_final_byte():
    assert escape_len("\x1b[1;5D", 0) == 6


def test_an_ss3_is_always_three_characters():
    assert escape_len("\x1bOAx", 0) == 3


def test_a_lone_trailing_esc_is_one_character():
    assert escape_len("\x1b", 0) == 1


def test_two_escapes_in_a_row_are_measured_one_at_a_time():
    # The second ESC starts its own sequence; consuming it as an alt-<char>
    # payload would eat half of the stop gesture.
    assert escape_len("\x1b\x1b", 0) == 1
    assert decode(b"\x1b\x1b") == ["esc", "esc"]


def test_an_alt_char_is_two_characters():
    assert escape_len("\x1bb", 0) == 2
