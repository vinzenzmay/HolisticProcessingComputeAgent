"""Tests for hpca.ui.keys: one read off the wire into key names.

The regression these guard is a real one — an escape sequence the table did not
know used to be read as the esc key followed by its letters, so reaching for
shift+arrow threw you out of the message box and into the chat.

`decode` returns `(keys, tail)`: what it could name, and the bytes it refuses
to name yet because the sequence may still be arriving. `final=True` says the
read timed out, so whatever is held is all there will ever be.
"""

import pytest

from hpca.ui.keys import (
    PASTE_END,
    PASTE_START,
    decode,
    escape_len,
    is_paste,
    paste_text,
)


def keys(data: bytes, *, final: bool = True) -> list[str]:
    """The names alone, for the cases where nothing is ever held back."""
    named, tail = decode(data, final=final)
    assert tail == "", f"unexpected tail {tail!r}"
    return named


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
    assert keys(data) == [name]


def test_a_sequence_nobody_mapped_is_dropped_whole():
    assert keys(b"\x1b[1;9Z") == []


def test_bare_esc_is_still_esc():
    assert keys(b"\x1b") == ["esc"]


def test_esc_then_typing_is_not_two_keys():
    # ESC[201~ is the bracketed-paste *end* marker with no start: unknown, and
    # dropped whole rather than arriving as "esc" plus the digits.
    assert keys(b"\x1b[201~") == []


def test_a_run_of_keys_decodes_in_order():
    assert keys(b"hi\r") == ["h", "i", "enter"]


def test_an_unknown_escape_does_not_swallow_what_follows_it():
    assert keys(b"\x1b[1;9Zab") == ["a", "b"]


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
    assert keys(b"\x1b\x1b") == ["esc", "esc"]


def test_an_alt_char_is_two_characters():
    assert escape_len("\x1bb", 0) == 2


# ------------------------------------------- a key split across two reads


def test_a_half_arrived_ss3_is_not_measured_as_three_characters():
    # The old bug: escape_len said 3 for a two-character buffer, so the next
    # read started one character into its own first sequence.
    assert escape_len("\x1bO", 0) == 2


def test_a_trailing_esc_is_held_back_rather_than_named():
    assert decode(b"\x1bO") == ([], "\x1bO")


def test_a_lone_esc_is_held_back_too_while_more_could_still_arrive():
    # Held, not dropped: this is either the escape key or the first byte of an
    # arrow. Only the timeout can tell, and this call does not know yet.
    assert decode(b"\x1b") == ([], "\x1b")


def test_but_a_timed_out_esc_is_the_escape_key():
    # Which is what keeps the stop gesture usable: nothing followed it inside
    # the read timeout, so the user pressed it.
    assert decode(b"\x1b", final=True) == (["esc"], "")


def test_an_arrow_split_across_two_reads_arrives_once_and_whole():
    named, tail = decode(b"x\x1b")
    assert (named, tail) == (["x"], "\x1b")
    assert decode(tail.encode() + b"[A") == (["up"], "")


def test_and_the_half_of_it_that_arrived_first_is_never_the_escape_key():
    # The defect this exists for: half a cursor movement used to arm the stop
    # gesture, and two of those in a second killed the turn.
    assert "esc" not in decode(b"\x1b")[0]


def test_a_half_arrived_csi_is_held_back():
    assert decode(b"\x1b[1;") == ([], "\x1b[1;")


def test_an_incomplete_csi_that_times_out_is_dropped_not_typed():
    assert decode(b"\x1b[1;", final=True) == ([], "")


def test_what_came_before_the_split_is_still_delivered():
    named, tail = decode(b"abc\x1b[")
    assert named == ["a", "b", "c"]
    assert tail == "\x1b["


# ------------------------------------------------------- bracketed paste


def bracketed(payload: str) -> bytes:
    return (PASTE_START + payload + PASTE_END).encode()


def test_a_paste_is_one_key():
    assert len(keys(bracketed("hello"))) == 1


def test_and_it_carries_its_payload():
    (key,) = keys(bracketed("hello"))
    assert is_paste(key)
    assert paste_text(key) == "hello"


def test_a_pasted_newline_is_not_an_enter():
    (key,) = keys(bracketed("one\ntwo"))
    assert paste_text(key) == "one\ntwo"


def test_a_pasted_carriage_return_becomes_a_newline_not_a_send():
    (key,) = keys(bracketed("one\r\ntwo\rthree"))
    assert paste_text(key) == "one\ntwo\nthree"


def test_a_pasted_tab_becomes_spaces():
    # A literal tab in the draft would be drawn by the terminal rather than by
    # us, and every row after it would be off by however far the tab stop is.
    (key,) = keys(bracketed("a\tb"))
    assert "\t" not in paste_text(key)


def test_a_pasted_escape_sequence_is_text_not_keys():
    (key,) = keys(bracketed("\x1b[1;5D"))
    assert paste_text(key) == "[1;5D"


def test_typing_around_a_paste_still_decodes():
    assert keys(b"a" + bracketed("b") + b"c") == ["a", "paste:b", "c"]


def test_a_paste_split_across_reads_is_held_until_its_end_marker():
    head = PASTE_START + "half a "
    assert decode(head.encode()) == ([], head)


def test_and_is_never_flushed_as_keystrokes_by_a_timeout():
    # The whole point: a paste that arrives in pieces must not turn into typed
    # characters, one of which is the newline that sends the message.
    head = PASTE_START + "half a\n"
    assert decode(head.encode(), final=True) == ([], head)


def test_and_arrives_whole_once_the_rest_lands():
    head = PASTE_START + "half a "
    named, tail = decode((head + "message" + PASTE_END).encode())
    assert tail == ""
    assert paste_text(named[0]) == "half a message"


# ------------------------------------------------------- the newline keys


NEWLINE_KEYS = [
    (b"\x1b\r", "alt-enter"),  # ESC CR: the alt/shift+enter workaround
    (b"\x1b\n", "alt-enter"),  # the same, on terminals that send LF
    (b"\x1b[13;2u", "shift-enter"),  # kitty keyboard protocol
    (b"\x1b[13;3u", "alt-enter"),
    (b"\x1b[13;5u", "ctrl-j"),
    (b"\x1b[27;2;13~", "shift-enter"),  # xterm modifyOtherKeys
    (b"\x1b[27;3;13~", "alt-enter"),
    (b"\n", "ctrl-j"),  # ^J is LF; Return in raw mode is CR
]


@pytest.mark.parametrize(
    "data,name", NEWLINE_KEYS, ids=[repr(d) for d, _ in NEWLINE_KEYS]
)
def test_the_newline_key_decodes(data: bytes, name: str):
    assert keys(data) == [name]


def test_plain_return_is_still_a_send():
    assert keys(b"\r") == ["enter"]
