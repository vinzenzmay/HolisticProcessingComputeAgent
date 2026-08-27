"""Tests for hpca.filetail: reading one end of a file without reading the file."""

import pytest

from hpca.filetail import read_head, read_tail


class TestReadTail:
    def test_a_file_under_the_window_comes_back_whole(self, tmp_path):
        path = tmp_path / "small.log"
        path.write_text("one\ntwo\nthree\n")
        text, whole = read_tail(path, 4096)
        assert text == "one\ntwo\nthree\n"
        assert whole is True

    def test_an_empty_file_is_empty_and_whole(self, tmp_path):
        path = tmp_path / "empty.log"
        path.write_bytes(b"")
        assert read_tail(path, 4096) == ("", True)

    def test_only_the_window_is_returned_and_it_says_so(self, tmp_path):
        path = tmp_path / "big.log"
        path.write_text("".join(f"line {n}\n" for n in range(100_000)))
        text, whole = read_tail(path, 1000)
        assert whole is False
        assert len(text) <= 1000
        # the *end* of the file, which is the whole point
        assert text.endswith("line 99999\n")

    def test_a_window_wider_than_the_file_is_not_an_error(self, tmp_path):
        path = tmp_path / "small.log"
        path.write_text("x\n")
        assert read_tail(path, 1 << 20) == ("x\n", True)

    def test_a_character_split_by_the_seek_does_not_raise(self, tmp_path):
        """The window boundary lands inside a multi-byte character.

        Decoding has to survive that — the fragment becomes replacement
        characters at the front, which is exactly why callers keep less than
        they read — rather than raising a UnicodeDecodeError at whoever asked
        for a log tail.
        """
        path = tmp_path / "utf8.log"
        path.write_text("é" * 100, encoding="utf-8")  # two bytes each
        text, whole = read_tail(path, 51)  # odd: cuts one 'é' in half
        assert whole is False
        assert text.endswith("é" * 25)
        assert "�" in text  # the split character, replaced not raised

    def test_what_is_appended_after_the_read_is_simply_not_in_it(self, tmp_path):
        """A log being written to is the normal case, not an edge case.

        The window is taken at a fixed offset from a size read off the open
        handle, so the answer is a consistent snapshot of the end of the file
        as it was — never a short read, a doubled line, or a chase.
        """
        path = tmp_path / "growing.log"
        path.write_text("first\n")
        text, _ = read_tail(path, 4096)
        with path.open("a") as handle:
            handle.write("second\n")
        later, _ = read_tail(path, 4096)
        assert text == "first\n"
        assert later == "first\nsecond\n"

    def test_a_missing_file_raises_for_the_caller_to_answer(self, tmp_path):
        with pytest.raises(OSError):
            read_tail(tmp_path / "nope.log", 4096)


class TestReadHead:
    def test_the_front_of_a_large_file(self, tmp_path):
        path = tmp_path / "big.log"
        path.write_text("".join(f"line {n}\n" for n in range(100_000)))
        text = read_head(path, 1000)
        assert text.startswith("line 0\n")
        assert len(text) <= 1000

    def test_a_file_under_the_window_comes_back_whole(self, tmp_path):
        path = tmp_path / "small.log"
        path.write_text("all of it\n")
        assert read_head(path, 4096) == "all of it\n"
