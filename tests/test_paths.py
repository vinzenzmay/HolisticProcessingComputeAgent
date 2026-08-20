"""Tests for hpca.paths: what a path argument means (§4.3).

This is the whole of what replaced PathRegistry, so the cases that used to be
the registry's — a relative path, a '.', a name that is not a path at all — are
the ones worth pinning down here.
"""

import os
from pathlib import Path

import pytest

from hpca.paths import PathError, contains, display_path, resolve_path


class TestResolvePath:
    def test_an_absolute_path_is_returned_as_it_stands(self, tmp_path):
        assert resolve_path("/data/cohort/x.bam", tmp_path) == Path(
            "/data/cohort/x.bam"
        )

    def test_a_relative_path_is_anchored_at_the_workdir(self, tmp_path):
        assert resolve_path("results/out.txt", tmp_path) == tmp_path / "results/out.txt"

    def test_a_bare_dot_is_the_workdir(self, tmp_path):
        """The call that raised UnknownKeyError in the field: create_file with
        dir_key '.', in a session whose registry was empty."""
        assert resolve_path(".", tmp_path) == tmp_path

    def test_dot_slash_is_the_workdir_too(self, tmp_path):
        assert resolve_path("./plan.md", tmp_path) == tmp_path / "plan.md"

    def test_dot_dot_is_folded_lexically(self, tmp_path):
        assert resolve_path("a/../b.txt", tmp_path) == tmp_path / "b.txt"

    def test_a_path_that_does_not_exist_still_resolves(self, tmp_path):
        """Every write tool resolves its target before creating it, so a
        missing path must be an ordinary answer, not an error."""
        assert resolve_path("nope/deep/x.md", tmp_path) == tmp_path / "nope/deep/x.md"

    def test_tilde_expands(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_path("~/notes.md", tmp_path) == tmp_path / "notes.md"

    def test_surrounding_whitespace_is_ignored(self, tmp_path):
        assert resolve_path("  x.md \n", tmp_path) == tmp_path / "x.md"

    def test_an_empty_path_says_what_to_send_instead(self, tmp_path):
        with pytest.raises(PathError, match="absolute"):
            resolve_path("   ", tmp_path)

    def test_symlinks_are_not_followed(self, tmp_path):
        """Lexical on purpose: resolving would rewrite the path the user gave
        into one they never wrote, and a result they cannot recognise is worse
        than one that names their own symlink back to them."""
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        assert resolve_path(str(link / "x.txt"), tmp_path) == link / "x.txt"


class TestDisplayPath:
    def test_a_relative_path_prints_absolute(self, tmp_path):
        """A result must never echo the relative form back: what it says
        happened has to be checkable against the filesystem."""
        assert display_path("out.txt", tmp_path) == str(tmp_path / "out.txt")


class TestContains:
    def test_a_child_is_inside(self, tmp_path):
        assert contains(tmp_path, tmp_path / "a" / "b.txt")

    def test_the_directory_itself_counts(self, tmp_path):
        assert contains(tmp_path, tmp_path)

    def test_a_sibling_is_not_inside(self, tmp_path):
        assert not contains(tmp_path / "a", tmp_path / "b")

    def test_an_escape_through_a_symlink_is_caught(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        inside = tmp_path / "inside"
        inside.mkdir()
        (inside / "out").symlink_to(outside)
        assert not contains(inside / "sub", inside / "out" / "x.txt")


class TestNothingIsStored:
    def test_resolving_touches_no_disk_and_no_state(self, tmp_path):
        """The gating predicates and describe helpers call this on every call,
        and again whenever a parked turn resumes. The registry's ``register=
        False`` flag existed for exactly this; a pure function needs none."""
        before = sorted(os.listdir(tmp_path))
        for _ in range(3):
            resolve_path("some/new/file.md", tmp_path)
        assert sorted(os.listdir(tmp_path)) == before
