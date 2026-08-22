"""Which editor `$EDITOR` means (§6.4): settings → $VISUAL → $EDITOR → nano.

One resolution order, three callers — the profile editor, the skill editor and
the expanded message box — and until this file it was asserted only in
`tests/test_tui_memory.py`, a file that goes with `src/hpca/tui/`. The function
itself is framework-free and survives, so the test belongs beside it rather
than beside whichever front-end happened to call it first.
"""

from hpca.editor import resolve_editor


class TestResolveEditor:
    def test_the_setting_wins_over_the_environment(self):
        assert resolve_editor("code --wait", {"VISUAL": "vim"}) == [
            "code",
            "--wait",
        ]

    def test_then_visual(self):
        assert resolve_editor(None, {"VISUAL": "vim", "EDITOR": "nano"}) == ["vim"]

    def test_then_editor(self):
        assert resolve_editor(None, {"EDITOR": "emacs -nw"}) == ["emacs", "-nw"]

    def test_and_nano_when_nothing_is_set(self):
        assert resolve_editor(None, {}) == ["nano"]

    def test_a_blank_setting_is_not_a_choice(self):
        # An empty string in the settings file is the absence of an answer,
        # not an editor called "": falling through is what keeps a stray
        # `editor = ""` from launching nothing at all.
        assert resolve_editor("   ", {"EDITOR": "vim"}) == ["vim"]
