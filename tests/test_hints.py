"""Invariants of hpca.agent.hints — the sentences a tool result addresses to
the model rather than to the user.

The module is only useful if the display layer can strip every one of them, in
an order that leaves no fragments, and if adding a hint to the tools cannot
quietly leave it off the list. None of that is visible at the call site, so it
is pinned here.
"""

import inspect

from hpca.agent import hints
from hpca.agent.hints import MODEL_HINTS, NOT_STRIPPED


def constants() -> dict[str, str]:
    """Every hint the module defines, by name."""
    return {
        name: value
        for name, value in vars(hints).items()
        if name.isupper() and isinstance(value, str) and not name.startswith("_")
    }


class TestTheListIsTheWholeList:
    def test_every_hint_defined_is_accounted_for(self):
        # A constant in neither list is a sentence someone tidied into this
        # module and then never decided about. Leaving one out of MODEL_HINTS
        # is allowed, but it has to be a decision written down in NOT_STRIPPED.
        listed = set(MODEL_HINTS) | set(NOT_STRIPPED)
        missing = {name for name, value in constants().items() if value not in listed}
        assert not missing

    def test_the_two_lists_do_not_overlap(self):
        assert not set(MODEL_HINTS) & set(NOT_STRIPPED)

    def test_nothing_is_listed_twice(self):
        assert len(set(MODEL_HINTS)) == len(MODEL_HINTS)


class TestOrderLeavesNoFragments:
    """Stripping is by substring, so a hint that contains another must come
    first — otherwise the shorter one is removed from inside the longer one and
    what is left of the longer one stays on screen."""

    def test_no_hint_contains_one_listed_after_it(self):
        for position, hint in enumerate(MODEL_HINTS):
            for later in MODEL_HINTS[position + 1 :]:
                assert later not in hint, f"{later!r} is inside an earlier {hint!r}"


class TestEachHintIsJustTheSentence:
    """The whitespace joining a hint to the state it follows stays at the call
    site: that is what keeps the string the model receives byte-identical to
    what it was before the sentence moved into this module."""

    def test_no_hint_is_padded(self):
        for hint in MODEL_HINTS:
            assert hint == hint.strip()

    def test_no_hint_is_empty(self):
        for hint in MODEL_HINTS:
            assert hint

    def test_no_hint_is_short_enough_to_match_by_accident(self):
        # These are matched by substring against arbitrary tool output, a file
        # read back included. A short one would eventually cut a phrase out of
        # the user's own prose — see NOT_STRIPPED for the one that did.
        for hint in MODEL_HINTS:
            assert len(hint) >= 20, hint


class TestItStaysALeaf:
    """The tool modules and the TUI both import it; if it imported either back,
    one of them would be importing the other through it."""

    def test_it_imports_nothing_from_hpca(self):
        source = inspect.getsource(hints)
        assert "import hpca" not in source
        assert "from hpca" not in source
