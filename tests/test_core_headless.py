"""The one rule `hpca.core` exists to enforce, checked for every module in it.

`tui/app.py` reached 4875 lines because agent logic could reach for
`self.notify`, `self._is_active_session` and a widget query whenever that was
convenient. Each of those reaches is a reason the runtime cannot run anywhere
else, and none of them announced itself — the file simply grew.

So the rule is checked rather than documented: import each `hpca.core` module
in a clean interpreter and assert Textual did not come with it. A module
discovered by the walk needs no test of its own, which is the point — the next
service added to the package is covered the moment it exists.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import pytest

import hpca.core

CORE_MODULES = sorted(
    module.name
    for module in pkgutil.iter_modules(
        hpca.core.__path__, prefix="hpca.core."
    )
)


def test_the_walk_actually_found_the_services():
    # A guard on the guard: an import error or a renamed package would make
    # every test below pass vacuously.
    assert len(CORE_MODULES) >= 2, CORE_MODULES
    assert "hpca.core.deps" in CORE_MODULES


@pytest.mark.parametrize("module", CORE_MODULES)
def test_a_core_module_pulls_in_no_front_end(module: str):
    # Both names, and neither is redundant. `hpca.ui` is the live rule: it is
    # the front-end that exists, and a core module reaching into it is the
    # coupling this whole package boundary is here to prevent. `textual` is a
    # ratchet: the dependency is gone as of M9 and this is what would notice
    # it coming back in through a core module rather than a UI one.
    code = (
        f"import sys, {module}; "
        "leaked = sorted(m for m in sys.modules "
        "if m.split('.')[0] == 'textual' or m.startswith('hpca.ui')); "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"{module} drags in a front-end:\n{result.stderr}"
    )


@pytest.mark.parametrize("module", CORE_MODULES)
def test_a_core_module_never_asks_what_is_on_screen(module: str):
    """`focused_session_id` is the only sanctioned answer to that question.

    Guarding the *names* is crude, but the failure being prevented is crude
    too: a service that grows an `_is_active_session` helper has quietly made
    itself un-runnable outside a UI, and the test suite would not otherwise
    notice until the split was undone.
    """
    path = sys.modules[module].__file__ if module in sys.modules else None
    if path is None:
        import importlib

        path = importlib.import_module(module).__file__
    source = open(path, encoding="utf-8").read()
    for banned in ("_is_active_session", "active_session", "query_one("):
        assert banned not in source, f"{module} reaches for the UI: {banned}"
