"""Fixtures and helpers shared by the whole suite.

Two things belong here. Isolation: the unit tests are meant to be hermetic, so
nothing about the machine (or checkout) they happen to run on may reach them.
And helpers that more than one test module needs — importing one test module
from another would drag its fixtures and collection along with it.
"""

import asyncio
from time import monotonic

import pytest


@pytest.fixture(autouse=True)
def isolated_project_dir(monkeypatch, tmp_path_factory):
    """Run every test from an empty working directory.

    Project-level skills (§5.1) live in ``<cwd>/.hpca/skills``, resolved from
    the process's working directory. Run from a checkout that carries its own
    project skills, the suite would load them into every test: assertions that
    no skill exists fail, and ``/skill-remove`` opens a picker modal that
    nothing dismisses, hanging the run.

    A fresh directory rather than ``tmp_path``, which is usually also
    ``HPCA_HOME`` — keeping the two apart means writing an app-dir skill never
    doubles as a project one.

    Tests that *want* a project dir chdir into their own ``tmp_path`` on top of
    this; the test body runs after this fixture, so their choice wins.
    """
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


async def wait_for_screen(app, pilot, screen_type, *, timeout_s=10.0):
    """Pause until the modal is actually up, rather than counting pauses.

    A fixed number of pauses is a guess about scheduling, and the guess is
    what fails: the work between the keypress and the screen being pushed can
    be a model round trip (/memorize) or just a local refresh (the skills
    list), and on a box running the suite across every core even the short one
    can outlast a handful of pauses. That is why these only ever failed in the
    parallel run and never when the test was run on its own. Waiting on the
    condition makes the test say what it means and stops the result depending
    on how busy the machine is.

    The worker cannot simply be awaited: it pushes the screen and then blocks
    on the user's answer, so ``wait_for_complete`` would deadlock against the
    very modal this is waiting for.
    """
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        if isinstance(app.screen, screen_type):
            return app.screen
        await pilot.pause()
        # Yield properly rather than spinning: the whole point is not to make
        # a loaded machine any busier.
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"{screen_type.__name__} never appeared within {timeout_s}s; "
        f"on screen: {type(app.screen).__name__}"
    )
