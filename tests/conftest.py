"""Fixtures and helpers shared by the whole suite.

Two things belong here. Isolation: the unit tests are meant to be hermetic, so
nothing about the machine (or checkout) they happen to run on may reach them.
And helpers that more than one test module needs — importing one test module
from another would drag its fixtures and collection along with it.
"""

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


@pytest.fixture(autouse=True)
def no_startup_backend_modal(monkeypatch):
    """Keep the startup "no backend is answering" screen out of the unit tests.

    Startup ends by probing the active backend and opening the manage-LLMs
    screen when nothing answers. Under the suite nothing ever answers —
    settings point at a default localhost URL and no server is running — so
    without this every UI test would find that screen on top of whatever it
    was driving, and would probe the network to get it. Tests that cover the
    check turn it back on for their own service.

    Switched off at the core, which is where the probing happens
    (``AgentService.startup`` → ``BackendRegistry.ensure_connected``);
    `ui/boot.py` only opens the screen on the answer it gets back. It used to
    have to be switched off in two places, because the Textual app probed for
    itself.

    Imported inside the fixture rather than at module scope: most of the suite
    never stands up a service, and this file is imported for every test.
    """
    from hpca.core.service import AgentService

    monkeypatch.setattr(AgentService, "startup_backend_check", False)
