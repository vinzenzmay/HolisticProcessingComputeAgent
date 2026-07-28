"""Fixtures shared by the whole suite.

Only isolation belongs here: the unit tests are meant to be hermetic, so
nothing about the machine (or checkout) they happen to run on may reach them.
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
