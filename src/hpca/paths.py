"""Resolving the path a tool was given (§4.3).

This module is what replaced the path registry. The registry gave every path a
short key and asked the model to name files by key: it saved a small model from
re-typing a 90-character cluster path, and it cost a round-trip to mint the key
plus a second vocabulary — ``dir_key``, ``subpath``, ``source_key`` — that no
agent corpus the model was trained on contains. Measured, the second cost was
the larger one (specs-path-registry.md): keys were invented, misremembered, and
mixed with paths, and the tool's answer to all three was an ``UnknownKeyError``
that listed keys the model had never chosen.

So a tool takes a path, and this is the whole of what a path argument means:

* ``~`` expands,
* a relative path is anchored at the session's working directory, and
* the result is normalized lexically — ``..`` is folded away without asking
  the filesystem, so a path that does not exist yet still resolves.

Nothing here touches the disk and nothing here is stateful, which is what lets
the gating predicates and the describe helpers call it on every call and again
whenever a parked turn resumes.
"""

from __future__ import annotations

import os
from pathlib import Path


class PathError(ValueError):
    """An unusable path argument. The message is model-facing, so it says what
    to send instead rather than naming the type that failed."""


def resolve_path(value: str, workdir: Path | str) -> Path:
    """The absolute path a tool argument names.

    ``workdir`` anchors a relative one. That is the launch directory for the
    whole session, not a per-tool cwd: nothing in HPCA chdirs, and run_bash
    starts its scripts without a cwd of its own, so "." means the same place in
    every tool — which is the property that makes a relative path safe to
    accept at all.
    """
    text = (value or "").strip()
    if not text:
        raise PathError(
            "No path given. Pass the path of the file, absolute (e.g. "
            "/data/project/notes.md) or relative to the working directory."
        )
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(workdir) / path
    # Lexical, not Path.resolve(): a file that is about to be *created* has no
    # inode to follow, and every write tool here resolves its target before the
    # write. normpath folds "." and ".." on the string alone.
    return Path(os.path.normpath(path))


def display_path(path: Path | str, workdir: Path | str) -> str:
    """A path as a result should print it: absolute, always.

    Kept as a function rather than an f-string at each site so that the day
    long cluster paths want shortening in the transcript, there is one place to
    do it — and so a result never prints the relative form the model sent,
    which is the form it cannot check against what actually happened.
    """
    return str(resolve_path(str(path), workdir))


def contains(parent: Path, child: Path) -> bool:
    """Whether ``child`` stays inside ``parent`` once both are made real.

    Used by the writes that accept a name relative to a directory. Resolved
    here (unlike ``resolve_path``) because the question is about symlinks: a
    name that escapes via one is exactly what this is asked to catch. Missing
    parents are tolerated — ``Path.resolve`` is non-strict.
    """
    return parent.resolve() == child.resolve() or child.resolve().is_relative_to(
        parent.resolve()
    )
