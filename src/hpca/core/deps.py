"""What every core service is handed, and the one way it talks back.

Written before the services that use it, and deliberately small. The domain
modules (`backends`, `pollers`, `scheduler`, `memory_service`) are extracted
from `HpcaApp` in parallel, and the thing that decides whether they compose is
whether they agree about their dependencies. So they do not each invent a
constructor: they take :class:`CoreDeps`, and they say things through
:attr:`CoreDeps.emit`.

Two shapes matter more than the field list.

**`emit` replaces four different reaches into the UI.** `self.notify(...)`,
`self._append_chat(...)`, `await self.refresh_watchers()` and
`self._refresh_session_row(...)` were all "tell the user"; here they are one
call taking a protocol event. A service that emits cannot accidentally depend
on a widget existing, which is what made the old calls untestable outside a
running app.

**`db` is the async seam, not a connection.** `DbIO` runs sqlite on its own
thread (see :mod:`hpca.db`). In the core that is an optimisation rather than
the correctness requirement it was on the UI loop — blocking here delays other
sessions' turns, not keystrokes — but the seam is kept because it is also the
only place that knows a store call may be slow, and because a service written
against `await deps.db(fn)` can be tested with a plain in-memory connection.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, TypeVar

if TYPE_CHECKING:
    from hpca.config import Settings
    from hpca.protocol import Event
    from hpca.slurm import SlurmClient

T = TypeVar("T")


class Emit(Protocol):
    """How a core service says something to whoever is listening.

    Synchronous on purpose: emitting must never be a place a service can block
    or fail. The implementation puts the event on a queue and returns; delivery
    (and a dead client) is the transport's problem, not the caller's.
    """

    def __call__(self, event: "Event") -> None: ...


class DbRunner(Protocol):
    """Run ``fn(conn)`` wherever sqlite is allowed to be slow."""

    def __call__(self, fn: Callable[[sqlite3.Connection], T]) -> Awaitable[T]: ...


@dataclass
class CoreDeps:
    """The runtime substrate, handed to every core service.

    Deliberately concrete rather than a bag of protocols: these are the app's
    own stores, and a service that wants to fake one in a test can pass a fake
    without an interface ceremony in between.
    """

    settings: "Settings"
    app_dir: Path
    # Sqlite work, off whatever loop the caller cares about. See the module
    # docstring for why this stays a seam even in the core.
    db: DbRunner
    # The one way to speak. See the module docstring.
    emit: Emit
    # The direct connection, for the callers that legitimately hold one for the
    # life of a session (`ToolContext`'s registry and runner). Anything on a
    # timer or in a turn should prefer `db`.
    conn: sqlite3.Connection | None = None
    slurm: "SlurmClient | None" = None
    # The profile the core is working under when nothing narrows it — a new
    # session's default, and whose watches an unfocused panel shows.
    profile: str = "default"
    # Which session the user is looking at, or None. Set from the `session.focus`
    # command and NOTHING else. It exists because one behaviour genuinely
    # depends on it (§4.4: a background completion reacts immediately only in
    # the open session), and making that an explicit, named piece of state is
    # the whole point — the old code read it from a widget, 14 times.
    focused_session_id: str | None = None
    # Set by services that need to hand extra context to each other without
    # widening this dataclass for every experiment.
    extras: dict[str, Any] = field(default_factory=dict)
