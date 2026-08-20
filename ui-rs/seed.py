"""Put a session and a watch in the scratch app dir, and print the session id.

The half-landed core cannot yet answer `session.list` or `session.new`
(`CoreService._dispatch` handles six commands; the session lifecycle is still
in `tui/app.py`), so there is no way to create one *through* the protocol. The
store underneath is real, though, so seeding it directly is enough to drive the
parts that do exist — `session.focus` makes the pollers rebuild the right
column, and that whole path is real runtime code.
"""

from __future__ import annotations

import sys

from hpca.config import Settings, app_dir
from hpca.db import connect, init_db
from hpca.dbcache import DbCache, local_dir_for
from hpca.sessions import SessionStore
from hpca.watches import WatchStore


def main() -> None:
    settings = Settings.load()
    directory = app_dir()
    cache = DbCache(
        directory,
        local_dir=local_dir_for(directory, configured=settings.database.local_dir),
        enabled=False,  # seeding writes straight to the app dir, before any lease
    )
    conn = connect(cache.path_for("hpca.db"))
    init_db(conn)

    session = SessionStore(conn).create(
        profile="default", title="align the reads", mode="auto"
    )
    watches = WatchStore(conn)
    watches.add(
        kind="log",
        target=str(directory / "align.err"),
        label="align.err",
        profile="default",
        session_id=session.session_id,
    )
    conn.commit()
    print(session.session_id)


if __name__ == "__main__":
    sys.exit(main())
