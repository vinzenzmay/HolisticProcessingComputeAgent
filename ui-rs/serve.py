"""A real core, on a socket — the missing wave 3, in prototype form.

`specs-core-process.md` splits the app into a UI process and a core process and
gets most of the way there: `hpca.core.build_service` assembles the runtime,
`hpca.transport.serve_unix` can carry it, and `hpca.protocol` says what crosses.
What never landed is the twenty lines that join them, because `tui/app.py` still
builds the runtime itself and `python -m hpca` has no `--serve` flag.

This is those twenty lines, kept *outside* `src/hpca` on purpose: the prototype
is meant to answer a question about the front-end, not to pre-empt how the real
`--serve` should be written. When wave 3 lands properly this file is deleted and
`hpca --serve` is what the Rust client dials.

    python serve.py /tmp/core.sock [--profile default]

It opens the databases the app opens, and a turn submitted through it runs the
real graph against the configured backend. That means it needs an LLM to be
reachable to do anything interesting; everything that is not a turn — sessions,
the sidebar, watches, the transcript — works regardless.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from hpca.config import Settings, app_dir
from hpca.core.service import build_service
from hpca.db import DbIO, connect, init_db
from hpca.dbcache import DbCache, local_dir_for
from hpca.runner import reconcile_orphans
from hpca.protocol import Hello, ProtocolError, parse
from hpca.transport import Connection, serve_unix

logger = logging.getLogger("hpca.proto.serve")


async def pump(service, conn: Connection) -> None:
    """One client: the handshake, then events out and commands in.

    The two directions are separate tasks because they must not gate each
    other — an event fanned out while a command is being handled has nowhere
    to wait, and a core that stalls its poll timers behind a slow client is a
    core that stops noticing that a job died.
    """
    await conn.send(Hello(profile=service._deps.profile, settings_digest="proto"))

    queue = service.subscribe()

    async def outbound() -> None:
        while True:
            event = await queue.get()
            with contextlib.suppress(Exception):
                await conn.send(event)

    task = asyncio.create_task(outbound())
    try:
        async for env in conn:
            try:
                command = parse(env)
            except ProtocolError as e:
                # A bad frame costs that frame, not the connection.
                logger.warning("rejecting %r: %s", env.type, e)
                continue
            try:
                await service.handle(command)
            except Exception:
                logger.exception("command %r failed", env.type)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        service.unsubscribe(queue)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("socket", type=Path)
    ap.add_argument("--profile", default="default")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    settings = Settings.load()
    directory = app_dir()

    cache = DbCache(
        directory,
        local_dir=local_dir_for(directory, configured=settings.database.local_dir),
        enabled=settings.database.local_cache,
    )
    if not cache.acquire() and settings.database.local_cache:
        print(f"databases: {cache.reason}", file=sys.stderr)

    conn = connect(cache.path_for("hpca.db"))
    init_db(conn)
    # Processes the previous run was watching would otherwise claim to be
    # running forever.
    reconcile_orphans(conn)
    db = DbIO(cache.path_for("hpca.db"))

    async with AsyncSqliteSaver.from_conn_string(
        str(cache.path_for("checkpoints.db"))
    ) as checkpointer:
        service = build_service(
            settings=settings,
            app_dir=directory,
            db=db,
            conn=conn,
            checkpointer=checkpointer,
            profile=args.profile,
        )
        if args.socket.exists():
            args.socket.unlink()

        async def handler(client: Connection) -> None:
            await pump(service, client)

        server = await serve_unix(args.socket, handler)
        print(f"core listening on {args.socket}", file=sys.stderr, flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            server.close()
            await service.stop()
            cache.release()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        pass
