"""How frames get across, and what happens when one of them is junk (§3).

`hpca.protocol` decides what a frame *is*; this module carries frames between
the UI process and the core. Like `protocol` it imports nothing else of hpca's:
a front-end that renders chat must not acquire the agent runtime by importing
the socket it talks over, and a core must not acquire Textual. The split is the
point of the whole exercise, and it only holds if the two modules underneath it
hold it.

Two implementations behind one `Connection` surface. `AF_UNIX` + NDJSON is the
deployed one; `InProcessConnection` is a pair of queues, used by the tests and
by `--no-fork`, where core and UI share a process so a debugger can step from a
keypress into the graph. They are interchangeable only if they behave
identically at the edges — close ending the peer's iteration, a send to a
departed peer being a no-op — so the tests drive both through the same cases.

Sequence numbers belong to the connection, not to the caller. §4 wants them
monotonic per sender from 1; a caller who has to remember to increment one is a
caller who will eventually forget, and nothing downstream can tell a skipped
seq from a dropped frame.

Two failures are treated as ordinary here rather than exceptional, because on
this wire they are: a frame that does not decode costs that frame and not the
connection, and a write to a peer that has already left is dropped instead of
raised. The peer of a UI process is a thing the user can close at any moment.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from hpca.protocol import Envelope, Message, ProtocolError, decode, encode

logger = logging.getLogger("hpca.transport")

# asyncio's StreamReader defaults to 64 KiB per line and, past that, does not
# grow: `readline` throws away what it has buffered and raises `ValueError:
# Separator is not found, and chunk exceed the limit`. 64 KiB is nowhere near
# enough here — a `chat.reset` carries a whole session's transcript and a tool
# result can be a file dump — so both ends are opened with a limit that a
# legitimate frame will not reach. 16 MiB is large enough that hitting it means
# something is wrong rather than merely big, and `__aiter__` survives it either
# way. Read at call time, so a test can shrink it.
FRAME_LIMIT = 16 * 1024 * 1024

# This endpoint runs `run_bash` (§3): whoever can connect can make the agent
# execute code, so on a shared login node the socket is the user's alone.
# `coreproc` owns the 0700 directory around it and states the same mode; the
# constant is repeated rather than imported because a module that knows how to
# bind a socket safely must not need the process supervisor to tell it.
SOCKET_MODE = 0o600


@runtime_checkable
class Connection(Protocol):
    """One end of the wire (§5.1).

    `send` takes a typed `Message`, never an `Envelope`: `seq` is the
    connection's to assign, and a caller who could hand in an envelope could
    hand in a second numbering. Iteration yields `Envelope`s rather than parsed
    messages for the opposite reason — `parse` is a routing decision, the two
    ends accept different halves of the message table, and a frame the peer had
    no business sending should be refused by the side that knows that.
    """

    async def send(self, msg: Message, *, id: str | None = None) -> None: ...

    def __aiter__(self) -> AsyncIterator[Envelope]: ...

    async def close(self) -> None: ...


class InProcessConnection:
    """Two paired queues; `peer` is the other end (§5.1).

    For the tests and for `--no-fork`. Nothing is serialised: `to_envelope`
    already renders the payload with `mode="json"`, so what crosses is the same
    primitives the codec would have produced, and the receiver holds the only
    reference to them.
    """

    def __init__(self) -> None:
        # `None` is end-of-stream — the queue's version of the empty read that
        # means EOF on a socket.
        self._inbox: asyncio.Queue[Envelope | None] = asyncio.Queue()
        self.peer: InProcessConnection | None = None
        self._seq = 0
        self._closed = False
        self._ended = False

    @classmethod
    def pair(cls) -> tuple[InProcessConnection, InProcessConnection]:
        """Two ends, where each one's `send` lands in the other's iterator."""
        a, b = cls(), cls()
        a.peer, b.peer = b, a
        return a, b

    async def send(self, msg: Message, *, id: str | None = None) -> None:
        self._seq += 1
        env = msg.to_envelope(seq=self._seq, id=id)
        peer = self.peer
        if self._closed or peer is None or peer._closed:
            # Same bargain as the socket end: a peer that has gone is the
            # normal end of a session, so an event emitted into the gap is
            # dropped rather than raised. The seq is spent regardless, exactly
            # as it would be for a frame the kernel accepted and never
            # delivered.
            logger.debug("dropping %s: no peer to take it", env.type)
            return
        peer._inbox.put_nowait(env)

    async def __aiter__(self) -> AsyncIterator[Envelope]:
        while not self._ended:
            env = await self._inbox.get()
            if env is None:
                self._ended = True
                return
            yield env

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Unread frames are discarded, mirroring the socket losing whatever is
        # still in the kernel buffer when the fd goes. A debug transport that
        # quietly delivered more than the real one would hide the very bug it
        # was being used to chase.
        while not self._inbox.empty():
            self._inbox.get_nowait()
        # Both inboxes: ours so a reader parked on the queue stops, the peer's
        # because closing a socket is precisely what its far end sees as EOF.
        self._inbox.put_nowait(None)
        if self.peer is not None:
            self.peer._inbox.put_nowait(None)


class _SocketConnection:
    """A `Connection` over an open AF_UNIX stream.

    Private: there are two ways to get one — `connect_unix` and `serve_unix` —
    and both hand back the `Connection` surface rather than this class.
    """

    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._seq = 0
        self._closed = False

    async def send(self, msg: Message, *, id: str | None = None) -> None:
        self._seq += 1
        env = msg.to_envelope(seq=self._seq, id=id)
        if self._closed:
            logger.debug("dropping %s: this end is closed", env.type)
            return
        # A `ProtocolError` from `encode` is ours, not the peer's, and is left
        # to propagate: it means this process built a frame it cannot send.
        raw = encode(env)
        try:
            self._writer.write(raw)
            await self._writer.drain()
        except (OSError, RuntimeError) as e:
            # ConnectionResetError / BrokenPipeError: the user closed the UI.
            # A core that raised here would fail a turn over a peer that has
            # merely stopped listening, which is not the turn's fault.
            logger.debug("dropping %s: %s", env.type, e)

    async def __aiter__(self) -> AsyncIterator[Envelope]:
        while True:
            try:
                line = await self._reader.readline()
            except (ValueError, asyncio.LimitOverrunError) as e:
                # Over FRAME_LIMIT. `readline` has already dropped what it had
                # buffered, so the frame's tail arrives as junk and is dropped
                # by `decode` below; the stream resynchronises at the next
                # newline, which is the next frame boundary.
                logger.warning("dropping an oversized frame: %s", e)
                continue
            except (OSError, RuntimeError) as e:
                logger.debug("read ended: %s", e)
                return
            if not line:
                return  # EOF: the peer closed, or we did
            try:
                env = decode(line)
            except ProtocolError as e:
                # One bad frame is one bad frame. A buggy — or hostile — peer
                # must not be able to end a live session by sending junk down
                # it, and every frame after this one is still readable.
                logger.warning("dropping a bad frame: %s", e)
                continue
            yield env

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._writer.drain()  # let what is queued reach the peer
        except (OSError, RuntimeError) as e:
            logger.debug("nothing to flush on close: %s", e)
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except (OSError, RuntimeError) as e:
            # Closing something the peer already dropped is a success, not a
            # failure: shutdown must not depend on the far end still being
            # there to shake hands with.
            logger.debug("peer had already gone on close: %s", e)


async def connect_unix(path: Path) -> Connection:
    """Connect to a core listening at `path`.

    Wrapping `open_unix_connection` earns its keep with one argument: the
    default 64 KiB line limit is smaller than a routine `chat.reset`, and a
    call site that forgot it would work until the first long session.
    """
    reader, writer = await asyncio.open_unix_connection(
        os.fspath(path), limit=FRAME_LIMIT
    )
    return _SocketConnection(reader, writer)


async def serve_unix(
    path: Path, handler: Callable[[Connection], Awaitable[None]]
) -> asyncio.Server:
    """Listen at `path`, calling `handler` once per accepted client.

    The chmod is here rather than in the caller because only the code that
    binds knows the moment the file exists; the server is created not yet
    serving so that no client can be accepted before the mode is right. That
    is the second of the two locks on this endpoint — `coreproc` owns the 0700
    directory — and it runs `run_bash`, so it gets both.

    A handler is allowed to raise. It costs that client its connection and
    nothing else: the same bargain the reader loop makes over a bad frame, for
    the same reason, since one session's bug is not the other sessions'
    problem.
    """

    async def _client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        conn = _SocketConnection(reader, writer)
        try:
            await handler(conn)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("connection handler failed; dropping this client")
        finally:
            await conn.close()

    server = await asyncio.start_unix_server(
        _client, os.fspath(path), limit=FRAME_LIMIT, start_serving=False
    )
    os.chmod(path, SOCKET_MODE)
    _unlink_on_close(server, Path(path))
    await server.start_serving()
    return server


def _unlink_on_close(server: asyncio.Server, path: Path) -> None:
    """Make `server.close()` take the socket file with it.

    Python 3.13 grew `cleanup_socket=True` for exactly this; on 3.11 and 3.12 a
    closed server leaves the file behind, and the next start then has to tell a
    stale file from a live core (§3). Wrapping the instance's `close` is the
    least invasive way to get one behaviour across the versions this project
    supports.

    The inode is checked first, so that a successor which has already rebound
    the path does not have its socket deleted out from under it — the same
    guard CPython's own cleanup uses.
    """
    try:
        inode = path.stat().st_ino
    except OSError:  # pragma: no cover - the bind just created it
        return
    close = server.close

    def close_and_unlink() -> None:
        close()
        try:
            if path.stat().st_ino == inode:
                path.unlink()
        except FileNotFoundError:
            pass  # 3.13+ cleaned it up already, or a peer did
        except OSError as e:
            logger.warning("could not remove the socket at %s: %s", path, e)

    server.close = close_and_unlink  # type: ignore[method-assign]
