"""Tests for hpca.transport: the socket, the queues, and the frames that fail.

Real AF_UNIX sockets, bound in a directory of the test's own — nothing here
leaves the machine, so it belongs in the default suite rather than behind the
integration marker.

Most behaviours are asserted against both transports through the `pair`
fixture. `--no-fork` runs the core over `InProcessConnection` and the deployed
thing over a socket, so a property that holds for only one of them is a bug
lying in wait for whichever mode is not under test.
"""

from __future__ import annotations

import asyncio
import shutil
import stat
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from hpca import transport
from hpca.protocol import Envelope, Notify, TurnSubmit, encode, parse
from hpca.transport import Connection, InProcessConnection, connect_unix, serve_unix

# Every wait in this file is bounded. A transport test that hangs takes the
# whole suite with it, and from the outside "slow" and "deadlocked" look the
# same; generous, because nothing here should ever actually wait.
TIMEOUT = 5.0


async def next_frame(
    frames: AsyncIterator[Envelope], *, timeout: float = TIMEOUT
) -> Envelope | None:
    """The next envelope, or None once iteration has ended."""

    async def pull() -> Envelope | None:
        try:
            return await frames.__anext__()
        except StopAsyncIteration:
            return None

    return await asyncio.wait_for(pull(), timeout)


class Peer:
    """The far end of a socket, under the test's control.

    `serve_unix` closes a connection the moment its handler returns, so the
    default handler parks until teardown and hands the test the server-side
    `Connection` to drive by hand.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._accepted: asyncio.Queue[Connection] = asyncio.Queue()
        self._live: list[Connection] = []
        self._release = asyncio.Event()
        self._servers: list[asyncio.Server] = []

    async def start(
        self, handler: Callable[[Connection], Awaitable[None]] | None = None
    ) -> asyncio.Server:
        async def _handle(conn: Connection) -> None:
            self._live.append(conn)
            self._accepted.put_nowait(conn)
            if handler is None:
                await self._release.wait()
            else:
                await handler(conn)

        server = await serve_unix(self.path, _handle)
        self._servers.append(server)
        return server

    async def accept(self) -> Connection:
        """The next server-side connection."""
        return await asyncio.wait_for(self._accepted.get(), TIMEOUT)

    async def shutdown(self) -> None:
        # Connections first: `wait_closed` waits for every client transport to
        # go, so a handler still parked on one would hang the teardown instead
        # of failing the test.
        self._release.set()
        for conn in self._live:
            await conn.close()
        for server in self._servers:
            server.close()
            await asyncio.wait_for(server.wait_closed(), TIMEOUT)


@pytest.fixture
def sock_path() -> AsyncIterator[Path]:
    """A socket path short enough to bind.

    `sun_path` is 108 bytes (§3), and pytest's `tmp_path` is nested deep enough
    — deeper still under xdist, which is how the default suite runs — that a
    socket under it can quietly exceed the limit. So: a shallow directory of
    our own, and an assertion that makes a future move back to `tmp_path` fail
    loudly rather than silently.
    """
    root = Path(tempfile.mkdtemp(prefix="hpca-t-", dir="/tmp"))
    path = root / "s"
    assert len(str(path).encode()) < 100, "socket path too long for sun_path"
    try:
        yield path
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
async def frames():
    """Open iterators over connections, closed when the test ends.

    An async generator abandoned mid-yield is finalised by the loop whenever it
    gets round to it, which at teardown means noise on stderr instead of a
    clean run.
    """
    opened: list[AsyncIterator[Envelope]] = []

    def _open(conn: Connection) -> AsyncIterator[Envelope]:
        it = conn.__aiter__()
        opened.append(it)
        return it

    yield _open
    for it in opened:
        await it.aclose()  # type: ignore[attr-defined]


@pytest.fixture
async def peer(sock_path):
    p = Peer(sock_path)
    yield p
    await p.shutdown()


@pytest.fixture
async def client(sock_path):
    """Connect to the fixture's socket. Every connection made is closed after."""
    made: list[Connection] = []

    async def _connect() -> Connection:
        conn = await asyncio.wait_for(connect_unix(sock_path), TIMEOUT)
        made.append(conn)
        return conn

    yield _connect
    for conn in made:
        await conn.close()


@pytest.fixture(params=["in-process", "unix socket"])
async def pair(request, sock_path):
    """Two connected ends, once per transport."""
    if request.param == "in-process":
        a, b = InProcessConnection.pair()
        yield a, b
        await a.close()
        await b.close()
        return
    peer = Peer(sock_path)
    await peer.start()
    a = await asyncio.wait_for(connect_unix(sock_path), TIMEOUT)
    b = await peer.accept()
    yield a, b
    await a.close()
    await peer.shutdown()


class TestEitherTransport:
    async def test_a_command_arrives_whole(self, pair, frames):
        a, b = pair
        msg = TurnSubmit(session_id="s1", text="which BAMs are in the cohort?")
        await a.send(msg)
        env = await next_frame(frames(b))
        assert env == msg.to_envelope(seq=1)
        assert parse(env) == msg

    async def test_an_event_comes_back_the_other_way(self, pair, frames):
        a, b = pair
        msg = Notify(severity="warning", text="job ✗ failed · 3 nodes")
        await b.send(msg)
        assert await next_frame(frames(a)) == msg.to_envelope(seq=1)

    async def test_the_correlation_id_crosses_with_the_frame(self, pair, frames):
        # §4: the sender of a command sets `id`, and a reply echoes it back.
        a, b = pair
        await a.send(TurnSubmit(session_id="s1", text="hi"), id="c7")
        env = await next_frame(frames(b))
        assert env is not None and env.id == "c7"

    async def test_the_connection_numbers_what_it_sends(self, pair, frames):
        # Monotonic from 1, and assigned here rather than by the caller.
        a, b = pair
        incoming = frames(b)
        seqs = []
        for _ in range(3):
            await a.send(Notify(text="tick"))
            env = await next_frame(incoming)
            assert env is not None
            seqs.append(env.seq)
        assert seqs == [1, 2, 3]

    async def test_the_two_directions_number_independently(self, pair, frames):
        # One counter per sender, not one per connection: a UI's third command
        # and a core's third event are both seq 3.
        a, b = pair
        at_a, at_b = frames(a), frames(b)
        await a.send(Notify(text="one"))
        await a.send(Notify(text="two"))
        await b.send(Notify(text="back"))
        assert [(await next_frame(at_b)).seq for _ in range(2)] == [1, 2]
        assert (await next_frame(at_a)).seq == 1

    async def test_closing_one_end_ends_the_peers_iteration(self, pair, frames):
        a, b = pair
        incoming = frames(b)
        await a.close()
        assert await next_frame(incoming) is None

    async def test_closing_ends_this_ends_own_iteration(self, pair, frames):
        a, b = pair
        incoming = frames(a)
        await a.close()
        assert await next_frame(incoming) is None

    async def test_close_is_idempotent(self, pair):
        # Shutdown has more than one path into it — the user quitting, the peer
        # vanishing, a handler returning — and they overlap.
        a, b = pair
        await a.close()
        await a.close()
        await b.close()
        await b.close()

    async def test_close_is_safe_while_a_reader_is_mid_iteration(self, pair, frames):
        a, b = pair
        incoming = frames(a)
        pending = asyncio.ensure_future(next_frame(incoming))
        await asyncio.sleep(0)  # let the reader park on the wire
        await a.close()
        assert await asyncio.wait_for(pending, TIMEOUT) is None

    async def test_sending_to_a_peer_that_is_gone_does_not_raise(self, pair):
        # The peer of a UI process is a thing the user can close at any moment,
        # so this is the ordinary end of a session and not an error. More than
        # one send: a socket write lands in the kernel buffer and it is usually
        # the next one that learns the far end has gone.
        a, b = pair
        await b.close()
        for _ in range(5):
            await a.send(Notify(text="nobody is listening"))
            await asyncio.sleep(0)

    async def test_sending_after_closing_this_end_does_not_raise(self, pair):
        a, _ = pair
        await a.close()
        await a.send(Notify(text="too late"))

    async def test_iterating_after_the_end_does_not_hang(self, pair, frames):
        a, b = pair
        await a.close()
        assert await next_frame(frames(b)) is None
        assert await next_frame(frames(b)) is None

    async def test_both_ends_satisfy_the_connection_protocol(self, pair):
        a, b = pair
        assert isinstance(a, Connection) and isinstance(b, Connection)


class TestSocket:
    async def test_the_socket_is_the_users_alone(self, peer, sock_path):
        # This endpoint runs `run_bash` (§3). `coreproc` states the same mode
        # for the same reason and deliberately leaves the chmod to whoever
        # binds, which is here.
        await peer.start()
        assert transport.SOCKET_MODE == 0o600
        assert stat.S_IMODE(sock_path.stat().st_mode) == 0o600

    async def test_a_frame_far_over_the_default_limit_survives_both_ways(
        self, peer, client, frames
    ):
        # asyncio's default is 64 KiB per line; a `chat.reset` with a real
        # session's transcript, or a tool result holding a file, goes past that
        # as a matter of course.
        text = "x" * (400 * 1024)
        await peer.start()
        conn = await client()
        server_side = await peer.accept()
        await conn.send(TurnSubmit(session_id="s1", text=text))
        out = await next_frame(frames(server_side))
        assert out is not None and parse(out).text == text
        await server_side.send(Notify(text=text))
        back = await next_frame(frames(conn))
        assert back is not None and parse(back).text == text

    async def test_a_frame_over_the_limit_is_dropped_and_the_next_arrives(
        self, peer, client, frames, monkeypatch, caplog
    ):
        # The limit is shrunk rather than 16 MiB actually being sent: what is
        # under test is the recovery, not the size. `readline` throws the
        # oversized frame away, so the stream resynchronises on the next
        # newline and the connection stays up.
        monkeypatch.setattr(transport, "FRAME_LIMIT", 4096)
        await peer.start()
        conn = await client()
        incoming = frames(await peer.accept())
        await conn.send(TurnSubmit(session_id="s1", text="x" * 20_000))
        await conn.send(TurnSubmit(session_id="s1", text="small"))
        out = await next_frame(incoming)
        assert out is not None and parse(out).text == "small"
        assert "oversized frame" in caplog.text

    async def test_a_malformed_line_costs_only_that_frame(
        self, peer, sock_path, frames, caplog
    ):
        # Written raw, because a buggy or hostile peer is exactly the thing
        # that will not be using this module to talk to us.
        await peer.start()
        _, writer = await asyncio.open_unix_connection(str(sock_path))
        incoming = frames(await peer.accept())
        writer.write(b"not json at all\n")
        writer.write(b'{"type": ""}\n')  # JSON, but not an envelope
        writer.write(encode(Envelope(type="shutdown")))
        await writer.drain()
        out = await next_frame(incoming)
        assert out is not None and out.type == "shutdown"
        assert "bad frame" in caplog.text
        writer.close()
        await writer.wait_closed()

    async def test_a_handler_that_raises_does_not_kill_the_server(
        self, peer, client, frames, caplog
    ):
        seen: asyncio.Queue[Envelope] = asyncio.Queue()
        calls = 0

        async def handler(conn: Connection) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("the handler for this client is buggy")
            async for env in conn:
                seen.put_nowait(env)

        await peer.start(handler)
        doomed = await client()
        # The raise costs that client its connection...
        assert await next_frame(frames(doomed)) is None
        # ...and nothing else: the next client is served as if nothing happened.
        second = await client()
        await second.send(Notify(text="still here"))
        env = await asyncio.wait_for(seen.get(), TIMEOUT)
        assert parse(env).text == "still here"
        assert "connection handler failed" in caplog.text

    async def test_the_socket_file_goes_when_the_server_closes(self, peer, sock_path):
        # §3 tells a stale socket from a live core by connecting to it, so a
        # cleanly stopped server must not leave one behind to be probed.
        server = await peer.start()
        assert sock_path.exists()
        server.close()
        await asyncio.wait_for(server.wait_closed(), TIMEOUT)
        assert not sock_path.exists()

    async def test_connecting_where_nothing_listens_raises(self, sock_path):
        # `coreproc`'s stale-socket check is "connect and expect the refusal",
        # so the failure has to arrive as itself.
        with pytest.raises(OSError):
            await connect_unix(sock_path)


class TestInProcess:
    def test_pair_gives_two_ends_that_point_at_each_other(self):
        a, b = InProcessConnection.pair()
        assert a.peer is b and b.peer is a

    async def test_close_drops_what_had_not_been_read(self, frames):
        # The socket end loses whatever is still in the kernel buffer when the
        # fd goes. A debug transport that quietly delivered more than the real
        # one would hide the very bug it was being used to chase.
        a, b = InProcessConnection.pair()
        await a.send(Notify(text="in flight"))
        await b.close()
        assert await next_frame(frames(b)) is None
