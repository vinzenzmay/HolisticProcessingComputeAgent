"""Starting the core child, and finding the socket it listens on (§2.1, §3).

The core is a *child process*, not a daemon: the deployment model is tmux on a
compute node, which already provides survive-disconnect, so a daemon would buy
nothing and add lease, GC and version-skew problems. What remains is a small
amount of genuinely fiddly process work — a handshake that must fail loudly
rather than hang, a socket path that has to fit in ``sun_path`` and must not be
reachable by the rest of a shared login node, and a shutdown that ends in
SIGKILL instead of a wedged child. It lives here so that neither the UI nor the
core has to carry it.

This module deliberately does not import :mod:`hpca.protocol` or
:mod:`hpca.transport`. The handshake line is a plain JSON dict read with
``json.loads`` because it *precedes* protocol negotiation: it is what decides
whether the thing on the other end is a core we may speak the protocol to at
all. A version check that had to agree about parsing before it could disagree
about versions would not be a check.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import socket
import stat
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NoReturn

from hpca.dbcache import local_root as node_local_root

logger = logging.getLogger("hpca.coreproc")

# Moves with protocol.PROTOCOL_VERSION but is a separate constant on purpose
# (see the module docstring). A mismatch means a core from another install, or
# one a crashed run left behind — the case this check exists for.
HANDSHAKE_VERSION = 1

SOCKET_NAME = "core.sock"

# AF_UNIX sun_path is 108 bytes including the NUL, and the failure mode of
# exceeding it is a silently truncated path rather than an error. Stopping well
# short turns that into a deliberate fallback.
MAX_SOCKET_PATH_BYTES = 100

# Where to go when the node-local root is too long to hold a socket, which a
# Slurm $TMPDIR can easily be. /tmp is node-local everywhere HPCA runs.
FALLBACK_ROOT = Path("/tmp")

# This socket accepts `turn.submit`, and a turn runs bash. On a shared login
# node the directory is the only thing standing between another user and
# arbitrary code execution as us, so it is checked on every reuse, not just
# created once. The Linux abstract namespace is not an option: it has no
# permissions at all.
DIR_MODE = 0o700

# The mode the socket file itself must carry. Set by whoever calls bind()
# (hpca.transport), recorded here because the security argument for it is the
# same one that owns DIR_MODE.
SOCKET_MODE = 0o600

# How long a core gets to die of SIGTERM before SIGKILL follows (§2.1).
KILL_AFTER_S = 5.0

# A core that never reported ready has no in-flight turn to protect, but may
# already hold the sqlite lease — so still SIGTERM first, just briefly.
_ABORT_TERM_S = 2.0

# Probing a Unix socket is a kernel-local operation; the timeout only guards a
# listener with a full backlog, which is a live core anyway.
_PROBE_TIMEOUT_S = 0.5


class CoreProcessError(RuntimeError):
    """The core could not be started, or its socket dir cannot be trusted."""


# ---------------------------------------------------------------- socket path


def socket_path_for(app_dir: Path, *, local_root: Path | None = None) -> Path:
    """Where the core for ``app_dir`` listens, with its directory made 0700.

    Under :func:`hpca.dbcache.local_root`, which already encodes the "never
    NFS" reasoning — and Unix sockets on NFS do not work at all, so there is no
    second option here the way there is for the databases.

    Keyed to the app dir so two ``$HPCA_HOME``s never share a socket, and
    stable for a given app dir so a UI can find a core it did not spawn (which
    is what makes a later ``--attach`` free). The digest is only eight hex
    digits because ``sun_path`` is 108 bytes and a Slurm ``$TMPDIR`` eats most
    of them; the collision it protects against is between app dirs of one user,
    and the ownership check below covers the rest.
    """
    # Resolved, so `~/.hpca`, `~/.hpca/` and a relative path all name one
    # socket — the same normalisation dbcache.local_dir_for does.
    digest = hashlib.sha256(str(Path(app_dir).resolve()).encode()).hexdigest()[:8]
    root = Path(local_root) if local_root is not None else node_local_root()
    path = root / f"hpca-{digest}" / SOCKET_NAME
    if len(str(path).encode()) > MAX_SOCKET_PATH_BYTES:
        path = FALLBACK_ROOT / f"hpca-{digest}" / SOCKET_NAME
        logger.warning(
            "%s is too long for an AF_UNIX socket path; using %s instead",
            root,
            path.parent,
        )
    _ensure_private_dir(path.parent)
    return path


def _ensure_private_dir(path: Path) -> None:
    """Create ``path`` as 0700, or verify an existing one is ours and 0700.

    Refusing is the only safe answer to a directory we did not make: whoever
    owns it can replace the socket with their own listener and be handed
    everything the UI sends, including bash to run.
    """
    try:
        path.mkdir(parents=True, mode=DIR_MODE)
    except FileExistsError:
        pass
    except OSError as e:
        raise CoreProcessError(f"cannot create the socket dir {path}: {e}") from e
    else:
        # umask can only clear bits from what mkdir was asked for, so set the
        # mode explicitly rather than trusting the caller's umask.
        path.chmod(DIR_MODE)
        return

    # lstat, not stat: a symlink planted at this path would otherwise be
    # checked at its target, which the attacker can point at a directory of
    # ours that happens to be 0700.
    try:
        st = os.lstat(path)
    except OSError as e:
        raise CoreProcessError(f"cannot inspect the socket dir {path}: {e}") from e
    if not stat.S_ISDIR(st.st_mode):
        raise CoreProcessError(f"{path} exists and is not a directory")
    if st.st_uid != os.getuid():
        raise CoreProcessError(
            f"the socket dir {path} is owned by uid {st.st_uid}, not {os.getuid()}"
        )
    mode = stat.S_IMODE(st.st_mode)
    if mode != DIR_MODE:
        raise CoreProcessError(
            f"the socket dir {path} is mode {mode:04o}, expected {DIR_MODE:04o}"
        )


# --------------------------------------------------------------- stale socket


def socket_is_live(path: Path) -> bool:
    """Whether something is actually listening on ``path``.

    A crashed core leaves its socket file behind, and ``bind()`` fails with
    EADDRINUSE on an existing path whether or not anyone is there — so the file
    alone cannot answer the question. Connecting can: ECONNREFUSED is the
    kernel saying the inode exists but has no listener.

    Every other error counts as live, which is the safe direction: refusing to
    start beats unlinking the socket of a running core and stranding its UI.
    """
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(_PROBE_TIMEOUT_S)
        probe.connect(str(path))
    except (ConnectionRefusedError, FileNotFoundError):
        return False
    except OSError:
        return True
    finally:
        probe.close()
    return True


def clear_stale_socket(path: Path) -> bool:
    """Remove a leftover socket file, if no core is listening. Did it remove one?

    Called before ``bind()``. Raises when a core *is* listening rather than
    unlinking: two cores on one app dir would fight over the sqlite lease, and
    the second one's socket file would silently orphan the first one's UI.
    """
    if not os.path.lexists(path):
        return False
    if socket_is_live(path):
        raise CoreProcessError(f"a core is already listening on {path}")
    logger.info("removing the stale socket %s", path)
    Path(path).unlink(missing_ok=True)
    return True


# ------------------------------------------------------------------ supervisor


def default_core_argv(socket_path: Path) -> list[str]:
    """The command that starts a core listening on ``socket_path``.

    ``sys.executable -m hpca`` rather than the ``hpca`` console script: it
    guarantees the child is the same interpreter and the same install as the
    UI, which is what keeps the handshake's version check a formality instead
    of a real risk. It also works when the script dir is not on ``$PATH``,
    which is normal inside a Slurm job.
    """
    return [sys.executable, "-m", "hpca", "--serve", "--socket", str(socket_path)]


class CoreSupervisor:
    """Spawns one core child, waits for its handshake, and ends it.

    One supervisor owns one child for its lifetime; ``start`` refuses a second.
    Everything after a failure is written so the caller can go straight to
    ``stop`` — the error paths here run when something has already gone wrong,
    which is exactly when a second exception is least welcome.

    ``argv0`` overrides the whole command (a stub in the tests, a wrapper such
    as ``srun`` in principle) and is used verbatim: an override is responsible
    for pointing its child at ``socket_path_for(app_dir)`` itself, because only
    the caller knows where in its own argv that belongs.
    """

    def __init__(
        self,
        *,
        app_dir: Path,
        argv0: list[str] | None = None,
        log_path: Path | None = None,
        ready_timeout_s: float = 15.0,
    ) -> None:
        self.app_dir = Path(app_dir)
        self.log_path = (
            Path(log_path) if log_path is not None else self.app_dir / "core.log"
        )
        self.ready_timeout_s = ready_timeout_s
        # Known only once start() has derived it; the handshake may correct it.
        self.socket_path: Path | None = None
        self._argv0 = list(argv0) if argv0 is not None else None
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def pid(self) -> int | None:
        """The pid of the child we can signal, or None before ``start``.

        Not necessarily the handshake's ``pid``: an override that wraps the
        core (``srun``, a shell) reports its own. This is the one that matters
        for stopping it.
        """
        return self._proc.pid if self._proc is not None else None

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode if self._proc is not None else None

    async def start(self) -> Path:
        """Spawn the core, read its handshake, and return the socket to dial.

        The default 15s ready timeout is generous because a cold import of
        langgraph off NFS is genuinely slow, and a timeout that fires during a
        normal start would be worse than no timeout at all.
        """
        if self._proc is not None:
            raise CoreProcessError("this supervisor has already started a core")
        requested = socket_path_for(self.app_dir)
        self.socket_path = requested
        argv = self._argv0 if self._argv0 is not None else default_core_argv(requested)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Appended rather than truncated: when a core dies during startup the
        # traceback worth reading is often the previous run's.
        log = self.log_path.open("ab")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                # start_new_session is load-bearing: without it the core shares
                # the terminal's foreground process group, so Ctrl-C in the UI
                # goes to the whole group and kills the core mid-turn. The UI
                # must be the only thing the terminal can signal.
                start_new_session=True,
                # Terminal hygiene in both directions (§2.1): a core that reads
                # the tty steals the user's keystrokes, and one that writes it
                # corrupts the TUI's screen. stdout is a pipe only for the
                # handshake line, which the core closes afterwards.
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
            )
        except OSError as e:
            raise CoreProcessError(f"could not start the core ({argv[0]}): {e}") from e
        finally:
            # The child holds its own dup of the fd from here on.
            log.close()
        return await self._handshake(requested)

    async def stop(self, *, timeout_s: float = 10.0) -> int | None:
        """End the core: SIGTERM, ``timeout_s``, SIGKILL. Returns its exit code.

        Safe to call twice, and safe to call when ``start`` never got as far as
        a child — both are shutdown paths after something already failed.

        This is the escalation, not the polite request: the caller is expected
        to have sent the protocol's ``shutdown`` first, so that the core can
        finish its turn and sync the databases before any signal arrives.
        """
        proc = self._proc
        if proc is None:
            return None
        if proc.returncode is not None:
            return proc.returncode
        try:
            # The child only, never its process group. The core's own tool
            # subprocesses are its to reap (runner.reconcile_orphans), and a
            # killpg would take out background work the user asked to keep.
            proc.terminate()
        except ProcessLookupError:
            pass  # it exited between the check and the signal
        code = await _wait_for_exit(proc, timeout_s)
        if code is not None:
            return code
        logger.warning(
            "core pid %s ignored SIGTERM for %ss; killing", proc.pid, timeout_s
        )
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        code = await _wait_for_exit(proc, KILL_AFTER_S)
        if code is None:
            logger.error("core pid %s survived SIGKILL; giving up on it", proc.pid)
        return code

    # ------------------------------------------------------------- handshake

    async def _handshake(self, requested: Path) -> Path:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            line = await asyncio.wait_for(
                proc.stdout.readline(), self.ready_timeout_s
            )
        except asyncio.TimeoutError:
            await self._abort(
                f"the core did not report ready within {self.ready_timeout_s:g}s"
            )
        except (ValueError, OSError) as e:
            # ValueError: a first "line" longer than the stream limit, i.e. not
            # a handshake at all.
            await self._abort(f"could not read the core's handshake ({e})")

        if not line:
            await self._abort("the core exited without reporting ready")
        try:
            hello = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            await self._abort(f"the core's first line was not JSON: {line[:200]!r}")
        if not isinstance(hello, dict) or not hello.get("ready"):
            await self._abort(f"the core did not report ready: {line[:200]!r}")
        version = hello.get("version")
        if version != HANDSHAKE_VERSION:
            await self._abort(
                f"core protocol version {version!r}, expected {HANDSHAKE_VERSION} "
                f"— a stale core, or a different install?"
            )

        bound = Path(str(hello.get("socket") or requested))
        if bound != requested:
            # Trust the child: it is the one that called bind().
            logger.warning("core bound %s, not the expected %s", bound, requested)
        self.socket_path = bound
        logger.info("core ready (pid %s) on %s", hello.get("pid"), bound)
        # Nothing reads stdout after this. Our end stays open even so: closing
        # it would deliver EPIPE to a core that prints anything later, and a
        # stray print deserves core.log, not a signal.
        return bound

    async def _abort(self, why: str) -> NoReturn:
        """Kill the child we could not talk to, then raise pointing at the log.

        The log path is in the message because it is the only place the child's
        side of the story exists — stderr went there, and by now the process is
        gone.
        """
        await self.stop(timeout_s=_ABORT_TERM_S)
        raise CoreProcessError(f"{why}; see {self.log_path}")


async def _wait_for_exit(
    proc: asyncio.subprocess.Process, timeout_s: float
) -> int | None:
    try:
        return await asyncio.wait_for(proc.wait(), timeout_s)
    except asyncio.TimeoutError:
        return None


# --------------------------------------------------------------- idle shutdown


class IdleShutdown:
    """Core-side: fires ``on_idle`` once the last client has been gone for
    ``grace_s``. Reset by every new connection.

    The core outlives a UI that died without saying goodbye — a SIGKILLed
    terminal, a lost tmux session — and it is the process holding the sqlite
    lease, so "nobody is attached any more" has to turn into "shut down
    cleanly" without anybody asking. A grace window rather than an immediate
    exit is what lets a UI restart (and, later, ``--attach``) reconnect to a
    warm core instead of paying for the langgraph import again.
    """

    def __init__(self, grace_s: float, on_idle: Callable[[], Awaitable[None]]) -> None:
        self._grace_s = grace_s
        self._on_idle = on_idle
        self._clients = 0
        self._timer: asyncio.Task[None] | None = None

    @property
    def pending(self) -> bool:
        """Whether a grace window is currently running."""
        return self._timer is not None and not self._timer.done()

    def client_connected(self) -> None:
        self._clients += 1
        self._cancel_timer()

    def client_disconnected(self) -> None:
        # Never below zero: a connection can be reported closed twice (the
        # handler's finally, then the server's teardown), and a negative count
        # would leave the timer disarmed while nobody is attached.
        self._clients = max(0, self._clients - 1)
        if self._clients:
            return
        # At most one timer ever: connect/disconnect churn must not stack up
        # tasks that all fire at the end of their own window.
        self._cancel_timer()
        self._timer = asyncio.ensure_future(self._countdown())

    def cancel(self) -> None:
        """Drop any pending countdown — the core is already shutting down for
        another reason, and must not shut down twice."""
        self._cancel_timer()

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    async def _countdown(self) -> None:
        await asyncio.sleep(self._grace_s)
        self._timer = None
        try:
            await self._on_idle()
        except Exception:
            # Nothing awaits this task, so an exception would surface only as
            # asyncio's "never retrieved" warning — on the one code path whose
            # whole job is to bring the core down.
            logger.exception("the idle shutdown handler failed")
