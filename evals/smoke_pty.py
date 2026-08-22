"""Boot the real binary in a real terminal and read what it paints.

Every other check in this repo drives the UI through a harness that never
opens a pty: `Screen` writes into a buffer, keys arrive as method calls, and
the terminal is assumed. That assumption is exactly what the first run on a
cluster node tests for the first time — so this opens a real pty, runs the
real console script under it, and prints the frames.

It exists because of one question the suite cannot answer: what does a fresh
install do when *nothing answers*? The two scenarios are the two ways that
happens, and they are different failures wearing the same face:

  bare   an empty $HPCA_HOME — no settings, no catalog, nothing configured
  down   a settings.toml naming a backend with nothing behind the port

`bare` is a UI question (does it tell the user, and offer the fix?). `down`
is a networking one: a TCP connect to a dead port is the startup path most
likely to *hang* rather than fail, and a hang here is a UI that never draws.
Both must reach a painted frame, an open manage-LLMs screen, and exit 0.

This is a release check, not a test — it takes tens of seconds, it sweeps the
network for endpoints, and what it finds depends on the box it runs on. Run
it deliberately, the way `test-live` and `edit_eval.py` are run:

    pixi run -e dev python evals/smoke_pty.py bare 14
    pixi run -e dev python evals/smoke_pty.py down 14

Read the output for three things: a layout that looks like the app, the
absence of a traceback, and `exit=0`. The frame times in the status line are
a bonus, not the point — `evals/` has benchmarks for that.
"""

from __future__ import annotations

import os
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Extra argv for the run under test. Empty since the row UI became the only
# front-end; kept as a seam because `--serve` (M11) will want a variant here.
LAUNCH_ARGS: list[str] = []

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "bare"
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0

home = Path(tempfile.mkdtemp(prefix=f"hpca-smoke-{SCENARIO}-"))

if SCENARIO == "down":
    # A backend that is configured and simply not answering: the tunnel case.
    # Port 20099 is chosen to be nothing; the point is the connect refusal.
    #
    # settings.JSON — `config.settings_path()` is `app_dir()/"settings.json"`.
    # This was written as settings.toml first, which HPCA ignores entirely, so
    # the scenario silently degraded into a second run of `bare` and reported
    # a pass for a case it had not exercised. Hence the assertion below: a
    # scenario that cannot prove it configured anything must fail loudly
    # rather than quietly test nothing.
    import json

    (home / "settings.json").write_text(
        json.dumps(
            {
                "llm": {
                    "base_url": "http://127.0.0.1:20099/v1",
                    "model": "nothing-listens-here",
                    "api_key": "unused",
                }
            },
            indent=2,
        )
    )
    # Read it back through the real loader, so a schema change breaks this
    # here rather than turning the scenario into a no-op again.
    import os as _os

    _os.environ["HPCA_HOME"] = str(home)
    from hpca.config import Settings

    assert Settings.load().llm.base_url.endswith(":20099/v1"), (
        "the down scenario did not configure the backend it claims to"
    )

env = dict(os.environ)
env["HPCA_HOME"] = str(home)
env["TERM"] = "xterm-256color"
env["COLUMNS"] = "120"
env["LINES"] = "40"
env.pop("HPCA_TEST_LLM_URL", None)
env.pop("HPCA_TEST_LLM_KEY", None)

primary, secondary = pty.openpty()
try:
    import fcntl
    import struct
    import termios

    fcntl.ioctl(secondary, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
except Exception as exc:  # pragma: no cover - best effort
    print(f"[smoke] could not size the pty: {exc}")

proc = subprocess.Popen(
    [sys.executable, "-m", "hpca", *LAUNCH_ARGS],
    stdin=secondary,
    stdout=secondary,
    stderr=secondary,
    env=env,
    close_fds=True,
    start_new_session=True,
)
os.close(secondary)

chunks: list[bytes] = []
deadline = time.time() + BUDGET
sent_quit = False

while time.time() < deadline:
    ready, _, _ = select.select([primary], [], [], 0.2)
    if ready:
        try:
            data = os.read(primary, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    if proc.poll() is not None:
        break
    # Give it most of the budget to draw, then ask it to leave politely.
    if not sent_quit and time.time() > deadline - 3.0:
        sent_quit = True
        os.write(primary, b"\x03")  # ctrl+c
        time.sleep(0.4)
        os.write(primary, b"\x04")  # ctrl+d

if proc.poll() is None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)

raw = b"".join(chunks)
out = raw.decode("utf-8", "replace")

print(f"[smoke] scenario={SCENARIO} exit={proc.returncode} bytes={len(raw)}")
print("[smoke] ---- decoded, escapes stripped ----")

plain = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", out)
plain = re.sub(r"\x1b[\]P][^\x07\x1b]*(\x07|\x1b\\)?", "", plain)
plain = plain.replace("\x1b", "<ESC>")
for line in plain.splitlines():
    stripped = line.rstrip()
    if stripped:
        print("   " + stripped)

# A grep, not a verdict. "Error" matches the word wherever it appears, and
# the UI is allowed to say it — a backend that refused a probe is news the
# screen is *supposed* to carry. So this prints the lines and lets the reader
# judge, and only "Traceback" is treated as unambiguous.
print("[smoke] ---- traceback check ----")
suspicious = [
    line.strip()
    for line in plain.splitlines()
    if any(m in line for m in ("Traceback", "Exception", "Error", "refused"))
]
for line in suspicious:
    print(f"   {line}")
if not suspicious:
    print("   nothing that looks like a crash")

log = home / "dbcache.log"
if log.exists() and log.read_text().strip():
    print("[smoke] ---- dbcache.log ----")
    print("   " + log.read_text().replace("\n", "\n   ").rstrip())

# Kept only when there is something to look at: a clean run's app dir holds
# nothing the next run will not rebuild, and these accumulate under $TMPDIR.
crashed = proc.returncode != 0 or "Traceback" in plain
if crashed:
    print(f"[smoke] FAILED — app dir kept at {home}")
else:
    shutil.rmtree(home, ignore_errors=True)
    print("[smoke] clean — app dir removed")

raise SystemExit(1 if crashed else 0)
