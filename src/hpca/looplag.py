"""Event-loop scheduling delay, measured (specs-core-process.md §8).

The case for moving the agent into its own process is that the TUI stutters
because synchronous work — file IO in tool handlers, sqlite, ast walks,
whole-chat rebuilds — runs on the same event loop as the UI. Nothing in that
refactor is allowed to be justified by feel, so it needs a number that can be
taken on ``main``, taken again afterwards, and compared.

The number is drift: the probe asks to run every ``interval_s`` and records how
much *later* than that it actually ran. Nobody else can run while the loop is
blocked, so a block on the loop is exactly the thing drift sees.

Two choices make the output readable rather than merely present:

* Drift is measured against the *intended* tick, not against the previous
  actual one. A 900ms block then appears as one 900ms sample instead of being
  attributed to whatever ran next.
* After a block longer than the interval the tick grid is resynchronised to the
  present rather than caught up. Nine immediate catch-up ticks would smear one
  stall into a decaying tail of nine smaller ones, which reads like nine
  problems.

The module knows nothing about the rest of hpca: no config import, no
``app_dir()``, no agent concepts. The caller passes the log path, and — if it
wants spikes annotated with what the agent was doing — a zero-arg ``label``
callable. That keeps the probe testable on its own and cheap enough to leave
wired in permanently: disabled, it starts no task and touches nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("hpca.looplag")

# The spec's tick (§8). Fine enough that a stutter a user can feel lands in a
# sample of its own, coarse enough that the probe costs nothing to leave on.
DEFAULT_INTERVAL_S = 0.1

# ~1h of samples at the default interval. This runs inside a TUI that stays up
# for a working day, so the window is a ring rather than a list that grows for
# as long as the session lives.
DEFAULT_CAPACITY = 36_000

# Enough worst offenders that a repeat offender is visible as a repeat, few
# enough to read in a log block.
DEFAULT_WORST_N = 20

# A frame at 60fps is 16ms; around 50ms a keypress starts to feel late. Below
# that, asking the caller "what were you doing" is not worth the call.
DEFAULT_LABEL_THRESHOLD_S = 0.05


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def _percentile(ordered: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Nearest rank rather than interpolation: a reported p99 is then a delay that
    really happened, which is what a latency claim needs. Hand-rolled because
    numpy is not a dependency and ``statistics.quantiles`` interpolates and
    needs two points — a probe that has taken one sample must still summarise.
    """
    if not ordered:
        return 0.0
    k = math.ceil(q * len(ordered)) - 1
    return ordered[min(max(k, 0), len(ordered) - 1)]


@dataclass(frozen=True)
class Spike:
    """One of the worst samples, with enough context to place it.

    ``at`` is wall clock (``time.time()``), not ``perf_counter``: the point of
    keeping it is to line a stall up against the transcript, the core log, or
    the user's memory of when the UI froze, and none of those speak in
    perf_counter units.
    """

    drift: float
    at: float
    label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"drift": self.drift, "at": self.at, "label": self.label}


class LoopLagProbe:
    """Event-loop scheduling delay over a bounded window.

    ``start()`` on the loop being measured, ``await stop()`` on the way out,
    ``write_report()`` to leave a block behind for the before/after diff.

    The ``label`` hook is called only for a sample that is both above
    ``label_threshold_s`` and large enough to be kept as a spike: the answer is
    only ever stored on a :class:`Spike`, so asking for one that will be
    discarded is pure cost during a stall.

    Note the two scopes in a summary: the percentiles describe the ring — the
    last ``capacity`` samples — while the worst list is the worst of the whole
    run, which is what "did it ever hang" is asking.
    """

    def __init__(
        self,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        capacity: int = DEFAULT_CAPACITY,
        worst_n: int = DEFAULT_WORST_N,
        label: Callable[[], str] | None = None,
        label_threshold_s: float = DEFAULT_LABEL_THRESHOLD_S,
        enabled: bool = True,
    ) -> None:
        self.interval_s = max(float(interval_s), 0.001)
        self.label_threshold_s = float(label_threshold_s)
        self.enabled = enabled
        self._label = label
        self._worst_n = max(int(worst_n), 0)
        self._samples: deque[float] = deque(maxlen=max(int(capacity), 1))
        # A min-heap of (drift, seq, Spike) capped at worst_n: the cheap way to
        # keep the top N of a stream. ``seq`` only breaks ties, so Spike itself
        # needs no ordering.
        self._worst: list[tuple[float, int, Spike]] = []
        self._seq = 0
        self._total = 0
        self._task: asyncio.Task[None] | None = None
        self._started_at = 0.0
        self._elapsed = 0.0

    # ----------------------------------------------------------------- state

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def elapsed(self) -> float:
        """Seconds spent sampling, across every start/stop cycle."""
        if self._started_at:
            return self._elapsed + (time.perf_counter() - self._started_at)
        return self._elapsed

    @property
    def samples(self) -> tuple[float, ...]:
        """A copy of the ring, oldest first — for histograms and ad-hoc work."""
        return tuple(self._samples)

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Begin sampling on the running loop. Safe to call twice.

        Synchronous on purpose: this is called from ``on_mount``-style setup,
        and a probe that had to be awaited first would miss the moment the app
        does its heaviest synchronous work.
        """
        if not self.enabled or self.running:
            return
        self._started_at = time.perf_counter()
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        """Cancel the sampler and wait until it is gone. Safe to call twice.

        Async because the shutdown paths that call it are, and because "no
        leaked task" only becomes true once the cancellation has been awaited —
        a fire-and-forget ``cancel()`` leaves a pending task at interpreter
        exit, with the warning that goes with it.
        """
        task, self._task = self._task, None
        if self._started_at:
            self._elapsed += time.perf_counter() - self._started_at
            self._started_at = 0.0
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        interval = self.interval_s
        next_tick = time.perf_counter() + interval
        while True:
            delay = next_tick - time.perf_counter()
            # Already late: yield rather than spin, so the probe never becomes
            # the thing starving the loop it is watching.
            await asyncio.sleep(delay if delay > 0 else 0)
            now = time.perf_counter()
            self.record(now - next_tick)
            next_tick += interval
            if next_tick <= now:
                # A block longer than the interval swallowed whole ticks. Move
                # the grid to the present instead of firing one catch-up tick
                # per missed slot: the stall is already recorded once, at its
                # full size, and catch-up ticks would only dilute it.
                next_tick = now + interval

    # --------------------------------------------------------------- samples

    def record(self, drift: float) -> None:
        """Fold one drift sample in.

        Public because a sample's provenance does not matter to the ring: the
        timer calls this, and so can a test feeding a known set, or a caller
        that timed one suspect synchronous call itself.
        """
        if not self.enabled:
            return
        # A loop may fire a timer up to its clock resolution early, and
        # negative lag is not a thing — it would only pull the mean away from
        # what is being measured.
        drift = max(float(drift), 0.0)
        self._samples.append(drift)
        self._total += 1
        self._remember(drift)

    def _remember(self, drift: float) -> None:
        """Keep this sample among the worst N, with context, if it belongs."""
        if self._worst_n <= 0:
            return
        if len(self._worst) >= self._worst_n and drift <= self._worst[0][0]:
            return  # the common case: not a contender, so not even a clock read
        label = ""
        if self._label is not None and drift >= self.label_threshold_s:
            label = self._ask_label()
        self._seq += 1
        entry = (drift, self._seq, Spike(drift=drift, at=time.time(), label=label))
        if len(self._worst) < self._worst_n:
            heapq.heappush(self._worst, entry)
        else:
            heapq.heappushpop(self._worst, entry)

    def _ask_label(self) -> str:
        """Ask the caller what was running, and never let the answer break us.

        The hook reaches into live app state, and it is called from inside a
        stall — precisely when that state may be half-built or being torn
        down. A measurement is not worth an exception.
        """
        assert self._label is not None
        try:
            return str(self._label() or "")
        except Exception:
            logger.debug("looplag label hook failed", exc_info=True)
            return ""

    # --------------------------------------------------------------- reports

    def summary(self) -> dict[str, Any]:
        """Percentiles over the ring, plus the run's worst offenders.

        JSON-ready — ``worst`` holds dicts, not dataclasses — so a caller can
        log it, ship it over the core protocol, or diff two of them without
        unpacking anything. All durations are seconds.
        """
        ordered = sorted(self._samples)
        n = len(ordered)
        worst = sorted(self._worst, key=lambda e: e[0], reverse=True)
        return {
            "count": n,
            "total": self._total,
            "interval": self.interval_s,
            "elapsed": self.elapsed,
            "mean": (sum(ordered) / n) if n else 0.0,
            "p50": _percentile(ordered, 0.50),
            "p90": _percentile(ordered, 0.90),
            "p99": _percentile(ordered, 0.99),
            "max": ordered[-1] if n else 0.0,
            "worst": [entry[2].as_dict() for entry in worst],
        }

    def format_report(self, *, note: str = "") -> str:
        """One human-readable block: header, percentiles, worst offenders.

        Local time throughout, including the spike stamps, because the person
        reading this is correlating it against a terminal they were sitting in
        front of.
        """
        s = self.summary()
        when = datetime.now().astimezone().isoformat(timespec="seconds")
        head = f"=== looplag {when}"
        if note:
            head += f" {note}"
        lines = [f"{head} ==="]
        lines.append(
            f"samples {s['count']} of {s['total']} taken over "
            f"{s['elapsed']:.1f}s at {_ms(s['interval'])} intervals"
        )
        if s["count"]:
            lines.append(
                f"mean {_ms(s['mean'])}  p50 {_ms(s['p50'])}  "
                f"p90 {_ms(s['p90'])}  p99 {_ms(s['p99'])}  max {_ms(s['max'])}"
            )
        else:
            lines.append("no samples recorded")
        if s["worst"]:
            lines.append("worst:")
            for item in s["worst"]:
                stamp = datetime.fromtimestamp(item["at"]).strftime("%H:%M:%S.%f")
                label = f"  {item['label']}" if item["label"] else ""
                lines.append(f"  {_ms(item['drift']):>10}  {stamp[:-3]}{label}")
        return "\n".join(lines) + "\n\n"

    def write_report(self, path: Path | str, *, note: str = "") -> bool:
        """Append one summary block to ``path``. Returns whether it wrote.

        Append rather than overwrite: §8's comparison is a diff of the baseline
        block against the one taken after the refactor, and both should survive
        in the one file. The path comes from the caller — this module never
        resolves ``app_dir()`` itself, which is what keeps it importable
        without the config layer.
        """
        if not self.enabled:
            return False
        block = self.format_report(note=note)
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(block)
        except OSError:
            # Measurement must never be able to take the app down, least of all
            # on the way out.
            logger.warning("could not write looplag report to %s", path)
            return False
        return True
