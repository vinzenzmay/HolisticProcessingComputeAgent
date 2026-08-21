#!/usr/bin/env python
"""Baseline the Textual TUI's per-interaction repaint cost, before it is deleted.

Run it with the repo's dev environment, from the repo root:

    pixi run -e dev python /path/to/bench_textual.py --iters 300
    pixi run -e dev python /path/to/bench_textual.py --only entries-300 --json out.json

WHAT IS TIMED — three units, because one is not enough
-----------------------------------------------------
1. ``press``: ``await pilot.press("down")`` then ``await pilot.pause()``. This
   is the whole interaction — key event, binding, widget action, scroll,
   layout, compositor update, ``render_update()``.

   Its WALL time is nearly useless: ``pilot.press`` and ``pilot.pause`` both
   call ``textual._wait.wait_for_idle``, which sleeps in 20 ms granules and
   returns only once process time stops advancing — a floor of roughly 100 ms
   per sample that has nothing to do with the app. So the headline for this
   unit is ``process_time`` (CPU), which the sleeps do not charge for. Wall is
   reported beside it so the floor is visible rather than hidden.

2. ``repaint``: ``screen.refresh(repaint=True, layout=False)`` followed by
   ``screen._on_timer_update()``, timed on wall clock with no sleeps in it at
   all. One full-screen repaint: every visible widget's ``render_line``, the
   compositor's cuts and chops, ``render_full_update()``. Deterministic, and
   the number that compares to the row prototype's "per scroll-and-render".

3. ``layout``: the same with ``layout=True``, so ``_refresh_layout()`` runs the
   arrange pass over the entire widget tree first. This is the one that is
   supposed to walk the whole conversation, and the reason a long chat is
   claimed to stutter.

WHAT IS *NOT* TIMED under ``--no-serialize`` (the default, and the honest one)
-----------------------------------------------------------------------------
``App._display`` early-returns when ``is_headless``, so segment→ANSI
serialisation and the terminal write are skipped. ``--serialize`` patches
``App.is_headless`` to False so ``_display`` runs the whole path; the driver's
``write`` is still a no-op, so only the syscall to the tty is missing. Both
numbers are reported when you pass ``--serialize``.

CONFOUNDS THE SCRIPT REMOVES (see --keep-timers / --animate to put them back)
----------------------------------------------------------------------------
The app arms 2s watcher/process/log pollers, a 60s db-cache sync and a curator
run at mount. Each of those wakes the loop and can land inside a timed sample.
They are stubbed to no-ops by default, and ``--keep-timers`` restores them so
the difference can be seen.

Textual *animates* the scroll a ↓ causes, and ``pilot.press`` waits for the
animator to go idle — so with animation on, every sample is pinned to the
animation's duration (~115 ms here) whatever the conversation contains. That
measures Textual's easing curve, not its rendering. The default therefore runs
with ``TEXTUAL_ANIMATIONS=none``, which is the number comparable to the row
prototype's; ``--animate`` restores the default look for the record.

The conversation is synthesised as ``hpca.transcript.Entry`` objects assigned to
``app._chat_entries`` and drawn with ``app._rerender_chat()``. That is exactly
the widget tree the real app builds; it skips the graph, the LLM and the
checkpointer, which are not part of a repaint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------- statistics


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def stats(prefix: str, samples: list[float]) -> dict:
    if not samples:
        return {f"{prefix}_n": 0}
    return {
        f"{prefix}_n": len(samples),
        f"{prefix}_p50_ms": round(pct(samples, 0.50), 4),
        f"{prefix}_p95_ms": round(pct(samples, 0.95), 4),
        f"{prefix}_p99_ms": round(pct(samples, 0.99), 4),
        f"{prefix}_max_ms": round(max(samples), 4),
        f"{prefix}_mean_ms": round(statistics.fmean(samples), 4),
    }


def summarise(name: str, measured: dict[str, list[float]], **extra) -> dict:
    out = {"scenario": name}
    for prefix, samples in measured.items():
        out.update(stats(prefix, samples))
    out.update(extra)
    return out


# ------------------------------------------------------------------ fake LLM


class FakeLLM:
    """Never called in this benchmark — nothing here starts a turn — but the
    app requires an LLM object at construction. Mirrors tests/test_tui_chat.py."""

    async def chat(self, messages, *, json_schema=None, **kwargs):
        from hpca.llm import ChatResponse

        return ChatResponse(content='{"action": "respond", "response": "ok"}')

    async def supports_constrained_decoding(self):
        return True

    async def aclose(self):
        return None


# ------------------------------------------------------------ content makers

PARAGRAPH = (
    "The job finished on node-042 after 14 minutes; the output landed in "
    "/fast/work/users/results/run_017 and the summary table has 3,214 rows. "
    "Nothing in the log looks like a failure, but the second replicate is "
    "missing two chromosomes, which is worth a look before anything downstream."
)


def message_text(i: int, lines: int = 4) -> str:
    return "\n".join([f"[{i:05d}] {PARAGRAPH}"] * lines)


def make_entries(n: int, *, big_at: int | None = None, big_bytes: int = 0):
    from hpca.transcript import Entry

    entries = []
    for i in range(n):
        kind = "user" if i % 2 == 0 else "assistant"
        if big_at is not None and i == big_at:
            unit = PARAGRAPH + "\n"
            text = (unit * (big_bytes // len(unit) + 1))[:big_bytes]
        else:
            text = message_text(i)
        entries.append(Entry(kind=kind, text=text, index=i))
    return entries


def make_thinking_entry(steps: int):
    """One turn's working with `steps` tool exchanges folded into it."""
    from hpca.transcript import Entry, Step

    parts = [
        Step(
            kind="call",
            text=f"edit_file /fast/work/users/data/chunk_{i:04d}.tsv\n"
            f"replace 3 lines with 5",
            tool="edit_file",
            target=f"chunk_{i:04d}.tsv",
            result=f"Edited /fast/work/users/data/chunk_{i:04d}.tsv (5 lines)",
            done=True,
        )
        for i in range(steps)
    ]
    return Entry(
        kind="thinking",
        text="\n\n".join(p.text for p in parts),
        steps=steps,
        reasoning_chars=4096,
        parts=parts,
    )


# ------------------------------------------------------------- app scaffolding


def stub_timers(monkey: list, keep: bool):
    """Silence the app's own pollers, which otherwise wake the loop inside a
    timed sample. Patched on the class before mount, because on_mount binds
    the methods when it arms the intervals."""
    from hpca.tui.app import HpcaApp

    HpcaApp.startup_backend_check = False
    if keep:
        return

    async def anoop(self, *a, **k):
        return None

    def noop(self, *a, **k):
        return None

    for name, fn in [
        ("refresh_watchers", anoop),
        ("watch_processes", anoop),
        ("poll_watched_logs", anoop),
        ("poll_watched_jobs", anoop),
        ("poll_jobs", anoop),
        ("_sync_db_cache", anoop),
        ("_discover_context_window", anoop),
        ("_startup_connect", anoop),
        ("run_curator_if_due", noop),
    ]:
        if hasattr(HpcaApp, name):
            monkey.append((HpcaApp, name, getattr(HpcaApp, name)))
            setattr(HpcaApp, name, fn)


def build_app():
    from hpca.config import Settings
    from hpca.tui.app import HpcaApp

    settings = Settings()
    # The db-cache lease/copy dance is startup cost and a background sync; it
    # has nothing to do with painting a frame.
    settings.database.local_cache = False
    return HpcaApp(settings=settings, llm=FakeLLM())


async def draw_entries(app, entries, expand_thinking=False):
    """Put a synthetic conversation on screen through the app's own renderer."""
    from hpca.tui.app import _ThinkingExpansion
    from hpca.transcript import THINKING

    app._chat_entries = list(entries)
    app._thinking_expanded.clear()
    if expand_thinking:
        for entry in entries:
            if entry.kind == THINKING:
                app._thinking_expanded[id(entry)] = _ThinkingExpansion(
                    open=True, parts={}
                )
    await app._rerender_chat()


async def measure_frames(app, iters: int, *, layout: bool, warmup: int = 5):
    """Force `iters` whole-screen frames and time each one.

    No sleeps and no message pump inside the timer: ``refresh`` only sets a
    flag, and ``_on_timer_update`` is the synchronous call that does the work.
    """
    screen = app.screen
    samples: list[float] = []
    for i in range(warmup + iters):
        screen.refresh(repaint=True, layout=layout)
        t0 = time.perf_counter()
        screen._on_timer_update()
        elapsed = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            samples.append(elapsed)
        await asyncio.sleep(0)
    return samples


async def fast_press(app, key: str, watch, timeout_yields: int = 5000):
    """Deliver one key and settle the frame without Pilot's barrier.

    ``Pilot._wait_for_screen`` posts a callback to *every widget on the screen*
    and waits for all of them; with 5000 rows that is 5000 messages of pure
    harness cost inside what is supposed to be one keystroke. Here the key goes
    straight into the app's queue, the loop is yielded to until the watched
    value changes (i.e. the widget has acted), and the frame is then forced.
    ``asyncio.sleep(0)`` yields without sleeping, so nothing but real work is
    inside the clock.

    Returns (milliseconds, whether the app actually reacted).
    """
    from textual import events

    before = watch()
    event = events.Key(key, key if len(key) == 1 else None)
    event.set_sender(app)
    t0 = time.perf_counter()
    app.post_message(event)
    reacted = False
    for _ in range(timeout_yields):
        await asyncio.sleep(0)
        if watch() != before:
            reacted = True
            break
    for _ in range(5):  # let the scroll / refresh messages land
        await asyncio.sleep(0)
    app.screen._on_timer_update()
    return (time.perf_counter() - t0) * 1000.0, reacted


async def measure_fast(app, listview, iters: int, *, key="down", warmup=10,
                       reset_margin=2):
    """`iters` cursor moves, each timed end-to-end with no harness overhead."""

    def watch():
        return listview.index

    async def reset():
        listview.index = 0
        listview.scroll_to(y=0, animate=False)
        for _ in range(5):
            await asyncio.sleep(0)
        app.screen._on_timer_update()

    listview.focus()
    await asyncio.sleep(0)
    await reset()
    samples: list[float] = []
    reactions = 0
    for i in range(warmup + iters):
        if listview.index is None or listview.index >= len(listview) - reset_margin:
            await reset()
        elapsed, reacted = await fast_press(app, key, watch)
        if i >= warmup:
            samples.append(elapsed)
            reactions += int(reacted)
    return samples, reactions


async def measure_resize(app, iters: int, width: int, height: int, warmup: int = 2):
    """Time a one-column terminal resize, which invalidates every cached wrap.

    This is the only interaction that makes Textual re-measure and re-wrap the
    *whole* conversation, so it is where a 1 MB message would be paid for over
    again rather than out of a cache.
    """
    from textual.events import Resize
    from textual.geometry import Size

    samples: list[float] = []
    arrived: list[bool] = []
    for i in range(warmup + iters):
        target = Size(width - 1 if i % 2 == 0 else width, height)
        if hasattr(app._driver, "_size"):
            app._driver._size = target
        # App._on_resize records the size and defers the real work to a 1/120 s
        # timer (_check_resize), which is what forwards the event to the screen
        # and triggers the relayout. Waiting on app._size alone would stop the
        # clock before any of that happened, so the deferred call is made here
        # rather than waited for.
        before = app._size
        t0 = time.perf_counter()
        app.post_message(Resize(target, target))
        for _ in range(5000):
            await asyncio.sleep(0)
            if app._size != before:
                break
        app._check_resize()
        for _ in range(5000):
            await asyncio.sleep(0)
            if app.screen.size == target:
                break
        for _ in range(5):
            await asyncio.sleep(0)
        app.screen._on_timer_update()
        elapsed = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            samples.append(elapsed)
            arrived.append(app.screen.size == target)
    if arrived and not all(arrived):
        # A sample where the screen never took the new size measured nothing;
        # better to fail loudly than to publish it.
        raise RuntimeError(
            f"resize did not reach the screen in {arrived.count(False)}/"
            f"{len(arrived)} samples"
        )
    return samples


async def measure_key(
    pilot,
    app,
    listview,
    iters: int,
    *,
    key: str = "down",
    warmup: int = 10,
    reset_margin: int = 2,
):
    """Press `key` `iters` times on `listview`, timing each press+settle.

    Two clocks per sample: wall (which carries ``wait_for_idle``'s ~100 ms
    sleep floor) and process_time (which does not, and is therefore the CPU
    the interaction actually cost).

    The cursor is walked from the top of the list downwards; once it nears the
    end it is put back to 0 *outside* the timer, so every timed sample is a
    genuine cursor move (and, past the first screenful, a one-row scroll).
    """

    async def reset():
        listview.index = 0
        listview.scroll_to(y=0, animate=False)
        await pilot.pause()

    listview.focus()
    await pilot.pause()
    await reset()

    wall: list[float] = []
    cpu: list[float] = []
    moved = 0
    for i in range(warmup + iters):
        if listview.index is None or listview.index >= len(listview) - reset_margin:
            await reset()
        before = listview.index
        t0, c0 = time.perf_counter(), time.process_time()
        await pilot.press(key)
        await pilot.pause()
        c1, t1 = time.process_time(), time.perf_counter()
        if i >= warmup:
            wall.append((t1 - t0) * 1000.0)
            cpu.append((c1 - c0) * 1000.0)
            if listview.index != before:
                moved += 1
    return wall, cpu, moved


async def list_metrics(pilot, app, listview, args):
    """Every unit, for a scenario whose interaction is ↓ in a ListView."""
    fast, reacted = await measure_fast(app, listview, args.iters)
    wall, cpu, moved = await measure_key(pilot, app, listview, args.pilot_iters)
    repaint = await measure_frames(app, args.frames, layout=False)
    layout = await measure_frames(app, args.frames, layout=True)
    measured = {
        "press": fast,
        "pilot_wall": wall,
        "pilot_cpu": cpu,
        "repaint": repaint,
        "layout": layout,
    }
    return measured, {"reacted": reacted, "pilot_moved": moved}


# ------------------------------------------------------------------ scenarios


async def scenario_chat(cfg, args):
    """N chat entries in the log; ↓ in the chat list."""
    app = build_app()
    width, height = cfg.get("width", 120), cfg.get("height", 40)
    async with app.run_test(size=(width, height)) as pilot:
        if args.serialize:
            unheadless(app)
        await app.start_new_session()
        await pilot.pause()
        entries = make_entries(
            cfg["entries"], big_at=cfg.get("big_at"), big_bytes=cfg.get("big_bytes", 0)
        )
        t0 = time.perf_counter()
        await draw_entries(app, entries)
        await pilot.pause()
        build_ms = (time.perf_counter() - t0) * 1000.0
        chat_list = app.query_one("#chat-list")
        measured, notes = await list_metrics(pilot, app, chat_list, args)
        measured["resize"] = await measure_resize(app, args.resizes, width, height)
        return summarise(
            cfg["name"],
            measured,
            rows=len(chat_list),
            entries=cfg["entries"],
            width=width,
            first_draw_ms=round(build_ms, 1),
            **notes,
        )


async def scenario_sessions(cfg, args):
    """N sessions in the sidebar; ↓ in the sessions list."""
    app = build_app()
    async with app.run_test(size=(120, 40)) as pilot:
        if args.serialize:
            unheadless(app)
        for i in range(cfg["sessions"]):
            app.session_store.create(
                profile="default", title=f"session {i:04d} — a working thread"
            )
        t0 = time.perf_counter()
        await app._reload_sessions()
        await pilot.pause()
        build_ms = (time.perf_counter() - t0) * 1000.0
        sessions_list = app.query_one("#sessions-list")
        measured, notes = await list_metrics(pilot, app, sessions_list, args)
        return summarise(
            cfg["name"],
            measured,
            rows=len(sessions_list),
            sessions=cfg["sessions"],
            first_draw_ms=round(build_ms, 1),
            **notes,
        )


async def scenario_steps(cfg, args):
    """One turn with N live tool rows.

    Two numbers: the cost of *inserting* the Nth live row (the M4 path,
    `report_step` + settle), and the cost of a ↓ afterwards with all N on
    screen.
    """
    app = build_app()
    async with app.run_test(size=(120, 40)) as pilot:
        if args.serialize:
            unheadless(app)
        await app.start_new_session()
        await pilot.pause()
        await draw_entries(app, make_entries(20))
        sid = app.active_session.session_id
        insert: list[float] = []
        for i in range(cfg["steps"]):
            payload = {
                "kind": "call",
                "tool": "edit_file",
                "arguments": {"path": f"/fast/work/users/data/chunk_{i:04d}.tsv"},
                "details": f"edit /fast/work/users/data/chunk_{i:04d}.tsv "
                f"(replace 3 lines with 5)",
            }
            c0 = time.process_time()
            app.report_step(sid, payload)
            await pilot.pause()
            insert.append((time.process_time() - c0) * 1000.0)
        chat_list = app.query_one("#chat-list")
        measured, notes = await list_metrics(pilot, app, chat_list, args)
        measured["insert_cpu"] = insert
        out = summarise(
            cfg["name"],
            measured,
            rows=len(chat_list),
            steps=cfg["steps"],
            **notes,
        )
        out["insert_first_ms"] = round(insert[0], 4)
        out["insert_last_ms"] = round(insert[-1], 4)
        out["insert_first50_p50_ms"] = round(pct(insert[:50], 0.50), 4)
        out["insert_last50_p50_ms"] = round(pct(insert[-50:], 0.50), 4)
        return out


async def scenario_thinking(cfg, args):
    """One expanded thinking box with N step rows under it; ↓ in the chat."""
    app = build_app()
    async with app.run_test(size=(120, 40)) as pilot:
        if args.serialize:
            unheadless(app)
        await app.start_new_session()
        await pilot.pause()
        entries = make_entries(20) + [make_thinking_entry(cfg["steps"])]
        t0 = time.perf_counter()
        await draw_entries(app, entries, expand_thinking=True)
        await pilot.pause()
        build_ms = (time.perf_counter() - t0) * 1000.0
        chat_list = app.query_one("#chat-list")
        measured, notes = await list_metrics(pilot, app, chat_list, args)
        return summarise(
            cfg["name"],
            measured,
            rows=len(chat_list),
            steps=cfg["steps"],
            first_draw_ms=round(build_ms, 1),
            **notes,
        )


async def scenario_draft(cfg, args):
    """A draft of N bytes in the input box; one printable keystroke.

    This is the one interaction that runs on *every* character typed, so it is
    timed as a key press into a focused TextArea rather than as a scroll.
    """
    from hpca.tui.app import ChatInput

    app = build_app()
    async with app.run_test(size=(120, 40)) as pilot:
        if args.serialize:
            unheadless(app)
        await app.start_new_session()
        await pilot.pause()
        await draw_entries(app, make_entries(100))
        chat_input = app.query_one("#chat-input", ChatInput)
        unit = PARAGRAPH + "\n"
        draft = (unit * (cfg["draft_bytes"] // len(unit) + 1))[: cfg["draft_bytes"]]
        t0 = time.perf_counter()
        chat_input.text = draft
        chat_input.focus()
        await pilot.pause()
        build_ms = (time.perf_counter() - t0) * 1000.0
        def watch():
            return len(chat_input.text)

        for _ in range(10):
            await fast_press(app, "x", watch)
        fast: list[float] = []
        reacted = 0
        for _ in range(args.iters):
            elapsed, ok = await fast_press(app, "x", watch)
            fast.append(elapsed)
            reacted += int(ok)
        wall: list[float] = []
        cpu: list[float] = []
        for _ in range(args.pilot_iters):
            t0, c0 = time.perf_counter(), time.process_time()
            await pilot.press("x")
            await pilot.pause()
            c1, t1 = time.process_time(), time.perf_counter()
            wall.append((t1 - t0) * 1000.0)
            cpu.append((c1 - c0) * 1000.0)
        repaint = await measure_frames(app, args.frames, layout=False)
        layout = await measure_frames(app, args.frames, layout=True)
        return summarise(
            cfg["name"],
            {
                "press": fast,
                "pilot_wall": wall,
                "pilot_cpu": cpu,
                "repaint": repaint,
                "layout": layout,
            },
            draft_bytes=cfg["draft_bytes"],
            final_len=len(chat_input.text),
            reacted=reacted,
            first_draw_ms=round(build_ms, 1),
        )


def unheadless(app):
    """Make App._display run its whole path (segment→ANSI serialisation).

    The headless driver's write() is a no-op, so nothing reaches a terminal;
    what this adds to the timed region is the Rich segment rendering Textual
    would otherwise hand to the tty. Patched after mount so startup is not
    affected."""
    type(app).is_headless = property(lambda self: False)


SCENARIOS: list[tuple[dict, object]] = [
    ({"name": "entries-100", "entries": 100}, scenario_chat),
    ({"name": "entries-300", "entries": 300}, scenario_chat),
    ({"name": "entries-1000", "entries": 1000}, scenario_chat),
    ({"name": "entries-5000", "entries": 5000}, scenario_chat),
    # A short chat so the cursor walks over the outsized entry roughly every
    # eighth press — long enough for p95 to see it if it costs anything.
    ({"name": "one-message-control", "entries": 12}, scenario_chat),
    (
        {"name": "one-message-10kb", "entries": 12, "big_at": 5, "big_bytes": 10_000},
        scenario_chat,
    ),
    (
        {
            "name": "one-message-1mb",
            "entries": 12,
            "big_at": 5,
            "big_bytes": 1_000_000,
        },
        scenario_chat,
    ),
    ({"name": "sessions-100", "sessions": 100}, scenario_sessions),
    ({"name": "sessions-1000", "sessions": 1000}, scenario_sessions),
    ({"name": "steps-500-live", "steps": 500}, scenario_steps),
    ({"name": "steps-500-folded", "steps": 500}, scenario_thinking),
    ({"name": "draft-100kb", "draft_bytes": 100_000}, scenario_draft),
    ({"name": "width-40", "entries": 300, "width": 40}, scenario_chat),
    ({"name": "width-120", "entries": 300, "width": 120}, scenario_chat),
    ({"name": "width-400", "entries": 300, "width": 400}, scenario_chat),
]


def machine_note() -> dict:
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = float("nan")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "loadavg": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_in_children(args, names: list[str]) -> list[dict]:
    """One scenario per fresh process.

    Not fastidiousness: a first run with all fourteen in one process made the
    *later* scenarios look 4x worse than identical earlier ones (width-120 is
    entries-300 by another name; it measured 32 ms p50 against 7.9 ms). Each
    scenario leaves a few thousand widgets behind, so by the end every gc pass
    walks the wreckage of all the ones before. A subprocess per scenario is the
    only way the rows compare to each other.
    """
    import subprocess

    results = []
    for name in names:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            out = handle.name
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--child",
            "--only",
            name,
            "--iters",
            str(args.iters),
            "--frames",
            str(args.frames),
            "--resizes",
            str(args.resizes),
            "--pilot-iters",
            str(args.pilot_iters),
            "--timeout",
            str(args.timeout),
            "--json",
            out,
        ]
        if args.serialize:
            cmd.append("--serialize")
        if args.keep_timers:
            cmd.append("--keep-timers")
        if args.animate:
            cmd.append("--animate")
        print(f"→ {name}", file=sys.stderr, flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            results.extend(json.loads(Path(out).read_text())["results"])
        except Exception as exc:
            results.append(
                {
                    "scenario": name,
                    "error": f"child failed ({exc}): "
                    f"{proc.stderr.strip().splitlines()[-1:]}",
                }
            )
        finally:
            Path(out).unlink(missing_ok=True)
    return results


async def main_async(args):
    import textual

    results = []
    chosen = [
        (cfg, fn)
        for cfg, fn in SCENARIOS
        if not args.only or cfg["name"] in args.only
    ]
    for cfg, fn in chosen:
        print(f"… {cfg['name']}", file=sys.stderr, flush=True)
        try:
            row = await asyncio.wait_for(fn(cfg, args), timeout=args.timeout)
        except asyncio.TimeoutError:
            row = {
                "scenario": cfg["name"],
                "error": f"timed out after {args.timeout}s",
            }
        except Exception as exc:  # a failed dimension is reported, not hidden
            row = {"scenario": cfg["name"], "error": f"{type(exc).__name__}: {exc}"}
        try:  # the box is shared; a spike in load is a confound worth recording
            row["loadavg1"] = round(os.getloadavg()[0], 2)
        except OSError:
            pass
        results.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)

    return {
        "machine": machine_note(),
        "textual": textual.__version__,
        "iters": args.iters,
        "serialize": args.serialize,
        "keep_timers": args.keep_timers,
        "animations": os.environ.get("TEXTUAL_ANIMATIONS"),
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=300, help="timed presses per scenario")
    ap.add_argument(
        "--frames", type=int, default=200, help="forced repaint/layout frames"
    )
    ap.add_argument(
        "--resizes", type=int, default=20, help="terminal resizes per chat scenario"
    )
    ap.add_argument(
        "--pilot-iters",
        type=int,
        default=40,
        help="presses driven through Pilot (slow: ~100 ms of sleep each)",
    )
    ap.add_argument("--only", nargs="*", default=None, help="scenario names to run")
    ap.add_argument("--json", default=None, help="write the full result set here")
    ap.add_argument(
        "--serialize",
        action="store_true",
        help="also run App._display (segment→ANSI) inside the timed region",
    )
    ap.add_argument(
        "--keep-timers",
        action="store_true",
        help="leave the app's 2s pollers armed (to see how much they cost)",
    )
    ap.add_argument(
        "--animate",
        action="store_true",
        help="leave Textual's scroll animation on (samples then measure the "
        "easing duration, not the render)",
    )
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--child", action="store_true", help="internal: run one scenario here"
    )
    ap.add_argument(
        "--in-process",
        action="store_true",
        help="run every scenario in this process (cross-contaminating; see "
        "run_in_children)",
    )
    args = ap.parse_args()

    if args.list:
        for cfg, _ in SCENARIOS:
            print(cfg["name"])
        return 0

    names = [
        cfg["name"] for cfg, _ in SCENARIOS if not args.only or cfg["name"] in args.only
    ]
    if not args.child and not args.in_process:
        report = {
            "machine": machine_note(),
            "iters": args.iters,
            "serialize": args.serialize,
            "keep_timers": args.keep_timers,
            "animations": "full" if args.animate else "none",
            "isolation": "one subprocess per scenario",
            "results": run_in_children(args, names),
        }
    else:
        home = tempfile.mkdtemp(prefix="hpca-bench-home-")
        os.environ["HPCA_HOME"] = home
        os.environ["TEXTUAL_ANIMATIONS"] = "full" if args.animate else "none"
        # A checkout's own project skills would otherwise be loaded into the app.
        os.chdir(tempfile.mkdtemp(prefix="hpca-bench-cwd-"))

        monkey: list = []
        stub_timers(monkey, args.keep_timers)

        report = asyncio.run(main_async(args))
        report["isolation"] = "child" if args.child else "single process"

    print()
    cols = [
        ("press p50", "press_p50_ms"),
        ("press p95", "press_p95_ms"),
        ("repaint p50", "repaint_p50_ms"),
        ("repaint p95", "repaint_p95_ms"),
        ("layout p50", "layout_p50_ms"),
        ("layout p95", "layout_p95_ms"),
        ("resize p50", "resize_p50_ms"),
        ("pilot cpu p50", "pilot_cpu_p50_ms"),
        ("pilot wall p50", "pilot_wall_p50_ms"),
    ]
    header = f"{'scenario':<18}" + "".join(f"{label:>15}" for label, _ in cols) + f"{'rows':>7}"
    print(header)
    print("-" * len(header))
    for row in report["results"]:
        if "error" in row:
            print(f"{row['scenario']:<18} {row['error']}")
            continue
        line = f"{row['scenario']:<18}"
        for _, key in cols:
            value = row.get(key)
            line += f"{value:>15.3f}" if isinstance(value, float) else f"{'—':>15}"
        line += f"{row.get('rows', ''):>7}"
        print(line)
    print()
    print(json.dumps(report["machine"]))

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
