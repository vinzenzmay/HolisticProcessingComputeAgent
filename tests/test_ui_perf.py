"""The claim the whole replacement rests on: frame cost is flat.

Only the visible slice of a pane is ever turned into lines, so scrolling a
5000-entry chat must cost what scrolling a 100-entry one costs. That is the
property, and these are the dimensions it has to hold in — each one chosen
because the measurement in specs-ui-baseline.md found a specific pathology
there in the UI this replaced. The Textual figure each test guards against is
named in its own docstring, so a future reader can tell what the threshold is
*for* rather than guessing at a number.

**Two kinds of assertion, and the second is the one that bites.**

An absolute budget (`BUDGET_MS`) catches a constant-factor blowup — something
that made every frame slower regardless of length. It is deliberately loose,
because a box running the suite across every core is not a benchmark rig, and
a threshold tight enough to be interesting would be flaky instead.

That looseness is exactly why it is not enough on its own. Frame time is
~0.08 ms and the budget is 2.0 ms, so a regression would have to be 25x before
the budget noticed — and an O(entries) frame is the failure this UI exists to
prevent. So the shape is asserted as a *ratio*: cost at 5000 entries over cost
at 100. It is ~1.0 today and would be in the tens if a frame started walking
the conversation, and — the point — it is invariant to how fast or how loaded
the machine is, because both halves are measured on the same box in the same
run.

**Median, not mean.** One GC pause or one scheduler hiccup drags a mean far
enough to fail a test that is measuring something else. The median of
per-frame samples is what specs-ui-baseline.md reports and what is asserted
here.

**The frame is not the only clock.** A frame slices a cached list of lines,
so it is flat for free; the list itself is built by whatever *changes* the
conversation, and that is the second cost, paid per arriving row rather than
per keypress. It only became worth a test when a chat row stopped being one
line, and a rebuild per arriving row would have put the O(conversation) event
back after all the work of taking it out. `TestARowArriving` is that
dimension.
"""

import statistics
import time

import pytest

from hpca import protocol
from hpca.ui.app import RowUI
from hpca.ui.client import UIClient
from hpca.ui.demo import DemoCore, build
from hpca.ui.state import ChatEntry

WIDTH, HEIGHT = 120, 40
FRAMES = 200

# A constant-factor backstop, not a benchmark. See the module docstring.
BUDGET_MS = 2.0

# Four is far above the ~1.0 these ratios measure and far below the tens an
# O(entries) frame would produce. The gap is the noise margin.
FLAT = 4.0


def frame_ms(ui, width: int = WIDTH, height: int = HEIGHT) -> float:
    """Median milliseconds for one scroll plus one full render."""
    ui.render(width, height)  # the first frame builds the caches; not the claim
    samples = []
    for _ in range(FRAMES):
        started = time.perf_counter()
        ui.handle("down", width, height)
        ui.render(width, height)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def wired(entries: list[protocol.Entry]) -> RowUI:
    """A UI holding exactly these entries, filled the way the real one is.

    Through `chat.reset` rather than by assignment: the pane caches its flat
    lines, so a test that set the state behind the cache would be measuring
    the cache. This is the same three-object wiring `demo.build` does, kept
    open so the entries can be arbitrary.
    """
    ui = RowUI()
    core = DemoCore(chat=1, sessions=1, watchers=0)
    client = UIClient(ui, send=core.handle)
    core.emit = client.apply
    core.start()
    client.apply(
        protocol.ChatReset(session_id=ui.sessions[0].session_id, entries=entries)
    )
    # The core and client are only reachable from here; the UI holds no strong
    # reference back, and a collected client stops answering mid-measurement.
    ui._perf_keepalive = (core, client)
    return ui


def append_ms(ui, rows: int = 200) -> float:
    """Median ms for one row arriving *and the frame that shows it*.

    Both halves, because the cost can hide in either. `invalidate` is nearly
    free in itself and hands the whole bill to the next `flat`, so an append
    timed on its own reads as instant however much work it just booked — the
    first version of this measured exactly that and could not tell `extend`
    from the rebuild it replaced.
    """
    ui.render(WIDTH, HEIGHT)  # so there is a cache to keep or to drop
    session, inner = ui.session, WIDTH - 2
    samples = []
    for n in range(rows):
        entry = ChatEntry(
            kind="assistant", text="a reply of ordinary length", seq=10**6 + n
        )
        started = time.perf_counter()
        session.append(entry)
        ui.chat.flat(inner)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def replies(count: int, text: str = "a reply of quite ordinary length") -> list:
    return [
        protocol.Entry(kind="assistant", text=text, seq=i + 1) for i in range(count)
    ]


# ------------------------------------------------------------------ flatness


@pytest.mark.parametrize("entries", [100, 1000, 5000])
def test_a_scroll_and_a_render_stay_under_the_frame_budget(entries: int):
    assert frame_ms(build(chat=entries)) < BUDGET_MS


def test_frame_cost_does_not_grow_with_the_conversation():
    """Textual: 6.8 ms at 100 entries, 103.9 at 5000, 881.9 at p95 — a
    keypress that freezes the terminal for the best part of a second. Here the
    two must be the same number, because only the visible slice is wrapped."""
    small = frame_ms(build(chat=100))
    large = frame_ms(build(chat=5000))
    assert large / small < FLAT, f"{small:.3f} -> {large:.3f} ms"


# -------------------------------------------------------------- pathologies


def test_one_enormous_message_costs_no_more_than_many_ordinary_ones():
    """Textual: 4.8 ms median but 587 ms at p95 — a cost not paid every frame,
    but paid brutally on the frames that touch it.

    A megabyte in one entry should if anything be *cheaper* than fifty
    ordinary ones, because there is one row to walk instead of fifty and only
    the visible slice of it is ever folded. Asserted as "no worse", not "must
    be faster", since which one wins is not the property.
    """
    ordinary = frame_ms(wired(replies(50)))
    huge = frame_ms(wired([protocol.Entry(kind="assistant", text="x" * 1_000_000, seq=1)]))
    assert huge < ordinary * FLAT, f"ordinary {ordinary:.3f} -> huge {huge:.3f} ms"


def test_a_thousand_sessions_in_the_sidebar_is_not_a_pathology():
    """Textual: 117 ms p95 in layout at 1000 sessions — a cost with nothing to
    do with the conversation at all. The sidebar is a pane like any other and
    draws only its visible rows."""
    few = frame_ms(build(chat=50, sessions=10))
    many = frame_ms(build(chat=50, sessions=1000))
    assert many / few < FLAT, f"{few:.3f} -> {many:.3f} ms"


def test_five_hundred_steps_in_one_turn_stay_flat():
    """A long tool-using turn is one entry with hundreds of parts, and folding
    it must not mean walking them every frame. Textual paid 18.8 ms live and
    19.7 ms folded."""
    parts = [
        protocol.Part(kind="call", text="", tool="read_file", result="ok", done=True)
        for _ in range(500)
    ]
    ui = wired(
        [protocol.Entry(kind="thinking", text="", seq=1, steps=len(parts), parts=parts)]
    )
    assert frame_ms(ui) < BUDGET_MS


def test_a_hundred_kilobyte_draft_does_not_slow_the_frame():
    """The message box wraps its own text, and a long draft must not be
    re-folded on every keystroke elsewhere. Textual was fine here (1.6 ms) —
    this guards the property rather than fixing anything."""
    ui = build(chat=100)
    ui.sessions[ui.active].draft.set_text("y" * 100_000)
    assert frame_ms(ui) < BUDGET_MS


class TestARowArriving:
    """A message landing costs that message, not the conversation behind it.

    This is `Pane.extend`, and it is the half of the append-only invariant
    (specs-ui-replacement.md §3.2) that is actually spent rather than merely
    kept: the flattened line list is added to, so what a row costs is the
    lines that row draws.

    It matters more than it used to. A chat row is a label and at least one
    line of what it holds, so the line list is longer than the conversation is
    deep — 8,300 lines at 5000 entries against 5000 when a row was one line —
    and a rebuild per arriving row would have been an O(conversation) event on
    the busiest path there is. During a turn these arrive per tool call.
    """

    def test_it_does_not_grow_with_the_conversation(self):
        small = append_ms(build(chat=100))
        large = append_ms(build(chat=5000))
        assert large / small < FLAT, f"{small:.4f} -> {large:.4f} ms"

    @pytest.mark.parametrize("entries", [100, 5000])
    def test_and_stays_far_under_a_frame(self, entries: int):
        # Well under, because it is on the path a turn takes per tool call and
        # a frame is the thing it must not delay.
        assert append_ms(build(chat=entries)) < BUDGET_MS

    def test_the_row_it_took_is_on_screen(self):
        # The measurement above is worthless if `extend` quietly dropped what
        # it was given, so the same path is checked for the row itself.
        ui = build(chat=100)
        before = len(ui.chat.flat(WIDTH - 2))
        ui.session.append(ChatEntry(kind="assistant", text="one\ntwo", seq=10**6))
        # Closed, so: the label and the one line of preview under it.
        assert len(ui.chat.flat(WIDTH - 2)) == before + 2


# ------------------------------------------------- cost tracks area, not n


@pytest.mark.parametrize("width", [40, 120, 400])
def test_cost_tracks_terminal_area_rather_than_entry_count(width: int):
    """0.11 ms at 80x24, 0.18 at 120x40, 0.25 at 200x50 — the right shape: the
    work is proportional to what is on the screen, not to what is behind it."""
    assert frame_ms(build(chat=5000), width=width) < BUDGET_MS


# --------------------------------------------------------- the structural half


def test_the_visible_slice_is_all_that_is_ever_flattened():
    """The half of the claim that cannot flake: the pane's flat list is
    cached, so scrolling never rebuilds it."""
    ui = build(chat=5000)
    ui.render(WIDTH, HEIGHT)
    flat = ui.chat.flat(WIDTH - 2)
    for _ in range(50):
        ui.handle("down", WIDTH, HEIGHT)
    assert ui.chat.flat(WIDTH - 2) is flat
