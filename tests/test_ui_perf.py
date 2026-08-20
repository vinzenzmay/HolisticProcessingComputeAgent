"""The claim the whole exercise rests on: frame cost is flat in entry count.

Only the visible slice of a pane is ever turned into lines, so scrolling a
5000-entry chat must cost what scrolling a 100-entry one costs. The prototype
measured ~0.015 ms per scroll+render at 5000 entries; the threshold here is two
orders of magnitude looser so that a loaded CI box cannot turn a real
regression test into a flaky one. A frame that has gone O(entries) misses it by
far more than that margin.

specs-ui-replacement.md §8 lists the other dimensions — message length, session
count, steps per turn, draft length, terminal width — which land as benchmarks
in M10.
"""

import time

import pytest

from hpca.ui.demo import build

BUDGET_MS = 2.0
FRAMES = 200


def cost_per_frame(entries: int) -> float:
    """Milliseconds for one scroll plus one full render, averaged."""
    ui = build(chat=entries)
    ui.render(120, 40)  # the first frame builds the caches; it is not the claim
    started = time.perf_counter()
    for _ in range(FRAMES):
        ui.handle("down", 120, 40)
        ui.render(120, 40)
    return (time.perf_counter() - started) / FRAMES * 1000


@pytest.mark.parametrize("entries", [100, 1000, 5000])
def test_a_scroll_and_a_render_stay_under_the_frame_budget(entries: int):
    assert cost_per_frame(entries) < BUDGET_MS


def test_five_thousand_entries_still_under_2ms_per_frame():
    assert cost_per_frame(5000) < BUDGET_MS


def test_the_visible_slice_is_all_that_is_ever_flattened():
    # The structural half of the same claim, and the one that cannot flake:
    # the pane's flat list is cached, so scrolling never rebuilds it.
    ui = build(chat=5000)
    ui.render(120, 40)
    flat = ui.chat.flat(118)
    for _ in range(50):
        ui.handle("down", 120, 40)
    assert ui.chat.flat(118) is flat
