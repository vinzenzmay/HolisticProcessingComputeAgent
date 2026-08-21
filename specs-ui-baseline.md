# Baseline: what the Textual UI actually costs

**Status:** measured 2026-08-20/21, before `src/hpca/tui/` is deleted.
Companion to [specs-ui-replacement.md](specs-ui-replacement.md) §8.

**Why it exists.** The whole replacement is justified by one line in a commit
message (`8b790e7`): *"0.012ms per scroll-and-render at 100 chat entries,
0.009ms at 5000. The Textual chat costs 44ms p95 at 300."* That number is not
reproducible from anything in the repo, and after `tui/` is gone it can never be
checked. So it was re-taken, on both sides, before the deletion.

**Headline: the specific claim does not hold, and the general one holds
comfortably.** Textual does not cost 44 ms at 300 entries — it costs about
13 ms. But it *does* grow with conversation length, and by 5000 entries a
keypress costs 104 ms at the median and 882 ms at p95, which is a UI that has
stopped working. The row UI is flat from 100 entries to 20,000 — at ~0.19 ms
when this was written, and at ~0.08 ms since M10 (§5, §7).

---

## 1. Method

`evals/bench_textual.py`, kept in the repo so the measurement can be repeated
while there is still something to measure. It imports `hpca.tui`, so it dies
with `src/hpca/tui/` at M9 and should be deleted in the same commit — the
numbers below are what outlives it. Like the other `evals/` scripts it is never
part of a test run. One subprocess per scenario, because a first pass showed
later scenarios inheriting the GC pressure of earlier ones.

Three units are timed separately, because no single one is honest on its own:

- **press** — `pilot.press("down")` then `pilot.pause()`: the whole interaction,
  from key event through binding, action, scroll, layout and compositor.
  Reported as **process time**, not wall clock. `pilot.press` and `pilot.pause`
  both call `wait_for_idle`, which sleeps in 20 ms granules, so wall time has a
  ~100 ms floor that belongs to the test harness and not to the app.
- **repaint** — `screen.refresh(repaint=True, layout=False)` plus
  `_on_timer_update()`, wall clock, no sleeps inside. One full-screen repaint.
- **layout** — the same with `layout=True`, so the arrange pass runs over the
  whole widget tree first.

Confounds removed, and the flags to put them back are in the script: the app's
2 s watcher/process/log pollers, its 60 s db-cache sync and the curator run are
stubbed (`--keep-timers` restores them), and **animations are off**
(`--animate` restores them).

That last one matters and is probably the explanation for the original figure.
Textual *animates* the scroll a `↓` causes, and `pilot.press` waits for the
animator to go idle — so with animation on, every sample is pinned to the
easing curve's duration (~115 ms here) no matter what the conversation
contains. That measures Textual's easing, not its rendering. A figure of 44 ms
is the kind of number that path produces.

The conversation is synthesised as `hpca.transcript.Entry` objects assigned to
`app._chat_entries` and drawn with `_rerender_chat()` — the same widget tree
the real app builds, skipping the graph, the LLM and the checkpointer, none of
which are part of a repaint.

**What is not measured:** `App._display` early-returns when headless, so
segment→ANSI serialisation and the write to the tty are excluded. The row UI's
numbers exclude the same two things, so the comparison is like for like.

Machine: 24 cores, Linux 7.0, Python 3.14.6, Textual 8.2.8, load average
2.5–5.5 during the run (noted per scenario in the JSON). 300 iterations per
scenario.

## 2. Textual

All figures in milliseconds.

| scenario | press p50 | press p95 | repaint p50 | layout p50 | layout p95 | first draw |
|---|---|---|---|---|---|---|
| entries-100 | 6.8 | 12.6 | 1.51 | 2.9 | 3.1 | 183 |
| entries-300 | 7.4 | 13.2 | 1.46 | 5.2 | 5.9 | 400 |
| entries-1000 | 18.8 | 36.9 | 1.49 | 18.4 | 21.8 | 2,933 |
| **entries-5000** | **103.9** | **881.9** | 1.54 | **126.1** | **994.0** | **15,172** |
| one-message-control | 4.2 | 6.0 | 1.41 | 1.8 | 2.0 | 48 |
| one-message-10kb | 4.4 | 6.8 | 1.44 | 1.9 | 2.0 | 78 |
| one-message-1mb | 4.8 | **587.2** | 1.63 | 2.1 | 2.2 | 218 |
| sessions-100 | 6.5 | 12.3 | 2.20 | 3.5 | 3.8 | 203 |
| sessions-1000 | 19.4 | 41.0 | 2.34 | 18.6 | 116.6 | 948 |
| steps-500-live | 18.8 | 36.4 | 2.18 | 11.6 | 15.8 | — |
| steps-500-folded | 19.7 | 25.6 | 2.13 | 10.4 | 12.7 | 733 |
| draft-100kb | 1.6 | 1.8 | 1.37 | 3.3 | 3.4 | 168 |
| width-40 | 7.1 | 13.1 | 1.03 | 8.0 | 14.7 | 415 |
| width-400 | 6.8 | 11.9 | 1.66 | 5.3 | 6.0 | 428 |

### What the split between the three units shows

**The repaint is flat and it is not the problem.** 1.4–1.6 ms whether the log
holds 100 entries or 5000. Textual's compositor only draws what is visible,
exactly as it should, and the row UI has no advantage there worth the name.

**The arrange pass is the problem, and it is O(conversation).** 2.9 ms at 100
entries, 18.4 at 1000, 126 at 5000 — and 994 ms at p95, i.e. a full second of
frozen terminal. This is precisely the thing the row UI's design note claims to
avoid by having no arrange pass at all, and it is the one place the claim is
worth making. Anything that invalidates layout — and a scroll does — pays it.

**Two pathologies that are not about length.** A single 1 MB message is cheap at
the median (4.8 ms) and 587 ms at p95: the cost is not paid per frame but is
paid, brutally, on the frames that touch it. And 1000 sessions in the sidebar
costs 117 ms p95 in layout, which has nothing to do with the conversation at
all.

**Startup is the least defensible number in the table.** 15 seconds to first
draw at 5000 entries, 2.9 s at 1000. That is not a repaint cost, it is mounting
one widget per entry, and it is what makes reopening a long session feel broken.

## 3. The row UI

Same machine, idle (load 0.3). One `handle("down")` plus one full `render()`,
which is the same interaction the `press` column measures, minus the serialise
and write that Textual's headless path also skips.

| scenario | p50 | p95 | max |
|---|---|---|---|
| entries-100 | 0.188 | 0.265 | 0.62 |
| entries-300 | 0.186 | 0.197 | 0.20 |
| entries-1000 | 0.195 | 0.205 | 0.33 |
| entries-5000 | 0.193 | 0.203 | 0.21 |
| **entries-20000** | **0.193** | **0.204** | 0.23 |
| one-message-1mb | 0.159 | 0.168 | 0.18 |
| width-40 | 0.162 | 0.172 | 0.18 |
| width-120 | 0.189 | 0.197 | 0.21 |
| width-400 | 0.303 | 0.317 | 0.33 |

Flat in entry count to 20,000 — the claim, and it holds. Cost tracks terminal
*area* instead (0.11 ms at 80×24, 0.18 at 120×40, 0.25 at 200×50), which is the
right shape: the work is proportional to what is on screen.

The 1 MB message costs *less* than the ordinary case rather than more, because
only the visible slice is ever wrapped. That is the single sharpest contrast
with the table above.

### The previously published row-UI figures do not reproduce either

`8b790e7` claimed 0.009–0.012 ms and the M0 work reported 0.016–0.018 ms. The
committed test measures 0.19 ms on an idle box today — an order of magnitude
slower than either. Two possible causes were checked and neither accounts for
it: M1's switch from `len()` to cell-accurate widths made `pad` 222× more
expensive, but fixing that with an ASCII fast path (18× back) moved frame time
by only 5%, so padding was never dominant; and terminal size explains a factor
of two, not ten.

The honest position is therefore that **the flatness is confirmed and the
absolute figure previously quoted was optimistic by about 10×**. 0.19 ms is
still a 5,000-frames-per-second budget and 60× cheaper than Textual's *best*
case, so nothing about the decision changes — but the number in circulation was
wrong and this document supersedes it.

## 4. Verdict

| | Textual | row UI | ratio |
|---|---|---|---|
| 100 entries | 6.8 ms | 0.19 ms | 36× |
| 1000 entries | 18.8 ms | 0.20 ms | 96× |
| 5000 entries | 103.9 ms | 0.19 ms | 540× |
| 5000 entries, p95 | 881.9 ms | 0.20 ms | 4,300× |
| 1 MB message, p95 | 587.2 ms | 0.17 ms | 3,500× |
| first draw, 5000 | 15,172 ms | ~0 | — |

The original "44 ms at 300" does not survive contact with a careful
measurement, and anyone repeating it should stop. What survives is better
founded and more useful: Textual's *repaint* is fine, its *layout* is
O(conversation), and layout is what a scroll invalidates. The replacement is
justified by the 1000-entry column and above, by the 15-second cold open, and by
the two pathologies that have nothing to do with length — not by the 300-entry
case, which is merely unpleasant.

## 5. What M10 did about it, and what it found

The prediction in this section was that `fit_index` was roughly half of frame
time, because the ASCII fast path missed: the UI's own furniture — `─ ▸ ▾ ● ○
› ↑ ⇧` — is non-ASCII, so nearly every drawn row contained at least one
character that dropped the whole row onto the per-character path. The fast
path was fastest on exactly the rows this UI does not have.

That was right, and the fix (`ansi.ONE_CELL_GLYPHS`, a set of the one-cell
glyphs the UI draws, checked with `frozenset.issuperset`) is worth more than
expected. Measured back-to-back in one process:

| | table off | table on |
|---|---|---|
| 100 entries | 0.224 ms | 0.082 ms |
| 1000 entries | 0.226 ms | 0.085 ms |
| 5000 entries | 0.219 ms | 0.082 ms |

**2.7× off the frame, and still flat.** The table is an optimisation and never
a second opinion: `char_width` remains the authority and `test_ui_ansi.py`
asserts it returns 1 for every character in the table, so a glyph that is not
one cell wide fails the suite rather than shifting a row.

### The regression it caused, and why the new tests exist

The first version of that fast path scanned the **whole string**, and a chat
entry can be a megabyte. `pad` measures with `fit_index` precisely so the cost
is the width of the row rather than the length of the string — the old
per-character loop is O(cells), because it stops the moment the budget is
used — so scanning made a 1 MB message cost **6.16 ms a frame**, against
0.061 ms for fifty ordinary ones. A hundredfold regression, in the exact case
§2 records Textual failing at.

`tests/test_ui_perf.py::test_one_enormous_message_costs_no_more_than_many_ordinary_ones`
caught it on the first run. The fix is to take the fast path on a window of
`cells + 1` characters, never the whole string; the `+ 1` is load-bearing,
because a zero-width combining mark follows the character it modifies and a
window cut exactly at the budget would drop one. That correctness case was
caught by an existing test the moment the window was introduced.

## 6. The dimensions as standing tests

§8 of specs-ui-replacement.md listed the dimensions; they are now
`tests/test_ui_perf.py`, one test each, with the Textual figure being guarded
against named in the docstring. Two kinds of assertion, and the second is the
one that bites:

- an **absolute** budget (2 ms) catches a constant-factor blowup, and is
  deliberately loose because the suite runs across every core and a tight
  threshold would be flaky rather than interesting;
- a **ratio** — cost at 5000 entries over cost at 100 — catches the shape.
  It is ~1.0 today and would be in the tens if a frame started walking the
  conversation, and it is invariant to how fast or how loaded the box is,
  because both halves are measured in the same run. Frame time is ~0.08 ms
  against a 2 ms budget, so a regression would have to be 25× before the
  absolute number noticed; the ratio notices at 4×.

Medians throughout, not means: one GC pause drags a mean far enough to fail a
test that is measuring something else.

## 7. Where it stands after M10

Same machine, idle. These supersede the row-UI figures in §3.

| scenario | median |
|---|---|
| entries-100 | 0.076 ms |
| entries-1000 | 0.077 ms |
| entries-5000 | 0.077 ms |
| **entries-20000** | **0.076 ms** |
| one-message-1mb | 0.115 ms |
| width-40 | 0.068 ms |
| width-120 | 0.075 ms |
| width-400 | 0.115 ms |

Flat in entry count to 20,000, cost tracking terminal *area* rather than
conversation length, and the 1 MB message now cheaper than the 0.159 ms §3
recorded for it. Against Textual's 103.9 ms median and 881.9 ms p95 at 5000
entries, the ratio is now ~1,350× and ~11,500×.
