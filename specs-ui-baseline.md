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
stopped working. The row UI is flat at ~0.19 ms from 100 entries to 20,000.

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

## 5. Known cost in the new UI, for M10

`fit_index` is roughly half of frame time. The ASCII fast path added here
(`text.isascii() and text.isprintable()` → one cell per character) misses more
often than it should, because the UI's own furniture — `─ ▸ ▾ ● ○ › ↑ ⇧` — is
non-ASCII, so most drawn rows contain at least one character that drops the
whole row onto the per-character path. A width table for the handful of glyphs
this UI actually draws would recover most of it. Not urgent at 10× inside
budget, and worth doing before anyone measures again.
