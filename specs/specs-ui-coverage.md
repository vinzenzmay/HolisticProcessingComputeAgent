# What the port covers, and what it does not

Companion to [specs-ui-acceptance.md](specs-ui-acceptance.md) and
[specs-ui-replacement.md](specs-ui-replacement.md) §7. Written immediately
before M9 — the milestone that deletes `src/hpca/tui/` and its 37
`tests/test_tui*.py` files — because after it there is no reference
implementation left to compare against.

**Status: re-audited at `e11b414`.** The first pass was taken at `599147e` and
blocked M9. Three commits worked the blockers — `63d44c4` (skills the model can
reach, and a menu that learns), `02feddb` (give the things that only
`tui/app.py` called somewhere else to live), `e11b414` (wire what the core
answers and the UI never asked). Every disposition below was re-checked against
the current tree rather than taken from those commit messages. The section
numbering is deliberately unchanged: `src/hpca/ui/__init__.py`,
`src/hpca/core/service.py`, `tests/test_core_service.py`, `tests/test_ui_boot.py`
and `tests/test_ui_commands.py` all cite these numbers.

**Method.** Every one of the 377 bullets in the acceptance list was read against
the tests that survive the deletion — everything in `tests/` except
`tests/test_tui*.py` — and against the implementation in `src/hpca/ui/`,
`src/hpca/core/` and `src/hpca/protocol.py`. Test bodies were opened rather than
test names trusted; where a name promises more than the assertion delivers it is
recorded as **weak** in §5. Where a module docstring promised three things, all
three were checked; that habit is what found §3.2, §3.4 and §3.7 the first time,
and §9.1 and §9.3 the second.

---

## 1. The verdict

**Delete `src/hpca/tui/`. Both halves are now safe — the test deletion
unambiguously, the front-end swap with a short list attached.**

The first audit split the question in two and answered it differently on each
side. That split has closed.

**The tests.** 571 Textual-coupled tests across 12,504 lines go; 1,109 tests in
`tests/test_ui_*.py` and 714 in the core suite stay — up from 601 and 678 at the
first pass. The two things the first audit said would be lost with nothing else
recording them are both properly closed, not worked around:

- `tests/test_context_bar.py` is repointed at `hpca.ui.meter`, and it kept the
  two assertions that had no successor in `test_ui_turn.py` — the overflowing
  bar staying inside its cells, and a small window reaching danger quickly. Its
  docstring now explains the copy-not-share history that made the repoint
  necessary.
- `tests/test_editor.py::TestResolveEditor` covers the `settings → $VISUAL →
  $EDITOR → nano` order in five tests, including both precedence pairs, and
  imports only `hpca.editor`.

**The front-end.** All three things that meant "the app does not work rather
than that it is missing a nicety" are closed, and closed at the seam rather than
in one half:

- `AgentService.startup()` runs the trash sweep, the curator, `auto_connect()`
  and `ensure_connected()` in that order; `Core.run()` schedules it and
  `boot.start()` opens manage-LLMs on a `False`. `ensure_connected` probes with
  the backend's own key and no pool keys, so a key-locked endpoint counts as not
  connected — all four startup-probe acceptance claims are now asserted
  (`test_core_backends.py::TestEnsureConnected::*`). Most convincingly,
  `test_ui_boot.py::TestTheWholeRun::test_nothing_answering_puts_manage_llms_on_the_screen`
  drives the *real* `AgentService` and the *real* UI over a real pipe with the
  probe stubbed to find nothing, and waits for both the warning and the screen.
  That is the seam, tested.
- The UI now sends `backend.scan` on opening manage-LLMs, fills the panel from
  the frames it answers with, stays keyable and closable while the scan is out,
  `r` removes after asking, and the form probes — autofill, the multi-model
  picker, the rejected-key warning, and `ctrl+s` to save anyway. `grep` for
  events with no client handler and commands never constructed in `ui/` now
  returns nothing but the abstract bases and three forward-looking commands.
- `ctx.queue_memory_edits` is bound in `_make_tool_ctx`, keyed to the session
  whose queue `/conclude` reads back, and
  `test_core_service.py::TestMemoryReview::test_the_memory_tool_reaches_the_queue_conclude_reads`
  goes the whole way: the model flags a fact mid-turn and `/conclude` offers it.

**On the one-caller sweep, which is the claim I was asked to second-guess.** I
ran it three ways independently — over public `def`s, over classes and
module-level constants, and over whole modules — counting a caller only when the
citing file actually imports the owning module. I validated the method by
running it at `599147e` first: it recovers all five of the original instances
(`cleanup`, `run_curator_if_due`, `warn_about_struggles`, `write_report` as
only-called-from-`tui`, and `auto_connect` as called-from-nowhere). At `e11b414`
the three sweeps agree on one answer:

| sweep | candidates | real residue |
|---|---:|---|
| public `def`s | 23 | `write_report` (looplag) |
| classes + constants | 17 | `LoopLagProbe` (looplag) |
| whole modules | 17 | `hpca.looplag` |

Everything else is a false positive of a name-based sweep — in-module callers
(`ui/approval.py`'s seven helpers, `core/pollers.py`'s three poll methods, which
`start_timers` schedules eleven lines further down), registry-dispatched
handlers (`ask_docs`), and `tui/`'s own same-named methods. Two names are dead
but harmless: `jobs.poll_active`, a wrapper superseded by `apply_statuses` which
`core/pollers.poll_jobs` calls directly, and `BackendRegistry.choices()`, whose
docstring says "for callers inside the core" and which has none.

**So the sweep is sound, and it did not clear everything: `hpca/looplag.py` is
still the shape it was written to find** (§3.12). `02feddb`'s message says "Five
of them"; looplag is a sixth, it was already named in §3.12 of the audit that
commit was answering, and it was not done. The method's one real blind spot is
generic names — a bare `cleanup` is only caught once callers are filtered by
import, which is why the tightened form is the one to keep.

**What would still be lost.** Six things, none of which stops the app working,
listed in §3 and ranked in §3.0. The two worth naming here:

1. **The prompt-injection warning on a flagged memory batch** (§3.5) — computed,
   carried as far as `ProposalSet.flagged`, and dropped by `_wire()` because
   `protocol.Proposal` is three strings. This is the one screen where the
   warning exists precisely because the user is about to persist
   attacker-influenced text, and it is unchanged since the first audit.
2. **The watchers column's arrangement** (§3.11) — `alt+↑/↓` reorders locally and
   the next `panel.update` puts the box back, because there is no `watch.move`
   on the wire; `shift+↑/↓`, the multiplexer fallback the old TUI bound, matches
   no branch; and the "how long ago the log was last written" line sits in the
   row's `body`, invisible until the row is expanded. A working feature becomes
   a non-working one.

The rest — memory not re-read at a session boundary (§3.8), loop-lag
instrumentation (§3.12), creating a profile from inside the new-session picker,
and the picker starting on the profile you are actually using (§7) — are
niceties, but they are real, and none of them is recorded as deliberately
dropped anywhere.

**And the fixes cost something, which §9 lists in full.** Nine new findings, two
of them medium-high and both in the newest code path: `/skill-creator <request>`
is handled ahead of the "wait for this turn" guard on the stated grounds that it
asks the core for nothing — which stopped being true in the same commit — and
overwrites a running turn's spinner label and elapsed clock (§9.7); and a draft
landing seconds later assigns `self.overlay`, which discards the whole screen
stack, so an open config editor or profile editor and its unsaved text vanish
(§9.8). Neither blocks M9. Both should be fixed before the row UI becomes the
default, because both are reachable by an ordinary user doing an ordinary thing.
One correction to my own first pass belongs here too: the escape-stop fix is
three-quarters done, not done — a turn parked on an approval still answers
`esc esc` with *"stopped the turn"* and does nothing (§4).

**One honest caveat, and it is not a coverage number.** `tests/test_ui_live.py`
still holds two tests. The 26 passing live tests are overwhelmingly bucket-A
core tests that predate the port; the new stack's live end-to-end coverage is
one real turn and one frame-width check. §9 of the replacement spec already
books real-terminal behaviour — alt screen, paste, resize, `$EDITOR` suspend —
as hand-verified at each checkpoint. The newly-wired startup path is unit-tested
thoroughly and joined-up-tested once, hermetically; it has never run against a
real cluster, and `startup_backend_check` is off suite-wide by design. Drive
`hpca --new-ui` on the cluster once, from a machine with no backend configured
and then with the tunnel down, before flipping the default.

---

## 2. The counts

Counted at the **bullet** level (377 bullets), each bullet classified by its
weakest constituent claim. A handful are judgement calls, so read these as ±5.

| bucket | at `599147e` | at `e11b414` |
|---|---:|---:|
| covered (incl. covered indirectly) | 299 | **333** |
| not covered — behaviour exists, nothing asserts it | 28 | **25** |
| not implemented — behaviour absent from the new stack | 38 | **7** |
| dropped deliberately, with a recorded reason | 11 | 11 |
| degraded — present but materially worse | 1 | 0 |

**The bucket the verdict rested on, re-derived.** All 38 previously-unimplemented
bullets were re-checked individually. Seven remain:

| # | bullet | section | §3 item |
|---|---|---|---|
| 1 | the watchers arrangement does not stick; no `shift+↑/↓`; the age line is hidden | Watchers | §3.11 |
| 2–4 | the lag probe is off unless `HPCA_LOOPLAG`; a spike names the step; a run leaves a report | Lag instrumentation | §3.12 |
| 5 | memories refresh at a session boundary | Core chat wiring | §3.8 |
| 6 | creating a profile from inside the new-session picker | Profiles | §3.13 |
| 7 | the chat entry only exists inside a session | Navigating the entry | §3.13 |

Two of the original 38 were **my error, not the tree's**, and are corrected here:
"opening a session switches to its profile" is covered UI-side by
`test_ui_app.py::test_the_profile_follows_the_session`, and the one-arg
per-session LLM provider (§3.14 row 16) *is* driven by every real-service turn
through `build_service`'s `llm=lambda sid: …`, so an arity regression would fail
loudly. Both are counted as covered above.

**The "not covered" bucket barely moved, and that is the honest headline of this
pass.** Of the 20 rows in §3.14, two are closed (the editor order, deliberately;
reusing a message twice, incidentally), one was my error, five moved from
nothing to part of it, and twelve are untouched. The three commits aimed at
*unimplemented* behaviour and hit it squarely; nothing was aimed at behaviour
that exists and is unasserted, and it shows.

Per section, gaps only:

| section | bullets | not covered | not impl. | dropped |
|---|---:|---:|---:|---:|
| Replies to a session you left | 8 | 2 | — | 1 |
| Rewind and fork | 15 | — | — | — |
| Core chat wiring | 21 | 1 | 1 | 1 |
| Agent modes | 8 | 1 | — | — |
| Thinking box, tool rows, logging | 22 | 1 | — | — |
| Titles, rename, delete | 20 | 2 | — | — |
| Profiles | 21 | 2 | 1 | — |
| Backends | 30 | 1 | — | 1 |
| Memory | 25 | 3 | — | — |
| Skills and self-review | 33 | 6 | — | — |
| Compaction | 9 | — | — | — |
| Node-local databases | 12 | 5 | — | — |
| Slash-command menu | 8 | — | — | — |
| Navigating the entry | 14 | — | 1 | 2 |
| Selecting text in the chat | 7 | — | — | 7 |
| Watchers column | 11 | 2 | 1 | — |
| Background processes and jobs | 7 | 1 | — | — |
| Lag instrumentation | 3 | — | 3 | — |

Backends went from 14 unimplemented bullets to none. Skills went from 8 to none.
Those two sections were 22 of the original 38, and each was one missing wire.

---

## 3. The blocker list, item by item

### 3.0 Disposition

| item | first pass | now |
|---|---|---|
| 3.1 a cluster user cannot connect, and is not told | not implemented (14) | **closed** |
| 3.2 the `memory` tool is dead | not implemented | **closed** |
| 3.3 two startup jobs lose their only caller | not implemented | **closed** |
| 3.4 the "updated" sidebar marker is gone | not implemented (2) | **partly closed** — built and wired, no test, and the failure path still does not set it |
| 3.5 the prompt-injection warning never reaches the screen | not implemented | **still open**, unchanged |
| 3.6 the context meter lies after `/compact` | not implemented | **closed** (the §9.5 nit closed with the compaction review) |
| 3.7 long answers clipped, inspect window unreachable | degraded | **closed** |
| 3.8 memories never re-read at a session boundary | not implemented | **still open**, unchanged |
| 3.9 the skill drafter and skill levels | not implemented (8) | **closed** |
| 3.10 the slash menu is in definition order | not implemented (2) | **closed** |
| 3.11 the watchers arrangement does not stick | not implemented | **still open**, unchanged |
| 3.12 the lag probe is unwired | not implemented (3) | **still open**, unchanged |
| 3.13 smaller unimplemented items | mixed | 2 closed, 1 was my error, 3 open |
| 3.14 not covered (20 rows) | not covered | 2 closed, 1 my error, 5 partial, 12 open |

Ranked by what a user would notice, what is left is: the watchers column
(§3.11), the injection warning (§3.5), the memory boundary (§3.8), the profile
picker (§3.13), loop-lag (§3.12), and — as a *testing* rather than behaviour gap
— the untested updated-marker (§3.4).

### 3.1 A cluster user cannot connect, and is not told — **CLOSED**

Both halves, and the seam. `BackendRegistry.ensure_connected()` (backends.py:1179)
probes the active backend with its own key and treats a 401 as not connected;
`AgentService.startup()` (service.py:2569) runs auto-connect first so a cluster
LLM counts as connected by consequence rather than special case;
`ui/boot.py::_open_llm_screen` opens the screen and declines to land on one the
user already opened — the guard I flagged in §7, reproduced with its rationale.

Convinced by: `test_core_backends.py::TestEnsureConnected` (8 tests: answers /
nothing answers and says why / key-locked / the key it carries is the one that
must work / nothing configured / a probe that explodes leaves the user alone /
an auto-connected cluster LLM counts as connected), `test_core_service.py::TestStartup`
(9 tests including `test_it_connects_before_it_checks` asserting the order),
`test_ui_boot.py::TestStartupPass` (4), and the end-to-end
`TestTheWholeRun::test_nothing_answering_puts_manage_llms_on_the_screen`.

Discovery: `test_ui_overlays.py::TestManageLlms` now runs to 34 tests including
`test_the_scan_goes_out_when_the_screen_opens`,
`test_and_the_panel_fills_from_the_frames_it_answers_with`,
`test_the_screen_takes_keys_while_the_scan_is_still_out`,
`test_a_rescan_asks_again`, all four empty-scan conditions, the recipe window
waiting for an open form, `r` and its denial, and — closing a §7 item —
`test_a_discovered_row_already_configured_is_not_drawn_twice`.
`TestBackendForm` runs to 20 including the probe, the autofill, the picker, the
rejected key, and `ctrl+s`, which were §7 items too.

### 3.2 The `memory` tool is dead — **CLOSED**

`_make_tool_ctx` binds `queue_memory_edits` to a session-keyed closure
(service.py:3043). Convinced by
`test_core_service.py::TestMemoryReview::test_the_memory_tool_reaches_the_queue_conclude_reads`,
which drives a real turn whose model calls the tool, asserts the tool's own
result text, then runs `/conclude` and asserts the flagged fact is in the offer.
Its docstring names why the three tests above it could not have caught this.

### 3.3 Two startup jobs lose their only caller — **CLOSED**

`AgentService._sweep_trash` and `self._memory.run_curator_if_due()` both run
from `startup()`, both best-effort, both reporting as events. Convinced by
`TestStartup::test_the_trash_is_swept`, `::test_a_backup_inside_its_ttl_is_left_alone`,
`::test_a_trash_that_cannot_be_swept_does_not_stop_startup`,
`::test_the_curator_runs`, `::test_the_curator_decides_for_itself_whether_it_is_due`.
Placing them in the core rather than in `boot` is right: both are driven by
settings keys that §4.2 rule 2 puts out of a front-end's reach.

### 3.4 The "updated" sidebar marker — **PARTLY CLOSED**

Back, and correctly built: `UPDATED_MARK = "*"` (app.py:146), set by
`RowUI.replied` only when the finishing session is not the one on screen
(app.py:531-543), cleared on open (app.py:617), and drawn only where the
decision and working marks are not — a row cannot be both still working and
finished. `client._finished` calls it (client.py:807) with the reason written
down.

**What remains.** Two things.

- **No test anywhere touches it.** `grep -rn 'UPDATED_MARK\|\.updated\b' tests/`
  returns nothing. A feature restored because its absence was invisible is now
  present in exactly the way that was invisible.
- **The failure path does not set it.** `client._failed` (client.py:809) appends
  the error entry and calls `refresh_sidebar()`, not `replied()`. The old TUI
  marked the row on both paths — `tui/app.py:2547` in the `except`, and 2572
  after a successful reply. So a background turn that *fails* clears its `⟳`,
  leaves the row blank, and puts an error entry in a transcript nothing points
  at. That is the original bug, narrowed to one path.

### 3.5 The prompt-injection warning never reaches the screen — **STILL OPEN**

Unchanged. `memory_ops` computes the flags, `ProposalSet.flagged` carries them
(`memory_service.py:127, 461`), and `_wire()` (memory_service.py:976) drops
them because `protocol.Proposal` is still `scope`, `kind`, `text`. The Textual
screen drew `⚠ text matches a prompt-injection pattern (…) — read it closely`
at `tui/memory_screens.py:112-121`; nothing draws it now.

The batch shape is also unchanged: `_emit_proposals` sends a flagged batch as N
separate `Proposal`s while `_apply_flagged` requires `all(flags)`, so answering
`y y n` to a three-op batch loses all three, explained only by a "Discarded the
flagged memory changes." toast — and `overlays/memory.py`'s docstring still says
a batch is "one question when the core sends it as one", which is not what the
core sends.

### 3.6 The context meter lies after `/compact` — **CLOSED**

`Context.superseded()` clears `measured` without clearing the fill, so the
number keeps drawing and starts drawing with the `~`, and the core's fresh
`context.estimate` is allowed to speak again. Called where the fold actually
happens — `RowUI._resolve_compact` on an accepted summary, since the review
landed and `/compact` no longer folds on the keystroke. (The review was a
screen when this was written; it is now the inline prompt in the session's own
column, `ui/compaction.py`, and the call moved with it.)
`test_ui_commands.py::TestWhatACompactDoesToTheMeter` cites this section by
name; §9.5's nit went with the move.

### 3.7 Long answers clipped, and the window unreachable — **CLOSED**

`RowUI.toast` sends a titled `Notify` whose body exceeds `toasts.MAX_BODY` to
`RowUI.window`, which is `RowUI.inspect` plus the rule that it does not land on
a screen the user opened for something else — parked and placed when it can be,
except for a screen that asked for it (`Overlay.welcomes_window`). The core now
gives `/compact`'s summary and `/skills-list` a `title`, so both open the
window. Convinced by `test_ui_commands.py::TestALongAnswer` (`test_a_titled_block_opens_the_window`,
`::test_and_the_heading_is_the_window_title`, `::test_and_it_stays_until_escape`,
`::test_a_one_liner_with_a_heading_is_still_a_toast`) and by
`test_ui_overlays.py::TestManageLlms::test_the_tunnel_recipe_opens_over_the_screen_that_asked`.
`RowUI.inspect` has two real callers now, and `overlays/inspect.py`'s docstring
names them accurately.

### 3.8 Memories are never re-read at a session boundary — **STILL OPEN**

Unchanged. `MemoryService.snapshot` caches per profile and `invalidate` is
called only after the core's own writes (memory_service.py:236-242). Nothing
invalidates on `session.open`. `tui/app.py:3552` did, with the reason written
down: *"A session boundary is a deliberate refresh point… memories written by
another session or instance are picked up here, while WITHIN a session the
frozen snapshot keeps the prompt prefix byte-stable."* The byte-stability half
survives and is well tested; the pick-up half does not exist.

### 3.9 The skill drafter and skill levels — **CLOSED**

`skill.draft` / `skill.drafted` carry the model's draft into the creator form;
`SkillSave` gained a level; `skill.list` gained a scope so shipped skills reach
the menu. Convinced by `test_ui_commands.py::TestSkillCreator` — 18 tests
including `test_an_argument_asks_the_core_for_a_draft`,
`::test_the_draft_opens_the_same_form_pre_filled`, `::test_and_it_saves_like_any_other_skill`,
`::test_a_failed_draft_still_opens_the_empty_form`,
`::test_and_a_global_skill_is_visible_but_not_own`,
`::test_and_a_project_skill_is_removable`,
`::test_a_name_taken_at_another_level_is_not_a_duplicate` — and by
`TestASkillByName::test_a_shipped_skill_runs_out_of_the_box`, which is `/plan`
on a fresh install.

### 3.10 The slash menu is in definition order — **CLOSED**

`command.list` / `command.counts` restore the ordering, restated after every
counted command rather than carried on `hello` (the reasoning is in
`protocol.CommandCounts`). `client.py:603` asks on connect; `_command_counts`
applies it. The UI still does not read `command_usage`, which
`TestFrequencyOrdering::test_the_ui_does_not_read_the_usage_table` continues to
assert in a clean interpreter.

### 3.11 The watchers column's arrangement does not stick — **STILL OPEN**

Unchanged in all three respects. There is no `watch.move` command in
`protocol.py` and no `MoveWatch` intent in `state.py`, so `alt+↑/↓` swaps two
`Item`s locally and the next `panel.update` — which arrives within seconds,
because a log box's age text changes every tick — puts the box back.
`shift+↑/↓` decode correctly in `keys.py` and match no branch in `app.py`, so
under tmux with alt swallowed there is no reorder key at all. And
`client._panel_item` (client.py:103-114) still puts `watch_lines()[0]` in `head`
and `[1:]` in `body`, while `watches.watch_lines` returns
`[first, _log_freshness(watch, now)]` — so "how long ago the log was last
written", the reason the box exists, is invisible until the row is expanded.

### 3.12 The lag probe is unwired — **STILL OPEN**

Unchanged, and it is the one instance the one-caller sweep did not clear.
`HPCA_LOOPLAG`, the label hook and the exit report exist only in
`tui/app.py:1467-1481`. `hpca/looplag.py` is the single non-`tui` module that
dies with `tui/` — all three of my sweeps agree — and `test_looplag.py` stays
green with no caller. §8 of the replacement spec plans to compare both sides at
M10; the instrumentation for one of them is about to become unreachable.

### 3.13 Smaller items

- **`↑` no longer leaves the message box**, and neither do `←/→`. Dropped
  deliberately (§5 key map, `55310bc`); the *removal* is still recorded nowhere
  but here.
- **`↑` in an empty log** — `ctrl-up` sets `focus = CHAT` unconditionally. Open,
  harmless.
- **The chat entry exists outside a session** — `_input_h` is unconditional; only
  the send is refused. Open.
- **"Nothing to save is silent"** — the core still emits *"Nothing durable to
  keep from this conversation."* (service.py:2482) and *"Nothing kept."*
  (service.py:2055). Neither string is asserted; either the claim or the code
  should move. Open.
- **Creating a profile from inside the new-session picker** — still absent;
  `overlays/newsession.py` has no `(new profile)` row. Open, no recorded reason.
- **"Opening a session switches to its profile"** — **my error.** Covered by
  `test_ui_app.py::test_the_profile_follows_the_session`. The core half of it
  lives in §3.14 row 2.

### 3.14 Not covered — the behaviour exists, nothing asserts it

Re-checked row by row. **2 closed, 1 was my error, 5 partial, 12 open.**

| # | claim | now |
|---|---|---|
| 1 | `d` deletes a session **and keeps the log file** | **open** — `TestSessionDelete` has five tests (row, history, watches, episodic index, queue) and none stats the file |
| 2 | profile memories reach a real system prompt | **open, and now the most important one** — see below |
| 3 | the editor resolution order | **closed** — `tests/test_editor.py::TestResolveEditor` |
| 4 | the dbcache notices reaching the user | **open** — `build_core` never passes `notices=`; nothing greps for `"Databases:"` |
| 5 | a new session starts in the configured **default mode** | **open** — `service.py:700` passes it; `test_core_service.py:301` mutates the setting only to shake the settings digest |
| 6 | `ctx.llm` routes to the session's backend | **open** — the primitive is covered (`test_core_backends.py::TestClients`), the join in `_make_tool_ctx` is not |
| 7 | job tools registered iff Slurm is available | **open** — nothing calls `tools.names()` on a built service |
| 8 | an **approved** script stays readable in the chat | **open** — the refused case is covered; the approved side asserts only that the prompt goes |
| 9 | the **second** Ctrl+C during the exit sync aborts | **partial** — the first press and the outer restore are covered; nothing asserts `on_interrupt` reinstalls `previous`, and `grep KeyboardInterrupt tests/` is empty |
| 10 | the watchers cursor: top / drop / clamp / log file | **partial** — two tests insert a row above and keep the cursor; nothing drops one, nothing clamps, nothing stats the log file after `watch.drop` |
| 11 | `MEMORY_GUIDANCE`, `SKILLS_GUIDANCE`, no skill names in the prompt | **open** — `test_prompts.py` covers one conditional block, `session_search` |
| 12 | the `/<skill>` **core** path | **open** — the UI half is now covered (`TestASkillByName`); `grep forced_skill tests/test_core_service.py` is empty, so the word-stripping, the sidecar directive and the clean transcript stay unasserted |
| 13 | a turn's tool context is per session | **partial** — two thirds now (the log, and the session-keyed memory queue); the runner and `session_id` are still unasserted |
| 14 | copying a profile inherits its **skills** | **open** — `copy_profile_skills` is covered directly, not through `duplicate_profile` |
| 15 | `skill_new` from a reflection | **open** — only the `skill_patch` branch is driven |
| 16 | the one-arg per-session LLM provider | **my error** — it *is* driven by every real-service turn through `build_service`. Routing is still unasserted (every such test has one fake LLM), but an arity regression would fail loudly |
| 17 | reusing a message twice stacks both | **closed** — `test_ui_queue.py::TestATakenBackMessageGoesToItsOwnSession::test_it_never_overwrites_a_draft` asserts `_into_draft`'s rule, and `reuse_message` now routes through it |
| 18 | the mouse stays released | **open** — `ENTER_MODES` is correct (`?1049h ?25l ?7l ?2004h`), and `test_ui_screen.py::TestTerminalModes::test_every_mode_set_is_reset` asserts symmetry, not absence. The one-line `"[?100" not in ENTER_MODES` still does not exist |
| 19 | `file_logger` / `quiet_terminal` | **open** — a near-copy is now tested (`test_core_backends.py::TestAutoconnectLogger`), for `core.backends.autoconnect_logger`, not for boot's pair |
| 20 | `sync_interval` from settings; `0` = exit-only | **open** — `TestPeriodicSync` sets `core.sync_interval` by hand and varies only `active` |

**Row 2 deserves promotion.** It is now the single most consequential unasserted
claim in the tree, and it is exactly the shape of the bug that started this
audit. `test_core_service.py::TestTurns::test_the_prompt_carries_the_sessions_own_profile`
is the *only* place in 4,600 lines of core-service tests that looks at a system
prompt (`llm.prompts[0][0]`), and it asserts `system["role"] == "system"` and
that `content` is a non-empty `str` — while its own comment claims it checks the
prompt was rendered for the running session. Every piece is tested in isolation
(`test_profiles.py::system_prompt_text`, `test_prompts.py::orchestrator_system_prompt`);
the join in `service.system_prompt_for` is not. If memories stopped reaching the
prompt tomorrow, nothing on screen would change and no test would fail. One test
that submits a turn under a profile with a distinctive memory and greps the
system message for it closes it.

---

## 4. The two latent bugs — one fixed, one three-quarters fixed

**`_escape` claiming a stop it may not have made — the tests are fixed properly,
the behaviour on one path is not.**
`RowUI._escape` (app.py:1680) now requires `self.active_id and
self.session.turn.interruptible` before sending, and answers the two other cases
distinctly: `NOTHING_TO_STOP` on an idle session, `NOT_A_TURN` when a spinner is
turning for a backend call with no turn behind it. Crucially **all four**
escape-stop tests were repaired, not one: `test_ui_app.py` now builds them
through `recorded()` and reads `stops(ui)` — the `Interrupt` intents — with the
comment naming why the note is not evidence, and two new tests cover the cases
that used to lie (`test_a_completed_gesture_with_nothing_running_claims_nothing`,
`test_and_on_a_backend_call_it_says_which`). The §5 rows for those tests are
struck.

**But the second half of that paragraph is untouched, and I initially wrote it
off wrongly.** `state.Turn.interruptible` is still just `working`, while the
core additionally requires a live `TurnState`. Two cases part company:

- *A turn parked on an approval.* `TurnScheduler` emits `DecisionRequested` and
  returns without a `TurnFinished` (scheduler.py:1128-1137), and the `finally`
  above it has already popped the `TurnState` (1110). So the UI keeps
  `working = True` — nothing clears it — and `esc esc` sends `Interrupt`, the
  core's handler finds no `TurnState` and returns `None` emitting nothing, and
  the footer says *"stopped the turn"*. The lie the fix was written to remove,
  surviving on the one path where a turn most often sits waiting.
- *A resume after an approval is fine*: `TurnScheduler` re-binds the anchor
  (scheduler.py:735-742) so `user_text` and `interrupt_keep` come back and the
  turn really is stoppable, which is what the acceptance list asks for. That is
  the case I checked first and then generalised from, incorrectly.

The fix is one line in `Turn.interruptible` — a turn parked on a decision is not
interruptible — or an explicit answer from the core; either way `NOTHING_TO_STOP`
already exists to say it. Nothing tests `esc esc` while parked.

**`reuse_message` copying into the wrong session — fixed.** It takes a
`session_id`, routes through `hand_back`, and `_rewind` passes
`overlay.session_id` (app.py:1576, 1587). One implementation of the rule instead
of two that disagreed.

---

## 5. Tests that assert less than their names promise

Re-checked. Struck rows are fixed; the rest stand.

| test | promises | actually asserts |
|---|---|---|
| ~~`test_ui_app.py` escape-stop tests~~ | ~~the gesture reaches the core~~ | **fixed** — all four now read the `Interrupt` intents (§4) |
| ~~`TestDrafts::test_a_reused_message_becomes_that_sessions_draft`~~ | ~~the draft belongs to the session it came from~~ | **fixed** — the implementation is now `hand_back`, covered by `test_ui_queue.py` |
| `test_core_service.py::TestTurns::test_the_prompt_carries_the_sessions_own_profile` | the prompt is rendered for the running session's profile | `isinstance(content, str) and content`. See §3.14 row 2 |
| `test_ui_turn.py::TestLiveStepRows::test_the_box_survives_reopening_the_session` | the box survives a reopen | feeds a `ChatReset` and greps "1 steps"; nothing is reopened |
| `test_ui_overlays.py::TestConfigEditor::test_invalid_values_show_an_error_and_keep_it_open` | invalid *values* keep the editor open | stubs `overlay._validate`. The shipped `client._settings_error` checks JSON syntax only |
| `test_ui_overlays.py::TestProfileLifecycle::test_an_empty_name_creates_nothing` | "and says so, on screen" | `isinstance(ui.overlay, PromptOverlay)`; the refusal text is never read |
| `test_ui_app.py::test_enter_elsewhere_in_the_log_goes_to_the_box`, `test_ui_client.py::test_a_row_that_is_not_a_message_cannot_be_rewound` | an assistant reply and a background event are not copyable | both aim at a `thinking` row. No `assistant` or `event` row is ever aimed at |
| the narrow-terminal sweep | columns survive a narrow terminal | `SIZES` still bottoms out at 40×10; the deleted suite swept to 1 column, and nothing asserts the four rows are still *present* at a narrow width |
| `test_ui_sessions.py::TestDrafts::test_a_parked_slash_draft_does_not_bring_its_menu_back` | that the menu does *not* come back | the comment is now correct and explains it asserts only the draft half — but the **name still says the opposite** of what `test_ui_commands.py::TestAParkedDraft` proves. Rename it |
| `test_ui_screen.py::TestTerminalModes::test_every_mode_set_is_reset` | the terminal modes are right | symmetry between enter and exit, not the absence of mouse tracking (§3.14 row 18) |

---

## 6. Stale prose

The first pass found nine. **Six are fixed**, and the fixes are good ones —
`overlays/llm.py` now explains the scan it runs, `overlays/backendform.py`
explains enter-probes/ctrl+s-saves, `state.py`'s `Context` explains where speed
and effort come from and no longer points at a `tests/test_ui_meter.py` that
never existed, `ui/__init__.py` is current and now points here,
`overlays/inspect.py` names its two real callers, and both contradicting test
comments were corrected in place with a note saying why.

Eight stand or are new. The recurrence is the point: this is the third pass in
which prose went stale *inside the milestone that made it stale*, and it is how
§3.1 stayed hidden for a milestone the first time.

| file | says | truth |
|---|---|---|
| `ui/__init__.py:42-45` | "M7 and M8 added the management overlays and the slash commands, **and M9 the scan, the probe and the read-only window**" | M9 is the *deletion* milestone and has not run; `tests/test_ui_overlays.py:970` dates the same work M8a. The package's front door now mis-dates the milestone about to happen — and the next line points at this document's §3 as "what an audit found still unwired", most of which is now wired |
| `specs-ui-replacement.md:438-449` | "**M9 is blocked until the list in specs-ui-coverage.md §3 is closed**", naming four blockers | all four are wired. This is the paragraph a planner reads to decide whether to start M9, and it now says no when the answer is yes |
| `boot.py:17-22` | "`tests/test_ui_boot.py` asserts them of this one" | true of the shutdown order and the wait message, overstated for the notices, the logging pair and the sync interval (§3.14 rows 4, 19, 20) — and it cites `tests/test_tui_dbcache.py`, which will not exist after M9 |
| `ui/app.py:2237` (`_profile_rows`) | the default-leads order "puts the profile the user is working in at the top of the list they are about to pick from" | only when the working profile *is* `default`. Under `hpc`, `default` still leads and the cursor still starts on it — the §7 finding, now with a docstring asserting the opposite |
| `overlays/llm.py:315` | "the core does restate the catalog after a remove" | true on the success path only (§9.1) |
| `ui/app.py:1810-1813` | the three screen commands are handled ahead of the busy check because they "draw a screen and **ask the core for nothing**" | `/skill-creator <request>` now sends `DraftSkill` from inside `_screen_command`. The comment is the justification for the bug in §9.7 |
| `core/memory_service.py:769-779` (`skill_body`) | own skills only, "matching what `save_skill_file` writes and `delete_profile_skill` removes" | `63d44c4` widened both — `save_skill_file` takes a `level=`, `delete_profile_skill` consults `load_project_skills`. `skill_body` is the one that stayed narrow, which is the seam §9.9 falls through |
| `overlays/skills.py:1-11`, `core/service.py:2318` (`_remove_skill`), `overlays/skillnew.py:58`, `ui/app.py:1868` | four copies of "this profile's **own** skills", and "the same rule the core enforces on the way in" | all four paths now include project skills, and the core no longer enforces the rule for delete. One sentence stopped being true in one commit and is still written down in four places |

---

## 7. What the acceptance list itself misses

Unchanged from the first pass except where noted. The list was extracted from the
old *tests*, so it can only contain what those tests happened to cover.

**Closed by the fixes**: `ctrl+s` on the backend form (save without probing),
choosing among several models served by one endpoint, the discovered panel
skipping what is already configured, the rescan key (now `s`, was `f5`), and the
startup screen's don't-land-on-an-open-screen guard. All five are now implemented
*and* tested — see §3.1.

**Still lost, and still claimed nowhere**: the prompt-injection warning (§3.5);
`matches: <keywords>` on a reflection proposal (`tui/memory_screens.py:166-171`)
— the user approves a *trigger*, not just a text, and `_wire()` keeps them only
incidentally for struggle notes; the scan progress line (`scanning localhost…
12,000/65,535 ports`), which is what makes a long sweep read as progress rather
than a hang; and the profile picker starting on the profile you are using
(§6, row 2).

**Carried over but unclaimed** — surviving only because someone hand-ported them,
so nothing would flag their loss: the "I have struggled with this before" toast
(now wired, per `02feddb`); a deleted session being forgotten from the episodic
index and dropping its watches; job state-change toasts and the terminal-state
event that wakes the agent with `triage_job`; the "remember this failure as a
signature?" offer after a tier-3 diagnosis; `reconcile_orphans` at startup;
"database sync is failing" said once rather than every tick; and writing the
discovered context window back into the catalog entry.

**New behaviour the port built that the list never asked for**, and which will be
lost the next time the list is used as the spec: `decision.cleared`; "the resumed
turn is still stoppable" and "a parked decision survives a core restart"; a fork
being deliberately ungated where a rollback is not; the cut carried by the core's
`Entry.index`; `Pollers._last_rows` suppressing an unchanged panel with `force=`
for a fresh client; and `_apply_settings` naming *which* setting waits for a
restart.

---

## 8. The M9 checklist

Done since the first pass: `tests/test_context_bar.py` repointed at
`hpca.ui.meter` with both orphan assertions kept; `tests/test_editor.py` added;
`tests/conftest.py::no_startup_backend_modal` extended to switch off
`AgentService.startup_backend_check` as well as the Textual one.

Still to do, all mechanical:

1. `src/hpca/__main__.py` — drop the `--new-ui` branch and the `HpcaApp` import;
   make the row UI the default. (The argparse-before-import rule stays; it is
   what `--serve` will need at M11.)
2. `pyproject.toml` — drop `textual>=0.80`.
3. `tests/conftest.py:85-87` — remove the two `HpcaApp` lines.
4. Delete `evals/bench_textual.py` in the same commit
   ([specs-ui-baseline.md](specs-ui-baseline.md) §1 already says so; the numbers
   it produced outlive it).
5. Update `README.md`, `project.md`, `docs/llm-key-registry.md` (which points at
   `tests/test_tui_llm_mgmt.py`) and `docs/concurrent-session-turns.md`, all of
   which still describe Textual workers and the `Pilot`.
6. Rename `test_ui_sessions.py::TestDrafts::test_a_parked_slash_draft_does_not_bring_its_menu_back`.
   This one **regressed rather than closed**: the comment was rewritten to say
   the menu *does* come back, so the file now carries a test whose name asserts
   the opposite of its own comment.
7. `src/hpca/looplag.py` — either wire the probe into `ui/run.py`'s loop or
   delete the module and `tests/test_looplag.py` with it. Leaving it is choosing
   the third option, which is a module that exists and cannot run.
8. Correct the two pieces of prose that now say M9 is blocked or already done:
   `specs-ui-replacement.md:438-449` (the four blockers it names are wired) and
   `src/hpca/ui/__init__.py:42-47` (which credits M9 with M8a's scan and probe,
   and points at a §3 that has mostly closed).

Worth doing soon after, in rough order of what a user would notice: the two
medium-high bugs the fixes introduced — the busy guard on `/skill-creator
<request>` (§9.7) and a guard on the landing draft so it cannot discard an open
editor (§9.8); `watch.move` and `shift+↑/↓` plus moving the freshness line into
`head` (§3.11); `Turn.interruptible` excluding a parked turn (§4); a `flagged`
field on `protocol.Proposal` and a batch sent as one question (§3.5); one test
for the updated marker and a `replied()` call in `client._failed` (§3.4); an
`invalidate` on `session.open` (§3.8); the system-prompt join test (§3.14 row 2);
and the `(new profile)` row in the new-session picker (§3.13).

---

## 9. What the fixes broke, or left behind

Fixes create bugs, and `e11b414` in particular added several "say it before the
core answers" paths. These are small — none is a blocker — but they are the kind
of thing this document exists to catch early.

**9.1 A refused `backend.remove` leaves the row off the screen.**
`overlays/llm.py:318` removes the row optimistically, with a comment saying "the
core does restate the catalog after a remove". `service._remove_backend`
(service.py:1376-1386) restates it only when the entry was found; the not-found
branch emits a warning and returns. So the row vanishes and the toast says
"There is no configured backend called X" about a row that is no longer there.
Narrow — the label comes from the row the UI drew — but the fix is one
`self._emit_catalog()` in the `None` branch. *Medium.*

**9.2 Adding a backend by hand now re-points every unpinned session at it.**
`a` → `BackendFormOverlay` → `SetBackend(backend)` with no session →
`BackendSet` → `BackendRegistry.set_default`, which calls
`settings.activate_backend`, saves, and rebuilds the client. The old TUI's
`manage_llms._save_backend` (`tui/manage_llms.py:534-539`) appended to
`settings.backends` and saved — it did **not** activate. So "add it to the
catalog" now silently means "and make it everyone's default", and the acceptance
list's "Enter on the configured panel does not set a global default" is honoured
for `enter` and quietly violated for `a`. The form's own comment
(`backendform.py:317`) calls it "the global half of `backend.set`, which is what
'add it to the catalog' means", which is the understatement that hides it.
*Medium — it changes which model answers, without saying so.*

**9.3 The updated marker has no test, and no failure path.** See §3.4. *Medium.*

**9.4 Two hand-kept lists of the seven built-in commands.**
`core/service.py:214`'s `SLASH_COMMANDS` and `ui/commands.py`'s `BUILTINS` name
the same seven, and `grep -rn SLASH_COMMANDS tests/` is empty. Add one to either
side and the other silently answers "Unknown command". A single test asserting
the two sets are equal costs three lines. *Low now, high the day someone adds
the eighth command.*

**9.5 `/compact` marks the meter superseded even when the core refuses it —
CLOSED.** It used to call `superseded()` on send, so a session parked on an
approval, an empty one and a backend failure each redrew an accurate measured
number with a `~`. The compaction review moved the call to the moment the fold
lands (`RowUI._closed`, accept), which is the only moment the number stops
describing the prompt — and the three refusals never reach it, nor does a
summary the user discards.

**9.6 `toasts.CLIPPED` still says "… more in the log".** After §3.7 the
remainder of a *titled* block is on screen in a window, not in a log — and for
an untitled multi-line notify it is still nowhere. Every multi-line `Notify` the
core sends today carries a title, so this is currently unreachable; it is one
string away from being right. *Low.*

**9.7 `/skill-creator <request>` can be typed mid-turn, and steps on the running
turn's spinner.** It is a `SCREEN_COMMAND`, so it is handled *ahead* of the
"wait for this turn — a command cannot be queued" guard, on the stated grounds
that the three screen commands ask the core for nothing. That stopped being true
when `63d44c4` made this one send `DraftSkill`. `service._draft_skill` then calls
`_working(session_id, "drafting a skill")` and, in its `finally`,
`_working(session_id, "")` — on a session that may have a live turn. Because
`Turn.activity_is` takes a fresh `started_at` whenever the activity changes, the
running turn's step label is replaced by "drafting a skill", then blanked to the
default text, and **its elapsed clock restarts twice** — while
`specs-ui-acceptance.md` is explicit that "a new step does not restart the
clock" and that the elapsed count answers "how long since I sent it". Neither
side checks `is_busy`. *Medium-high, and the cheapest fix is to move
`/skill-creator` behind the busy check when it carries an argument.*

**9.8 A landing draft discards whatever screen is open, unsaved edits and all.**
`RowUI.skill_drafted` does `self.overlay = self._skill_form(...)`, and the
`overlay` setter starts a fresh stack (`self.overlays = []`, app.py:325) without
calling `child_closed`/`_closed` on what it drops. A draft is a real model
generation of several seconds, during which the user is free to press `c` for
the config editor, open a profile's memories, or start a backend form. When the
draft lands, that screen and its unsaved text are gone. This is the exact hazard
`RowUI.window`/`_waiting_window` was built for two hundred lines above, and that
`boot._open_llm_screen` guards with `if ui.overlay is not None: return`; the one
new asynchronous screen-opener got no guard. *Medium-high.*

**9.9 `removable` is being used to mean "own", and they are different
predicates.** `RowUI._remember_skill` adds a freshly-saved skill to the
profile's own list when `fresh.removable`, which is `level in ("profile",
"project")`. But `ProfileInfo.skills` is the *own*-scoped list `SkillsOverlay`
draws and opens through `skill.get`, and `memory_service.skill_body` still
answers own-only. So a skill created at the **project** level appears on the
profile's own-skills screen, where Enter answers *"Profile X has no skill Y of
its own."* until the screen's next `skill.list` corrects it. *Medium.*

**9.10 Two protocol fields added and never read.** `SkillDrafted.request` is
documented as "echoed so a front-end can tell which request this answers and put
it in front of the user again if the draft is not what they meant"; it travels
core → `client._skill_drafted` → `RowUI.skill_drafted` → `_skill_form` →
`SkillCreatorOverlay.request`, where it is assigned (skillnew.py:118) and never
read. `SkillDrafted.profile` is dropped outright at `client.py:975`, so a draft
requested under `hpc` that lands after a profile switch is saved into whichever
profile is current. *Medium — the second half is a real, if narrow, misfile.*

**9.11 The frequency sort systematically under-counts three of the seven
built-ins.** `service.py:2175` calls `command.run` "the one place the menu's
sort can learn it ran", but `/thinking` always, and `/skill-creator` and
`/skill-remove` when bare, are answered by `_screen_command` and never reach the
core. The ordering `63d44c4` shipped will never promote them. *Low, and
arguably correct — but it is not what the docstring says.*

**9.12 A one-paragraph warning is still clipped, and it is the most
consequential one in the app.** `RowUI.toast` opens the window on
`len(text.splitlines()) > MAX_BODY` — *logical* lines. `XHIGH_WARNING` is a
single 600-character logical line, so it is still cut to three wrapped lines
plus "… more in the log". `test_a_one_liner_with_a_heading_is_still_a_toast`
blesses this deliberately and names the xhigh warning as the case, so it is a
judgement rather than an oversight; the judgement is worth revisiting, because
the sentence that gets cut is "Use low or medium". *Medium.*

**9.13 `window`'s stated rule is not the rule it implements.** The docstring
says "a modal must not land on top of someone mid-typing", but
`_open_waiting_window` defers only when an `Overlay` is on the stack. With no
overlay open — the user typing in the message box — the `InspectOverlay` lands
and takes the keyboard. `/compact` is asynchronous and can take a minute.
*Low-medium.*

**9.14 The `*` can be drawn where `⟳` belongs.** `_marks` computes the working
mark as `session.turn.busy or "working" in flags`, but the `*` overwrite excludes
only `session.turn.busy`. A background session the core's poll flags as
`working` with no local `turn.started` — a reconnect, or a second front-end —
and `updated` set will draw `*`, which the adjacent comment says cannot happen.
*Low.*

**9.15 `probed` adopts only the top screen.** `RowUI.probed` ends with
`self._adopt(self.overlays[-1])`; if a second probe answers while the model
picker is up, the form beneath it sets a `child` that is never pushed. *Low.*
