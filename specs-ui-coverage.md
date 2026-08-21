# What the port covers, and what it does not

Companion to [specs-ui-acceptance.md](specs-ui-acceptance.md) and
[specs-ui-replacement.md](specs-ui-replacement.md) §7. Written immediately
before M9 — the milestone that deletes `src/hpca/tui/` and its 37
`tests/test_tui*.py` files — because after it there is no reference
implementation left to compare against.

**Baseline.** Commit `599147e` ("M8b: slash commands, toasts, clipboard,
`$EDITOR`, quit"), with a dirty working tree in which `protocol.py`,
`core/service.py`, `core/memory_service.py` and their tests are being edited to
add `skill.draft` / `skill.drafted` and `command.list` / `command.counts`.
Those two changes, once they reach the UI, close three of the gaps below
(§3.9, §3.10); everything else in this document is true of the tree as
committed, and nothing in flight touches it.

**Method.** Every one of the 377 bullets in the acceptance list was read
against the tests that survive the deletion — that is, everything in `tests/`
except `tests/test_tui*.py` — and against the implementation in
`src/hpca/ui/`, `src/hpca/core/` and `src/hpca/protocol.py`. Test bodies were
opened rather than test names trusted; where a name promised more than the
assertion delivered it is recorded as **weak** in §5. Where a module docstring
promised three things, all three were checked; that habit is what found §3.2,
§3.4 and §3.7.

---

## 1. The verdict

**Deleting `src/hpca/tui/` is safe as a *test* proposition and unsafe as a
*product* proposition, and the two must not be conflated.**

The tests are in good shape. 571 Textual-coupled tests across 12,504 lines are
replaced by 601 tests in `tests/test_ui_*.py` and 678 in the core suite, and
they are better tests: faster, hermetic, and asserting against a protocol
rather than through a widget tree. Roughly **297 of 377 acceptance bullets are
asserted by something that survives** — and where the port changed a
behaviour deliberately (the queue moving into `TurnScheduler`, entry
addressing replacing whole-transcript rebuilds, `esc esc` replacing the
interrupt modal), what replaced it is usually covered *better* than what it
replaced. The performance claim that justified the whole exercise is already
banked in [specs-ui-baseline.md](specs-ui-baseline.md), measured on both
sides, so `evals/bench_textual.py` can die with `tui/` without losing the
number.

What is not safe is the **product**. `hpca` still defaults to Textual
(`__main__.py`: "Textual is still the default — the row UI is opt-in behind
`--new-ui` until M9"), and M9 flips that switch. On the day it does:

- **A cluster user cannot get connected.** `BackendRegistry.auto_connect()` is
  fully implemented and has ten passing tests, and **has no caller anywhere
  outside its own module**. Nothing probes the active backend at startup,
  nothing opens manage-LLMs when nothing answers, and **no module under
  `src/hpca/ui/` ever sends `backend.scan`, `backend.probe` or
  `backend.remove`** — all three of which the core answers and tests
  thoroughly. The discovered panel is permanently empty and says so. The only
  way into the catalog is typing a URL into the manual form, which also
  silently re-stars the global default.
- **The `memory` tool is dead.** `ctx.queue_memory_edits` is set only in
  `tui/app.py:3506`; `core/service.py:_make_tool_ctx` never sets it. The tool
  is still registered and the prompt still tells the model to use it, so every
  call returns *"Memory flagging is not available in this context."*
- **Two startup housekeeping jobs lose their only caller**: the trash is never
  swept (`trash.cleanup` — one caller, `tui/app.py:1554`) and the curator never
  runs (`MemoryService.run_curator_if_due` — no caller in `src/hpca/core/` or
  `src/hpca/ui/`). Both grow the NFS home this project goes out of its way to
  keep small, and both have a settings key that will silently do nothing.
- **A safety warning is dropped on the wire.** The prompt-injection flag on a
  flagged memory batch (`⚠ text matches a prompt-injection pattern`) is
  computed by `memory_ops`, carried on `ProposalSet.flagged`, and then dropped
  by `_wire()` because `protocol.Proposal` is three strings. A memory harvested
  out of injected tool output is now approved with no warning.

None of those four is a testing gap. They are wiring that was never done, and
they are invisible to the suite because each half is green on its own.

**What is lost that nothing else records.** Two things, and only two:

1. **`tests/test_context_bar.py` does not survive the deletion.** It imports
   `hpca.tui.context_bar`, and §7's claim that `render_bar`/`severity` are
   "pure and untouched" is not what happened — they were *copied* into
   `src/hpca/ui/meter.py`. At M9 that file is a collection error, not a passing
   test. Most of its assertions are duplicated by
   `test_ui_turn.py::TestTheContextMeterState`; two are not
   (`test_overflow_does_not_exceed_the_bar`,
   `test_small_window_reaches_danger_quickly`). Repointing the import is a
   one-line M9 chore, but it must be a deliberate one.
2. **`tests/test_tui_memory.py::TestResolveEditor`** is the only assertion
   anywhere of the `settings → $VISUAL → $EDITOR → nano` order.
   `test_ui_edges.py:396` says the order "is its own test
   (`tests/test_memory_ops.py`)" — **that test does not exist**. Move it before
   the delete; it is three lines.

Everything else that goes is either replaced, or was never asserted in the
first place.

**Recommendation.** M9 as a *test* deletion — remove `tui/`, drop `textual`,
delete the 37 files — is ready once those two files are dealt with and
`tests/conftest.py`'s `no_startup_backend_modal` (which imports
`hpca.tui.app`) and `evals/bench_textual.py` are updated. M9 as a *front-end
swap* should wait for §3.1 and §3.2 below, which are three afternoons of
wiring, not a milestone.

---

## 2. The counts

Counted at the **bullet** level (377 bullets in the acceptance list), each
bullet classified by its weakest constituent claim: a bullet holding four
claims of which one is unimplemented counts as unimplemented. A handful of
bullets are judgement calls, so read these as ±5.

| bucket | bullets | share |
|---|---:|---:|
| covered (incl. covered indirectly) | 299 | 79% |
| not covered — behaviour exists, nothing asserts it | 28 | 7% |
| not implemented — behaviour absent from the new stack | 38 | 10% |
| dropped deliberately, with a recorded reason | 11 | 3% |
| degraded — present but materially worse | 1 | <1% |

Per section, gaps only (sections not listed are fully covered):

| section | bullets | not covered | not impl. | dropped |
|---|---:|---:|---:|---:|
| Queueing while a turn runs | 16 | — | — | — |
| Stopping a turn | 21 | — | — | — |
| Per-session concurrency | 10 | — | — | — |
| Replies to a session you left | 8 | 1 | 2 | 1 |
| Rewind and fork | 15 | 1 | — | — |
| Core chat wiring | 21 | 1 | 2 | 1 |
| Inline approval prompts | 12 | — | — | — |
| Declining with a reason | 10 | — | — | — |
| Agent modes | 8 | 1 | — | — |
| Thinking box, tool rows, logging | 22 | 1 | — | — |
| The spinner | 12 | — | — | — |
| Titles, rename, delete | 20 | 2 | — | — |
| Profiles | 21 | 1 | 2 | — |
| **Backends** | **30** | **1** | **14** | **1** |
| Memory | 25 | 3 | 2 | — |
| **Skills and self-review** | **33** | **8** | **8** | — |
| Compaction | 9 | — | 1 | — (1 degraded) |
| Node-local databases | 12 | 5 | — | — |
| Drafts | 7 | — | — | — |
| Slash-command menu | 8 | — | 2 | — |
| Navigating the entry | 14 | — | 1 | 2 |
| Selecting text in the chat | 7 | — | — | 7 |
| Thinking effort | 5 | — | — | — |
| Watchers column | 11 | 2 | 1 | — |
| Background processes and jobs | 7 | 1 | — | — |
| Lag instrumentation | 3 | — | 3 | — |
| Context meter | 7 | — | — | — |
| Carried over from Textual mechanics | 3 | — | — | — |

Two sections carry 22 of the 38 unimplemented bullets. **Backends** is one
missing wire (the UI never sends the three discovery commands the core
answers); **Skills** is two (no draft channel, no level field). Neither is a
distributed problem.

The best-covered sections are the ones the port rebuilt from the protocol up:
queueing, stopping a turn, approvals, the spinner and the thinking effort dial
have no gaps at all, and the queue/interrupt behaviour that moved from UI state
into `TurnScheduler` is materially better asserted than it was.

---

## 3. Everything not covered or not implemented, ranked

Ranked by what a user would actually notice. §3.1–§3.8 are things a user hits;
§3.9 onward are progressively quieter.

### 3.1 A cluster user cannot connect, and is not told — **not implemented**

*Backends, 14 bullets; §4.3 item 40.*

Three separate holes with one cause: the UI never sends the commands.

| what is missing | where the core already does it |
|---|---|
| the port scan on `m`, the discovered panel, the incremental fill, responsiveness during a slow scan | `test_core_service.py::TestBackendDiscovery::test_a_scan_fills_the_catalog_as_it_goes_and_then_closes`; `test_core_backends.py::TestScanning::*` |
| cluster endpoints from the manifest | `test_core_backends.py::TestScanning::test_a_cluster_endpoint_reaches_the_catalog_the_sweep_cannot_see` |
| the tunnel recipe and its four conditions | `test_core_backends.py::TestWhatAnEmptyScanMeant::*` |
| `r` to remove a configured backend | `test_core_service.py::TestBackendDiscovery::test_a_configured_entry_is_removed_by_its_label` |
| the form's probe, model autofill, multi-model picker, rejected-key warning | `test_core_backends.py::TestProbingAnEndpoint::*` |

`grep -r 'BackendScan\|BackendProbe\|BackendRemove\|BackendScanned\|BackendProbed' src/hpca/ui/` returns nothing, and neither
`state.py` nor `client.py` has an intent or a handler for any of them.

Separately and worse: **`BackendRegistry.auto_connect()` has no caller.** On a
cluster, `hpca --new-ui` will not wire up the live vLLM servers that
`specs-auto-connect.md` exists for, and the startup check shipped in 0.23.0
(`tui/app.py:_ensure_backend_connected`, `NO_BACKEND_MESSAGE`) has no
counterpart, so nothing says why. The first sign of trouble is a failed turn.

The gap is **hidden by its own docstrings**, which is why it deserves the top
slot. `ui/overlays/llm.py:15-21` says "Discovery is not implemented anywhere…
there is no rescan key… `r` is absent rather than present and inert", and
`ui/overlays/backendform.py` says "the probe is not here… there is no command
on the wire for it yet". Both were true when written and both were made false
by `efb9b3f` / `805b172`. A reader checking whether the gap is tracked will
conclude it is settled.

Three key-registry behaviours are absent outright rather than merely
unreachable: no dedup of a discovered row once it is configured (the endpoint
appears twice, under two labels), no save-time sentinel guard, no re-probe once
a key joins the pool.

### 3.2 The `memory` tool is dead — **not implemented**

*Memory, 1 bullet, silently invalidating 3 more.*

`agent/memory_tools.py:147` returns *"Memory flagging is not available in this
context."* when `ctx.queue_memory_edits is None`. It is set in
`tui/app.py:3506` and nowhere else; `core/service.py:_make_tool_ctx` does not
set it. The tool is still registered (`service.py:2467`) and `MEMORY_GUIDANCE`
still instructs the model to use it, so the model will reach for it every
session and be refused every time.

The three bullets that look covered — the flagged batch offered at
`/conclude`, a full scope rejecting it, a batch that frees room and adds in one
call — are covered at the *service* level by tests that prime the queue with
`service._memory.queue_edits(...)` in Python. No test can reach that method
from a turn, so none of them can catch this. Neither is the "not available"
branch asserted anywhere.

### 3.3 Two startup jobs lose their only caller — **not implemented**

*Not on the acceptance list at all (trash); half a bullet (curator).*

- `TrashManager.cleanup(settings.safety.trash_ttl_days)` — one caller in the
  repo, `tui/app.py:1554`. `core/service.py:2933` builds the `TrashManager` for
  the tool context and never sweeps it. Every file the agent edits leaves a
  backup under `<app_dir>/trash`, and `<app_dir>` is the NFS home.
  `config.py:133` still defines `trash_ttl_days: int = 7`.
- `MemoryService.run_curator_if_due()` — implemented, five passing tests in
  `test_core_memory_service.py::TestCurator`, and **no caller** outside
  `tui/app.py:1567`. RAG entries are never archived. The acceptance bullet
  ("The curator runs at startup at most once every few days…") reads as
  covered because the tests call the method directly.

Both are the same failure mode: a well-tested unit whose only invocation lives
in the module being deleted.

### 3.4 The "updated" sidebar marker is gone — **not implemented**

*Replies to a session you left, 2 bullets; §4.3 item 15.*

`core/service.py:_flags` emits exactly two flags, `decision` and `working`;
`ui/app.py:_marks` draws exactly `!` and `⟳`. The old TUI carried a third
state, `session-updated` — `tui/app.py:1382`, *"replies that landed while
switched away"* — and §4.3 item 15 asks for it by name ("the
updated/pending/working row colours"). M6 shipped without it and without
recording the omission.

The consequence: a background turn's row shows `⟳` while it runs and goes
completely blank the instant it finishes — the exact moment there is something
new to read. Running two conversations, there is now no way to tell which one
answered.

The row *colours* went the same way, and that at least is defensible: the new
sidebar uses two glyphs and reserves colour for "this row is open". It is
undocumented, though — no commit or spec section records the swap.

### 3.5 The prompt-injection warning never reaches the screen — **not implemented**

*Not on the acceptance list.*

`memory_screens.py:112-121` draws `⚠ text matches a prompt-injection pattern
(…) — read it closely` above the y/n on a flagged batch. `memory_ops.py:68`
computes the flags and `core/memory_service.py:459` still carries them on
`ProposalSet.flagged` — but `protocol.Proposal` is three strings and `_wire()`
drops the field, so `overlays/memory.py` cannot draw it. This is the one screen
where the warning exists precisely because the user is about to persist
attacker-influenced text.

Related and also invisible: `_emit_proposals` sends a flagged batch as N
separate `Proposal`s, so the review walks them one at a time asking y/n each —
while `_apply_flagged` requires `all(flags)`. Answer `y y n` to a three-op
batch and all three are lost, explained only by a "Discarded the flagged memory
changes." toast. `overlays/memory.py`'s own docstring claims a batch is "one
question when the core sends it as one", which is not what the core sends.

### 3.6 The context meter lies after `/compact` — **not implemented**

*Compaction, 1 bullet.*

`Context.estimate` returns early `if self.measured` (`state.py:407`), and
`measured` is cleared only by `Context.reset()`, which runs on `chat.reset` —
which `/compact` deliberately never sends. The core does its half correctly
(`forget_session` then `estimate_context`, asserted by
`TestCompact::test_the_chat_is_not_re_stated`). So a user compacts a session at
92% and the bar stays at 92%, unmarked, indefinitely.

Both halves are individually green. The seam is not tested.

### 3.7 Long answers are clipped to three lines, and the window that would show them has no caller — **degraded**

*Compaction, 1 bullet ("the chat shows what was kept — the summary itself").*

`/compact` writes no chat row; the summary rides a `Notify`, and
`toasts.MAX_BODY = 3` cuts it with `… more in the log`. The same clipping hits
`/skills-list`, `/skill-remove`'s listing and `/thinking`'s level list, all of
which are now multi-line `Notify`s. `/skills-list` prints two lines per skill,
so a profile with two skills is cut mid-list.

And the pointer is wrong twice over: the session log records turns, not command
answers, and **`RowUI.inspect` (`app.py:1965`) has no caller** — the read-only
window of §4.3 item 31 is built, tested (`TestInspect`) and unreachable. Its
own docstring names two callers, `/skills-list` and the tunnel recipe, and
neither exists.

### 3.8 Memories are never re-read at a session boundary — **not implemented**

*Core chat wiring, half a bullet.*

`tui/app.py:3552-3556` refreshed the snapshot on every session open, with the
reason written down: *"A session boundary is a deliberate refresh point…
memories written by another session or instance are picked up here, while
WITHIN a session the frozen snapshot keeps the prompt prefix byte-stable."*
The core invalidates only on its own writes, so a note added by a second HPCA
instance, or edited on disk, is invisible for the life of the process. The
byte-stability half survives and is well tested
(`TestSnapshotFreeze::test_a_write_on_disk_is_not_picked_up_silently`); the
pick-up half is gone.

### 3.9 The skill drafter and skill levels are gone — **not implemented**

*Skills, 8 bullets.* Honestly documented in-source
(`ui/overlays/skillnew.py:13-17, 79-86`) and the shortfall is *asserted*
(`test_an_argument_is_not_drafted_and_the_form_says_so`), which is the right
way to leave a gap — but it is not in §4.3's "Dropped deliberately" list, which
names only Textual's command palette.

- `/skill-creator <what it should do>` opens a bare form. `agent/skill_drafter.py`
  and its full test file survive with no caller.
- Neither screen can offer the level; `SkillSave` carries a profile and nothing
  else, so global and project skills are unreachable from the UI.
- **`/plan` reports itself unknown on a fresh install**: `skill.list` answers
  `own` only, so no shipped skill is callable by name.

*In flight:* the working tree adds `skill.draft` / `skill.drafted` and a
`SkillList.scope`. Neither has a UI caller yet.

### 3.10 The slash menu is in definition order — **not implemented**

*Slash-command menu, 2 bullets; §4.3 item 25 required most-used-first.*

The core counts every `command.run`; nothing served the counts back, and rule 2
of §4.2 forbids the UI reading `command_usage`. `commands.py`'s docstring says
so plainly and `TestFrequencyOrdering` asserts definition order, that
`hpca.db` is not importable from `hpca.ui.app`, and that the sort works when
handed counts. This is the model for how to leave a gap.

*In flight:* `command.list` / `command.counts` in the working tree.

### 3.11 The watchers column's arrangement does not stick — **not implemented**

*Watchers, 1 bullet (5 sub-claims).*

`alt+↑/↓` swap two `Item`s in the client's `Pane` and tell the core nothing:
there is no `watch.move` command, and `WatchStore.move` is now reachable only
from `test_watches.py`. Because a log box's age text changes every tick,
`panel.update` re-sends the store's order within seconds and puts the box back.
Hold `alt+↓` and the box walks down and then jumps back.

`shift+↑/↓` — the multiplexer fallback the old TUI bound (`app.py:1076-1077`) —
decode correctly in `keys.py` and match no branch, so under tmux with alt
swallowed there is no reorder key at all.

Related, and the reason the box exists: `client._panel_item` puts
`watch_lines()[0]` in `head` and `[1:]` in `body`, and `Pane` draws a body only
when the row is expanded. **"How long ago the log was last written" is
invisible until the user presses `→`.**

### 3.12 The lag probe is unwired — **not implemented**

*Lag instrumentation, all 3 bullets.* `HPCA_LOOPLAG`, the label hook and the
exit report live only in `tui/app.py:1467-1481`. `looplag.py` and
`test_looplag.py` stay green with no caller. §8 plans to compare both sides at
M10; the instrumentation for one of them is about to be deleted.

### 3.13 Smaller unimplemented items

- **`↑` no longer leaves the message box** (dropped, §5 key map + `55310bc`) and
  neither do `←/→` (dropped). Defensible — one escape cannot be both — but it is
  the biggest muscle-memory change in the port, and nothing documents the
  *removal*, only the replacement.
- **`↑` in an empty log**: `ctrl-up` sets `focus = CHAT` unconditionally; the
  old guard is gone. Harmless.
- **The chat entry now exists outside a session.** The message row is always
  drawn and typeable; only the send is refused.
- **"Nothing to save is silent"** is no longer true: the core emits *"Nothing
  durable to keep from this conversation."* and *"Nothing kept."* Neither string
  is asserted; either the claim or the code should move.
- **Creating a profile from inside the new-session picker** is gone. No recorded
  reason.
- **"Opening a session switches to its profile"** is replaced by per-session
  profile resolution — a better design, but nothing asserts a turn running
  under session-profile X while the working profile is Y.

### 3.14 Not covered — the behaviour exists, nothing asserts it

Ranked by what a regression would cost. All are one short test each.

| # | claim | where the behaviour lives |
|---|---|---|
| 1 | `d` deletes a session **and keeps the log file** | `service.py:_delete_session` (docstring says so; nothing checks). The doomed `test_tui_session_titles.py:373` asserted `log.exists()` after the delete |
| 2 | profile memories reach a real system prompt; the chosen profile's memories reach it | `service.py:system_prompt_for` — three layers tested apart, the join not at all |
| 3 | the editor resolution order `$VISUAL` → `$EDITOR` → nano | `hpca/editor.py:9`; only `test_tui_memory.py::TestResolveEditor` asserts it |
| 4 | the dbcache **notices** reaching the user — declined cache, second instance, corrupt home copy | `boot._open_db_cache` → `Core.notices` → `Notify("Databases: …")`. No test builds a `Core` with a notice. Three acceptance claims lean on it |
| 5 | a new session starts in the **configured default mode** | `service.py:655`. A user who set `manual` would get sessions that run scripts unasked, silently |
| 6 | `ctx.llm` routes to the session's backend | `_make_tool_ctx`; a regression runs every `ask_docs` sub-loop on the bootstrap backend |
| 7 | job tools registered iff Slurm is available | `service.py:2461`; the only test was `test_tui_jobs.py` |
| 8 | an **approved** script stays readable in the chat after the prompt closes | the refused case has a core test; the approved one has none on either side |
| 9 | the second Ctrl+C during the exit sync aborts | correct by construction (`KeyboardInterrupt` is not an `Exception`); widening `suppress(Exception)` would make a hung NFS mount unkillable |
| 10 | the watchers cursor: first paint on the top box; holding position when a box is dropped; clamping on the last; `d` leaving the log file alone | `Pane.replace` / `Pane.current` |
| 11 | `MEMORY_GUIDANCE`, `SKILLS_GUIDANCE`, and "skill names never appear in the prompt" | `agent/prompts.py:279-280`; `test_prompts.py` tests one conditional block |
| 12 | the `/<skill>` **core** path: command word stripped, directive on the sidecar, stored transcript clean | `service.py:2586-2610`; nothing passes `skill_directive=` or a `forced_skill` in any surviving test |
| 13 | a turn's tool context is per session (its own runner and session id) | `_make_tool_ctx`; only the log third is asserted. Two concurrent turns sharing a runner would corrupt the `processes` table |
| 14 | copying a profile inherits its **skills** through the service | `duplicate_profile`'s docstring promises memories *and* skills; only memories asserted |
| 15 | `skill_new` from a reflection: creation, duplicate refusal, first-skill tool registration | `_apply_skill_reflection`'s `else` branch — the docstring promises three things and one is tested |
| 16 | the per-session LLM provider branch `build_service` actually wires (`llm=lambda sid: …`) | `test_graph.py::TestPerSessionLLM` uses the zero-arg form; the one-arg branch of `graph._resolve` is never driven |
| 17 | "reusing a message twice stacks both" | `_into_draft`; only one reuse is exercised |
| 18 | the mouse stays released | every "Selecting text in the chat" claim is satisfied by *not interfering*, and nothing asserts the absence. One line — `"[?100" not in ENTER_MODES` — closes it |
| 19 | `file_logger` / `quiet_terminal`: dbcache logging never reaching the terminal, the trail in the app dir, the app dir created on a first run | untouched by any surviving test |
| 20 | `sync_interval` coming from settings; `0` meaning exit-only | `TestPeriodicSync` varies `active`, never the interval |

---

## 4. Two latent bugs found on the way

Neither is an acceptance claim; both are one keystroke from being user-visible.

**`_escape` claims a stop it may not have made.** `ui/app.py:1536-1541` sends
`Interrupt` only `if self.active_id`, then sets `note = "stopped the turn"`
unconditionally — no turn required. Press `esc esc` on an idle session and the
footer says the turn was stopped. It is worse than a cosmetic bug because
**four surviving tests use that note as their proof the gesture works**
(`test_ui_app.py::test_the_second_esc_stops_the_turn`, `test_the_fourth_stops_again`,
`test_esc_esc_stops_from_the_row_too[3 rows]`). They would all still pass if
`send(Interrupt(...))` were deleted. Only
`test_ui_client.py::test_esc_esc_interrupts_the_open_session` watches the wire,
and only from the default focus.

Adjacent, and untested on both sides: `state.Turn.interruptible` is just
`working`, while `scheduler.can_interrupt` additionally requires
`interrupt_keep`. In the window between, the working row offers the interrupt,
confirming sends the command, the core returns `None` and emits nothing — and
the UI says "stopped the turn" while the spinner keeps turning. The same
disagreement makes a turn *parked on an approval* render as
`LLM processing… (enter or esc esc to interrupt)` when both stop gestures are
dead there.

**`reuse_message` copies into the session on screen, not the one the rewind was
opened in.** `_rewind`'s own docstring promises "an intent aimed at the session
it was opened in — which is not necessarily the one on screen by the time it
closes", and then `COPY` calls `reuse_message` → `_into_draft(self.session, …)`.
`QueuedOverlay`'s copy does it correctly, via `hand_back(overlay.session_id, …)`.
The test that names the behaviour
(`TestDrafts::test_a_reused_message_becomes_that_sessions_draft`) only ever
copies into the on-screen session, so it cannot catch it. Not reachable by
keyboard today; one added handler away from putting one conversation's words
into another's box.

---

## 5. Tests that assert less than their names promise

Worth knowing, because each is the strongest-looking evidence for a claim.

| test | promises | actually asserts |
|---|---|---|
| `test_ui_app.py::test_esc_esc_stops_from_the_row_too[*]`, `test_the_second_esc_stops_the_turn`, `test_the_fourth_stops_again` | the stop gesture reaches the core from any row | `ui.note == "stopped the turn"`, which is set unconditionally (§4) |
| `test_ui_turn.py::TestLiveStepRows::test_the_box_survives_reopening_the_session` | the box survives a reopen | feeds a `ChatReset` and greps "1 steps"; nothing is reopened. The core test is the real one |
| `test_core_service.py::TestTurns::test_the_prompt_carries_the_sessions_own_profile` | the prompt is rendered for the running session's profile | `isinstance(content, str) and content` — passes against the wrong profile or none |
| `test_ui_overlays.py::TestConfigEditor::test_invalid_values_show_an_error_and_keep_it_open` | invalid *values* keep the editor open | stubs `overlay._validate` by hand. The shipped `client._settings_error` checks JSON syntax only; a bad value closes the editor and comes back as a note |
| `test_ui_sessions.py::TestDrafts::test_a_reused_message_becomes_that_sessions_draft` | the draft belongs to the session it came from | only ever copies into the on-screen session (§4) |
| `test_ui_overlays.py::TestProfileLifecycle::test_an_empty_name_creates_nothing` | "and says so, on screen" | `isinstance(ui.overlay, PromptOverlay)`; the refusal text is never read |
| `test_ui_app.py::test_enter_elsewhere_in_the_log_goes_to_the_box`, `test_ui_client.py::test_a_row_that_is_not_a_message_cannot_be_rewound` | an assistant reply and a background event are not copyable | both aim at a `thinking` row, which is the first non-`user` row in the demo. No `assistant` or `event` row is ever aimed at |
| the narrow-terminal sweep | columns survive a narrow terminal | `SIZES` bottoms out at 40×10; the deleted suite swept to 1 column. Nothing asserts the four rows are still *present* at a narrow width, only that the frame is rectangular |
| `test_ui_turn.py::TestTheContextMeterState::test_speed_*` | the tok/s suffix works | builds `Context(speed=4.2)` by hand. Nothing in `src/` ever assigns `speed` (§6) |

---

## 6. Stale prose, which is how three of these gaps stayed hidden

Each of these reads as a settled decision and is a false statement about the
tree. Fixing them is part of M9.

| file | says | truth |
|---|---|---|
| `ui/overlays/llm.py:15-21` | "Discovery is not implemented anywhere… there is no rescan key… `r` is absent" | the core implements and tests all three; the UI does not send them |
| `ui/overlays/backendform.py` | "the probe is not here… there is no command on the wire for it yet" | `backend.probe` landed in `efb9b3f` |
| `ui/state.py:337-344` | "Nothing on the wire carries either yet" (speed, effort); "see `tests/test_ui_meter.py`" | `TurnUsage` carries `completion_tokens` and `request_seconds`; `effort` *is* wired (`client.py:596`); **`speed` is dropped by `client._usage`**, which is why `· 14.2 tok/s` can never appear. `tests/test_ui_meter.py` does not exist |
| `ui/__init__.py:42` | "deliberately still missing: … the slash commands and the management overlays" | M7 and M8 delivered both |
| `ui/overlays/inspect.py:3-6` | "`/skills-list` opens it, and so does the tunnel recipe" | neither exists; `RowUI.inspect` has no caller |
| `test_ui_edges.py:396` | the editor order "is its own test (`tests/test_memory_ops.py`)" | no such test |
| `test_ui_sessions.py::TestDrafts::test_a_parked_slash_draft_does_not_bring_its_menu_back` | "there is no menu in this build for a draft to bring back" | `test_ui_commands.py::TestAParkedDraft` asserts the opposite. Delete one before M9 or the surviving suite contradicts itself |
| `test_ui_sessions.py:418-421` | "`session.retitle` emits no `turn.activity`" | it does, `service.py:1036`, asserted at `test_core_service.py:1636` |
| `boot.py:19-22` | "the same guarantees hold here, and `tests/test_ui_boot.py` asserts them" | accurate for the shutdown order and the wait message; overstated for the notices, the logging and the disabled-cache path |

---

## 7. What the acceptance list itself misses

The list was extracted from the old *tests*, so it can only contain behaviour
those tests happened to cover. This is the complement, found by reading
`src/hpca/tui/` one last time. Ranked by what their loss would cost.

**Lost outright, and not claimed anywhere** (the four in §3.2–§3.5 plus):

- **`ctrl+s` on the backend form: save without probing** (`tui/backend_form.py:78,
  122, 205-219`) — the deliberate escape hatch for an endpoint that is
  momentarily down but whose URL and model id you already know, including the
  "Fix it, or press ctrl+s to save anyway" recovery from a rejected key.
- **Choosing among several models served by one endpoint**
  (`tui/backend_form.py:26-70`). `protocol.BackendProbed.models` exists and is
  built (`core/backends.py:601-618`); no overlay consumes it.
- **The profile picker no longer starts on the profile you are using**
  (`tui/profiles_screen.py:439`: *"the cursor starts on the current profile, so
  the common case is just enter"*). The new picker leads with `default` at
  `cursor = 0`, so a reflexive Enter creates the session under the wrong
  profile — while the acceptance claim ("a new session runs under the profile
  chosen in the picker") stays technically true.
- **The startup manage-LLMs screen's guard** (`tui/app.py:5334`:
  `if self.is_running and len(self.screen_stack) == 1`) — so a modal arriving
  seconds later never lands under the hands of someone who has meanwhile opened
  a screen. The part most likely to be forgotten when §3.1 is rebuilt.
- **`matches: <keywords>` on a reflection proposal**
  (`tui/memory_screens.py:166-171`) — the user approves a *trigger*, not just a
  text. `_wire()` keeps them only incidentally, for struggle notes.
- **`f5` rescan, and the discovered panel skipping what is already configured**
  (`tui/manage_llms.py:396-431`, plus the `_superseded` save-time guard at
  `547-553`). `core/backends.py:406-427` emits discovered rows *without*
  excluding configured ones, so when §3.1 is reconnected the duplicate will be
  there waiting.
- **The scan progress line** (`scanning localhost… 12,000/65,535 ports`) — what
  makes a 60-second sweep read as progress rather than a hang.
- **The command palette on bare `p`** — the one thing §4.3 *does* record as
  dropped deliberately, worth noting only because `overlays/help.py` is now the
  sole discovery surface.

**Carried over but unclaimed** — these survive only because someone hand-ported
them, so nothing would flag their loss: the "I have struggled with this before"
toast; a deleted session being forgotten from the episodic index and dropping
its watches ("patient-data environment: a deleted conversation must not
resurface"); job state-change toasts and the terminal-state event that wakes
the agent with `triage_job`; the "remember this failure as a signature?" offer
after a tier-3 diagnosis; `reconcile_orphans` at startup; "database sync is
failing" said once rather than every tick; and writing the discovered context
window back into the catalog entry.

**New behaviour the port built that the list never asked for**, and which will
be lost the next time the list is used as the spec:

- `decision.cleared` — the core withdrawing a prompt when the turn dies or
  another front-end answers (`TestAnsweringIt::test_the_core_clearing_it_also_takes_it_off_the_screen`).
- "the resumed turn is still stoppable" and "a parked decision survives a core
  restart" (`TestAParkedApproval`, `TestAResumeAfterACoreRestart`).
- A fork is deliberately *not* gated where a rollback is
  (`TestFork::test_forking_is_allowed_while_the_source_is_busy`).
- The cut is carried by the core's `Entry.index`, not by screen position or
  `seq` — the mechanism the whole rewind now rests on
  (`TestTheRewind::test_a_fork_carries_the_message_index`).
- The decision-pending marker for an off-screen session — better covered than
  the "updated" one it outlived.
- `Pollers._last_rows` suppressing an unchanged panel, with `force=` for a
  fresh client — real failure modes (a reconnecting client that never gets its
  column), well tested, unclaimed.
- `_apply_settings` naming *which* setting waits for a restart, instead of the
  old blanket claim. Better than what it replaced; keep it.

---

## 8. The M9 checklist this audit implies

Mechanical, before the deletion:

1. Repoint `tests/test_context_bar.py` at `hpca.ui.meter` and drop its widget
   half, or delete it and port `test_overflow_does_not_exceed_the_bar` and
   `test_small_window_reaches_danger_quickly` into
   `test_ui_turn.py::TestTheContextMeterState`.
2. Move `test_tui_memory.py::TestResolveEditor` somewhere surviving.
3. `tests/conftest.py::no_startup_backend_modal` imports `hpca.tui.app`; delete
   `evals/bench_textual.py` in the same commit (specs-ui-baseline.md §1 already
   says so); update `README.md`, `project.md`, `pyproject.toml`,
   `docs/llm-key-registry.md`, `docs/concurrent-session-turns.md`.
4. Delete or rename
   `test_ui_sessions.py::TestDrafts::test_a_parked_slash_draft_does_not_bring_its_menu_back`.

Before the front-end swap: §3.1, §3.2, §3.3. Each is wiring an existing,
tested unit to an existing, tested caller.
