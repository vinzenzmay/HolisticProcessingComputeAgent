# Spec: replacing Textual with the row UI

**Status:** plan, 2026-08-20. Branch `proto/row-ui`.

**One-line purpose:** make `prototypes/row_ui.py` the real front-end, delete
`src/hpca/tui/` and the Textual dependency, and reach a build that can be used
for real work on a cluster and refined from there.

---

## 1. Why this is smaller than it looks

Three findings decide the shape of the work.

**There is already a seam, and it is finished.** `feat/core-subprocess`
(merged, `37b334f`) landed `protocol.py`, `transport.py`, `coreproc.py` and
`hpca/core/` — a UI-agnostic agent runtime behind a typed command/event
protocol. It is tested (`test_core_service.py` drives a real graph through
protocol commands with no terminal) and `test_core_headless.py` enforces that
no `hpca.core` module can even import Textual. **Nothing under `tui/` calls any
of it.** `specs-core-process.md` §2 names a second front-end as the intended
additive next step.

We skip that spec's unfinished "Wave 2b — the surgery", which turns `HpcaApp`
into a forwarder. It exists to keep 34 `test_tui_*` files green while the
runtime moves out from under them. If the Textual UI is being deleted anyway,
refactoring it first produces only throwaway scaffolding. The new UI is written
directly as a protocol client and `tui/` is removed whole.

**Textual does not leak.** Outside `src/hpca/tui/` there is not one
`import textual` — the single grep hit, `agent/middleware.py:304`, is the
English adjective in a docstring. `rich` is never imported anywhere in the
repo; it is only a transitive dependency of Textual and leaves with it. All
styling is `textual.content.Content` plus 22 `DEFAULT_CSS` blocks, every one of
which is *replaced* by the row UI's own ANSI rather than ported.

**Four modules are already framework-free** and move unchanged: `transcript.py`
(the `Entry`/`Step` render model, already twinned on the wire as
`protocol.Entry`/`protocol.Part`), `clipboard.py` (OSC 52 — it takes an
injected `emit` callable and never knew what a driver was), `editor.py`
(`$EDITOR` resolution), and all seven helpers in `tui/approval_screen.py`, 103
lines of pure functions over the interrupt dict.

## 2. Topology

```
  hpca                       UI process: the row UI, keys, clipboard, $EDITOR
    │  in-process queues first, AF_UNIX + NDJSON last
    └─ AgentService          graph, tools, sqlite, checkpointer, pollers,
                             turn scheduler, approvals
```

The UI runs one asyncio loop owning both `stdin` (via `loop.add_reader`) and
the `Connection`. Verified by spike before this plan was written: raw terminal
reads and protocol events interleave on a single loop with no threads. So the
prototype's synchronous `render(width, height) -> list[str]` core is kept
exactly as it is, and only the shell around it becomes async.

`InProcessConnection` for the whole port. The subprocess split (`--serve`,
Wave 3 of the old spec) is deliberately **last**: it buys crash isolation, not
correctness, and every milestone before it is testable without a socket.

## 3. Module layout

`src/hpca/ui/` — new package, and the permanent home. Not `tui/`: the old name
dies with the old front-end, and a half-migrated package sharing a name with
what it replaces makes every import during the port ambiguous.

| module | from the prototype | contents |
|---|---|---|
| `ui/ansi.py` | `_pad`, `_rule`, `_reverse`, colours | plus the width fix (§4.1) |
| `ui/screen.py` | `Screen` | alt screen, differential paint, synchronised output, SIGWINCH |
| `ui/keys.py` | `KEYS`, `_escape_len`, `decode` | plus bracketed paste (§4.2) |
| `ui/editor.py` | `Editor` | wrapping, word motion, selection |
| `ui/pane.py` | `Item`, `Pane` | the row list, expand/collapse, reorder |
| `ui/overlays/` | the five overlay classes | one module per screen; §5 lists the rest |
| `ui/app.py` | `RowUI` | layout, focus, key dispatch — no I/O |
| `ui/state.py` | — | events in, UI state out: sessions, chat, panel, turn |
| `ui/client.py` | — | protocol client: keys become commands, events become state |
| `ui/run.py` | `main()` | the asyncio loop, terminal setup, teardown |

`RowUI` stays free of I/O and of asyncio: it takes keys and events and returns
frames. That property is what makes the whole UI testable by calling `render()`
and comparing strings, which the prototype's 221-check harness already relies
on, and it is the single most valuable thing to protect during the port.

### 3.1 The three layers, and what may know what

```
  run.py     asyncio: stdin, the Connection, SIGWINCH, terminal modes
     │         knows about I/O. Knows nothing about rows or keys.
  client.py  events -> state mutations; UI intents -> commands
     │         knows the protocol. Draws nothing.
  app.py     state -> frames; keys -> intents
               knows rows and keys. Has never heard of a socket.
```

The rule that keeps this honest: **`app.py` must not import `hpca.protocol`.**
If a `RowUI` method takes a `protocol.Entry`, the boundary has already leaked
and the render tests start needing pydantic models to say anything. `client.py`
translates; `ui/state.py` holds the plain dataclasses that both sides share.

This is the same discipline `test_core_headless.py` enforces on the core, and it
should be enforced the same way — by a test that imports `hpca.ui.app` in a
clean interpreter and asserts `hpca.protocol` did not come with it.

### 3.2 Event to state

Each event has exactly one job. Anything not listed is dropped, per §4.2's rule
that a client ignores what it cannot draw.

| event | effect |
|---|---|
| `hello` | version check, profile into the header |
| `session.rows` | rebuild the sessions pane, preserving cursor **by session id**, not by index — a row inserted above the cursor must not move the selection |
| `chat.reset` | replace the open session's chat; only on open |
| `chat.append` | append one entry; the *only* path by which a chat grows |
| `turn.started` / `turn.finished` / `turn.failed` | the working row appears and goes; failure becomes an `error` entry |
| `turn.activity` | the working row's label and `started_at`; a repeated activity must not restart the clock |
| `turn.usage` / `context.estimate` | the context meter, measured beating estimated |
| `decision.requested` / `decision.cleared` | the inline approval prompt for that session; a sidebar `!` for any other |
| `confirm.requested` | the generic yes/no |
| `panel.update` | the watchers pane, updated in place by row `key` so the cursor survives a repaint |
| `memory.proposals` | the review queue |
| `notify` | a toast |

Two properties the UI must hold to, both of which the Textual app got wrong at
some point and paid for:

1. **Events are addressed and most are not for the visible session.** Every
   handler starts by asking which session the event names. State for
   non-visible sessions is updated silently — the sidebar marker is the only
   thing that may change on screen.
2. **The chat is append-only between resets.** No path may rebuild it from a
   snapshot. The queued-message bugs in `test_tui_queue.py` — "survives the
   transcript rebuild triggered by the finishing turn's reply" — are entirely
   the consequence of a UI that rebuilt, and they disappear if nothing rebuilds.

---

## 4. What the prototype is missing

The prototype's own docstring is honest about being a sketch. This is the full
list, and it is the bulk of the work.

### 4.1 Terminal correctness

1. **Character width.** `_pad` counts with `len()`. Real chat carries emoji,
   box drawing and occasionally CJK, all of which occupy two cells or zero.
   Every row is padded to an exact width, so a single wrong count shifts the
   rest of the line and corrupts the differential repaint. Needs an
   east-asian-width aware cell counter (`unicodedata.east_asian_width` plus a
   combining-mark check; no new dependency), applied in `_pad`, `_rule`,
   `_wrap_spans`, `_reverse` and the cursor column maths.
2. **Bracketed paste.** `decode()` currently drops `ESC[200~`/`ESC[201~` as
   unknown escapes, which is the right default but means a paste arrives as
   raw keystrokes — a pasted newline submits the message mid-way. Paste must
   be bracketed on (`?2004h`), captured whole, and inserted literally.
2b. **A key split across two reads decodes as garbage.** `decode()` consumes
   its buffer to the end and keeps nothing back, so an escape sequence
   straddling a read boundary is mis-measured: `escape_len` returns 3 for
   `ESC O` even when only two bytes have arrived, and the next `read()` starts
   mid-sequence. Worse, a lone trailing `ESC` from a split arrow key is decoded
   as the escape key and **arms the stop gesture** — one half of a cursor
   movement can begin killing the turn. Unreachable at 4096-byte reads today,
   and reachable the moment paste starts delivering multi-KB payloads, so it is
   fixed in the same milestone: `decode()` must return the undecodable tail and
   the loop must prepend it to the next read.
3. **Resize.** No `SIGWINCH` handling. The loop must re-read the terminal size
   and force a full repaint.
4. **Newline in the box.** Textual needed `termkeys.patch_alt_enter()` to tell
   `alt+enter` from `enter`. The row UI decodes escapes itself, so this is a
   key-table entry rather than a monkeypatch — but `shift+enter`, `alt+enter`
   and `ctrl+j` must all still insert a newline.
5. **Mouse.** Implemented and deliberately reverted (`22a503b`, reverted by
   `b5d2cbb`) because grabbing the mouse suppresses the terminal's own
   drag-to-select, and copying paths out of the log is constant here. Restore
   it late, off by default, with the `M` toggle the reverted commit already
   had.
6. **Selecting text out of the chat.** With the mouse released this is the
   terminal's job and works for free — which is the argument for leaving it
   released. A keyboard-driven copy of the row under the cursor covers the
   rest (§5, `y`).

   "For free" turned out to be half true, and the other half is v0.27.0.
   A terminal selects whole screen columns, so every column the UI spent in
   front of a line of the conversation landed in the paste buffer with it:
   the chat drew `▌ ▸ you   what is the coverage?`, and an opened message's
   body under six spaces of indent. What the terminal gave back was the words
   with the furniture still attached.

   The fix is a property of the chat pane and not of the copy path:
   `Pane(flush=True)` draws content lines at column 0 with no gutter and no
   indent, and `state.entry_item` moves the speaker onto a label line of its
   own so there is nothing left on the words' own lines. The rule the two
   make together — **flush is verbatim, indented is the UI talking** — is
   what a reader can rely on: a turn's working, a notice and a step's head
   keep their furniture, and a message's text and a tool's output do not.
   `tests/test_ui_pane.py::TestTheChatSelectsClean` is the promise.

   **What a closed row shows.** Moving the words off the head line takes the
   preview with them, and a column of bare `you` / `hpca` labels says who
   spoke and not a word of what was said. So `Item.preview` is the one line a
   closed row draws under its label — the message on one line, clipped to the
   terminal with `ansi.clip`. The mark is `[...]` and deliberately not the
   `…` `pad` truncates with: that one means the terminal is this wide, this
   one means there is more of this message and → will show it, and only the
   second has a gesture attached. A row whose head already summarises it — a
   turn's working, "8 steps · edit_file → run_bash …" — has no preview and
   closes to the head alone.

   So the shape is two lines a message closed, the full wrapped text open.
   Opening is what a selection wants anyway: the clipped line carries a
   `[...]` that is not part of what anyone said.

   **And when it happened.** The label line is what made room for it: a row
   reads `▸ you 21-08-2026 19:04:11`, because a chat log people scroll back
   through is one they ask "when was that" of. `protocol.Entry.at` carries the
   instant as UTC and `ui.state.when` writes it in the reader's own clock —
   the core need not be on the same machine (specs-core-process.md), and only
   the front-end knows which clock a person reads. The stamp is put on the
   *message* by the reducer that appends it, `agent.graph._append_messages`,
   which is the one door every path into a thread goes through; the transcript
   reads it back rather than reading a clock of its own, so a transcript built
   twice comes out the same. A turn's working carries none — it folds several
   messages and cannot honestly name an instant — and neither do the steps and
   tool calls under it. The sidebar says the same thing about a whole
   conversation, from `SessionRow.last_active`.

   The line list still grew — 8,300 lines at 5000 entries against 5000 —
   which is why `Pane.extend` exists: a row arriving is added to the cached
   list instead of dropping it. See §4.2 property 1 and
   `test_ui_perf.py::TestARowArriving`.

### 4.2 Protocol gaps

The protocol covers most of the surface. Four things are missing and must be
added to `protocol.py` with tests, in the same style as what is there:

7. **`session.fork` / `session.rollback`.** `Entry.index` exists precisely so a
   UI can offer the rewind, and `agent/graph.py` has `fork_thread` and
   `rollback_thread` — but no command carries the request. Without these the
   `RewindOverlay` cannot do anything real.
8. **`watch.peek`.** `watch.drop` exists; the `enter`-to-peek tail does not.
9. **A queued-message channel.** In Textual the type-ahead queue is UI state
   (`_pending_work`); in the core it is `TurnScheduler`. The `queued` chat kind
   and the "cancel it" gesture need a wire representation — most likely
   `chat.append` with `kind="queued"` plus a `turn.unqueue` command.
10. **`--serve` argv.** `coreproc.default_core_argv` already builds
    `python -m hpca --serve --socket …` and `__main__.py` parses nothing.
    Deliberately deferred to M11: the argv has no production caller today
    (`CoreSupervisor` is only ever constructed in tests, always with a stub),
    so parsing the flag early can only make it lie — a core that prints a
    handshake and exits leaves the supervisor dialling a socket nobody bound,
    which is a worse failure than the current honest absence. `__main__.py` is
    edited once, at the milestone that also swaps the front-end, and `main()`
    must argparse *before* importing either side so a core process never
    imports the UI.
11. **Entry addressing.** `chat.append` can only append, but a tool call must
    appear as a row and have its result land *in that row* (`Part.done`
    already documents that shape), and a `queued` entry must become an ordinary
    `user` entry in place. Textual dodged both by rebuilding the whole chat
    every turn — exactly the cost this protocol exists to delete, and
    incompatible with §3.2's append-only invariant. Needs a core-assigned entry
    identity and an update event.

**Found during M7, and it breaks an invariant rather than merely missing a
feature.** The overlays that *edit* a profile's memories, its archive, a skill
file and `settings.json` have **no read path on the wire**: `profile.rows`
carries a count, not a body, and there is no `profile.get`, `skill.list`,
`settings.get` or `settings.save`. Since `profile.save` writes verbatim, an
editor opened over an unfetched body would truncate the file — so the UI
currently reads all four out of the app dir itself.

That is `specs-core-process.md` §4.2 rule 2 — *"The UI never reads the
database"* — and the rule is not decorative: it is what deletes the poll
timers, `DbIO` and the `_is_active_session` checks from the front-end, and it is
what makes a socket-separated core possible at all in M11. A stopgap here is
the kind of thing that silently becomes the architecture. **Close before M9.**

Three smaller consequences of the same missing surface, all in M8's way:

- **`ctrl+l` cannot switch to a key-locked backend.** `backend.set` is the only
  command that pins a session and it wants a whole `LLMBackend` including the
  key, which the catalog rightly never carries. Wants a by-label form.
- **No `backend.probe`**, so the connection form cannot check an endpoint,
  auto-fill the model, or offer the multi-model picker.
- **Discovery is still not on the wire** (above), so manage-LLMs has no
  discovered panel, no rescan, and no `r` — there is no remove command either.

**Found during M6/M7, and the most serious thing on this list.** The core
writes **no per-turn transcript log and no episodic index entries.**
`TurnScheduler`'s `on_turn_result` hook names three consumers — "the transcript
log, the episodic index and session titling" — and only titling is wired.
`tui/app.py:_log_turn` has no counterpart, so in a core-driven run
`<app_dir>/sessions/*.log` is never written and `session_search` indexes
nothing, which silently breaks recall across sessions. Nine acceptance claims
under "Thinking box, tool rows, logging" and two under "Core chat wiring"
depend on it. **This must close before M9**, because after M9 there is no
implementation left to compare against.

Two smaller ones from the same work:

- **Discovery is not on the wire.** `LLMEntry.discovered` exists and is always
  `False`: nothing runs `scan_local_ports` or `discover_cluster_endpoints`, so
  manage-LLMs' *discovered* panel and its rescan have no core behind them.
  Adding a catalog entry is `backend.set`; **removing one has no command at
  all**, which the screen's `r` key needs. (M8.)
- **`specs-core-process.md` §4.1/§4.2 are behind again** — `llm.list`,
  `profile.list`, `llm.catalog` and `profile.rows` are absent.
  `tests/test_protocol.py` carries them with a comment saying so.

Found during M4, recorded for the milestone that hits each:

- **`turn.interrupt` recovers the user's message and then throws it away.**
  `scheduler.interrupt()` returns the text so it can be put back in the entry
  box — the behaviour `specs-ui-acceptance.md` calls "hands the user's message
  back", including "to *its own* session, not whichever session is now on
  screen". No event carries it; `turn.unqueued` belongs to the queue. (M5.)
- **A resume after a core restart draws nothing.** With no live turn record —
  the parked half belonged to a dead process — the reconcile deliberately emits
  nothing rather than a second working box, so the reply appears only on the
  next `chat.reset`. Reachable exactly through the newly-fixed "a parked
  decision survives a restart" path, which is what makes it M5's. (M5.)
- **A turn that *fails* leaves its live rows unreconciled**, because there is no
  result to fold. They settle as drawn. (M5.)
- **`turn.usage` carries no completion tokens and no wall clock**, so the
  meter's `· 14.2 tok/s` has no channel. The meter state is built and tested;
  nothing can set it. (M8.)
- **Thinking effort has no event.** `ThinkingSet` is a command only, so the
  meter's `· think medium` likewise has state and no source. (M7, with
  `/thinking`.)
- **`TurnStarted` carries no `started_at`**, so the elapsed clock only starts at
  the first `turn.activity`. (M5.)
- **`TurnActivity(activity="")` has two senders** — the scheduler just before
  `turn.finished`, and the memory service, which has no turn at all — so the UI
  disambiguates on whether a turn is working. Worth a wire distinction. (M8.)
- **`boot.py` emits its own `hello`** before subscribing, and `subscribe()` now
  sends one too. Harmless, since the handler is idempotent, but one of them
  should go. (M5.)

Three smaller gaps, recorded here for the milestone that hits each:

- **`chat.reset` must re-emit queued rows.** A session re-opened while a turn
  is running silently loses its type-ahead entries otherwise; `tui/app.py:4585`
  is where the old UI appended them. (M5.)
- **`notify` has no `title`.** Several Textual call sites used one. Memory and
  curation warnings will want it. (M8.)
- **`decision.requested.payload` is an opaque dict**, so the UI must understand
  the agent's interrupt shape after all. Accepted by `specs-core-process.md`
  §4.4, but it means `tui/approval_screen.py`'s seven helpers are load-bearing
  on the UI side rather than a convenience — they must be moved, not
  reimplemented. (M5.)

### 4.3 Screens and features with no prototype equivalent

Everything below exists in Textual today and has to be rebuilt as an overlay or
an inline region. The prototype has non-functional sketches for three of them
(`LlmOverlay`, `ProfilesOverlay`, `ConfigOverlay`); the rest do not exist.

**Session management** — 11. rename (`r`), 12. retitle (`t`), 13. delete (`d`,
confirmed), 14. the new-session flow (profile picker, then LLM picker), 15. the
sidebar markers (`!` decision pending, `⟳` working, and the updated/pending/
working row colours).

**The turn** — 16. the working indicator (braille spinner, activity name,
elapsed seconds, and the `enter or esc esc to interrupt` hint only when
interruptible), 17. live step rows where a tool call appears immediately and
its result is attached *into the same row* rather than appended below,
18. the context bar (`context [████──] 12,345 / 32,768 (38%) · 14.2 tok/s ·
think medium`, with warn ≥70% / danger ≥90% and `~` marking an estimate),
19. the mode bar and `shift+tab` cycling, 20. the model line.

**Approvals** — 21. the two-stage inline decision prompt: `y` approves, `n`
opens a "what should be different?" reason box, `escape` refuses without one.
Deliberately *not* a modal, so it cannot cover another session's chat; that
rationale is written down at `tui/approval_screen.py:9-15` and should survive
the port. 22. The generic confirm (`y`/`n`) used by eleven call sites.
23. `confirm.requested`, the separate yes/no channel triage uses to offer a
learned log signature — never wired into the Textual UI at all, so this is new
behaviour arriving with the port.

**Commands** — 24. the seven built-ins (`/memorize`, `/conclude`, `/compact`,
`/skill-creator`, `/skills-list`, `/skill-remove`, `/thinking`) plus `/<skill>`
by name, 25. the autocomplete menu with most-used-first ordering, closing as
soon as the token contains whitespace so ↑/↓ stay free for a multi-line draft.

**Overlays** — 26. settings (raw JSON, validated on close, refuses to close
while invalid), 27. profiles & learnings (list, memory editor, archive, skills,
copy, delete), 28. manage LLMs (discovered/configured columns, the port scan
with live progress, add-manually, the backend form with its probe, the
model picker), 29. switch LLM (`ctrl+l`), 30. thinking effort, 31. the
read-only inspect window, 32. the three memory screens (proposal, batch,
reflection), 33. the two skill screens (creator, picker), 34. the queued-message
dialog.

**Around the edges** — 35. toasts (`notify`), 36. clipboard copy through the
tiered `ClipboardManager`, 37. `$EDITOR` suspend for `ctrl+e` (drop out of the
alt screen, restore on return), 38. quit confirmation, 39. the exit DB-sync
message with its SIGINT guard, 40. opening manage-LLMs at startup when no
backend answers.

**Dropped deliberately** — Textual's command palette (`p`). It is a framework
feature, not an HPCA one, and everything it reached is on a key already.

---

## 5. Key map

The prototype already follows `app.py`'s bindings. The port keeps them, adds
what §4 requires, and resolves one collision.

| context | key | effect |
|---|---|---|
| global | `q` | quit (confirmed), sessions row only |
| global | `←` `→` | previous / next column |
| global | `esc esc` | stop the turn (1.0s window; one escape means nothing) |
| global | `c` | config, anywhere but the chat column |
| global | `m` / `a` | manage LLMs / profiles, sessions row only |
| global | `?` | help |
| sessions | `enter` | open (and focus the message box), or start a new session |
| sessions | `r` `t` `d` | rename, retitle, delete |
| sessions | `alt+↑` `alt+↓` | reorder |
| chat | `→` `←` | expand / collapse an entry |
| chat | `enter` | rewind on your own message; interrupt on the working row |
| chat | `y` | copy the row under the cursor to the clipboard |
| chat | `ctrl+l` | switch this session's LLM |
| chat | `shift+tab` | cycle mode |
| watchers | `enter` `d` | peek, drop |
| watchers | `alt+↑` `alt+↓` | reorder |
| message box | `enter` | send |
| message box | `shift+enter` `alt+enter` `ctrl+j` | newline |
| message box | `ctrl+←/→`, `ctrl+bksp/del`, `shift+`motion | word motion, word delete, selection |
| message box | `ctrl+↑` | leave the box |
| decision | `y` `n` `esc` | approve, refuse with a reason, refuse without |

`y` on the chat row is new and collides with nothing: the decision prompt owns
`y` only while a decision is pending, and it takes the key first.

---

## 6. Milestones

Each is a push to `proto/row-ui`. Each ends green: `pixi run -e dev test`, plus
the new UI's own checks. The old Textual tests stay green until M9 — they are
the regression reference, and losing them early means porting blind.

**M0 — the package.** Split `prototypes/row_ui.py` into `src/hpca/ui/` per §3,
no behaviour change, no hpca imports yet. Convert the 221 headless checks into
real pytest modules (`tests/test_ui_*.py`). The prototype file stays as a
runnable demo against fake data until M3, then goes.
*Done when:* `pixi run -e dev python -m hpca.ui.run --demo` draws the same
frames the prototype does, and the checks run under pytest.

**M1 — terminal correctness.** §4.1 items 1-4: cell widths, bracketed paste,
SIGWINCH, the newline keys. These are cheap, they are pure functions, and every
later milestone renders on top of them, so they come first.
*Done when:* a message containing emoji, CJK and a pasted multi-line block
renders with every row exactly `width` cells, asserted by test.

**M2 — the client, against a fake core.** `ui/state.py` + `ui/client.py`:
events become UI state, keys become commands. Tested by driving a scripted
`InProcessConnection` peer — no database, no LLM, no terminal.
*Done when:* a scripted `chat.reset` + `chat.append` + `turn.*` sequence
produces the right frames, and the right commands come back out.

**M3 — first real conversation.** `hpca --new-ui` builds an `AgentService` over
`InProcessConnection` and talks to a live backend. Nothing but send/receive and
the session list; no approvals, no overlays.
*Done when:* a real turn against `http://localhost:20001/v1` (key `cubi`)
appears in the chat. **First live checkpoint — stop and use it.**

**M4 — the turn, fully.** §4.3 items 16-20: spinner, live steps, context bar,
mode bar, model line, interrupt. This is where the performance claim gets its
first real test, because live step rows are what a long turn actually produces.

**M5 — approvals and the queue.** §4.3 items 21-23, §4.2 item 9. The
approval is the reason a turn can stall, so it lands before the overlays.

**M6 — sessions.** §4.3 items 11-15, plus the `session.fork`/`session.rollback`
protocol additions (§4.2 item 7) that make the existing `RewindOverlay` real.

**M7 — the overlays.** §4.3 items 26-34. Largest milestone by line count,
smallest by risk: each overlay is an independent `render`/`handle` pair, so
this is the one milestone that parallelises cleanly across subagents.

**M8 — commands and edges.** §4.3 items 24-25 and 35-40.

**M9 was blocked until the list in [specs-ui-coverage.md](specs-ui-coverage.md)
§3 was closed; it no longer is.** The audit taken before the step found 38
acceptance claims unimplemented, and three of them meant the app did not work
rather than that it was missing a nicety: `BackendRegistry.auto_connect()` had
no caller and nothing in the UI sent `backend.scan`/`probe`/`remove`, so a
cluster user could not connect at all; `ctx.queue_memory_edits` was set only in
`tui/app.py`, so the `memory` tool was dead while the prompt still instructed
the model to use it; and `trash.cleanup` and `run_curator_if_due` lost their
only caller with the file being deleted, which is unbounded growth in an NFS
home behind settings keys that silently do nothing.

All three are wired and tested, as is the fourth blocker — the lag probe of §8,
which `ui/boot.py` now builds and `ui/run.py`'s loop starts and stops. What §3
still lists as open is the injection warning on a flagged memory batch, the
watchers column's arrangement (there is no `watch.move` on the wire), and
memories not being re-read at a session boundary. None of the three stops the
app working; all three should be booked as work, and the second is a working
feature becoming a non-working one, so it is worth doing before the row UI
becomes the default rather than after.

Two of the original three were hidden by docstrings that had gone stale *within
this milestone* — they still said discovery was unimplemented and the probe had
no wire command, both of which earlier commits had made false. It happened
again in the round of fixes that closed them, in eight more places. Anything
the port still owes should be checked against behaviour, not against comments.

**M9 — the deletion.** Remove `src/hpca/tui/`, drop `textual` from
`pyproject.toml`, point `hpca` at the new UI, delete or rewrite the Textual
tests per §7. This is the milestone that makes the change irreversible, so it
comes only once M3-M8 have been used for real work.

**M10 — performance.** §8. Benchmarks land as tests with thresholds, so a
regression fails the suite rather than being noticed months later.

**M11 — the subprocess.** `--serve`, `CoreSupervisor` spawn, socket transport,
shutdown paths. Optional, and explicitly last: everything above works
in-process.

### Parallelism

M0-M3 are sequential — they build the spine. From M4 the milestones touch
mostly disjoint files and can overlap, and M7 splits across as many subagents as
there are overlays. The one shared file is `ui/app.py`'s key dispatch; the rule
is that an overlay subagent adds its class and one dispatch line, nothing else.

---

## 7. Tests

The default suite stays hermetic: no terminal, no LLM, no database.

**What the new harness offers.** The equivalent of Textual's `run_test()`/pilot
is three lines, because `RowUI` is synchronous and pure: construct it, feed keys
through `handle()`, read frames from `render(width, height)`. State assertions
read the UI object directly. There is no async, no widget query, and no
mounting, so the tests are faster and considerably shorter than what they
replace.

**Triage.** 100 test files, 34,633 lines.

| bucket | files | lines | disposition |
|---|---|---|---|
| A — core, no UI | 63 | ~22,000 | untouched; must stay green throughout |
| B — Textual-coupled, real behaviour | 34 | 12,277 | rewritten against the new harness |
| C — Textual mechanics only | 3 | 227 | deleted |

Bucket C is `test_tui_termkeys.py` (a monkeypatch of Textual's key parser),
`test_tui_notify.py` (that toasts default to `markup=False`) and
`test_tui_resize.py` (CSS min-widths). Each states a requirement that outlives
its test — ESC-CR must still decode as `alt+enter`, a toast must never crash on
arbitrary text, columns must survive a narrow terminal — so each is re-asserted
once against the new renderer rather than carried over. `test_context_bar.py`
splits: `render_bar`/`severity` are pure and untouched, the widget-state half
is rewritten.

Bucket A is worth naming because it is the reason this port is affordable:
`test_transcript.py` (the fold from graph state into chat entries, including
the rewind cut points), `test_core_*.py`, `test_protocol.py`,
`test_transport.py` and `test_clipboard.py` all already test, headlessly, the
machinery the new UI sits on. `test_core_headless.py` gets *more* valuable
during the port, not less.

**The acceptance checklist.** The subagent triage produced a per-file list of
the behaviours bucket B asserts — roughly 400 plain-English claims, from "a
queued message survives the transcript rebuild triggered by the finishing
turn's reply" to "the first Ctrl+C during the exit DB copy is answered, not
obeyed". That list is the port's real specification and is kept alongside this
document. Every claim must end up asserted by a new test or be explicitly and
visibly dropped.

**Rule for the port:** a Textual test is deleted only in the same commit that
adds its replacement, or in the same commit that deletes the behaviour it
covers. Never on its own.

**The harness the port must build.** There is no shared TUI harness today —
530 `run_test()` call sites re-declare the same helpers, and `submit_chat` is
copy-pasted into fifteen files. Building it once is the single biggest
simplification available:

- construct the app with an injected fake LLM, at a given terminal size;
- feed keys; settle; resize — the equivalents of `pilot.press`/`pause`/
  `resize_terminal` (613 `pause()` calls today);
- *wait until all in-flight work is done* — today `app.workers.wait_for_complete()`;
- *wait until a named view is open* — today `conftest.wait_for_screen`, which
  polls because the worker that pushed the modal is blocked on the answer.
  Condition-waiting, never pause-counting: the `-n auto` parallel run is what
  made pause-counting flaky in the first place;
- address a row by index or predicate, move the cursor to it, activate it;
- render any region to plain text and search it — generalising the
  `chat_log_texts()` / `bar_texts()` pattern that the better existing tests
  already prefer over widget queries;
- inject a mouse press/move/release at a cell coordinate (for M-restored mouse);
- a recordable output sink plus an "application mode stopped" marker, so the
  exit DB-sync ordering test survives without monkeypatching a driver.

One shared library of fake backends replaces the dozen hand-rolled ones
(`SlowLLM`, `BlockingLLM`, `ToolThenAnswerLLM`, `LoopingLLM`, `UsageLLM`, …),
all of which special-case the titling call by JSON schema — so the title
request must stay distinguishable by schema.

`conftest.py`'s `isolated_project_dir` is not Textual-specific and stays. The
`hpca_home` fixture, redefined identically in ~20 files, is consolidated during
the port. `no_startup_backend_modal` keeps its meaning: an opt-out for the
startup backend probe.

**Markers are unchanged.** `-m 'not integration'` in `addopts`, `integration`
meaning "needs a live LLM" and nothing else, and the 9 integration tests are
all bucket A. `test-live` keeps working throughout.

---

## 8. Performance

The whole reason for the exercise, so it is measured rather than asserted.
**Both sides have now been measured: see [specs-ui-baseline.md](specs-ui-baseline.md).**

The figure this plan originally quoted — Textual at 44 ms p95 for 300 messages,
from `8b790e7`'s commit message — **did not reproduce**. Textual costs about
13 ms there. The justification is nonetheless stronger than that number was,
and better located:

- Textual's *repaint* is flat in conversation length (~1.5 ms at any size). It
  is not the problem and the row UI has no real advantage there.
- Textual's *arrange pass* is O(conversation): 2.9 ms at 100 entries, 18 ms at
  1000, 126 ms at 5000 and 994 ms at p95. A scroll invalidates layout, so a
  scroll pays it.
- 15 seconds to first draw at 5000 entries, from mounting one widget per entry.
- A single 1 MB message costs 587 ms at p95; 1000 sessions cost 117 ms of
  layout, neither of which is about conversation length at all.
- The row UI is flat at ~0.19 ms from 100 entries to 20,000, and its cost tracks
  terminal *area* rather than content.

The row UI's own previously published figures (0.009–0.018 ms) were optimistic
by about 10× and are superseded by the same document. M10 keeps the remaining
dimensions and adds thresholds as tests.

Benchmarks to land as tests with thresholds:

| dimension | scale to break | what it exercises |
|---|---|---|
| chat entries | 1k, 10k, 100k | the visible-slice claim; must stay flat |
| one message's length | 10 KB, 1 MB | `_wrap_spans` is O(text), and it runs per frame for the *visible* rows only — a 1 MB entry must not cost a 1 MB wrap |
| sessions | 100, 1000 | the sidebar's flat list and its per-session state dicts |
| steps in one turn | 500 tool calls | live-row insertion, the path M4 adds |
| draft length | 100 KB in the box | wrapping under a keystroke, which is the one thing that runs on every input |
| terminal width | 40, 200, 400 cols | wrap cost scales with rows-per-entry |

Two things the table is designed to catch, both of which the prototype would
currently fail: `Pane.flat()` builds a list over *all* items to find the visible
window (fine at 5000, not obviously fine at 100k), and a single enormous entry
is wrapped in full even when three of its lines are on screen. If either shows
up, the fix is an index rather than a rewrite — but the number decides that, not
the plan.

Baselines are taken on `main` under Textual with the same content, so the
comparison is like-for-like, and `looplag.py` records event-loop delay on both
sides: `tui/app.py` starts it in `on_mount`, and on this side `ui/boot.py`
builds it and `ui/run.py`'s loop starts and stops it. Both are off unless
`$HPCA_LOOPLAG` is set and both append their block to `<app_dir>/looplag.log`,
noted with whatever the variable was set to — which is what makes the two
blocks tellable apart in one file.

---

## 8.1 Found while splitting the package

M0 was the first close reading of all 2100 prototype lines since they were
written. What it turned up, recorded against the milestone that owns each:

- **`expanded` is keyed by position, not identity** (`Pane`). Correct today
  because entries are only ever appended, and a landmine for M4: any insertion
  that is not at the end silently reassigns which entries are open. `reorder()`
  already hand-patches around it. M4 inserts live step rows, so M4 changes the
  key to an identity — this is not optional there.
- **`SessionState` imports `demo`**, lazily, to manufacture its own chat. The
  one place the "no I/O" property is a fiction: the model layer calls a content
  generator. M2's `ui/state.py` takes the chat as an argument instead, and the
  lazy import goes.
- **`flat()` and `expand_all()` are O(items) per keystroke** — `expand_all`
  rebuilds the whole `expanded` set and then `_go_to` scans linearly, three
  passes over every entry for one press. Invisible at 5000; the 100k row of §8
  is where it shows. M10 decides whether an index is warranted.
- **The key table is deliberately asymmetric and says so nowhere**: `ESC[1;3C/D`
  (alt+←/→) fold onto `ctrl-left`/`ctrl-right` for word motion, while
  `ESC[1;3A/B` stay `alt-up`/`alt-down` for reorder. Right, but it needs a
  comment before someone "fixes" it.
- **`Editor.cursor_visual` indexes `rows[-1]` unguarded.** Unreachable today —
  `wrap_spans` always returns at least one span — and one arithmetic change from
  being reachable.
- **`HelpOverlay` pages four rows short** because `height - 2` is subtracted
  twice. Cosmetic, shipped that way, left alone.

Two things it checked that turned out to be *fine*, recorded so they are not
re-investigated: the message box clamping to two rows on an 8-row terminal is a
clamp and not a lie, because the editor scrolls to keep the cursor visible; and
frames are byte-identical across the package split, verified over 234 frames at
six terminal sizes.

## 9. Risks

- **The core layer is tested but never used in anger.** `AgentService` has
  never driven a real interactive session. M3 exists to find out early what
  that costs, before eight milestones are built on top of it.
- **Approvals end the turn and resume a new one.** A UI that assumes the turn
  is still running while parked will double-count or lose state. The Textual
  app's `_interrupt_anchor` outliving a single turn is the workaround; the port
  must reproduce that or the stop gesture breaks after an approval.
- **Deleting `tui/` deletes the reference.** Hence M9's ordering, and hence the
  rule in §7 about never deleting a test on its own.
- **No terminal in CI.** Everything is tested through `render()` and a fake
  connection. Real-terminal behaviour — alt screen, paste, resize, `$EDITOR`
  suspend — is verified by hand at each live checkpoint, and the milestones are
  shaped so there is always something runnable to verify by hand.
