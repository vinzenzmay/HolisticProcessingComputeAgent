# Spec: Concurrent per-session turns + per-session model line

## Status

Design settled (grilling session, 2026-07-21). Not yet implemented. This
document is the hand-off: a coding agent should be able to implement the whole
thing from here without re-deriving the decisions.

## Motivation

Two symptoms, one root cause, plus one UI cleanup.

The user runs two vLLM instances on one node, different GPUs: a 35B and a 27B.
They open session 1 (35B), start a prompt, then open session 2 with a different
profile pointed at the 27B and send a message. Session 2's message shows the
**"queued"** marker and does not start until session 1's turn finishes.

The user's first theory was that both sessions share one LLM backend. **That is
wrong** — per-session backends already exist and work (commit `78a11be`, "Choose
the LLM per session; drop the global default"). Each session stores its own
backend blob (`Session.backend`); the graph resolves its client per running
session via `_client_for(self._busy_turn or self.active_session)`.

The real cause is that the TUI orchestrator runs **exactly one turn at a time,
globally, regardless of backend**:

- `_busy_turn` is a single app-wide slot (`app.py`). On submit,
  `queued = self._busy_turn is not None` — *any* in-flight turn in *any* session
  marks the new message queued.
- `drain_work` returns early if `_busy_turn is not None`, so nothing starts
  until the current turn ends.
- The turn runs as an **exclusive** Textual worker
  (`run_worker(..., exclusive=True)`), so the worker layer itself cancels a
  second turn.
- Every per-turn field is an app-wide singleton, overwritten each turn:
  `_busy_turn`, `_turn_ctx`, `_turn_worker`, `_turn_memory`, `_turn_skills`,
  `_interrupt_text`, `_interrupt_keep`.
- The graph reads those globals: `llm=lambda: self._client_for(self._busy_turn
  or self.active_session)` and `ctx=lambda: self._turn_ctx if ... else
  self._tool_ctx`. With two turns live, "the current turn" is ambiguous — the
  lambdas would hand session 2's turn session 1's client/context.

The code comment justifies the global lock as "concurrent ainvoke on a thread
would interleave checkpoint writes." That hazard is **per-`thread_id`** (per
session). Two turns on two *different* sessions write to *different* checkpoint
keys and do not conflict. The global serialization was simplicity, not
necessity.

Second, unrelated cleanup the user asked for: the TUI shows the current model in
the **`TopBar` at the absolute top of the app**, which reads as a global app
setting. It should move to the **top of the chat column** so it clearly belongs
to the session on screen.

## Goal

- A turn in session 2 can run **while** session 1's turn is still in flight
  (concurrent execution across sessions).
- **Same-session** concurrency stays forbidden: two turns in one session still
  serialize (that is the real checkpoint-write hazard). A second message typed
  into a session whose turn is running still queues *for that session*.
- The per-session model in use is shown at the **top of the chat column**, not
  the app top bar.

## Environment reality (why this is worth doing)

The two models are served by **two separate vLLM instances on separate GPUs**,
so concurrent HPCA turns produce genuinely parallel replies — not just a moved
queue. No shared-server bottleneck.

Turns run as **asyncio tasks on Textual's single event loop**, not OS threads.
Concurrency is *cooperative*: two `ainvoke`s interleave only at `await` points.
Consequences that shape the whole design:

- No OS-thread data races on Python app state. No locks needed **as long as each
  turn only reads/writes its own per-session state** and no `await` sits in the
  middle of a read-modify-write of shared state.
- Different `thread_id`s hit different checkpoint keys; `aiosqlite`
  (`AsyncSqliteSaver`) serializes writes on its own single connection, so
  concurrent checkpoint writes for *different* sessions are safe.

## Decisions (settled — do not relitigate)

1. **Goal is concurrent turns across sessions.** The shared-backend theory is
   discarded; per-session backends already exist and are correct. Same-session
   concurrency is out of scope (stays serialized).
2. **No HPCA-side concurrency cap** — unbounded at the app level. Each backend
   server is its own natural throttle. (Here the two servers are distinct, so
   real parallelism.)
3. **Mechanism = single stateless graph, resolve dependencies by `thread_id`**
   (option A, chosen over per-session graphs). Keep one `self.graph`. Change the
   `llm` / `ctx` / `mode_fn` callables from "read the global current turn" to
   "take a `session_id` (thread_id) and return that session's
   client / ctx / mode." The graph nodes already receive LangGraph's `config`;
   pull `config["configurable"]["thread_id"]` and pass it to the resolver. The
   graph then only ever reads the `thread_id` it was invoked with — correct by
   construction, no per-turn globals feed it.
4. **Per-session turn state.** Replace the singleton turn fields with a
   `dict[session_id, TurnState]` (or equivalent). `TurnState` holds at least:
   busy flag, worker handle, tool ctx, interrupt bookkeeping
   (`interrupt_keep` / `interrupt_text`), turn memory, turn skills, and usage.
5. **Queue semantics become per-session.** `queued = <this session has a turn in
   flight>`, not the global flag. `drain_work` starts the next waiting item for
   any session **not currently running a turn** (and not awaiting approval),
   instead of bailing whenever anything is busy. A second message for a busy
   session still queues for that session.
6. **Sidebar working indicator.** Add a per-session in-flight spinner/indicator
   in the Sessions sidebar (left column), reusing the existing row-marker
   machinery (`_session_row_text`, the `session-updated` class pattern). This is
   how the user sees that an off-screen session is still working.
7. **In-chat spinner is scoped to the visible session's turn** only. Background
   turns show progress only via the sidebar indicator, never in a chat the user
   is not looking at. `report_activity` currently sets one global `_activity`;
   it becomes per-turn (the visible chat's `WorkingIndicator` reflects the
   visible session's turn).
8. **Interrupt is scoped to the visible session's turn.** Interrupt keeps its
   current UX (select the working line / hold, per the esc-interrupt design),
   backed by the visible session's per-session interrupt state. There is **no**
   way to abort a background session's turn without switching to it first.
   Accepted by the user.
9. **Per-session context meter.** `_context_used`, `_discovered_window`, and the
   `on_usage` callback are keyed per session. The meter **always reflects the
   session on screen**: it ticks live if that session's own turn is running; a
   background turn silently updates *its own* stored numbers, so switching to it
   later shows the correct current value (not a stale zero, not another
   session's number). This also fixes an existing latent bug — the window probe
   is global today and is already wrong the moment two backends are in play.
10. **`switch_backend` / start gating goes per-session.** Today `switch_backend`
    refuses while *any* session is busy; it should refuse only if **this**
    session is mid-turn. Same for starting a turn.
11. **Model line moves to the top of the chat column, option A: a dedicated
    line directly above the context meter** (`ContextBar`), e.g. `model:
    qwen-27B` on its own row, the `context [▓▓▓──] …` meter on the next. Remove
    `model:` from the `TopBar`. Keep `profile:` in the `TopBar` for now (it is
    also echoed on each sidebar row); only the model moves. The model line is
    session-specific and updates on session switch.

## Design detail

### Graph dependency resolution (decision 3)

- `build_graph(llm=..., ctx=..., mode_fn=...)` currently accepts values or
  zero-arg callables resolved inside nodes (`llm() if callable(llm) else llm`,
  `graph.py`; `ctx() if callable(ctx) else ctx`). Change the contract so these
  callables take the running **thread_id** (session_id). Inside each node, read
  the thread_id from the node's `config` (`config["configurable"]["thread_id"]`)
  and pass it in. Keep backward-compat for a plain value / zero-arg callable if
  cheap, but the app will pass thread_id-aware resolvers.
- App-side resolvers become: `llm_for(session_id)` → `_client_for(session)`,
  `ctx_for(session_id)` → that session's `TurnState.ctx` (falling back to the
  UI/`_tool_ctx` only for the on-screen session with no running turn), and
  `mode_for(session_id)` → that session's mode.
- `run_turn` already passes `config = {"configurable": {"thread_id":
  session_id}}` into `ainvoke`; no change needed there beyond making sure the
  per-turn resolvers key off it.

### Per-session turn state (decision 4/5)

- Introduce `TurnState` and a `dict[session_id, TurnState]` on the app.
- `_run_agent` / `_agent_turn`: write into the session's `TurnState` instead of
  the singleton fields; clear that session's entry in the `finally` block.
- Drop `exclusive=True` on the turn worker (it cancels sibling turns). Use a
  per-session worker group instead so a session's own new turn still cannot
  double-run, but other sessions are untouched. Verify Textual worker groups
  behave as expected here.
- `drain_work`: iterate pending work, start the first item whose session is
  **not** already running a turn and **not** in `_awaiting_approval`; keep
  draining so multiple sessions can start in one pass (respecting the
  one-item-per-pass reasons only where they were about *same-thread*
  interleaving — re-examine that comment).
- `_on_chat_submitted`: `queued` is now "this session already has a turn in
  flight", not the global flag.

### UI (decisions 6/7/9/11)

- Sidebar: extend `_session_row_text` / row classes with a working indicator for
  any session with a live `TurnState`. Repaint on turn start and turn end.
- `report_activity` / `WorkingIndicator`: activity is per-turn; the in-chat
  spinner shows the visible session's turn activity only.
- `ContextBar`: keep it, but store `used` / `window` per session and bind the
  displayed values to the active session. Add a `ModelLine` (or fold a model row
  into the chat-column header) directly above it.
- `TopBar.render_text`: drop the `model:` segment. Keep app name, `profile:`,
  and `(c) config`.

### Interrupt (decision 8)

- `_interrupt_keep` / `_interrupt_text` move into `TurnState`. The interrupt
  action reads the **visible** session's `TurnState`. Background turns are not
  interruptible from another session's view.

## Files in play (starting points, not exhaustive)

- `src/hpca/tui/app.py` — orchestrator: `_busy_turn`, `_pending_work`,
  `drain_work`, `_run_agent`, `_agent_turn`, `_on_chat_submitted`,
  `switch_backend`, `reload/rebuild graph`, `_client_for` / `_backend_of` /
  `_active_*`, `report_activity`, `show_working` / `hide_working`, `_on_usage`,
  `_reload_sessions` / `_session_row_text`, `TopBar`.
- `src/hpca/agent/graph.py` — `build_graph` (the `llm` / `ctx` / `mode_fn`
  resolution at the two call sites), `run_turn` (already thread_id-scoped).
- `src/hpca/tui/context_bar.py` — `ContextBar`; add the model line above it.
- `src/hpca/agent/modes.py` — mode resolution if `mode_fn` signature changes.

## Test surface (TDD — write these first)

New / changed behavior to pin down. The harness pattern is in
`tests/test_tui_queue.py` and `tests/test_tui_session_llm.py`: a gated
`SlowLLM` that holds a turn open until released, driven via Textual's `Pilot`.

- **Two sessions' turns overlap.** Start a turn in session A (held open by the
  gate), switch to session B, send a message → B's turn **starts** (not queued)
  while A is still held. Both `TurnState`s are live simultaneously.
- **No cross-contamination.** With A and B live on different fake backends, each
  turn sees its own client, its own tool ctx, its own usage — assert A's usage
  does not land on B's meter and vice versa.
- **Same-session still serializes.** A second message in session A while A's
  turn runs is queued for A (per-session `queued`), and does not start a second
  A-thread.
- **Per-session context meter.** Usage reported by a background turn updates that
  session's stored number, not the visible session's; switching sessions shows
  the right value.
- **Sidebar working indicator** appears for a session with a live turn and
  clears when it ends.
- **Interrupt** targets only the visible session's turn.
- **Model line** renders at the top of the chat column and updates on session
  switch; `TopBar` no longer contains `model:`.
- **Rewrite `tests/test_tui_queue.py::test_only_one_turn_runs_at_a_time`** and
  `test_a_message_sent_while_busy_is_accepted`: they currently assert the global
  single-turn invariant via `app._busy_turn`. Under the new design the invariant
  is *per session*. Keep the "typing while your own session is busy queues"
  guarantee; drop the "any busy blocks everything" assertion.

Also re-scan for other tests asserting the global lock: `tests/test_tui_working.py`,
`tests/test_tui_interrupt.py`, `tests/test_tui_background_reply.py`,
`tests/test_tui_session_llm.py`, `tests/test_context_bar.py`.

## Work plan (staged, one feature branch, TDD, commit per step)

Order matters: the concurrency core is the foundation; the rest is UI that
depends on `TurnState` existing. All stages touch `app.py`, so build them
**sequentially**, not in parallel worktrees.

1. **Concurrency core.** `TurnState` per session; graph resolves
   `llm` / `ctx` / `mode` by `thread_id`; drop `exclusive=True`; `drain_work`
   runs one turn *per session*; `queued` and `switch_backend` / interrupt gate
   per-session. Tests first (overlap, no cross-contamination, same-session still
   serializes). This is the risky commit — land and verify it before UI.
2. **Per-session context meter.** Key `_context_used` / `_discovered_window` /
   usage by session; meter binds to the visible session.
3. **Sidebar working spinner.** Per-session in-flight indicator.
4. **Model line move.** Dedicated line atop the chat column; remove `model:`
   from `TopBar`.

## Execution instructions (agreed with the user)

- Branch: `feature/concurrent-session-turns` off `main`.
- TDD throughout: failing test → implementation → green, per stage.
- The user asked for sub-agents to build the features; because every stage
  touches `app.py`, dispatch them **sequentially** (each builds on the prior),
  not in parallel.
- After all stages: run the **full** test suite (`uv run pytest`, per
  `README.md`).
- This is a **feature** → on green, **merge to `main` and bump the version**
  (`pyproject.toml` and `src/hpca/__init__.py`, currently `0.5.2`; minor bump →
  `0.6.0`). Follow the existing merge-commit style,
  e.g. "Merge feature/concurrent-session-turns: concurrent per-session turns +
  per-session model line; bump to 0.6.0".

## Out of scope

- Same-session concurrent turns.
- Any HPCA-side concurrency cap / scheduler.
- Aborting a background session's turn from another session's view.
- Moving `profile:` out of the `TopBar`.
- Changing the backend-selection or key-registry flows (already shipped).
