# Prototype: a ratatui front-end, instead of Textual

**Status:** prototype, measured, on `proto/ratatui`. Not a proposal to merge as
it stands — a report on whether the road is open, and what it would cost.

**One-line purpose:** find out whether the TUI can be replaced by a Rust
(ratatui) front-end, and whether that actually fixes the performance problem
Textual has here.

---

## 1. What was asked, and the short answer

Two questions:

1. **Can ratatui run here?** Yes. `ui-rs/` builds and runs on this box against
   ratatui 0.30.2 / crossterm 0.29. One obstacle, described in §7: there is no
   system C compiler, so the Rust linker had nothing to call.
2. **Can the current TUI be translated to it?** Yes, and much more cheaply than
   the 8,292 lines of `src/hpca/tui/` suggest — because **the port is a
   front-end port only**. `specs-core-process.md` already cut the seam this
   needs, and it is a *published protocol*, not an internal boundary.

The thing that makes this tractable is worth stating plainly: **no part of the
agent runtime has to be ported to Rust, and none of it is.** `hpca/protocol.py`
says so itself — "a front-end that renders chat has no business importing the
agent runtime" — and the spec anticipated exactly this case: *"the protocol is
designed so it is a later, additive front-end, not a redesign."*

So the question is not "rewrite HPCA in Rust". It is "write a second client for
a protocol that already exists". That is what `ui-rs/` is.

## 2. What was built

`ui-rs/`, ~2,600 lines of Rust, additive — **not one line under `src/hpca` was
changed**.

| file | what it is |
|---|---|
| `src/protocol.rs` | the wire, mirroring `hpca/protocol.py`: 19 commands, 17 events, `Entry`/`Part`/`SessionRow`/`PanelRow` |
| `src/transport.rs` | AF_UNIX + NDJSON client, framing and sequence numbers as `transport.py` defines them |
| `src/app.rs` | front-end state, and the flattened chat line model that §4 is about |
| `src/ui.rs` | the draw: top bar, three columns 1:2:1, chat boxes, context meter, mode line, approval bar, footer |
| `src/input.rs` | the keymap, mirroring the `BINDINGS` tables |
| `src/bench.rs` | off-screen render and timing harness |
| `fakecore.py` | a core that speaks the protocol and runs no agent |
| `serve.py` | **the missing wave 3**: `build_service` + `serve_unix`, ~90 lines |
| `seed.py`, `interop.sh`, `realcore.sh`, `liveturn.sh` | the harnesses |

39 Rust tests, all passing.

### Running it

```
cd ui-rs
cargo build --release
./target/release/hpca-ui --snapshot        # draw a frame, print it as text
./target/release/hpca-ui --bench 3000      # the timing table in §4
./interop.sh                               # against a protocol-speaking fake core
HPCA_LLM_KEY=… ./liveturn.sh               # against a real core and a real LLM
```

`--snapshot` and `--probe` render through ratatui's `TestBackend`, so the whole
thing can be developed and checked without a TTY — which is also how every
figure in this document was produced.

## 3. It talks to the real thing

Three levels of verification, weakest first.

**Against the real protocol modules.** `fakecore.py` deliberately imports
`hpca.protocol` and `hpca.transport` rather than reimplementing them, and parses
every incoming frame with `protocol.parse`. Every model there is
`extra="forbid"`, so a field the Rust client renamed, mis-typed or dropped fails
loudly on the Python side. All eight command types the probe sends are accepted;
23 event frames decode; zero bad frames.

```
ok session.list   ok session.open   ok session.focus   ok turn.submit
ok command.run    ok mode.set       ok watch.drop      ok decision.resolve
```

**Against a real core.** `serve.py` puts `hpca.core.build_service` behind
`serve_unix` — the real databases, session store, graph, backends, scheduler and
pollers. It boots, binds, handshakes, and the Rust client attaches.

**Against a real LLM.** `liveturn.sh` runs a whole turn: Rust front-end →
NDJSON over AF_UNIX → `CoreService` → the langgraph graph → the cluster's
`Qwen3.8-27B-FP8`, and the answer back. Two 200s on
`/v1/chat/completions`, `turn.started → turn.activity ×3 → context.estimate
(3,961 tokens) → turn.finished`, and the reply rendered in the chat column.

That last one is the result that matters: **the ratatui front-end drove a real
HPCA agent to a real answer.** Nothing was stubbed below the socket.

## 4. Does it fix the performance problem?

The Textual problem is not vague, and it is not "Python is slow". It is written
down in `WorkingIndicator._render_frame` (`tui/app.py:767`):

> `Static.update` lays out by default, and a layout pass walks the whole chat
> log — O(messages). At 12.5 frames a second that made the TUI crawl for as long
> as a reply was in flight, and worse the longer the conversation: measured
> event-loop lag went from 0.7ms to 44ms (p95) at 300 messages.

That is the shape to beat: **frame cost proportional to conversation length,
paid continuously while the user is waiting**. Textual's own mitigation is the
`layout=len(text) != self._width` trick — a workaround at one call site for a
cost the widget model imposes everywhere.

`ui-rs` attacks the cause. The log is flattened *once* into a vector of styled
lines; a frame copies the visible slice. The spinner's text is written at draw
time, so a spinner frame invalidates nothing.

`--bench 3000`, release build, 120×40, 500 frames per row:

```
 messages    lines     layout  frame p50  frame p95     rebuilds
       10       36     0.07ms    0.104ms    0.216ms            0
       50      172     0.07ms    0.086ms    0.090ms            0
      100      341     0.11ms    0.086ms    0.092ms            0
      300     1021     0.32ms    0.086ms    0.091ms            0
     1000     3401     1.01ms    0.086ms    0.091ms            0
     3000    10201     3.41ms    0.086ms    0.091ms            0
```

Read this carefully, because it is easy to overclaim:

- **`frame` is flat.** 0.091ms p95 at 50 messages and at 3000 — a 60× increase
  in transcript for no increase in frame cost. `rebuilds = 0` says why: no
  layout pass happens during a turn at all.
- **`layout` is linear** — 3.41ms at 3000 messages — and that is fine, because
  it is paid only on a resize, not 12.5 times a second.
- **The two numbers are not the same statistic.** 44ms is Textual's event-loop
  lag; 0.091ms is time spent in `draw`. This is not "44ms became 0.09ms". What
  is established is the *shape*: the O(messages) term that made the Textual
  version degrade as a conversation grew is gone, and a test asserts it
  (`frame_cost_does_not_grow_with_the_length_of_the_conversation`).

A like-for-like number against the Textual build would need `looplag.py` running
in both; that is the honest next measurement, and §8 lists it.

## 5. What the port is actually like

Faithful, and less painful than expected, because most of `tui/app.py` is not
drawing — it is the runtime, and the runtime does not move.

Reproduced: the three columns at 1:2:1 with the `min-width: 74` guard; message
boxes titled by speaker ("who is speaking is a colour, not a prefix"); the
collapsed thinking box and its expand-into-rows behaviour, including per-part
collapse and the `── result ──` divider; the working spinner with its glyph set,
0.08s period, turn-owned clock and interrupt hints; the context meter with its
28 cells and 70%/90% thresholds; the mode line; the approval bar *at the foot of
the chat, not as a modal*; per-session drafts; the footer.

The keymap is ported with its stated subtleties intact, and they are the parts
that would be easy to get wrong: `q` quits from the sessions column but types a
letter in the entry; escape is *not* a priority binding, so a prompt that uses
escape to mean "leave this" still wins; the two-press `esc esc` stop gesture;
`shift+tab` for mode because `ctrl+m` arrives as Enter.

Two things ratatui makes *easier* than Textual, both visible in §4:

- **Immediate mode removes the whole class of bug.** There is no widget tree to
  fall out of step with the data, so no `_live_call` reference to keep pointing
  at a row, no "redrawing the log destroys the widget so the state must survive
  it" (`WorkingIndicator.clone`), no expansion state to protect from an
  unrelated re-render. The state is the model; the frame is a pure function of it.
- **A resize cannot crash it.** The Textual CSS carries a comment about a
  resize to ~50 columns taking the whole app down through Rich's wrapping. Two
  tests here render at 20–74 columns and 3–8 rows.

What is **not** built, and would be real work: the modal screens (`manage_llms`
585 lines, `profiles_screen` 462, plus settings, memory, rewind, skills — ~1,700
lines of Textual), mouse selection and clipboard, `$EDITOR` integration, and
autocomplete. None is hard; all is volume. Call it the larger half of the
remaining front-end effort.

## 6. The real obstacle is on the Python side

This is the finding that should drive the decision, and it has nothing to do
with Rust.

`specs-core-process.md` describes three waves. Waves 1–2 landed: the services
are extracted, `protocol`/`transport`/`coreproc` exist and are tested. **Wave 3
never did.** `tui/app.py` is still the monolith that builds the runtime, and
`python -m hpca` has no `--serve`. The consequence is that `CoreService` has
never had a client, and it shows:

**`CoreService._dispatch` handles 6 of 19 commands** — `session.focus`,
`turn.submit`, `turn.interrupt`, `decision.resolve`, `confirm.resolve`,
`shutdown`. Everything else answers *"Unhandled command"*: the whole session
lifecycle (`list`/`new`/`open`/`close`/`rename`/`retitle`/`delete`),
`command.run`, `mode.set`, `thinking.set`, `backend.set`, the profile and skill
commands, `process.kill`, `job.cancel`, `watch.drop`.

**The core never emits `chat.reset` or `chat.append` at all.** Grep `src/hpca/core`
for either: nothing. The transcript events — the ones §4.2 calls the reason the
protocol is not slower than what it replaces — are unimplemented. A reply
reaches a client only as `TurnFinished.reply`. (The Rust client renders that,
with a guard so it will not double-draw once real `chat.append` lands.)

So: **the front-end language is not the blocker; the unfinished core is.** That
work is identical whether the next front-end is Rust, a rewritten Textual, or
the row-oriented Python prototype on the other branch. It is worth doing on its
own merits — it is what the spec already committed to.

## 7. Environment notes

- **No system C compiler.** `cargo build` failed with ``linker `cc` not
  found``; there is no `gcc` in `/usr/bin`. Fixed with `pixi global install gcc`
  (16.1.0), which exposes `cc` in `~/.pixi/bin` without touching the system or
  the repo. Anyone building this needs `~/.cargo/bin` and `~/.pixi/bin` on PATH.
- `~/.cargo` was already present (Rust 1.97.1) but not on PATH.
- crates.io is reachable; the dependency set is small and mainstream (ratatui,
  crossterm, tokio, serde, unicode-width).
- Shipping a Rust binary to a cluster is a real question this prototype does not
  answer: HPCA installs as a Python package today, and adding a compiled
  artefact changes that. Options — a build step in `pixi`, prebuilt binaries per
  target, or `maturin` — are unexplored.

## 8. What would settle it

1. **Finish wave 3** (Python, front-end-agnostic, the prerequisite for anything):
   the missing 13 commands and the `chat.reset`/`chat.append` emission, plus a
   real `hpca --serve`. `ui-rs/serve.py` is a working sketch of the last part.
2. **Measure like for like**: `looplag.py` under both front-ends, same
   transcript, same turn. §4 shows the shape; this would give the number.
3. **Decide the distribution question** before, not after, the modal screens get
   written.
4. Only then: port the modal screens, or decide the Rust front-end stays a
   focused "fast chat client" and the Python one keeps the management UI.

## 9. Honest summary

Ratatui runs here, the translation is faithful, the protocol interoperates
exactly with Python's own definition, and a real agent turn works end to end
through it. The performance claim holds in the specific way that matters: the
cost that grew with conversation length is gone, and stays gone at 3000
messages.

But the reason this was easy is that someone already did the hard architectural
work in `specs-core-process.md` — and the reason it is not yet *usable* is that
the same work is two-thirds finished. Throwing out Textual is a smaller decision
than it looks; finishing the core split is the larger one, and it is owed
regardless of which front-end wins.
