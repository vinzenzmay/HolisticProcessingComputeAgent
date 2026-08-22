# Spec: the core as a child process

**Status:** design agreed (2026-07-31), implemented on `feat/core-subprocess`.

**One-line purpose:** run the agent runtime (graph, tools, databases, pollers)
in its own child process, and reduce the TUI to a renderer that speaks a small
command/event protocol over a Unix socket — so UI latency stops depending on
what the agent is doing, and the agent stops depending on Textual.

---

## 1. Context — why

Today `HpcaApp` *is* the application. `on_mount` (`tui/app.py:1346`) builds the
DbCache, sqlite, the `AsyncSqliteSaver`, the graph, the tool registry, RAG,
embeddings, trash and Slurm. A turn is a Textual worker coroutine on the app's
own event loop (`tui/app.py:1984` → `_agent_turn` → `run_turn` →
`graph.ainvoke`), and the tool context is manufactured by the widget class
(`_make_tool_ctx`, `tui/app.py:2935`) holding the loop-thread sqlite connection.

Two consequences.

**Latency.** The long waits are already async and harmless — `llm.py` is
`httpx.AsyncClient`, `slurm.py` uses `asyncio.create_subprocess_exec`, the
checkpointer is aiosqlite. The hiccups come from short synchronous blocks on
the UI loop:

| Source | Where |
|---|---|
| tool handlers are `async def` with fully sync bodies (`read_text`, `write_text`, `iterdir`, `rglob`) | `agent/builtin_tools.py:159,222,255`, `agent/doc_tools.py:323`, `symbols.py:290` |
| whole-log reads during triage | `triage.py:135,222,373` |
| CPU on the loop: ast walk + `verify_script`, pydantic validation, schema retries | `verify_code.py`, `agent/middleware.py` |
| sync sqlite on the loop: `ToolContext` gets `self._conn`, not `DbIO` | `tui/app.py:2947` |
| `MemoryIndex.search` before the worker even starts | `tui/app.py:1573`, called from `tui/app.py:1954` |
| full chat rebuild at the end of every turn | `tui/app.py:4095` |

**Coupling.** `tui/app.py` is 4875 lines and holds the agent runtime: the turn
scheduler (`_pending_work`/`drain_work`/`_turns`/`_awaiting_approval`), the
pollers, memory/curator/reflection, prompt assembly, the LLM catalog, and the
approval state machine. There is no seam a second front-end could attach to,
and a Textual exception can take down a running turn — the `notify()` override
at `tui/app.py:1329` exists precisely because a `MarkupError` from LLM output
used to do that.

The process split fixes every row of the latency table except the last (that
one is fixed by incremental chat events, §4.3), and fixes the coupling
structurally.

**Explicitly out of scope for this spec:** the `--simple-ui` front-end. The
protocol is designed so it is a later, additive front-end, not a redesign.

## 2. Topology

```
  terminal (tmux, compute node)
  └─ hpca                    UI process: Textual, widgets, keys, clipboard, $EDITOR
       │  AF_UNIX, NDJSON
       └─ hpca --serve        core process: graph, tools, sqlite, checkpointer,
                              LLM clients, pollers, turn scheduler, approvals
```

A **child process**, not a daemon. The deployment model (project.md §2) is tmux
on a compute node, so tmux already provides survive-disconnect; a daemon would
add lease, GC and version-skew problems to buy something we already have. The
lifecycle below leaves `--attach <socket>` available as a later flag without a
protocol change.

### 2.1 Lifecycle

**Spawn.** The UI starts the core with

```python
asyncio.create_subprocess_exec(
    sys.executable, "-m", "hpca", "--serve", "--socket", str(path),
    stdout=PIPE, stderr=log_fd, start_new_session=True,
)
```

`start_new_session=True` is load-bearing: without it, Ctrl-C in the terminal
goes to the whole foreground process group and kills the core mid-turn. The UI
must be the only thing the terminal can signal.

**Handshake.** The core binds the socket, writes exactly one line to stdout —
`{"ready": true, "pid": <int>, "socket": "<path>", "version": <int>}` — then
closes stdout. The UI reads that line with a timeout (default 15s; a cold NFS
import of langgraph is not fast) and then connects. `version` is asserted even
though both sides come from one install: it stops a stale core left by a
crashed run from being attached to later.

`coreproc.HANDSHAKE_VERSION` is deliberately a separate constant from
`protocol.PROTOCOL_VERSION` — the supervisor must not import the protocol,
since it has to be able to reap a core that cannot speak it. The two must be
bumped in lockstep; wave 3 owes a test asserting they agree.

**Terminal hygiene.** The core must never write a byte to the inherited tty; a
stray traceback corrupts the TUI's screen. stderr is redirected to
`<app_dir>/core.log` at spawn, and stdout is used only for the handshake line
and then closed.

**Shutdown, UI-initiated.** UI sends `shutdown` → core stops accepting work,
lets in-flight turns finish or cancels them (§4.4), syncs the DbCache, releases
the lease, exits 0. The UI waits up to 10s, then SIGTERM, then 5s, then SIGKILL.

**Shutdown, UI died.** The core watches its client connections. On EOF with no
reconnect inside a grace window (default 60s) it performs the same clean
shutdown. This covers a SIGKILLed UI, and is what makes `--attach` trivial
later.

**Lease.** The core takes `db.lease` (`dbcache.py:47`). The UI opens no
database at all. That is a better invariant than today's, where the lease pid
is the UI's.

## 3. Transport

`AF_UNIX` SOCK_STREAM with newline-delimited JSON, via
`asyncio.start_unix_server` / `asyncio.open_unix_connection`. No new
dependency.

**Socket location.** Derived from `dbcache.local_root()`, which already encodes
the "never NFS" reasoning — and Unix sockets on NFS do not work at all.

Two constraints that bite:

- **`sun_path` is 108 bytes.** Slurm `$TMPDIR` can be long. The name must be
  short: `<local_root>/hpca-<first 8 hex of sha256(app_dir)>/core.sock`. If the
  resulting path still exceeds 100 bytes, fall back to
  `/tmp/hpca-<8hex>/core.sock` and log the fallback.
- **This socket is a remote-code-execution endpoint** — `turn.submit` runs
  `run_bash`. On a shared login node it must be mode `0600` inside a `0700`
  directory, both created before bind, and the directory ownership checked on
  reuse. Not the Linux abstract namespace, which has no permissions at all.

A stale socket file from a crashed core is removed before bind only after
confirming no live process is listening (connect, expect `ECONNREFUSED`).
`coreproc.clear_stale_socket()` is that check, and calling it is **mandatory**
before `transport.serve_unix()`: asyncio's own unix-server setup unlinks any
existing path that stats as a socket, unconditionally and without asking
whether someone is listening on it. `EADDRINUSE` therefore never surfaces, and
a second core would silently steal the path from a running one — two cores on
one app dir, fighting over the sqlite lease.

Wave 3's shutdown path must also close the accepted connections *before*
awaiting `Server.wait_closed()`: since 3.12.1 that waits for every client
transport to drop, so a core that awaits it while a handler is still parked on
a live client hangs instead of exiting.

## 4. Protocol

One envelope in both directions:

```json
{"seq": 12, "id": "c7", "type": "turn.submit", "payload": {...}}
```

- `seq` — monotonic per sender, starting at 1. Cheap now, expensive to
  retrofit, and it is what any future reconnect needs.
- `id` — correlation id set by the sender of a command; a core reply that
  answers one command echoes it in `payload.reply_to`. Most commands are
  answered by ordinary events, not replies.
- `type` — dotted name, see below.

### 4.1 Commands (UI → core)

| type | payload |
|---|---|
| `session.list` | — |
| `session.new` | `profile`, `backend?` |
| `session.open` | `session_id` |
| `session.close` | — |
| `session.rename` | `session_id`, `title` |
| `session.retitle` | `session_id` |
| `session.delete` | `session_id` |
| `session.fork` | `session_id`, `index` — branch the conversation into a new session, cut before that entry; answered by `session.created`. The *entry index* crosses, not the `keep` count `fork_thread` takes: the core issued the index and is the only side that can still say whether it means what the user saw once a turn has appended. Deliberately allowed while the source is busy (see `session.rollback`) |
| `session.rollback` | `session_id`, `index` — trim the conversation to before that entry, in place. Refused with a warning `notify` while a turn is running, a decision is unanswered or a message is queued (`TurnScheduler.rewind_blocker`). Gated where the fork is not because the consequences differ: a fork at a stale cut point leaves a session the user deletes, a rollback at one destroys messages that cannot come back |
| `session.focus` | `session_id \| null` — which session the user is looking at (§4.4) |
| `turn.submit` | `session_id`, `text`, `forced_skill?` |
| `turn.interrupt` | `session_id` |
| `turn.unqueue` | `session_id`, `seq` — take a typed-ahead message back out of the queue, named by the `Entry.seq` of the `queued` row the core drew for it. Not a queue position: the turn ahead can finish while the user is deciding, and a position would then quietly name the neighbour |
| `decision.resolve` | `session_id`, `approved: bool`, `reason: str` |
| `command.run` | `name`, `args`, `session_id?` — `/compact`, `/memorize`, `/conclude`, `/skill-*`. Session-scoped commands carry the id explicitly rather than letting the core infer it from the last `session.focus`, which may have moved on between the keystroke and the frame arriving |
| `confirm.resolve` | `id`, `confirmed` — answers a `confirm.requested`; the core holds the continuation, only the yes/no crosses |
| `memory.resolve` | `session_id`, `approved: [bool]` — answers a `memory.proposals` offer, positionally. The core holds the proposal objects; only the yes/no crosses, so a front-end cannot smuggle an edited memory back in an approval |
| `mode.set` | `session_id`, `mode` |
| `thinking.set` | `session_id`, `effort` — the other per-session dial (§3.6); a plain string for the same reason as `mode`, since which levels exist is the served model's business |
| `backend.set` | `backend` (JSON blob), `session_id?` |
| `profile.set` | `name` |
| `profile.save` | `name`, `kind` (`memories\|archive`), `text` |
| `profile.create` / `profile.delete` / `profile.duplicate` | `name`, `source?` |
| `skill.save` / `skill.delete` | `profile`, `name`, `text?` |
| `process.kill` | `pid` |
| `job.cancel` | `job_id` |
| `watch.peek` | `watch_id` — the tail of a watched log, or a job's state; answered by `watch.peeked`. The only read-only command here, and still a command: the file is on a node the UI may not share (§4.2 rule 2) |
| `watch.drop` | `watch_id` |
| `shutdown` | — |

### 4.2 Events (core → UI)

| type | payload |
|---|---|
| `hello` | `version`, `profile`, `settings_digest`, `display` — first frame on connect. `display` is the carve-out the digest implies: a front-end may not read the settings file (§4.2 rule 2), but a handful of keys are about nothing except what a frame looks like, and a digest cannot answer "draw this how". Those keys, and strictly those, ride the first frame so nothing is drawn before they land |
| `display.settings` | `display` — the same payload again, because it has just been edited. `hello` alone would make these restart-only, which is the toast `settings.save` exists to stop printing; its own event rather than a re-sent `hello` because that one is a per-subscriber handshake and re-sending it would have every attached front-end re-run its version check and re-ask for the sidebar |
| `session.rows` | `rows: [{session_id, title, profile, mode, last_active, flags}]` — `last_active` is when something last happened in that conversation, ISO-8601 **UTC**: the core and the front-end need not share a machine, so the wire carries the instant and the UI decides whose clock to write it in (`ui.state.when`). Stored on the session rather than read off the thread, because the sidebar draws every row at once and answering it from the transcript would mean opening every conversation to paint a list — deliberately **not** `session.list`: `parse()` sees a frame without knowing which direction it travelled, so one type string cannot carry two payload shapes |
| `session.created` | `row: SessionRow` — a session the core just made and the UI is expected to open: the reply to `session.fork`, and what `session.new` needs too. The whole row, so one frame both opens it and fills the sidebar; `session.rows` re-states the sidebar but cannot say which line is new |
| `chat.reset` | `session_id`, `entries: [Entry]` — on open only; also re-bases the row numbering (see `chat.update`). Each `Entry` carries `at`: when the message it reads was added, ISO-8601 UTC, stamped once by the reducer that appends it (`agent.graph._append_messages`) and read back by `transcript.build_entries`. Empty for the entries that are not one message — a `thinking` box folds several and cannot honestly name an instant |
| `chat.append` | `session_id`, `entry: Entry` |
| `chat.update` | `session_id`, `entry: Entry` — a row already on screen, revised in place; `Entry.seq` says which. Without it the chat is not append-only: a tool call that gains its result, and a `queued` entry becoming a `user` one, could only be expressed by resending the transcript — the per-turn rebuild this protocol exists to delete |
| `turn.started` | `session_id` |
| `turn.activity` | `session_id`, `activity`, `started_at` |
| `turn.usage` | `session_id`, `prompt_tokens`, `max_model_len?` |
| `turn.finished` | `session_id`, `reply?` |
| `turn.failed` | `session_id`, `error` |
| `turn.unqueued` | `session_id`, `seq`, `text` — a queued message was taken back: drop that row, and the text returns to the entry box, where an interrupt would also have left it |
| `decision.requested` | `session_id`, `payload` (the graph interrupt value) |
| `decision.cleared` | `session_id` |
| `panel.update` | `profile`, `session_id?`, `rows: [PanelRow]` |
| `watch.peeked` | `watch_id`, `title`, `text` — the answer to a `watch.peek`. Deliberately not a `notify`: it answers a keypress (so it echoes `reply_to`), two peeks can cross so the answer must name its box, and how long a tail stays on screen is the renderer's decision, not a timeout the core sets |
| `memory.proposals` | `session_id`, `proposals` |
| `confirm.requested` | `id`, `question` — a yes/no that is not a tool approval (triage offering a learned log signature). Deliberately not `decision.requested`: nothing is parked on it, and conflating them would make an unanswered offer look like a stalled session |
| `context.estimate` | `session_id`, `used`, `window` |
| `notify` | `severity`, `text` |

Three properties do the real work:

1. **Snapshot then delta.** `session.open` yields one `chat.reset` with the
   full entry list, then `chat.append` per new entry and `chat.update` per
   revised one, each row addressed by the `Entry.seq` the core gave it. This is where the
   rebuild-the-whole-chat-per-turn cost at `tui/app.py:4095` dies. If it is not
   in the protocol from the start, the whole history crosses the socket every
   turn and the result is slower than today.
2. **The UI never reads the database.** That single invariant is what deletes
   the poll timers, `DbIO`, and the `_is_active_session` checks from agent code.
3. **Events are addressed.** Every event names its `session_id`. The core does
   not know or care which session is on screen; the UI drops what it is not
   showing.

### 4.3 Entry and PanelRow

`Entry` is the existing `transcript.Entry` (kind, text, parts) serialised as a
dict, plus one wire-only field: `seq`, the core-assigned per-session row name
that `chat.update` addresses. It has no transcript twin because the transcript
has no notion of a row being revised — it is rebuilt, which is the cost this
protocol removes. `seq` is not `index`: `index` says which thread *message* an
entry is (−1 for the many that are none), `seq` names the row on screen. `PanelRow` is the existing `tui/app.py:242` dataclass minus its widget
payloads: `{key, text, classes, title, kind, ref}` where `ref` is the pid, job
id or watch id the UI needs for `process.kill` / `job.cancel` / `watch.drop`.

### 4.4 The three hard boundary cases

**Approvals.** The graph parks at `interrupt()` and the resume value must cross
the socket. Split the state: the *decision* moves to the core (it is session
state), the *half-typed refusal reason* stays in the UI (it is a draft, same
class as `_drafts`). This fixes a live latent bug — `_pending_decision`
(`tui/app.py:1288`) is UI-process memory, and `open_session` surfaces a parked
decision only from that dict (`tui/app.py:3373`), so today a restart leaves a
session parked on an interrupt with nothing on screen to answer it. A
core-owned decision is re-emitted on subscribe.

**`_deliver_event`'s policy leak.** `tui/app.py:3940` decides *whether the
model runs at all* from whether the session is on screen. The core cannot know
that. Keep the behaviour and make the dependency explicit: the UI sends
`session.focus`. Turning an implicit coupling into one named command is the
point of the exercise — do not silently change the semantics instead.

**`suspend()` for `$EDITOR`** (`tui/app.py:2899`) and the OSC-52 clipboard stay
UI-side. The editor writes a file the core owns, so the flow ends with
`profile.save` or a reload command.

## 5. Module layout

New:

| module | contents |
|---|---|
| `hpca/protocol.py` | envelope, command/event models, NDJSON codec |
| `hpca/transport.py` | `Connection` interface, `InProcessConnection`, socket client + server |
| `hpca/coreproc.py` | socket path derivation, `CoreSupervisor` (spawn/handshake/stop), core-side `IdleShutdown` |
| `hpca/core/service.py` | `AgentService` — the extracted runtime |
| `hpca/core/scheduler.py` | the turn scheduler lifted out of `drain_work` |
| `hpca/core/pollers.py` | the 2s/30s pollers, emitting `panel.update` |
| `hpca/core/backends.py` | LLM catalog, per-session clients, autoconnect, probes |
| `hpca/looplag.py` | event-loop lag probe (measurement, §7) |

Changed: `hpca/__main__.py` gains `--serve`, `--socket`, `--no-fork`;
`hpca/tui/app.py` shrinks to rendering.

### 5.1 Pinned interfaces

These are contracts between work packages. Implementations must match exactly.

```python
# hpca/protocol.py
PROTOCOL_VERSION: int = 1

class Envelope(BaseModel):
    seq: int = 0
    id: str | None = None
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)

def encode(env: Envelope) -> bytes:   # one NDJSON line, trailing b"\n"
def decode(line: bytes | str) -> Envelope:   # raises ProtocolError on bad input

class ProtocolError(Exception): ...

# Typed bodies. Every Command/Event subclass carries TYPE and round-trips
# through Envelope via .to_envelope() / .from_envelope().
class Message(BaseModel):
    TYPE: ClassVar[str]
    def to_envelope(self, *, seq: int = 0, id: str | None = None) -> Envelope: ...
    @classmethod
    def from_envelope(cls, env: Envelope) -> "Message": ...

COMMANDS: dict[str, type[Message]]   # type -> model
EVENTS: dict[str, type[Message]]
def parse(env: Envelope) -> Message:  # looks in both tables, raises ProtocolError
```

```python
# hpca/transport.py
class Connection(Protocol):
    async def send(self, msg: Message) -> None: ...
    def __aiter__(self) -> AsyncIterator[Envelope]: ...
    async def close(self) -> None: ...

class InProcessConnection:
    """Two paired queues; .peer is the other end. Used by tests and --no-fork."""
    @classmethod
    def pair(cls) -> tuple["InProcessConnection", "InProcessConnection"]: ...

async def connect_unix(path: Path) -> Connection: ...
async def serve_unix(path: Path, handler: Callable[[Connection], Awaitable[None]]) -> Server: ...
```

```python
# hpca/coreproc.py
def socket_path_for(app_dir: Path, *, local_root: Path | None = None) -> Path: ...

class CoreSupervisor:
    def __init__(self, *, app_dir: Path, argv0: list[str] | None = None,
                 log_path: Path | None = None, ready_timeout_s: float = 15.0): ...
    async def start(self) -> Path:      # spawn, read handshake, return socket path
    async def stop(self, *, timeout_s: float = 10.0) -> int | None
    @property
    def pid(self) -> int | None: ...
    @property
    def returncode(self) -> int | None: ...

class IdleShutdown:
    """Core-side: fires `on_idle` when the last client has been gone for
    `grace_s`. Reset by every new connection."""
    def __init__(self, grace_s: float, on_idle: Callable[[], Awaitable[None]]): ...
    def client_connected(self) -> None: ...
    def client_disconnected(self) -> None: ...
```

## 6. Sequencing

**Wave 1 — infrastructure (parallel, all new files, nothing existing edited).**
`protocol.py`, `transport.py`, `coreproc.py`, `looplag.py`. Each with tests.
**Done** (`5e1e6ad`, `eb16b45`, `c3a7e81`, `4ebd7b3`), plus the probe wired
into the TUI behind `$HPCA_LOOPLAG` (`85d0a72`).

**Wave 2 — extraction, in-process.** `AgentService` gains everything listed in
§7's "moves" column; `HpcaApp` keeps method names as thin forwarders so the 34
`test_tui_*` files stay green. No transport yet: the UI holds an
`InProcessConnection`. This is the whole job; the socket afterwards is
mechanical.

*Done, except the surgery.* The four services (`3eff9ec`, `583bedc`,
`750bee1`, `58bf34e`, `9bcefc1`) and their composition (`04d41e7`) exist and
are proven headless: `test_core_service.py` drives a real graph through
protocol commands with no terminal, and `test_core_headless.py` enforces that
no module in the package can import Textual.

**Wave 2b — the surgery.** `HpcaApp` still owns its own copy of everything;
nothing under `tui/` calls the core yet. What this step is:

- `on_mount` builds an `AgentService` instead of a graph, stores, registry and
  clients. Delete the duplicated construction, *not* the method names.
- Every extracted method becomes a forwarder. Keep the names: 34 `test_tui_*`
  files call them, and refactoring the code and its tests at the same time
  leaves nothing green in between.
- ~~`format_started` and `PROCESS_HISTORY_LIMIT` now exist in both `tui/app.py`
  and `core/pollers.py`, and `test_tui_processes.py` imports them from the
  former. Re-export, do not delete.~~ Moot: the right column lost its process
  and job rows (project.md §3.3), so both names and that test module are gone
  from either side. The panel is watch boxes only, in the store's order.
- `on_mount`'s `EmbeddingClient` and `on_unmount`'s `embedder.close()` must go:
  `BackendRegistry` owns that client now, and leaving both is a double close.
- Wire the UI's existing `ConfirmScreen` to `confirm.requested`, or triage
  stops offering to learn log signatures.
- `_paint_panel` renders `panel.update` instead of polling; `_set_chat_messages`
  becomes `chat.reset` + `chat.append` (§4.3), which is the last row of §1's
  latency table.

**Wave 3 — wiring.** `--serve` entry point, `CoreSupervisor` spawn from the UI,
`--no-fork` for debugging, shutdown paths.

**Wave 4 — the payoff.** Incremental `chat.append` rendering; `asyncio.to_thread`
for the synchronous tool bodies; delete `DbIO` from the UI side.

## 7. What moves, what stays

| To the core | Stays in the UI |
|---|---|
| graph, tools, `ToolContext`, `ProcessRunner`, checkpointer | widgets, CSS, keybindings, `check_action` |
| sqlite, `DbCache`, all stores, RAG, embeddings, symbols | `_drafts`, `_thinking_expanded`, reason drafts |
| LLM catalog, per-session clients, autoconnect, probes | `_paint_panel` diffing, `WorkingIndicator` |
| turn scheduler, `_pending_work`, `_turns`, approvals | clipboard, `suspend()`, `notify()` toasts |
| memory, curator, reflection, titler, compaction | command menu / autocomplete |
| all pollers, transcript logs, episodic index | |

Simplifications this unlocks, to be claimed only once measured or observed:

- `DbIO` (`db.py:182`) exists to keep sqlite off the *UI* loop; a UI with no
  database has no such loop to protect. It becomes an optimisation in the core,
  not a correctness requirement.
- 14 uses of `_is_active_session` leave the agent paths.
- `show_working` / `hide_working` / `report_activity` / `_backend_working`
  collapse into one `turn.activity` event. `TurnState.started`
  (`tui/app.py:329`) stops being a workaround for a rebuilt widget.
- `runner.py` carries five workarounds for "the TUI builds a fresh runner per
  turn" (`:179`, `:208`, `:289`, `:317`, `:481`). A core with a long-lived
  per-session runtime removes most of them. `list_processes` and
  `reconcile_orphans` stay — the table is still the record across restarts.
- A Textual exception can no longer kill a running turn.

## 8. Measurement

Nothing here is allowed to be justified by feel. `hpca/looplag.py` records
event-loop scheduling delay (a 100 ms timer, measuring drift) into
`<app_dir>/looplag.log`, with a percentile summary on exit. Take a baseline on
`main` under a representative load (a turn that runs `read_file` on a large
file, an `index_docs` over a real tree) before Wave 2 lands, and the same
afterwards.

## 9. Testing

Everything stays hermetic and in the default suite: no real sockets to the
outside, no LLM, no cluster. Socket tests use a `tmp_path` socket and a real
`asyncio` server — that is fast and local. `CoreSupervisor` tests spawn a
*stub* core (a tiny python `-c` script that prints a handshake line and
sleeps), never the real one, so the suite never pays for a langgraph import.

Follow the existing convention: TDD, and comments that explain *why* a choice
was made rather than restating the code.
