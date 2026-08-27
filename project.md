# HolisticProcessingComputeAgent (HPCA)

## 1. General

HPCA is a terminal-based AI agent that uses local or remote LLMs to help users of our
HPC Slurm cluster with biomedical data processing and data juggling. It runs on a
compute node, talks to an LLM served elsewhere (typically a GPU node reached through an
SSH-tunneled port), and gives the user a three-column TUI to chat, browse past
sessions, and supervise running sub-processes and cluster jobs.

**Guiding constraint:** the backend LLM is assumed to be a *small* local model
(e.g. Qwen-3 class). Every design decision follows from this:

1. **Deterministic middleware does the hard work.** Parsing Slurm output, triaging
   logs, validating paths, dry-running scripts — all done in Python code, never by the
   model. The model explains, decides, and suggests; code observes and verifies.
2. **The model never reproduces literal paths.** All paths/URIs live in a path
   working directory; tools accept paths.
3. **Retry loops with validation feedback.** Malformed or semantically invalid tool
   calls are caught by pydantic validation; the error message is fed back to the model
   for a bounded number of retries.
4. **Human-in-the-loop for anything destructive.** Always. The sole opt-out
   is the explicitly user-chosen full-auto mode (§3.5), where the recovery
   layer (§5.3) remains the safety net.
5. **Grounded answers over recall.** Technical claims about tools, APIs, and flags
   come from indexed docs/source via the doc-researcher subagent, or are explicitly
   marked as ungrounded; generated scripts are verified against the same index
   before execution (§5.2).

## 2. Deployment model

* The agent process runs **on a compute node**, started inside a **tmux, screen, or
  zellij** session (lost SSH connections to the cluster are therefore not our problem).
* The LLM is served anywhere reachable via a **port the user SSH-tunnels** (GPU node,
  login node, laptop, remote provider). HPCA only needs a `base_url` (+ optional API
  key).
* **Standardize on the OpenAI-compatible chat completions API.** vLLM,
  llama.cpp-server and commercial providers all speak it, so "local or remote
  LLM" is a single config switch, one code path.
* Prefer backends that support **structured output / JSON-schema-constrained
  decoding** (vLLM `guided_json`, llama.cpp grammars). Use it for all
  tool calls so they are syntactically valid *by construction*; retries then only
  handle semantic errors.
* **Open question to verify before building the job tools:** does our site permit
  `sbatch`/`squeue`/`sacct` from compute nodes? If submission is restricted to login
  nodes, the job-runner tool must wrap its commands in an SSH hop to a submit host.
  The tool interface must hide this behind a `submit_host` setting (`null` = run
  locally).

## 3. Interface (TUI)

### 3.1 Framework

**None.** The UI is a few thousand lines that turn state into a list of strings and
write the ones that changed, over asyncio and the standard library.

It was built on **Textual** until v0.26.0, for reasons that were good at the time: a
real layout system for the three-column design, CSS-like styling, `Header`/`Footer`
widgets that render the hotkey bar for free, modal screens, and an async event loop
so streaming output never freezes the UI. What replaced it keeps every one of those
properties except the styling, and the styling was never load-bearing.

The reason for the move is measured, and written up in
[specs/specs-ui-baseline.md](specs/specs-ui-baseline.md): Textual's *repaint* is flat and
perfectly fine, but its *arrange* pass is O(conversation) — 2.9 ms at 100 chat
entries, 126 ms at 5000, and 994 ms at the 95th percentile, which is a second of
frozen terminal on a keypress. A scroll invalidates layout, so a long session pays
it on every arrow key. Cold-opening a 5000-entry session took 15 seconds, because it
mounts one widget per entry. The replacement is flat at ~0.19 ms from 100 entries to
20,000, because only the visible slice is ever turned into lines.

Two consequences worth stating plainly, because they are the cost side: there is no
style cascade and no arrange pass, so anything the layout does it does explicitly;
and cell widths, escape decoding and the alternate screen are ours to get right
rather than a dependency's. §3.2 onward is that layout; `ui/ansi.py` is the width
and escape handling.

Dropping into an external editor no longer needs `App.suspend()` — the UI restores
the terminal, runs the editor, and takes it back (see §6.4).

### 3.2 Layout

Terminals in 2026 are assumed wider than 80 columns

```
┌──────────────────────────────────────────────────────────────┐
│ Top bar: app name | profile | model | (c) config editor      │
├───────────────┬───────────────────────────┬──────────────────┤
│ LEFT          │ CENTER                    │ RIGHT            │
│ Sessions      │ Chat window of the        │ Watchers: the    │
│ (past +       │ selected session          │ logs and jobs    │
│  current)     │ (scrollable message list) │ this session     │
│               │                           │ asked to be      │
│               │                           │ shown            │
├───────────────┴───────────────────────────┴──────────────────┤
│ Bottom bar: context-sensitive hotkeys, e.g. (i) inspect ...  │
└───────────────┴───────────────────────────┴──────────────────┘
```

### 3.3 Controls

* **← / →** switch focus between the three columns.
* **↑ / ↓** move through the vertically listed elements of the focused column.
* **Enter** starts the interaction with the selected element.
* **ESC** ends the current interaction / closes modal / returns focus.
* Interactions are resolved via modal prompts or the bottom hotkey bar:
  * *Session (left column):* open into the center chat window.
  * *Chat message (center):* **Enter** on one of your own messages opens the
    rewind dialog — the way a conversation is trimmed once the agent has gone
    in an unwanted direction. **(f)** forks the session from just before that
    message (the original stays whole, its turn can even keep running);
    **(r)** rolls this conversation back to just before it (refused while a
    turn runs, a decision is pending, or messages are queued — those all write
    to the thread being cut); **(c / Enter)** copies its text back into the
    entry — appended on its own line to whatever is already being written, so
    nothing typed is lost — which is how a command is re-sent with one word
    changed, and Enter-Enter keeps the old copy reflex working. Fork and
    rollback both hand the message back to the entry, ready to re-edit; both
    drop a compaction summary that covered trimmed messages (the raw history
    it stood for is still there, so folding can be redone). A queued message
    is not in the thread yet — nothing to trim — so Enter on it offers the
    take-back instead: **(x)** cancels it, dropping it from the queue and the
    log and handing the text back to the entry exactly as an interrupt does,
    and **(c / Enter)** copies it and leaves it queued.
    Enter on a thinking box expands
    it into its parts — each block of reasoning, and each tool *exchange* — every
    one its own collapsible row, so the script the agent wrote and the call it
    made stay readable long after the approval prompt that showed them is
    answered (and are there at all in auto mode, where no prompt showed them).
    One exchange is one row: the call on top, with the script or command or diff
    it would actually run, and the result underneath it. They used to be two
    rows, which put the tool's name on the screen twice and left the reader
    scrolling between a question and its answer. Those rows do not wait for the
    turn to end: the call appears the moment it is made, above the working line,
    and the row *fills in* when the result lands rather than a second one
    arriving below it — so a turn that spends minutes in tools shows *what* it is
    doing while it does it, can be opened and read while it runs, and does not
    move under the cursor when it finishes. The turn's last act is to fold them
    into its box.
    What a row shows is what happened, not what the agent was told about it. A
    tool result carries both: "Created x.tsv (8 lines)" is news, and "Change it
    with edit_file, not by writing it again" is an instruction for the model's
    next call. The second kind is stripped for display — the sentences live in
    `hpca.agent.hints`, which is where the tools themselves get them, so the two
    cannot drift apart — along with the `[tool result] <tool>:` framing, which
    the row's own header already says. Arguments are shown the same way: one
    `key: value` line each rather than indented JSON, minus the payload the
    script block below already is, and minus the call altogether when the tool
    resolved it into real paths itself (`edit /work/x.tsv (replace 3 lines with
    5)` says everything the raw arguments would, better).
    Enter on anything else (the agent's replies, background events, recalled
    memory) simply hands focus to the entry. `(c)` copy content to clipboard.
  * *Working line (center, while a turn runs):* names the step in flight and
    counts up from the moment the turn started. **Enter** on it asks whether to
    interrupt the turn and hand the message back to the entry for editing — in
    any phase, waiting on the model or running a tool. It ends the turn, not
    the work already started: a script keeps running under its own monitor
    until it exits or times out. It is the *last* line of the log for as long
    as a turn runs — tool rows, typed-ahead messages and background events are
    written above it — because "go to the end of the log and press enter" is
    how the user stops the agent, and a row taking that place turned the stop
    key into something else. It also survives an approval: answering a
    destructive-op prompt starts a fresh turn on the same exchange, which
    inherits the message and rollback point of the one that parked, so the
    second half of an approved turn is as stoppable as the first. Enter on a
    line that genuinely cannot be stopped (a silent backend call such as
    `/conclude`) says so rather than doing nothing.
    **Escape twice** (within a second) does the same abort from anywhere on the
    main screen, with no dialog — the doubling is the confirmation. It is the
    route that needs no aiming: enter has to land on this one row, which is the
    awkward thing to do exactly when the agent is misbehaving and filling the
    log, whereas the user's hands are already wherever they are. One escape on
    its own still means nothing (terminals emit it as the prefix of arrow keys
    and pastes), and on a modal or the approval bar escape keeps meaning "leave
    this prompt" — the binding is deliberately not a priority one, so anything
    nearer the user claims the key first. Because it cancels the turn's worker,
    it lands wherever the turn is waiting: on the model, in a tool, or on the
    next attempt of the decision retry loop (§4.3) — the one place a turn can
    spin without ever asking the user anything.
  * *Watch (right column):* a box the user asked for, pinned by the agent's
    `watch_log` / `watch_job` tools — see *Watches* below. **Enter** flashes
    the last 300 characters of the log (a toast that expires, not a screen to
    dismiss), `(d)` stops watching and removes the box, `alt+↑` / `alt+↓` carry
    it past its neighbour. The log itself is never touched.
  * *Entry (message box):* **Enter** sends and `alt`/`shift`+**Enter** opens a
    new line; **⇧tab** changes the mode and `ctrl+l` the model; `ctrl+e` opens
    the draft in `$EDITOR`; `ctrl+u` empties it. **`ctrl+z` / `ctrl+y`** undo
    and redo inside it — a run of typing undoes a word at a time, a paste
    undoes whole, and a send starts the stack again, because a message already
    sent is reached through the history rather than by putting a copy of it
    back in the box. The same two keys work in every other buffer that takes
    typing (the config editor, a profile's learnings): the undo lives in
    `ui.editor.Editor`, which all three share.
    **↑ / ↓** walk this session's own messages, the way a shell walks its
    history — read out of the chat log itself, so there is no second copy and a
    session closed and reopened still has it. They step only from the *edges*:
    ↑ from the top screen line and ↓ from the bottom one, so a recalled
    multi-line message is still navigable with the arrows that recalled it. The
    draft in the box when the walk starts is stashed as the newest entry, so
    one ↓ brings it back verbatim and a stray ↑ mid-sentence costs nothing.
    Editing ends the walk. `/`-commands are not in it — only a real submission
    becomes a chat entry — and while a command is being named the `/` menu owns
    ↑/↓ as it always did.

The right column held the session's own run history too — every subprocess and
sbatch hpca had started, under a "── this session ──" heading, with **Enter** to
inspect and `(k)` to kill or cancel. That is gone. Every one of those calls is
already in the chat log one column to the left, so it was a second copy of what
the user had just read, and it grew without bound while the boxes they had
actually asked for were pushed off the bottom of a short terminal. Nothing else
changed: the `processes` table is still written and still wakes the agent when a
background script exits (§5.4), and a job worth keeping on screen is a job worth
`watch_job`. There is now no keyboard route to killing a local process from the
TUI; the agent still has one.

**Watches (right column, "Watchers").** What hpca started is only a slice of
what runs on a cluster, and it is a slice already in the chat log — so the
column's job is somebody *else's* work: an sbatch script submitted by hand, a
pipeline run spawning tool after tool, the log some long-running tool appends
to. Finding out whether any of that is alive otherwise means `squeue`, then ssh
to the node, then `tail`. The agent pins one to the column instead:

* `watch_log <path>` — a file. Its mtime is the signal: "last write 4s ago" is
  alive, "last write 40m ago" is dead or wedged. Registering a log that does
  not exist yet is normal (the job has not created it), and the box says so.
* `watch_job <id>` — refreshed from `squeue` every 15 s; once the job leaves the
  queue `sacct` supplies the final state, so the box settles on COMPLETED or
  FAILED rather than vanishing. A finished job stops being polled.
* `list_watches`, `unwatch <name>` — the same operations from the model's side.

A watch belongs to the **session** that registered it, not to the profile: the
column describes the conversation being read. It was profile-scoped originally,
on the reasoning that a watch describes the machine — but sessions on one
profile are the normal case, so every session showed every other session's
boxes. It survives a restart (table `watches`, see `hpca.watches`). Polling is
deliberately *not* scoped the same way: state is refreshed for every watch in
the store, so a session returned to shows a current clock rather than one frozen
at the moment the user switched away. Logs are stat'ed every 5 s on the DB
thread — on an NFS home the stat is the slow half, and a wedged filesystem must
cost a late repaint, never the event loop. A log going quiet, and any job state
change, raise a toast.

**Order is the user's.** Boxes come out in registration order until `alt+↑` /
`alt+↓` move one (column `watches.position`, `WatchStore.move`), and nothing a
poll learns may reorder them — the cursor lives in this list, and a column that
reshuffles itself every few seconds cannot be arrowed through. Sorting by
freshness or state was rejected for the same reason it cannot work: only the
reader knows which box matters, and it is not a property the code can compute.

**Reserved hotkeys — never bind these** (they are eaten or made unreliable by the
terminal, by zellij/tmux, or by the flow-control layer, so a future UI addition
must avoid them):

* `ctrl` + `q p t n h s o g` — `ctrl+s`/`ctrl+q` are terminal flow control
  (XOFF/XON: `ctrl+s` *freezes* output), and `ctrl+p`/`ctrl+t`/… are common
  multiplexer/zellij prefixes.
* `alt` + `n f`, the arrow keys, `+`, `-` — commonly grabbed by zellij/tmux.

One deliberate exception: `alt+↑` / `alt+↓` reorder the watchers column. It was
asked for by name, and it is the gesture every editor uses for move-a-line. The
reserved rule still holds — under zellij or tmux the keypress may never arrive —
so `shift+↑` / `shift+↓` are bound to the same action as a fallback, the same
belt-and-braces as `ChatInput.NEWLINE_KEYS`. Treat that pairing as the pattern
for any future binding that has to use a reserved key.

`ctrl+z` is *not* on the list, which is worth saying because it looks as though
it should be: the UI puts the terminal in raw mode, so ISIG is off and ^Z
arrives as a byte instead of suspending the process. It and `ctrl+y` are undo
and redo in the editors (§3.3).

Otherwise prefer a bare letter gated (via `check_action`) to a non-typing
column, or a safe `ctrl` combo (`ctrl+l`, `ctrl+e`, `ctrl+r`, …). Keep this list
in sync with the `RESERVED HOTKEYS` comment above `HpcaApp.BINDINGS`.
* **Config editor** `(c)`: edit the settings JSON, persisted to
  `~/.HolisticProcessingComputeAgent/settings.json`.
* **Profiles & learnings** `(a)`: manage profiles and their memories (see §6).
* **Chat commands** (typing `/` or `\` lists the built-in commands *and* the
  profile's skills; ↑/↓ select, `⇥` completes):
  * `/memorize [NOTE]` — the agent forms memories from NOTE plus the conversation
    so far and proposes them for approval (see §6).
  * `/conclude` — the agent analyses the conversation and proposes memories to write
    into the profile (user approves before write, see §6).
  * `/compact [BRIEF]` — fold the conversation into a summary now rather than
    waiting for the automatic fold, freeing the window. BRIEF is what the summary
    must carry: material to keep, or the step the user is about to take — the
    summary is then written for it, and the brief itself stays in the folded view
    so it keeps framing the turns that follow. The chat is not rewritten; only the
    view the model receives is folded.
    The summary is **reviewed before it lands**, because a fold cannot be undone:
    it comes back as a prompt in that conversation's own column — inline, where
    the message box was, so the conversation it summarizes is still on screen
    behind it and a summary offered in one session never covers another — and the
    answer is accept (Enter), *again* (`r`, with a line saying what it has to do
    differently — that sentence and the rejected text both steer the next
    attempt), or discard (`d`). Escape answers nothing and the offer is kept, so
    a bare `/compact` re-opens it rather than paying for a second summary. A
    summary waiting in a session the user is not reading marks its row in the
    sidebar and nothing else. The prompt also says when the model's own length
    budget cut the summary short, which is the complaint the *again* answer
    exists for.
  * `/skill-creator [WHAT IT SHOULD DO]`, `/skills-list`, `/skill-remove` — manage
    this profile's skills (see §5.1). With a description, the model drafts name,
    description and body from it (and the conversation so far) and the form opens
    pre-filled; bare, the form opens empty as before.
  * `/<skill> [PROMPT]` — invoke a user-defined skill directly: its procedure is
    handed to the model inline for that turn (see §5.1). Unlike the built-in
    commands, this is a real turn — queued and run like any message, not an
    exclusive UI worker.

**Typing is never blocked.** One turn runs at a time — two invocations on a single
`thread_id` would interleave checkpoint writes — but that is the orchestrator's
constraint, not the user's. A message sent while a turn is running is accepted,
shown in the transcript as `queued`, and started when the orchestrator frees up;
the entry field clears immediately, so the next thought can be typed while the
current one is still being answered. Queued work drains in arrival order, one item
per pass, and shares the queue with background completions (§5.4) so both go
through the same one-at-a-time discipline. A queued message can still be taken
back — Enter on it, then **(x)** — right up until the orchestrator starts it;
after that it is a turn, and stopping it is the interrupt's job (§ interrupt).
A session parked on an approval holds
only its own queued messages; other sessions keep draining. Slash commands are not
turns — they act on the UI and run their own exclusive workers — so they are still
refused while busy rather than queued.

### 3.4 Clipboard (must work under tmux, screen, and zellij)

Copying must never silently fail. Implement a `ClipboardManager` with a tiered
strategy; the primary mechanism is **OSC 52** (escape sequence instructing the user's
*local* terminal emulator to set the clipboard — the only mechanism that survives
compute-node → SSH → laptop). Multiplexers intercept escape sequences, so:

| Environment | Detection | OSC 52 handling | Guaranteed fallback |
|---|---|---|---|
| tmux | `$TMUX` set | native if `set-clipboard on`; else wrap in passthrough `\ePtmux;...\e\\` (needs `allow-passthrough on`, tmux ≥ 3.3) | `tmux set-buffer` (paste with prefix+]) |
| screen | `$STY` set | not understood; chunk into DCS passthrough blocks `\eP...\e\\` (flaky on some builds) | `screen -X register . <text>` + `readbuf` |
| zellij | `$ZELLIJ` set | forwarded by default in recent versions; user `copy_command` config may alter behavior | (no CLI buffer API — rely on OSC 52) |
| none | — | emit directly | — |

On every copy action the manager does **both**: (1) emit the (wrapped) OSC 52
sequence targeting the system clipboard, and (2) write the multiplexer's own paste
buffer where an API exists. Last-resort tier for any failure or oversized payload:
write to `~/.HolisticProcessingComputeAgent/clipboard.txt` and show a toast saying so.

Notes:
* Terminals cap OSC 52 payloads (commonly ~100 KB base64). Above
  `clipboard.osc52_limit_kb`, skip OSC 52 and go straight to the file fallback with a
  notice.
* OSC 52 is written as raw sequences, with the multiplexer wrapping tmux and screen
  need — a plain unwrapped OSC 52 is silently swallowed inside either. (This was
  originally a warning not to use Textual's `copy_to_clipboard`, which emitted
  exactly that unwrapped form.)
* README must state: tmux users need `set -g set-clipboard on` (or
  `allow-passthrough on`) for the system-clipboard path; the tmux buffer fallback
  works regardless.

Settings block:

```json
"clipboard": {
  "mode": "auto",
  "command": null,
  "osc52_limit_kb": 74
}
```

`mode`: `auto | tmux | screen | zellij | osc52 | command | file` — the pre-baked
switches. `command` (e.g. `"xclip -selection clipboard"`, `"wl-copy"`) covers exotic
setups: content is piped to the command's stdin.

### 3.5 Agent modes (manual / auto / full-auto)

Each session has an interaction mode, indicated on a single line directly above
the chat entry and cycled with **shift+tab** (ctrl+m is bound as well, but most
terminals deliver it as Enter, so it only works under keyboard protocols that
can tell them apart). The mode is stored per session (`sessions.mode`, empty =
the `agent.default_mode` setting, default `manual`) and read fresh every graph
round, so switching applies immediately — even to a turn already in flight.

* **manual** — every execution tool call (`start_background_script`,
  `run_bash`, `submit_job`) pauses at the same `interrupt()` gate as
  destructive operations; an inline approval bar at the foot of the chat column
  (`DecisionBar`, deliberately non-modal so the other columns and sessions stay
  visible) shows the actual script text and offers *run script* / *skip
  script*. A skip without a reason is fed back to the model as a
  SKIPPED tool result that forbids retrying, rephrasing, or reaching the same
  outcome another way (bare denials make small models re-propose the same
  command); a skip *with* one says the opposite — see the refusal box in §5.3.
* **auto** — structurally today's behaviour (only the §5.3 destructive gate),
  plus a prompt block telling the model that no confirmation will ever arrive,
  so it must work to completion instead of narrating and waiting for "do it".
* **full-auto** (shown as "full auto") — auto with the §5.3 destructive gate
  waived as well: nothing pauses for approval. This is the one deliberate,
  user-chosen exception to guiding constraint 4 ("human-in-the-loop for
  anything destructive"); the §5.3 recovery layer (hardlink trash, copy
  backups, TTL restore) still stands behind every deletion and overwrite,
  and the prompt tells the model to verify paths itself and to list every
  destructive action in its final report.
A fourth mode, **plan**, was removed. It withdrew `create_script`,
`start_background_script` and `submit_job` from the tools offered to
`decide()`, made the model hand its checklist over through a `present_plan`
tool, and let the inline decision bar start execution on auto (`ctrl+r`) or
step-by-step (`ctrl+e`). The shipped `/plan` skill (§5.1) does the job better:
it grills the user to a shared understanding first and writes the plan to a
`specs.md` that outlives the session, with no mode to enter and leave. Its
structural guarantee had also always been softer than it read — `run_bash`
stayed available and runs arbitrary bash, so a script registered in an earlier
turn could be executed by naming its path.

The `update_plan` checklist outlived the mode: the model maintains it in every
mode, it lives in the checkpointed graph state (`AgentState.plan`), and it is
re-injected into the system prompt every round, so it survives restarts and
context compaction — a prompt-only checklist is forgotten as soon as compaction
folds the instruction away.

Mode guidance is appended to the system prompt per render (§4.3), never stored,
and the per-mode gating decision is made in the graph's `execute_tool` node —
the same machinery as the destructive gate, with a different question.

### 3.6 Thinking effort (off / low / medium / xhigh)

The second per-session dial, with the same lifecycle as the mode: stored on the
session (`sessions.thinking`, empty = the `agent.default_thinking` setting,
default `off`), read fresh every graph round so a change reaches a turn already
in flight, chosen through the `/thinking` chooser, and always visible — it rides
the context bar at the top of the chat column, next to the fill and the
generation speed. Implementation in `hpca.thinking`.

`off` sends `enable_thinking: false` and **no** `reasoning_effort` at all, so a
backend that has never heard of the parameter sees exactly the request HPCA
always sent. The other three send both. The three levels are not ours to pick:
vLLM validates `reasoning_effort` against the served model's own enum, and
Qwen3.8 accepts only `low`, `medium` and `xhigh` — there is no `high` (a 400
says so), and with thinking on and no level given the server defaults to
`xhigh`, the slowest one. `thinking_budget`, which the chat template also
accepts, is silently a no-op.

Why per session rather than per request: mechanically the level is a prompt
injection at **position 0**, before HPCA's own system prompt. xhigh prepends a
"think carefully, validate key assumptions…" paragraph, low a two-line "keep it
brief", medium nothing at all — measured against the cluster's Qwen3.8-27B-FP8,
the same real decision costs 2659 prompt tokens at xhigh, 2647 at low and 2621
at medium. Two levels therefore differ from the first token onward, so changing
one throws away the session's whole prefix KV cache on a `--enable-prefix-caching`
server. It is a setting a conversation is put into, not a knob turned per call.

It replaced a per-**backend** `enable_thinking` toggle on the manage-LLMs
screen. That flag answered "does this model reason?", where the question a user
actually has is "how hard should *this conversation* think?" — and answering it
by editing the catalog changed every session on that backend at once. Old
settings files still carry the key on their catalog entries; it is ignored on
load.

The cost is real and is why `off` stays the default. On an idle backend an easy
decision took 2.7s off against 17.2s low, 14.9s medium and 33.9s xhigh. The
interesting measurement is the hard one — a multi-part diagnostic question, one
full agent decision (system prompt, tool listing, 4096 cap), backend at ~17
tok/s:

| level | decision | requests | outcome |
|---|---|---|---|
| `off` | 74s | 1 | answered (1284 completion tokens) |
| `low` | 234s | 1 | answered, but 4039 of the 4096 tokens went to thinking |
| `medium` | 411s | 2 | first request truncated at the cap; the retry produced a tool call |
| `xhigh` | 474s | 2 | **both requests truncated at the cap — the decision failed** |

Every decision of a turn pays this, not just the first. Three things follow,
and they are the operational content of the feature:

* **The 120s `request_timeout_s` does not size a thinking request.** Each of
  those generations ran ~237s — the cap divided by the decode rate — so all of
  them would have died mid-flight on an httpx read timeout, having already
  spent the time. The deadline is therefore raised per request when the request
  thinks (`LLMClient._timeout_for`, `THINKING_TIMEOUT_S = 600`), and left alone
  otherwise: 120s is also what makes a wedged backend fail fast, and every
  other call would pay for a blanket increase.
* **Thinking tokens come out of the decision cap, so the cap scales with the
  level.** `decision_cap(effort)` (§4.3) keeps the base 4096 at `off` and
  `low`, and applies ×1.5 at `medium` and ×2.0 at `xhigh`
  (`DECISION_TOKEN_SCALE`). This is a token limit, not a time limit, so a
  faster backend hits it the same way, only sooner.

  The multipliers are round numbers, chosen rather than fitted, and that is
  deliberate. The table above is one question: a second run of the same one
  finished at *every* level inside 1400 completion tokens, and a third — an
  easy decision — inside 400. What a decision costs is dominated by the task,
  not by the level, so a percentile fitted to any fixed task set would carry a
  precision it does not have. The intent is that the payload keeps roughly its
  own 4096 whatever the level; the scale factors buy that and nothing more is
  claimed for them.

  **And at `xhigh` they do not buy it.** Measured on the write that provokes
  the problem best — a 200-line `create_file`, which costs 3415–3732
  completion tokens with thinking off, i.e. ~85% of the base cap before a
  single thinking token:

  | level | flat 4096 | scaled |
  |---|---|---|
  | `off` | 200 lines, 197.5s | 200 lines, 215.8s (cap unchanged) |
  | `low` | 21-line skeleton, 271.0s | 6-line skeleton, 268.4s (cap unchanged) |
  | `xhigh` | **turn lost**, 473.4s | **turn lost**, 947.4s |

  Doubling `xhigh`'s budget did not rescue the turn; it spent the whole 8192
  twice and failed the same way, for twice the wall time. The reason is that
  thinking at this level is not sized by the task — it expands into whatever
  budget is available — so headroom handed to the decision is taken by the
  deliberation rather than left for the answer. `thinking_budget`, which would
  bound the two separately, is a no-op on this server (hpca.thinking), so
  there is no way to give the payload room that thinking cannot take.

  What the scaling does buy is a smaller gap for `medium`, and what it costs is
  exactly this: the cap also bounds how long a looping generation hangs, so
  ×2.0 doubles the price of an `xhigh` failure. The `low` rows are the
  reassuring ones — the cut-off-decision retry (§4.3) turns the same overrun
  into a skeleton to fill rather than a dead turn, which is the behaviour the
  feature is supposed to have.
* **`xhigh` is the slowest, and it is not flagged.** It is slowest by a wide
  margin — ~237s per thinking generation against 19.5s for the same decision at
  `off` — and a tool-heavy turn pays that on every round, so `off` remains the
  default. It is not, however, unusable: it was shipped for a while with a
  "⚠ NOT USABLE" flag in the chooser and a warning toast on selection, on the
  strength of the lost-write rows above, and use since has been good. The flag
  and the toast are gone (`hpca.thinking` no longer has `XHIGH_WARNING`); what
  the chooser carries now is a one-line hint per level, which is where the
  practical differences live: `off` suits simple tasks, `low` spends much of
  its deliberation on correction steps, `medium` is the quickest of the three
  that think, and `xhigh` overthinks everything.

## 4. Agent design

### 4.1 Framework

**LangGraph** (with `langchain-core`), *not* the legacy LangChain `AgentExecutor`.
LangGraph provides exactly the primitives required here:

* `interrupt()` → human-in-the-loop gates for destructive operations and `\conclude`
  approval.
* **Checkpointing** with the sqlite checkpointer → persistent, resumable sessions;
  the left "sessions" column is essentially a view over checkpoints.
* Explicit graph nodes → the retry/validation middleware is a normal node, not a
  monkey-patch.

### 4.2 Orchestrator and subagents

One **orchestrating agent** does the work, delegating a couple of narrowly-scoped
jobs to firewalled sub-loops. Each sub-loop gets a minimal system prompt and only
its own tools:

| Sub-loop | Purpose | Tools |
|---|---|---|
| doc-researcher | answer technical questions & verify API usage against man pages, docs, source (RAG) | `search_docs`, `lookup_symbol`, `read_manpage`, `read_source` |
| log-explainer | diagnose failing/finished jobs from triaged logs | (fed a structured triage report; no tools) |

**Implementation note (design vs. code).** The original design imagined a
router node dispatching to five distinct subagents (script-writer, job-runner,
log-explainer, doc-researcher, process-QA), each a separate agent. That router
was **not** built. What ships instead is a single orchestrator LangGraph loop
(`orchestrator` + `execute_tool` nodes) holding the whole tool registry, plus
**two** firewalled sub-loops that are invoked *as tools*, not routed to:
**doc-researcher** (reached via the `ask_docs` tool, `researcher.py`) and
**log-explainer** (run inside `get_job_report`, `explainer.py`). The
would-be script-writer, job-runner and process-QA "subagents" are simply regular
tools on the one orchestrator (`create_script`, `submit_job`, …). The
context-firewall contract below still holds for the two real sub-loops: raw
retrieval and raw logs never enter the orchestrator's context.

The orchestrator itself holds the full tool registry — file/script/job/doc/memory
tools plus `list_scripts` — and answers directly only for trivial
conversational queries (see the grounded answering policy below).

**Grounded answering policy.** Small models hallucinate API details, so the agent
is steered away from answering technical questions from model weights. The policy
is that any query naming an *external* program, library, API/function, file
format, or error message should go to the doc-researcher, and the orchestrator
answers directly only for trivial conversational turns. It explicitly does **not**
cover the agent's own tools: their schemas are already in its prompt, so
researching them before a call is pure overhead and is forbidden. The
hallucination risk lives in the command-line programs the agent drives from
generated scripts, not in its own toolbox. Doc-researcher answers are asked to
carry citations (source + section), and to mark an answer
`[ungrounded — not in indexed docs]` when retrieval finds nothing relevant.

**Enforcement is prompt-steered, not deterministic (design vs. code).** There is
no router node that inspects a query and *forces* the doc-researcher route:
"routing" happens only when the orchestrator chooses to call the `ask_docs` tool,
guided by prompt text (`GROUNDED_ANSWERING_GUIDANCE`). Likewise the citation
requirement and the `[ungrounded …]` marker are instructions to the sub-model, not
code-enforced backstops — the doc-researcher returns its text verbatim, with no
check that a citation is present or that the marker was added on empty retrieval.
(The log/process explainers *do* have a deterministic quote-or-admit backstop; the
doc-researcher does not.) So the policy is real but only as reliable as the small
model's compliance. Accepted cost: a technical question still takes ≥2 model calls
(the orchestrator turn plus the grounded sub-loop).

**Context firewall (subagent I/O contract).** Every subagent call follows a fixed
contract: the caller passes a focused question plus optional context references
(code excerpt, target library/tool, paths); the subagent may read as many
raw chunks, man pages, or source excerpts as it needs *inside its own context*, and
returns only a bounded, structured, cited answer. Raw retrieval results never enter
the orchestrator's or another subagent's context.

### 4.3 Middleware (deterministic layer around every model call)

* **Tool-call validation:** every tool has a pydantic schema. On validation failure,
  the pydantic error text is appended to the conversation and the model retries
  (bounded, `max_retries` in settings, default 3). Combined with constrained decoding
  (§2), retries handle semantic errors only.
* **Argument order:** array-valued arguments come **last** in every tool's schema,
  and nothing else may follow them. Constrained decoding cannot reject anything
  written inside a JSON *string*, so a model part-way through a long `list[str]`
  that decides it is done and starts on the next key emits `"timeout_s: 60"` as one
  more element instead — the array swallows the key and no error is raised anywhere.
  A live `run_bash` failed exactly this way (`line 98: timeout_s:: command not
  found`) after writing a ~100-line document through a heredoc. Ordering the schema
  leaves no key to reach for; the middleware also drops such an element when it
  appears anyway (last element, exact sibling field name, JSON value), and reports
  the repair with the call rather than silently shortening a script. Both are
  needed: measured against the live 27B backend *after* reordering, a `run_bash`
  heredoc still produced a trailing `timeout_s:` element in 1 of 4 generations —
  the model repeats the key out of habit, not because the grammar offered it.
* **Cut-off decisions:** a decision that stops at `max_tokens` is retried once with
  guidance to write the file in parts, rather than killing the turn (before this it
  raised out of the retry loop entirely — `LLMError` is not `DecisionError` — and
  ended the turn). The base cap is 4096: measured live at the default (thinking
  off), a document costs ~12 completion tokens per line, so 4096 holds ~330 lines,
  and raising it would mainly double how long a looping generation hangs. It is a
  base rather than a constant because thinking is spent from the same budget —
  `decision_cap(effort)` applies ×1.5 at `medium` and ×2.0 at `xhigh` so the
  payload keeps its own room whatever the level (§3.6).
* **Paths are paths** (`hpca.paths`). A tool argument that names a file is the
  path: `~` expands, a relative one is anchored at the session's working
  directory, and `..` is folded lexically so a file that does not exist yet
  still resolves. Nothing is stored and nothing is looked up.
  This replaced a **path registry** — a `{key → absolute path}` map per
  profile+session in sqlite, with `register_path` to mint a key and
  `dir_key`/`subpath`/`source_key` arguments to spend one. Its original premise
  ("the model mis-copies long paths") had already failed measurement: asked to
  read a 161-character cluster path, the 27B reproduced it byte-for-byte 20
  times out of 20. What kept it alive after that was **durable naming** — a key
  resolves from sqlite whatever is in the context window, while a long path
  that has scrolled out of a compacted conversation is gone.
  That argument is the one the removal had to answer, and it was answered by
  measuring it rather than by reasoning about it: see specs/specs-path-registry.md
  for the head-to-head at ~95k of context, which is where a key is supposed to
  win. What the registry cost in the meantime was constant and visible in the
  field — an `UnknownKeyError` listing keys the model never chose, in a session
  whose registry was empty, for a `create_file` whose `dir_key` was `"."`.
  A second vocabulary that no agent corpus the model was trained on contains is
  paid for on every call; durable naming is collected on the few where the path
  has scrolled away.
  What survives of naming is the one handle that was never a path: a **script's
  name**. `create_script(name=...)` writes `<name>.<suffix>` into the scripts
  dir, `{name}` in a run_bash line expands to it, and `start_background_script`
  / `submit_job` take it — resolved by looking in the directory, so there is
  still no table.
  `create_file` remains the one file tool that will build a mistyped path
  instead of failing on it — every other one needs its target to exist — so
  when it has to create the directory it says so as a **caution** naming the
  path, inviting a spelling check before ten more calls are built on the wrong
  tree.
* **The model sees its own actions.** A tool round appends the model's own call
  as an assistant turn and then the result, rather than the result alone — the
  shape agent-trained models are post-trained on. Without it the model
  reconstructed what it had done purely from the result text and re-issued
  calls it had already made. A create_file's content is already on disk, so
  carrying it in that copy as well spends the window on the file twice — but
  *when* it stops being carried is the whole design. Payloads are folded out by
  age, not at the moment of writing: the last few calls keep theirs, and older
  ones are replaced by a `<<HPCA: …>>` descriptor (`history.fold_old_payloads`,
  a view over a history that is never rewritten, exactly as compaction is).
  Folding at write time was a real bug — a model that reproduces its own record
  does it *immediately*, while it is still finishing the job that record
  belongs to, so it copied the descriptor back as `content_lines` and the
  descriptor went to disk; the file collapsed to what had survived, and folding
  *that* confirmed the loss on every retry. Deferring the fold removes the
  failure instead of guarding against it, because the record in reach is
  intact. The threshold moved with it (only genuinely large payloads are worth
  describing at all), and the file tools still refuse content carrying the
  sentinel — a few lines, no runtime cost, and the only thing standing between
  an already-checkpointed session and the old behaviour. Measured on the live
  27B (`second_file_after_first`, 10 reps): 0.8 success and two corrupted files
  before, 1.0 and none after, with fewer tool calls per task.
* **How a call travels** is `settings.llm.tool_protocol`, and there are two
  answers. `envelope` (default) is the hand-rolled decision object,
  `{"action": "tool_call", "tool": …, "arguments": …}`, held to an `anyOf`
  grammar by constrained decoding, with the tool list spelled out in prose
  because a grammar constrains syntax and cannot say which tools exist. It
  works on any OpenAI-compatible backend, including one too old to have a
  tool-call parser — and as of v0.22.0 that portability is no longer the only
  argument for it: re-measured on Qwen3.8 it is also cheaper, faster and the
  only channel that can salvage a cut-off long write (specs/specs-edit-eval.md
  §7.3, §7.4).
  `native` puts the call on the backend's own tool-calling channel: the branch
  choice becomes the backend's, the tool list moves out of the prompt into the
  `tools` array the chat template renders, and the exchange becomes a real
  assistant `tool_calls` message answered on the `tool` role. Argument
  validation does not move — the array constrains shape, not meaning, so a
  wrong key still returns through the same feedback loop, and on this channel
  the complaint answers the call on the tool role, because a template handed a
  call with no matching result renders a broken conversation.
  It is a setting rather than a probe, unlike constrained decoding: the
  protocol shapes the whole conversation rather than one request — the system
  prompt's respond-vs-tool guidance moves with it — so a verdict arriving
  mid-session would leave the prompt describing a format the model can no
  longer emit, which is the most expensive bug this area has had
  (specs/specs-edit-eval.md §7). It needs the server started for it
  (vLLM: `--enable-auto-tool-choice` plus a `--tool-call-parser` matching the
  model — `qwen3_coder` for the Qwen3.8 the cluster serves). A backend
  without it rejects the request with a 400 rather than degrading quietly, and
  the fix is `llm.tool_protocol`, or the per-entry `tool_protocol` override on
  a backend in the catalog when only *some* of the servers in reach are old.
  Because it can vary per backend, the two prompt builders ask the session's
  client (`middleware.uses_native_tools`) rather than the global setting —
  one source, so the prompt and the request can never disagree.
  The call's id is checkpointed with the pending call, not just held in
  memory: an approval parks the turn for as long as the user takes, and the id
  is the only thing tying the result to the call it answers.
  **A cut-off call is invisible on this channel**, which is the one thing the
  native protocol is genuinely worse at. The envelope raises `TruncatedOutput`
  on `finish_reason: "length"` and salvages the complete prefix of a long
  write. A tool parser instead reconstructs a call from whatever it managed to
  parse: measured on vLLM's `qwen3_coder` (2026-08-17), a `create_file` cut
  off inside its 200 `content_lines` comes back as a well-formed call of
  `{path}` alone — the array simply gone, `content` null, and
  `finish_reason` reading `"tool_calls"`. Nothing says truncation except
  `completion_tokens == max_tokens`. Read as a shape error it invites the same
  too-long write again, and the turn dies on the retry budget (reproduced
  live). So `middleware._hit_token_cap` reinterprets an argument-validation
  failure that spent the whole budget as truncation, and routes it to the
  skeleton-then-fill feedback instead. It is checked only *after* validation
  fails, so a complete call that merely ends at the cap is still kept. There
  is no salvage on this path — the fragment never arrives — which is why no
  tool may take an optional array argument: a dropped one has to fail
  validation, not execute silently as an empty list.
  Each tool's description carries a **filled example** of its arguments on
  both protocols. The `tools` array's JSON schema alone is not enough for a
  27B: with the schema only, `edit_file` came back with `old_lines` as a bare
  string instead of a list in 6 of 6 generations, and the model answered in
  prose rather than fix the shape on the retry. Same example source
  (`_example_args`) both ways, so the two cannot show different shapes.
  Measured head to head (specs/specs-edit-eval.md §7.2, n=36): the two are level on
  success, and fail differently — the envelope's grammar can loop until
  `max_tokens` (one run cost 224s and the turn), which is a tail the native
  channel structurally lacks; native spends more completion tokens for fewer
  tool calls.
* **Output size control:** long tool outputs are kept small before they reach the
  model. *As built,* this is per-tool truncation rather than a single generic
  middleware layer: `read_file` returns head/tail, `read_manpage`/`read_source`
  are bounded, and `run_bash` clips each stream to a tail and points at the full
  log file — registered by key at that moment, not on every run, so a check
  whose output fit leaves no key behind. There is **no** universal
  "nothing above N tokens ever passes" guard in `execute_tool`, so a tool that
  returns a large string directly (e.g. several full `search_docs` chunks) is not
  spilled to disk and summarized — a known gap versus this design.
* **Dry-run & verification gates:** see §5.2.
* **Destructive-op gate:** see §5.3.
* **Prompt assembly:** system prompts are rendered per call; current date/time and
  environment facts are injected dynamically (never stored as memories).

### 4.4 Self-reflection

If the agent fails at or struggles with a task (e.g. exhausts retries, user aborts,
job repeatedly fails), the orchestrator characterizes the problem in 1–2 sentences and
proposes a "struggle note" for the profile memory (user approves; stored in the
`rag` scope, §6.1). On future similar tasks — detected by a simple whole-word
keyword match against the profile's struggle notes — the agent warns the user up
front ("I have struggled with X before") and lets the user decide whether to
attempt it anyway.

## 5. Tools & safety harnesses

### 5.1 Tool suite (initial; extensible)

The tool suite must be a plugin-style registry so new tools can be added without
touching core code. Additionally, users can provide **skills**: user-defined
markdown/YAML files describing procedures the agent should follow for specific tasks,
loaded per profile (three levels — **project > profile > global** — with the most
specific winning a name collision).

**Skills stay out of the standing prompt.** Listing every skill each turn scaled
the prompt with the skill count and pulled the small model toward procedures the
user had not asked for. Instead the system prompt carries only a one-line note that
skills exist; a skill body reaches the model just two ways, cheapest first:

1. **Direct invocation** — the user types `/<skill> [PROMPT]`. The named skill's
   full procedure is dropped onto the *API copy* of that user message (the same
   sidecar that carries recalled memory and the volatile date, so the stored
   transcript keeps the clean `/<skill> …` and the cacheable prompt prefix is
   untouched). The turn is queued and run like any other message. Skills also
   appear in the `/` autocomplete menu next to the built-in commands; single-token
   names only (the parser splits on the first space), and a built-in command wins a
   name clash.
2. **On-demand fetch** — the model calls the `read_skill` tool itself when a
   request seems to match one (passing any name returns the available skills, so it
   can still discover them without the list being spent on every turn). The tool is
   registered whenever any skill exists anywhere reachable.

**Writing one is drafted, never automatic.** A blank creator form asks the user to
be an author on the spot, which is why most profiles have no skills; `/skill-creator
<what it should do>` instead spends one firewalled generation (`agent/skill_drafter`)
on a first draft — name, description, body — from the request plus the conversation
so far, and opens the *same* form pre-filled. The draft is a head start, not an
author: the fields are ordinary editable fields, escape still asks before writing,
and a backend that will not draft opens the empty form rather than eating the
command. The name is normalised to a kebab-case handle before it reaches the form,
since a name with spaces can never be invoked as `/<skill>`. Nothing here writes a
skill without the user reading it — the same rule as memory (§6).

Core tools:

* `create_script(kind: bash|python, name, content)` — writes the script,
  immediately syntax-checks it (§5.2), registers the path.
* `run_bash(timeout_s, content_lines)` — runs *and blocks*, returning captured
  output as the tool result. Writes a throwaway script, `bash -n`-checks it, and
  runs it through the same tracked runner (it is not a free-form shell — see
  below). A `{name}` in a line expands to that script's path, which is
  how a *kept* script is run synchronously: `{my_script} --flag`. Only keys that
  exist are substituted, so `awk '{print $1}'` and `${VAR}` survive untouched;
  an unmatched `{…}` is named in the result if the run then fails. Its script is
  bounded at 2000 characters — the same acknowledgement its output bound already
  makes, applied on the way in. Over that, the call is refused *in validation*, so
  it never becomes a pending call the user could be asked to approve, and the model
  is pointed at `create_file` (a document) or `create_script` (work to run). This is
  what keeps run_bash from being used as a file-writing tool, which is the one thing
  it is measurably bad at: asked for a design document with only run_bash available,
  the live model put all 5–8k characters into a single array element and skipped the
  §5.2 gate that `create_script` faces.
  *(The design had a second blocking tool, `run_script(name, args)`, for
  registered scripts. It was removed: splitting the two by where the script came
  from gave the model two tools advertising one job, while `{key}` expansion
  covers the case without carving an exception into the "keys, never paths"
  rule.)*
* `start_background_script(name, args)` — runs locally as a tracked **background**
  subprocess, stdout/stderr captured to files and
  registered; its completion is delivered back into the conversation (§5.4).
  The split from `run_bash` is *when the result arrives*, not what is run — the
  one choice the model cannot recover from on its own, which is why it is a
  separate tool rather than a flag.
* `submit_job(name, args)` — submits an sbatch script to the cluster,
  records the job ID and log paths in the job DB (§5.4), starts periodic tracking.
* `job_status(job_id)` / `get_job_report(job_id)` — structured status / triaged
  failure report (§5.5).
* `cancel_job(job_id)` — HITL-gated.
* `search_docs(query)`, `lookup_symbol(name, kind)`, `read_manpage(name)`,
  `read_source(path, range)`, `index_docs(...)`, `ask_docs(question)` —
  retrieval over man pages, tool documentation, and source code (§5.6):
  `lookup_symbol` is the exact-match path used by the verification gate (§5.2),
  `search_docs` the embedding path for prose questions, `ask_docs` the firewalled
  doc-researcher sub-loop (§4.2).
* File operations (`move_file`, `copy_file`, `delete_file`, `restore_file`,
  `read_file`, `create_file`, `edit_file`) — destructive ones gated per §5.3. There is **no**
  `list_dir` tool: kept scripts are listed by `list_scripts`, and directory contents
  are read via `run_bash`. `read_file` lists a directory rather than refusing it,
  which is the answer often enough to be worth not costing a second call.
* `create_file(path, content_lines)` — writes a **new** text file at that
  path, one array element per line. It exists because prose had no
  tool: `create_script` writes only into the scripts dir under a language suffix and
  `edit_file` needs a file to already exist, so authoring a `specs.md` meant a
  `cat << 'EOF'` heredoc through `run_bash` — a hundred lines of documentation
  squeezed through bash quoting, where one stray line costs the file. It creates and
  does not overwrite (an existing path is refused and pointed at `edit_file`), which
  keeps it non-destructive by construction: writing a document never stops for
  approval. A file whose suffix names a script language faces §5.2's gate on a
  scratch copy exactly as `edit_file` does — a second way to put content into a
  script file must not be a second way around the gate.
* `edit_file(path, old_lines, new_lines)` — replaces one run of
  whole lines in place, so changing a 400-line script costs the two lines rather
  than the file twice (once read, once rewritten). Matching is whole-line and must
  be unique: no match and an ambiguous match are both refused with the line numbers
  that nearly matched, never guessed at. It is the only tool besides `create_script`
  that writes file content, and is held to the same rules as everything else that
  overwrites: the previous version is copied to the trash first (§5.3 — a *copy*,
  not a hardlink, because the file survives the write), the call always gates, and
  when the file is a script the edited content faces §5.2's syntax and code-vs-docs
  gate — run on a scratch copy, so a refused edit leaves the real file untouched
  rather than momentarily broken. Undo is `move_file` the edited file aside, then
  `restore_file` the path; deleting it instead would trash the *edited* version
  under that same path and restore would bring back what was being undone.

All subprocess execution goes through **one internal runner** (timeouts, output
capture, cwd tracking, env control). There is deliberately **no free-form shell tool
in v1**: even `run_bash` funnels through a syntax-checked throwaway script and the
tracked runner, so every terminal operation stays typed and captured.

### 5.2 Dry-run & verification middleware (mandatory, automatic)

Every script and every job submission is dry-run/dummy-tested **before** the real
execution, using native mechanisms — no LLM involved:

| Artifact | Check |
|---|---|
| bash script | `bash -n` |
| Python script | `python -m py_compile` |
| sbatch submission | `sbatch --test-only` |

Missing checker binaries are reported as `skipped`, never a hard failure. Failures
are parsed and fed back to the model as structured errors for a bounded
fix-and-retry loop.

**Semantic verification gate (code-vs-docs).** Syntax checks miss the small model's
dominant failure mode: plausible-but-wrong API usage — invented CLI flags,
deprecated or misspelled kwargs, wrong subcommands. After a clean syntax dry-run,
every script therefore passes a second, mostly deterministic gate that compares it
against the indexed documentation and source (§5.6):

0. **On-demand indexing.** A command nobody indexed is a command the gate cannot
   check, which is exactly the case for the tools that matter (`minimap2`,
   `samtools`). So before the gate runs, `create_script` learns the flags of every
   external program the script drives, from the man page **unioned with** `--help`
   output — neither source alone is sufficient (`sort` exposes no parseable
   OPTIONS section; `find` hides `-maxdepth` outside its own). This is
   deterministic and unskippable on purpose: the model never decides whether to
   look a tool up, because under instruction load a small model reliably decides
   not to. A parse yielding too few flags counts as a failed probe and leaves the
   command unindexed — a half-parsed flag list would turn correct scripts into
   gate failures. Probes are bounded and cached per session. What may be executed
   for its `--help` is decided by `safe_to_execute`: a small `NEVER_EXECUTE`
   denylist of destructive commands, and anything resolving inside the agent's own
   `scripts_dir` (the model's generated code, which must not run before the §5.3
   gate sees it), are refused — everything else is probed. *(The original design
   gated this on a command resolving into a `bin/` directory; that heuristic was
   found wrong and replaced by the denylist + own-scripts rule, though the safety
   intent — never run the agent's own scripts early — is preserved.)* Man pages are
   still read for refused commands, since fetching one never runs anything.
1. **Deterministic extraction** of used APIs: Python via `ast` (imports, calls,
   keyword names), and bash via command tokenization (command + flags). Wrapper
   prefixes (`conda run -n env …`, `time`, `nohup`) are unwrapped and absolute
   paths reduced to their basename, so flags are attributed to the program that
   owns them rather than to the wrapper.
2. **Exact lookup, not embeddings:** extracted symbols are checked against the
   symbol table (§5.6) — CLI flags against the learned flag set, functions and
   kwargs against indexed signatures. Flag matching allows attached values and
   clustering (`-k1,1`, `-q20`, `-bh`): the target is an invented flag *name*, and
   blocking a valid command costs far more than passing a malformed value through
   to the tool's own error message.
3. Mechanical mismatches (flag absent from the man page, kwarg absent from the
   signature) are flagged by code alone. *(The design also routed fuzzy cases —
   ambiguous parse, partial match — to the doc-researcher for a judgment call;
   that escalation is not implemented: as built, mismatches are decided entirely
   in code, with no model-in-the-loop for borderline cases.)*
4. The result is a structured per-symbol report `confirmed | mismatch |
   not_indexed`. Mismatches feed the same bounded fix-and-retry loop as syntax
   failures. `not_indexed` surfaces as a visible warning to the user, never a hard
   block — index coverage is always partial.

Only after a clean dry run and verification report does the destructive/execution
gate (§5.3) or actual submission proceed.

### 5.3 Destructive operations: human-in-the-loop + recovery

* **Every** destructive operation (delete, overwrite, move-over-existing, kill,
  cancel) requires an explicit interactive confirmation in the TUI
  (LangGraph `interrupt()` → modal with the exact operation and resolved real paths
  shown to the user). The prompt shows *that* operation and nothing else: the
  command bash would run, the diff a file would be replaced with, the lines a
  file would be created from. What it deliberately does not show is the tool's
  schema description — that is the blurb written for the model to pick the tool
  by, it describes the tool in general and never this call, and on a prompt
  asking a yes/no about one concrete action it is prompt content leaking onto the
  screen. Nor the raw arguments when the tool already resolved them into a
  sentence about real paths.
* **Refusing asks why.** Saying no has a second step: the prompt keeps the call
  on screen and opens a box for what should be different, sent with enter
  (empty is allowed, and esc refuses without explaining). The graph resumes
  with `Command(resume={"approved": bool, "reason": str})`, and the reason
  turns the refusal into a correction — the SKIPPED/DENIED tool result then
  tells the model to work it into a fixed version and put *that* up for
  approval, instead of the "do not retry" a bare refusal carries. The point is
  the common case: a script that is nearly right, where retyping the whole
  request is the only way to say "wrong partition". Half-written reasons are
  parked per session like chat drafts, so switching away does not lose them.
* **Recovery for small files (< 1 GB, configurable):**
  * *Deletions:* do **not** copy — **hardlink the file into a trash directory**
    (`~/.HolisticProcessingComputeAgent/trash/<timestamp>/…`) before unlinking.
    Hardlinks cost zero extra space on the same filesystem (Lustre/GPFS support
    them) and fully protect against the unlink. Fall back to copy only if the trash
    dir is on a different filesystem.
  * *Content-overwriting operations* (in-place edits, overwrites): make a real copy
    first (this is the rarer case) — `TrashManager.backup()`, used by `edit_file`.
    A hardlink is no good here: the file survives the operation and is rewritten
    through the same inode, which the "backup" would follow.
  * Trash entries carry a TTL (`trash_ttl_days`, default 7) and are cleaned up on
    app start. Recovery is reached by *asking the agent*: the `restore_file` tool
    wraps `TrashManager.list()`/`.restore()`, taking the file's original path (or
    its name, or empty to list the trash) — the file it names is gone, so
    there is nothing else left to identify it by. It never gates — restoring only creates a file, and refuses outright when
    the original path is occupied. *(Still no TUI affordance: the "restore from the
    inspect view" action is not wired up, so there is no trash browser to click.)*
* Files ≥ 1 GB: no automatic backup (quota!), but the confirmation modal states this
  explicitly.

### 5.4 Job tracking: sqlite DB

One sqlite database at `~/.HolisticProcessingComputeAgent/hpca.db` (WAL mode). It is
the backbone of the right column, of `job_status`, and of log triage. *(LangGraph
conversation checkpoints deliberately live in a **separate** `checkpoints.db`, not
in `hpca.db`, to keep the checkpointer's heavy writes off the app's own tables. The
shipped schema also carries a few additive columns/tables beyond the minimum below
— e.g. `sessions.mode`/`backend`, a `symbols` table, a message store.)*

Tables (minimum):

```
jobs(job_id PK, kind, session_id, profile, submit_time, state,
     script_key, sbatch_stdout_path, sbatch_stderr_path,
     last_checked, exit_info)
job_logs(job_id FK, rule_or_step, log_path, tool_name)
sessions(session_id PK, profile, title, created_at, checkpoint_ref)
processes(pid, session_id, cmd, state, stdout_path, stderr_path, started_at,
          notified, background)
```

A background asyncio task polls the cluster (interval `job_poll_seconds`,
default 30) and updates states; a sibling timer polls the `processes` table for
local subprocesses that have ended. *(As built, cluster polling uses `sacct`
exclusively — `sacct --parsable2` — not `squeue`; `sacct` covers both running and
finished jobs, so the live-queue path was dropped.)*

**Completion reaches the agent, not just the user.** A turn ends when the model
answers, and nothing else starts one — so "I'll check on it in a moment" was a
promise the runtime could not keep, and a background script could fail silently
until the user noticed. Terminal transitions are therefore delivered *into the
conversation*: the change is appended to the session's LangGraph thread as new
input. For a local background process the event carries its exit code and a log
tail (and any triage finding); the cluster-job event is currently terser — it
announces the new state and points at the logs, without an exit code or tail
inlined. Reacting immediately is
`run_turn` on that thread (the agent speaks unprompted); deferring is
`aupdate_state`, which leaves the message in checkpointed history for the next
turn at no model cost. Both are the same primitive — new input on an existing
`thread_id` — and the choice is only whether the model runs now.

Three constraints shape it. Delivery is serialised against live turns, because
concurrent `ainvoke` on one `thread_id` interleaves checkpoint writes. Only
`start_background_script` work qualifies (`background = 1`): `run_bash`
blocks and returns its output as the tool result, so an event for those runs
would report the same failure twice. And `notified` lives in the table rather than in
memory, so a job that ends while the TUI is closed is still announced on the
next start — the watcher reads the table, since a fresh `ProcessRunner` is built
per turn and no single instance knows about processes an earlier one started.

### 5.5 Log triage (the killer feature — ~80% deterministic)

The model must never see raw `squeue`/`sacct` dumps or multi-MB logs. A triage
pipeline turns "job 48812 failed" into a compact structured report:

1. From the job DB, collect *all* related logs (slurm stdout/stderr, tool logs).
2. Parse `sacct` fields (State, ExitCode, Elapsed, MaxRSS, ReqMem, Timelimit) into
   JSON.
3. Scan logs (tail-first) against a **signature library** of known error patterns:
   OOM-kill (`oom-kill`, `Out Of Memory`), `DUE TO TIME LIMIT`, command not found,
   Python tracebacks, missing input files, permission denied, quota exceeded, …
   The signature library is a data file (`error_signatures.yaml`) and
   user-extensible.
4. Extract the ~30 most relevant lines per matched signature.
5. Hand the structured report `{job meta, matched signatures, excerpts}` to the
   log-explainer subagent, which produces: **why the job failed, its current state, a
   suggested fix, and a finickiness estimate** (`trivial | easy | fiddly | hard`,
   with one sentence of justification). Given the small model, the suggestion is
   always framed as a suggestion; fixes touching files re-enter the dry-run and HITL
   gates.

**Three tiers of error identification.** The same pipeline reads the logs of local
background processes (§5.4), and there the naive answer — send the tail of stderr —
fails badly: a tool that errors on its last line after printing 25 lines of usage
buries the cause, and a grep for `error|fail` on a chatty aligner log matches
statistics (`Error rate: 0.012%`) and cleanup messages (`0 errors during cleanup`)
just as readily as the failure. So identification escalates, each tier running only
when the cheaper one above it found nothing:

1. **Signature library** — deterministic, and the only tier that names the failure
   *class* and carries a `hint`. For a small model that hint is worth more than the
   log text: it says what kind of mistake was made, not just what was printed.
2. **Scored keyword scan** — a broad sweep whose candidates are *ranked*, not picked
   by position. Structured markers (`error:`, `[E::`, `Traceback`) score up; things
   shaped like counters or rates score down and drop out; lateness contributes a
   little and never outranks a real marker. Several candidates survive, each with
   three lines of context either side. Deliberately proposes rather than decides.
3. **The log-explainer** — chooses among those candidates. Grep is better at
   *finding* and a model is better at *judging*, so the model never searches: it is
   handed retrieved lines and asked which one caused the failure. It must quote a
   candidate verbatim or return an empty cause; a quote that is not in the log is
   discarded in code and marked unverified. Admitting "cannot tell from this log" is
   a correct answer, and cheaper than a confident wrong one.

Tier 3 running at all is evidence that a signature is missing, so the explainer may
propose one — id, patterns, hint — offered to the user for approval and appended to
their `error_signatures.yaml`. The expensive path teaches the cheap path, and a tool
that fails oddly on this cluster costs a model call once rather than every time.

### 5.6 RAG for documentation

Embedded, daemon-free stack (HPC users have no root, no services). *Bulk* indexing
runs as an explicit command, not implicitly, over: man pages of cluster-relevant
tools, user-supplied docs directories, selected source trees. The one implicit path
is the narrow, bounded flag lookup the verification gate performs on the specific
commands a script is about to run (§5.2 step 0) — without it the gate has no data
precisely when it matters. Indexing builds **two** structures:

1. **Symbol table (exact retrieval)** — plain sqlite tables: function/class
   signatures extracted from indexed Python source via `ast`, CLI flags parsed from
   man-page OPTIONS sections and `--help` output. This is what the gate (§5.2) and
   `lookup_symbol` query. "Does `samtools view -e` exist?" is an exact-match
   question; embeddings are the wrong tool for it. Requires no embedding model, so
   it works in a slim base install.
2. **Vector store (semantic retrieval)** — `sqlite-vec` (one file, embedded,
   dimension fixed by the first insert), queried by `search_docs` for prose
   questions ("how do I subset a BAM by region?"). Embeddings come from the LLM
   backend's embedding endpoint (vLLM and llama.cpp-server serve one; keeps the
   agent itself free of local ML dependencies and GPU-vendor concerns).

   *(Design vs. code: the doc originally offered `chromadb` **or** `sqlite-vec`
   and a local `sentence-transformers` fallback when no embedding endpoint is
   available. Neither shipped — the store is `sqlite-vec` only (the `rag.store`
   config switch is currently inert), and there is no local embedding fallback: if
   the backend serves no embedding endpoint, `search_docs`/indexing return an error
   rather than falling back. `sentence-transformers` is not a dependency; the
   default `rag.embedding` value is a model **name** sent to the remote endpoint,
   not a locally-loaded model.)*

## 6. Agent profiles & memory (two-scope model)

An **agent profile** is a per-user, named memory document recording what the agent
has learned, and on which LLM backend each learning was made (small models differ —
a workaround for one backend may not apply to another). On session start the user
picks an existing profile or creates a new one (managed from the `a` screen).

> **Design vs. code.** This section originally described a **tier1 / tier2** split
> (plus a deferred tier 3). The shipped implementation collapsed that into **two
> scopes** — `system-prompt` and `rag` — and this section has been rewritten to
> match the code. (A now-stale `memory_redesign.md` at the repo root captures an
> intermediate design with *character* budgets; the code moved past it to a single
> *token* budget on one injected scope.)

### 6.1 Scopes

* **`system-prompt` — injected memory.** The entries injected verbatim into the
  **orchestrator's** system prompt each turn: stable facts about the cluster and
  filesystem, user preferences, backend-specific workarounds (tagged with the
  backend). A single hard **write** budget applies (`memory.system_prompt_token_cap`,
  default **2400** tokens): a full scope refuses new memories until the user
  condenses it, but injection itself never truncates the file. Subagents/sub-loops
  get their own minimal prompts and are **not** given this memory. Time/date facts
  are never stored here — they are injected dynamically at render time (§4.3).
* **`rag` — retrieved memory.** Not injected wholesale; only the entries matching
  the current request are pulled in, within a per-turn prefetch budget
  (`rag_prefetch_chars` 800, `rag_prefetch_count` 3). Task learnings and struggle
  notes (§4.4) live here. A background **curator** ages `rag` entries out to a
  `<profile>.archive.md` (never deletes) so retrieval quality does not decay as
  notes accumulate (`curator_*_days` settings).

### 6.2 Storage format

Human-editable markdown per profile at
`~/.HolisticProcessingComputeAgent/profiles/<name>.md`: YAML front-matter for
metadata, then one memory per block under `## [system-prompt]` / `## [rag]`
headings, each block with a small inline metadata line
(`<!-- backend: qwen3-6b, created: 2026-07-16, kind: struggle -->`). Markdown, not
JSON, because users will edit this in vim/nano (§6.4) and it must survive hand edits;
the parser is lenient and reports problems clearly (unknown headings become
`problems` rather than raising).

### 6.3 Writing memories

* `/memorize [NOTE]` — the model forms durable memories from NOTE and the
  conversation so far; each is proposed for approval (choosing its own scope).
* `/conclude` (also `\conclude`) — the orchestrator analyses the conversation and
  **proposes** memory blocks; the approval screen shows each for **approve/reject**
  (HITL, consistent with §5.3 — a small model writes these, so review is essential).
  Editing a proposal's text is done in the separate profile editor rather than
  inline in the approval screen.
* Struggle notes (§4.4) follow the same proposal/approval path and are stored in the
  `rag` scope.

### 6.4 Size warnings & manual cleanup

The injected `system-prompt` scope's token count is tracked against its cap
(`tiktoken` when available, chars/4 as the last-resort fallback — there is no
backend `/tokenize` call in the shipped code). When a write would exceed the cap it
is blocked with a notification prompting the user to condense; the `rag` scope is
not metered. Cleanup paths:

* **Edit externally** — the UI restores the terminal, opens
  `$VISUAL`/`$EDITOR`/`nano` on the profile file, and takes the terminal back on
  exit. On return the file is re-parsed and validated; the user can move a memory
  between scopes simply by moving its block between the two headings.
* **Condense with the LLM** — the model drafts a shorter version for approval,
  never auto-applied.
* **Demote** — an op that moves a `system-prompt` entry into `rag` to free the
  injected budget without losing the note.

## 7. Configuration

`~/.HolisticProcessingComputeAgent/settings.json`, editable via the in-app config
editor (`c`) and by hand. Sketch:

```json
{
  "llm": {
    "base_url": "http://localhost:8000/v1",
    "api_key": null,
    "model": "qwen3-6b",
    "constrained_decoding": "auto",
    "max_retries": 3,
    "request_timeout_s": 120
  },
  "cluster": {
    "submit_host": null,
    "job_poll_seconds": 30
  },
  "safety": {
    "backup_limit_gb": 1,
    "trash_ttl_days": 7
  },
  "agent": { "default_mode": "manual", "default_thinking": "off" },
  "memory": {
    "system_prompt_token_cap": 2400,
    "rag_prefetch_chars": 800,
    "rag_prefetch_count": 3
  },
  "clipboard": { "mode": "auto", "command": null, "osc52_limit_kb": 74 },
  "editor": null,
  "rag": {
    "store": "sqlite-vec",
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_base_url": "http://localhost:51943/v1"
  }
}
```

## 8. Dependencies

Constraints: no root, no daemons, installable into a venv/conda env on the cluster.

* `python3` (≥ 3.11)
* `langgraph`, `langchain-core` — agent graph, checkpointing, interrupts
* `httpx` + OpenAI-compatible client (or `langchain-openai`) — any backend behind an
  SSH tunnel
* `pydantic` — tool schemas & validation-driven retries
* `sqlite3` (stdlib) — job DB, sessions; LangGraph sqlite checkpointer
  (`langgraph-checkpoint-sqlite`)
* `sqlite-vec` — RAG vector store (the only store shipped; `chromadb` was dropped)
* `pyyaml` — signature library, skills, front-matter
* embeddings — from the LLM backend's `/v1/embeddings` endpoint; there is **no**
  bundled `sentence-transformers` local fallback in the shipped code
* `tiktoken` — optional; token estimates fall back to chars/4 when it is absent

Explicitly avoided: anything requiring a service daemon, X11 clipboard tools as a
hard dependency, LangChain `AgentExecutor`.

## 9. Suggested build order (milestones for implementation)

1. **Skeleton TUI:** three columns, arrow-key focus model, bottom hotkey bar,
   settings modal reading/writing `settings.json`.
2. **ClipboardManager** with multiplexer detection and all fallback tiers
   (independently testable; test matrix: tmux/screen/zellij × set-clipboard on/off).
3. **LLM client** against the OpenAI-compatible API, incl. streaming into the chat
   window and constrained-decoding probe.
4. **LangGraph core:** orchestrator, checkpointed sessions (left column live),
   pydantic validation + retry middleware.
5. **Runner + tools:** internal subprocess runner, `create_script`/`start_background_script`
   with dry-run gate. *(This milestone also built a right-column process list —
   Enter inspects, `k` kills — since removed; see §3.3.)*
6. **Slurm layer:** job DB, `submit_job`/`job_status`/`cancel_job`, background
   poller, log collection.
7. **Log triage:** signature library + report generator + log-explainer subagent.
8. **Safety layer:** destructive-op interrupts, hardlink trash, restore, TTL cleanup.
9. **Profiles & memory:** two-scope storage (`system-prompt`/`rag`), `/memorize`,
   `/conclude` with approve/reject, size warnings, `App.suspend()` editor round-trip.
10. **Symbol index & verification gate:** man-page/source indexing into the sqlite
    symbol table, `lookup_symbol`/`read_manpage`/`read_source`, doc-researcher
    subagent (exact-lookup mode), semantic code-vs-docs gate wired into §5.2. No
    embedding model needed.
11. **Vector RAG:** embedding store + indexing command, `search_docs`, grounded
    prose Q&A per the grounded answering policy (§4.2).
12. **Skills** (user-defined procedure files) and self-reflection struggle notes.

Each milestone is shippable and testable on its own; 1–5 already yield a useful
local-script assistant before any Slurm integration exists.

## 10. Open questions

* Is `sbatch` available from compute nodes on our site, or is a `submit_host` SSH
  hop required? (Decides the job-runner implementation, §2.)
* Which exact model(s) will be served first, and does the serving stack support
  JSON-schema-constrained decoding? (Decides how much retry middleware is exercised.)
