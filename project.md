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
   registry; tools accept registry keys.
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
  llama.cpp-server, Ollama, and commercial providers all speak it, so "local or remote
  LLM" is a single config switch, one code path.
* Prefer backends that support **structured output / JSON-schema-constrained
  decoding** (vLLM `guided_json`, llama.cpp grammars, Ollama `format`). Use it for all
  tool calls so they are syntactically valid *by construction*; retries then only
  handle semantic errors.
* **Open question to verify before building the job tools:** does our site permit
  `sbatch`/`squeue`/`sacct` from compute nodes? If submission is restricted to login
  nodes, the job-runner tool must wrap its commands in an SSH hop to a submit host.
  The tool interface must hide this behind a `submit_host` setting (`null` = run
  locally).

## 3. Interface (TUI)

### 3.1 Framework

**Textual** (Python, async-first). Reasons: proper layout system for the three-column
design, CSS-like styling, native `Header`/`Footer` widgets (the footer renders
keybindings exactly like the required bottom hotkey bar), modal screens for
interactive resolution, runs in any terminal incl. over SSH, and — critically for an
agent — an async event loop so streaming LLM output and tool execution never freeze
the UI. `App.suspend()` is used to drop into external editors (see §6.4).

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
  * *Chat message (center):* **Enter** on one of your own messages (including a
    queued one) puts its text back in the entry — appended on its own line to
    whatever is already being written, so nothing typed is lost — which is how
    a command is re-sent with one word changed. Enter on a thinking box expands
    it into its parts — each block of reasoning, each tool call with the script
    or command it would run, and each result — every one its own collapsible
    row, so the script the agent wrote and the call it made stay readable long
    after the approval prompt that showed them is answered (and are there at all
    in auto mode, where no prompt showed them). Those rows do not wait for the
    turn to end: each call appears the moment it is made and its result the
    moment it lands, above the working line, so a turn that spends minutes in
    tools shows *what* it is doing while it does it — and can be opened and
    read while it runs. The turn's last act is to fold them into its box.
    Enter on anything else (the agent's replies, background events, recalled
    memory) simply hands focus to the entry. `(b)` go back in conversation to
    this point, `(c)` copy content to clipboard.
  * *Working line (center, while a turn runs):* names the step in flight and
    counts up from the moment the turn started. **Enter** on it asks whether to
    interrupt the turn and hand the message back to the entry for editing — in
    any phase, waiting on the model or running a tool. It ends the turn, not
    the work already started: a script keeps running under its own monitor
    until it exits or times out.
  * *Watch (right column):* a box the user asked for, pinned by the agent's
    `watch_log` / `watch_job` tools — see *Watches* below. **Enter** flashes
    the last 300 characters of the log (a toast that expires, not a screen to
    dismiss), `(d)` stops watching and removes the box, `alt+↑` / `alt+↓` carry
    it past its neighbour. The log itself is never touched.

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
snakemake run spawning tool after tool, the log some long-running tool appends
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
  * `/skill-creator`, `/skills-list`, `/skill-remove` — manage this profile's
    skills (see §5.1).
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
through the same one-at-a-time discipline. A session parked on an approval holds
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
* Textual's built-in `copy_to_clipboard` emits plain OSC 52 without multiplexer
  wrapping — do **not** use it; write raw sequences through the Textual driver
  (~150 lines total).
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
`start_background_script` and `submit_job` from the registry offered to
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
tools plus `list_paths` (registry) — and answers directly only for trivial
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
(code excerpt, target library/tool, registry keys); the subagent may read as many
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
  the repair with the call rather than silently shortening a script.
* **Path registry:** a named map `{key → absolute path/URI}` per profile+session,
  stored in sqlite. Tools accept **keys**, middleware resolves to real paths and
  errors out on unknown keys (error fed back for retry). New paths discovered by
  tools (e.g. output of a job) are auto-registered and announced to the model as
  their key. The model never has to reproduce a literal path correctly.
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

Core tools:

* `create_script(kind: bash|python|R|snakemake, registry_key, content)` — writes the
  script, immediately syntax-checks it (§5.2), registers the path.
* `run_bash(timeout_s, content_lines)` — runs *and blocks*, returning captured
  output as the tool result. Writes a throwaway script, `bash -n`-checks it, and
  runs it through the same tracked runner (it is not a free-form shell — see
  below). A `{registry_key}` in a line expands to the registered path, which is
  how a *kept* script is run synchronously: `{my_script} --flag`. Only keys that
  exist are substituted, so `awk '{print $1}'` and `${VAR}` survive untouched;
  an unmatched `{…}` is named in the result if the run then fails.
  *(The design had a second blocking tool, `run_script(registry_key, args)`, for
  registered scripts. It was removed: splitting the two by where the script came
  from gave the model two tools advertising one job, while `{key}` expansion
  covers the case without carving an exception into the "keys, never paths"
  rule.)*
* `start_background_script(registry_key, args)` — runs locally as a tracked **background**
  subprocess, stdout/stderr captured to files and
  registered; its completion is delivered back into the conversation (§5.4).
  The split from `run_bash` is *when the result arrives*, not what is run — the
  one choice the model cannot recover from on its own, which is why it is a
  separate tool rather than a flag.
* `submit_job(registry_key, args)` — submits an sbatch script to the cluster,
  records the job ID and log paths in the job DB (§5.4), starts periodic tracking.
  *(The design imagined a `kind: sbatch|snakemake` switch; only the sbatch path is
  implemented — a snakemake-cluster submission tool does not exist yet.)*
* `job_status(job_id)` / `get_job_report(job_id)` — structured status / triaged
  failure report (§5.5).
* `cancel_job(job_id)` — HITL-gated.
* `search_docs(query)`, `lookup_symbol(name, kind)`, `read_manpage(name)`,
  `read_source(registry_key, range)`, `index_docs(...)`, `ask_docs(question)` —
  retrieval over man pages, tool documentation, and source code (§5.6):
  `lookup_symbol` is the exact-match path used by the verification gate (§5.2),
  `search_docs` the embedding path for prose questions, `ask_docs` the firewalled
  doc-researcher sub-loop (§4.2).
* File operations (`move_file`, `copy_file`, `delete_file`, `restore_file`,
  `read_file`, `create_file`, `edit_file`) — destructive ones gated per §5.3. There is **no**
  `list_dir` tool: registry keys are listed by `list_paths`, and directory contents
  are read via `run_bash`. `restore_file` is the one file tool taking a path instead
  of a key: the key died with the file.
* `create_file(dir_key, name, content_lines)` — writes a **new** text file into a
  registered directory, one array element per line. It exists because prose had no
  tool: `create_script` writes only into the scripts dir under a language suffix and
  `edit_file` needs a file to already exist, so authoring a `specs.md` meant a
  `cat << 'EOF'` heredoc through `run_bash` — a hundred lines of documentation
  squeezed through bash quoting, where one stray line costs the file. It creates and
  does not overwrite (an existing path is refused and pointed at `edit_file`), which
  keeps it non-destructive by construction: writing a document never stops for
  approval. A file whose suffix names a script language faces §5.2's gate on a
  scratch copy exactly as `edit_file` does — a second way to put content into a
  script file must not be a second way around the gate.
* `edit_file(registry_key, subpath, old_lines, new_lines)` — replaces one run of
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
| R script | `Rscript -e 'parse(file="...")'` |
| snakemake workflow | `snakemake -n` (dry run) |
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
   owns them rather than to the wrapper. *(R and snakemake `shell:` extraction were
   designed but are not yet implemented — the extractor returns nothing for them,
   so those scripts pass the semantic gate on syntax alone.)*
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
  shown to the user).
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
    its name, or empty to list the trash) because the deletion dropped its registry
    key. It never gates — restoring only creates a file, and refuses outright when
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
     snakemake_log_path, last_checked, exit_info)
job_logs(job_id FK, rule_or_step, log_path, tool_name)
sessions(session_id PK, profile, title, created_at, checkpoint_ref)
path_registry(profile, session_id, key, path, created_at)
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

1. From the job DB, collect *all* related logs (slurm stdout/stderr, snakemake main
   log, per-rule logs, tool logs).
2. Parse `sacct` fields (State, ExitCode, Elapsed, MaxRSS, ReqMem, Timelimit) into
   JSON.
3. Scan logs (tail-first) against a **signature library** of known error patterns:
   OOM-kill (`oom-kill`, `Out Of Memory`), `DUE TO TIME LIMIT`, command not found,
   Python tracebacks, R errors (`Error in …`), snakemake rule failures
   (`Error in rule …`), missing input files, permission denied, quota exceeded, …
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

* **Edit externally** — via Textual's `App.suspend()`: restore the terminal, open
  `$VISUAL`/`$EDITOR`/`nano` on the profile file, reinstate the TUI on exit. On
  return the file is re-parsed and validated; the user can move a memory between
  scopes simply by moving its block between the two headings.
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
  "agent": { "default_mode": "manual" },
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
* `textual` — TUI
* `langgraph`, `langchain-core` — agent graph, checkpointing, interrupts
* `httpx` + OpenAI-compatible client (or `langchain-openai`) — any backend behind an
  SSH tunnel
* `pydantic` — tool schemas & validation-driven retries
* `sqlite3` (stdlib) — job DB, sessions, path registry; LangGraph sqlite checkpointer
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
   pydantic validation + retry middleware, path registry.
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
* Snakemake profiles/executors in use on the cluster (affects `snakemake -n`
  invocation and log locations).