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
│ Sessions      │ Chat window of the        │ Sub-processes of │
│ (past +       │ selected session          │ the current      │
│  current)     │ (scrollable message list) │ session (jobs,   │
│               │                           │ subagents, local │
│               │                           │ processes)       │
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
  * *Chat message (center):* `(b)` go back in conversation to this point,
    `(c)` copy content to clipboard.
  * *Sub-process / job (right column):* **Enter** inspects (opens the
    logs/status view), `(k)` kills the selected local process (with
    confirmation). (The design once envisaged an `(a)` "ask a Q&A subagent about
    this process" action; that is not implemented — failure explanation is done
    automatically by the triage pipeline in §5.5, not on demand from this
    column.)

**Reserved hotkeys — never bind these** (they are eaten or made unreliable by the
terminal, by zellij/tmux, or by the flow-control layer, so a future UI addition
must avoid them):

* `ctrl` + `q p t n h s o g` — `ctrl+s`/`ctrl+q` are terminal flow control
  (XOFF/XON: `ctrl+s` *freezes* output), and `ctrl+p`/`ctrl+t`/… are common
  multiplexer/zellij prefixes.
* `alt` + `n f`, the arrow keys, `+`, `-` — commonly grabbed by zellij/tmux.

Prefer a bare letter gated (via `check_action`) to a non-typing column, or a safe
`ctrl` combo (`ctrl+l`, `ctrl+e`, `ctrl+r`, …). Keep this list in sync with the
`RESERVED HOTKEYS` comment above `HpcaApp.BINDINGS`.
* **Config editor** `(c)`: edit the settings JSON, persisted to
  `~/.HolisticProcessingComputeAgent/settings.json`.
* **Profiles & learnings** `(a)`: manage profiles and their memories (see §6).
* **Chat commands** (typing `/` or `\` lists them):
  * `/memorize [NOTE]` — the agent forms memories from NOTE plus the conversation
    so far and proposes them for approval (see §6).
  * `/conclude` — the agent analyses the conversation and proposes memories to write
    into the profile (user approves before write, see §6).

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

### 3.5 Agent modes (manual / auto / plan)

Each session has an interaction mode, indicated on a single line directly above
the chat entry and cycled with **shift+tab** (ctrl+m is bound as well, but most
terminals deliver it as Enter, so it only works under keyboard protocols that
can tell them apart). The mode is stored per session (`sessions.mode`, empty =
the `agent.default_mode` setting, default `manual`) and read fresh every graph
round, so switching applies immediately — even to a turn already in flight.

* **manual** — every execution tool call (`run_script`, `start_script`,
  `run_bash`, `submit_job`) pauses at the same `interrupt()` gate as
  destructive operations; an inline approval bar at the foot of the chat column
  (`DecisionBar`, deliberately non-modal so the other columns and sessions stay
  visible) shows the actual script text and offers *run script* / *skip
  script*. A skip is fed back to the model as a
  SKIPPED tool result that forbids retrying, rephrasing, or reaching the same
  outcome another way (bare denials make small models re-propose the same
  command).
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
* **plan** — nothing executes. Enforcement is structural, not prompt-trust:
  script tools are withdrawn from the registry offered to `decide()` (a tool
  never offered cannot be called). `run_bash` is kept so the plan can be
  grounded in what is actually on disk; it runs unattended (only the §5.3
  destructive gate still applies) rather than pausing for approval like manual —
  look-around commands are cheap and gating each one made planning tedious. The
  model maintains a checklist through an `update_plan` tool; the checklist lives
  in the checkpointed graph state (`AgentState.plan`) and is re-injected into the
  system prompt every round, so it survives restarts and context compaction.
  When a plan-mode turn ends with a plan, the same inline decision bar shows the
  editable checklist and offers: execute on auto (`ctrl+r`), execute
  step-by-step under manual approval (`ctrl+e`), or keep planning (`esc`).
  Approval switches the session's mode and starts execution with a
  `[plan approved]` event turn; during execution the plan stays in the prompt
  and the model checks steps off via `update_plan`.

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
* **Path registry:** a named map `{key → absolute path/URI}` per profile+session,
  stored in sqlite. Tools accept **keys**, middleware resolves to real paths and
  errors out on unknown keys (error fed back for retry). New paths discovered by
  tools (e.g. output of a job) are auto-registered and announced to the model as
  their key. The model never has to reproduce a literal path correctly.
* **Output size control:** long tool outputs are kept small before they reach the
  model. *As built,* this is per-tool truncation rather than a single generic
  middleware layer: `read_file` returns head/tail, `read_manpage`/`read_source`
  are bounded, and `run_script`/`run_bash` clip each stream to a tail and point at
  the full log file (which is registered by key). There is **no** universal
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
loaded per profile.

Core tools:

* `create_script(kind: bash|python|R|snakemake, registry_key, content)` — writes the
  script, immediately syntax-checks it (§5.2), registers the path.
* `run_script(registry_key, args)` / `run_bash(content)` — run *and block*,
  returning captured output as the tool result; `run_bash` writes a throwaway
  script, `bash -n`-checks it, and runs it through the same tracked runner (it is
  not a free-form shell — see below). These are the built-in execution tools
  beyond the §5.1 "core" list.
* `start_script(registry_key, args)` — runs locally as a tracked **background**
  subprocess (appears in the right column), stdout/stderr captured to files and
  registered; its completion is delivered back into the conversation (§5.4).
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
* File operations (`move_file`, `copy_file`, `delete_file`, `read_file`) — destructive
  ones gated per §5.3. There is **no** `list_dir` tool: registry keys are listed by
  `list_paths`, and directory contents are read via `run_bash`.

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
* **Recovery for small files (< 1 GB, configurable):**
  * *Deletions:* do **not** copy — **hardlink the file into a trash directory**
    (`~/.HolisticProcessingComputeAgent/trash/<timestamp>/…`) before unlinking.
    Hardlinks cost zero extra space on the same filesystem (Lustre/GPFS support
    them) and fully protect against the unlink. Fall back to copy only if the trash
    dir is on a different filesystem.
  * *Content-overwriting operations* (in-place edits, overwrites): make a real copy
    first (this is the rarer case).
  * Trash entries carry a TTL (`trash_ttl_days`, default 7) and are cleaned up on
    app start. *(`TrashManager.restore()`/`.list()` exist and are unit-tested, but
    are not yet surfaced anywhere in the TUI — the "restore from the inspect view"
    action is not wired up, so recovery is currently a library capability rather
    than a user-facing button.)*
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
`start_script` work qualifies (`background = 1`): `run_script` and `run_bash`
block and return their output as the tool result, so an event for those would
report the same failure twice. And `notified` lives in the table rather than in
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
5. **Runner + tools:** internal subprocess runner, `create_script`/`start_script`
   with dry-run gate, right-column process list (Enter inspects, `k` kills).
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