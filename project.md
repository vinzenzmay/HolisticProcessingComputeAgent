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
4. **Human-in-the-loop for anything destructive.** Always. No exceptions.
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
│ Top bar: app name | profile | model | (s) settings           │
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
  * *Sub-process (right column):* `(i)` inspect (open logs/status view),
    `(k)` kill (with confirmation), `(a)` ask — spawn a Q&A subagent about this
    sub-process.
* **Top settings menu** `(s)`: edit configuration, persisted to
  `~/.HolisticProcessingComputeAgent/settings.json`.
* **Slash commands** in the chat input:
  * `\memorize [TEXT]` — add TEXT verbatim as a profile memory (tier 2).
  * `\conclude` — the agent analyses the conversation and proposes memories to write
    into the profile (user approves before write, see §6).

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

One **orchestrating agent** routes work to specialized **subagents**. Each subagent
gets a minimal system prompt and only its own tools:

| Subagent | Purpose | Tools (max ~5) |
|---|---|---|
| script-writer | create/edit bash, Python, R, snakemake scripts | `create_script`, `read_file`, `search_docs` |
| job-runner | submit & manage cluster jobs and local runs | `start_script`, `submit_job`, `job_status`, `cancel_job` |
| log-explainer | diagnose failing/finished jobs from triaged logs | `get_job_report`, `read_log_excerpt` |
| doc-researcher | answer technical questions & verify API usage against man pages, docs, source (RAG) | `search_docs`, `lookup_symbol`, `read_manpage`, `read_source` |
| process-QA | the `(a) ask` action in the right column | `get_process_info`, `read_log_excerpt` |

The orchestrator itself has: route-to-subagent, `list_paths` (registry), memory tools
(§6), and direct answers for trivial conversational queries only (see the grounded
answering policy below).

**Grounded answering policy.** Small models hallucinate API details, so technical
questions are never answered from model weights. Any query that names a tool,
library, API/function, file format, or error message is routed to the
doc-researcher; the orchestrator answers directly only for trivial conversational
turns. Doc-researcher answers must carry citations (source + section). If retrieval
finds nothing relevant, the answer is explicitly marked
`[ungrounded — not in indexed docs]` so the user always knows which kind of answer
they are reading. Accepted cost: a technical question takes ≥2 model calls
(routing + grounded answering); on a small local model that latency is the right
trade against confidently wrong flags on cluster CLIs.

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
* **Output size control:** no raw tool output above a token budget ever reaches the
  model. Long outputs are stored to disk, summarized deterministically (head/tail +
  signature extraction), and referenced by path-registry key.
* **Dry-run & verification gates:** see §5.2.
* **Destructive-op gate:** see §5.3.
* **Prompt assembly:** system prompts are rendered per call; current date/time and
  environment facts are injected dynamically (never stored as memories).

### 4.4 Self-reflection

If the agent fails at or struggles with a task (e.g. exhausts retries, user aborts,
job repeatedly fails), the orchestrator characterizes the problem in 1–2 sentences and
proposes a "struggle note" for the profile memory (user approves). On future similar
tasks — detected by simple keyword/tag match against struggle notes injected via
tier 2 — the agent warns the user up front ("I have struggled with X before") and
lets the user decide whether to attempt it anyway.

## 5. Tools & safety harnesses

### 5.1 Tool suite (initial; extensible)

The tool suite must be a plugin-style registry so new tools can be added without
touching core code. Additionally, users can provide **skills**: user-defined
markdown/YAML files describing procedures the agent should follow for specific tasks,
loaded per profile.

Core tools:

* `create_script(kind: bash|python|R|snakemake, registry_key, content)` — writes the
  script, immediately syntax-checks it (§5.2), registers the path.
* `start_script(registry_key, args)` — runs locally as a tracked subprocess
  (appears in the right column), stdout/stderr captured to files and registered.
* `submit_job(kind: sbatch|snakemake, registry_key, args)` — submits to the cluster,
  records job ID and *all* log paths in the job DB (§5.4), starts periodic tracking.
* `job_status(job_id)` / `get_job_report(job_id)` — structured status / triaged
  failure report (§5.5).
* `cancel_job(job_id)` — HITL-gated.
* `search_docs(query)`, `lookup_symbol(name, kind)`, `read_manpage(name)`,
  `read_source(registry_key, range)` — retrieval over man pages, tool documentation,
  and source code (§5.6): `lookup_symbol` is the exact-match path used by the
  verification gate (§5.2), `search_docs` the embedding path for prose questions.
* File operations (`move`, `copy`, `delete`, `read_file`, `list_dir`) — destructive
  ones gated per §5.3.

All subprocess execution goes through **one internal runner** (timeouts, output
capture, cwd tracking, env control). There is deliberately **no free-form shell tool
in v1**: every terminal operation is funneled through typed tools. (A guarded
`run_shell` can be added later as its own subagent tool if needed.)

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

Failures are parsed and fed back to the script-writer subagent as structured errors
for a fix-and-retry loop (bounded).

**Semantic verification gate (code-vs-docs).** Syntax checks miss the small model's
dominant failure mode: plausible-but-wrong API usage — invented CLI flags,
deprecated or misspelled kwargs, wrong subcommands. After a clean syntax dry-run,
every script therefore passes a second, mostly deterministic gate that compares it
against the indexed documentation and source (§5.6):

1. **Deterministic extraction** of used APIs: Python via `ast` (imports, calls,
   keyword names), bash and snakemake `shell:` blocks via command tokenization
   (command + flags), R best-effort (`library()`, `pkg::fn` calls).
2. **Exact lookup, not embeddings:** extracted symbols are checked against the
   symbol table (§5.6) — CLI flags against man-page OPTIONS, functions and kwargs
   against indexed signatures.
3. Mechanical mismatches (flag absent from the man page, kwarg absent from the
   signature) are flagged by code alone. Only fuzzy cases (ambiguous parse,
   partial match) go to the doc-researcher as a focused judgment call: "code calls
   `X(a=…, b=…)`; indexed signature is `X(a, c)` — mismatch?"
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
    app start; a `restore` action is offered in the inspect view.
* Files ≥ 1 GB: no automatic backup (quota!), but the confirmation modal states this
  explicitly.

### 5.4 Job tracking: sqlite DB

One sqlite database at `~/.HolisticProcessingComputeAgent/hpca.db` (WAL mode). It is
the backbone of the right column, of `job_status`, and of log triage.

Tables (minimum):

```
jobs(job_id PK, kind, session_id, profile, submit_time, state,
     script_key, sbatch_stdout_path, sbatch_stderr_path,
     snakemake_log_path, last_checked, exit_info)
job_logs(job_id FK, rule_or_step, log_path, tool_name)
sessions(session_id PK, profile, title, created_at, checkpoint_ref)
path_registry(profile, session_id, key, path, created_at)
processes(pid, session_id, cmd, state, stdout_path, stderr_path, started_at)
```

A background asyncio task polls `squeue`/`sacct` (interval `job_poll_seconds`,
default 30) and updates states; state changes surface as TUI notifications.

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
   The signature library is a data file (YAML) and user-extensible.
4. Extract the ~30 most relevant lines per matched signature.
5. Hand the structured report `{job meta, matched signatures, excerpts}` to the
   log-explainer subagent, which produces: **why the job failed, its current state, a
   suggested fix, and a finickiness estimate** (`trivial | easy | fiddly | hard`,
   with one sentence of justification). Given the small model, the suggestion is
   always framed as a suggestion; fixes touching files re-enter the dry-run and HITL
   gates.

### 5.6 RAG for documentation

Embedded, daemon-free stack (HPC users have no root, no services). Indexing runs as
an explicit command, not implicitly, over: man pages of cluster-relevant tools,
user-supplied docs directories, selected source trees. Indexing builds **two**
structures:

1. **Symbol table (exact retrieval)** — plain sqlite tables: function/class
   signatures extracted from indexed Python source via `ast`, CLI flags parsed from
   man-page OPTIONS sections. This is what the verification gate (§5.2) and
   `lookup_symbol` query. "Does `samtools view -e` exist?" is an exact-match
   question; embeddings are the wrong tool for it. Requires no embedding model, so
   it works in a slim base install.
2. **Vector store (semantic retrieval)** — `chromadb` in embedded mode **or**
   `sqlite-vec`, queried by `search_docs` for prose questions ("how do I subset a
   BAM by region?"). Embeddings come from the LLM backend's embedding endpoint if
   it serves one (vLLM and llama.cpp-server do; keeps the agent itself free of
   local ML dependencies and GPU-vendor concerns), else from a local
   `sentence-transformers` model as an optional install.

## 6. Agent profiles & memory (two-tier model)

An **agent profile** is a per-user, named memory document recording what the agent
has learned, and on which LLM backend each learning was made (small models differ —
a workaround for one backend may not apply to another). On session start the user
picks an existing profile or creates a new one.

### 6.1 Tiers

* **Tier 1 — global standing notes.** Injected into **every system prompt of every
  agent** (orchestrator and subagents). Only *stable*, universally useful facts:
  what cluster this is, scheduler, filesystem layout, module system quirks, site
  policies. Hard cap enforced (default 300 tokens). *Not* for time/date — those are
  injected dynamically at prompt render time (§4.3).
* **Tier 2 — profile memories.** Injected into the **orchestrator's** system prompt
  only: task learnings, user preferences, struggle notes (§4.4), backend-specific
  workarounds (tagged with the backend). Cap default 800 tokens.

A third, RAG-retrieved tier was considered and deliberately **deferred** to keep v1
simple; the storage format below is chosen so it can be added later without
migration (a `tier: 3` value is simply not used yet).

### 6.2 Storage format

Human-editable markdown per profile at
`~/.HolisticProcessingComputeAgent/profiles/<name>.md`: YAML front-matter for
metadata (profile name, created, default backend), then one memory per block under
`## [tier1]` / `## [tier2]` headings, each block with a small inline metadata line
(`<!-- backend: qwen3-6b, created: 2026-07-16, kind: struggle -->`). Markdown, not
JSON, because users will edit this in vim/nano (§6.4) and it must survive hand edits;
the parser must be lenient and report problems clearly.

### 6.3 Writing memories

* `\memorize [TEXT]` — appended verbatim to tier 2 (user chooses tier via a quick
  modal, default 2).
* `\conclude` — the orchestrator analyses the conversation and **proposes** memory
  blocks; a modal shows the proposals; the user approves/edits/rejects each
  (HITL, consistent with §5.3 — a small model writes these, so review is essential).
* Struggle notes (§4.4) follow the `\conclude` approval path.

### 6.4 Size warnings & manual cleanup

Token counts per tier are tracked (tokenizer of the backend if exposed via
`/tokenize`, `tiktoken` as fallback, chars/4 as last resort — this is a soft limit).
When a tier exceeds its cap, a persistent notification offers, in hotkey-bar style:

* **(e) edit externally** — via Textual's `App.suspend()`: restore the terminal,
  `subprocess.call([editor, profile_path])`, reinstate the TUI on exit. Editor
  resolution: settings override → `$VISUAL` → `$EDITOR` → `nano`. Works inside
  tmux/screen/zellij. On return: re-parse, validate, report problems; the user can
  move memories between tiers simply by moving blocks between headings.
* **(s) summarize with LLM** — the model drafts a condensed version into a *proposal
  file*, then the same external editor opens for the user to approve/adjust
  (never auto-applied).
* **(d) defer** — dismiss until next threshold crossing or app start.

## 7. Configuration

`~/.HolisticProcessingComputeAgent/settings.json`, editable via the top-bar settings
menu and by hand. Sketch:

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
  "memory": {
    "tier1_token_cap": 300,
    "tier2_token_cap": 800
  },
  "clipboard": { "mode": "auto", "command": null, "osc52_limit_kb": 74 },
  "editor": null,
  "rag": { "store": "chromadb", "embedding": "sentence-transformers/all-MiniLM-L6-v2" }
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
* `chromadb` (embedded) **or** `sqlite-vec` — RAG store
* `sentence-transformers` — embeddings (optional if backend provides an embedding
  endpoint)
* `pyyaml` — signature library, skills, front-matter
* `tiktoken` — token estimates (soft limits)

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
   with dry-run gate, right-column process list with (i)/(k)/(a).
6. **Slurm layer:** job DB, `submit_job`/`job_status`/`cancel_job`, background
   poller, log collection.
7. **Log triage:** signature library + report generator + log-explainer subagent.
8. **Safety layer:** destructive-op interrupts, hardlink trash, restore, TTL cleanup.
9. **Profiles & memory:** two-tier storage, `\memorize`, `\conclude` with approval
   modal, size warnings, `App.suspend()` editor round-trip.
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