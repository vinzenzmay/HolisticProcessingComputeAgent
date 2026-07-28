# HolisticProcessingComputeAgent (HPCA)

A terminal-based AI agent that uses local or remote LLMs to help users of an HPC
Slurm cluster with biomedical data processing. It runs on a compute node, talks to
an OpenAI-compatible LLM backend (e.g. vLLM behind an SSH tunnel), and provides a
three-column TUI: sessions, chat, and running sub-processes/jobs.

See [project.md](project.md) for the full design document.

## Install

HPCA is a Python package (requires Python ≥ 3.11) with a single console script,
`hpca`. Install it into a virtual environment with either `uv` or `pixi`.

### With uv

```bash
uv venv
uv pip install -e '.[dev]'   # drop [dev] to skip the test dependencies
```

### With pixi

```bash
pixi install            # base environment
pixi install -e dev     # environment including the test dependencies
```

This installs the `hpca` entry point (defined in [pyproject.toml](pyproject.toml)).
Semantic doc search (RAG) uses your LLM backend's `/v1/embeddings` endpoint — no
local embedding model is bundled, so that feature needs a backend that serves
embeddings.

## Run

```bash
hpca            # or: python -m hpca
```

Configuration lives at `~/.HolisticProcessingComputeAgent/settings.json` and can be
edited from the in-app config editor — press `c` while the sessions column is
focused. The same directory holds the job database (`hpca.db`), agent profiles,
skills, and the deletion trash.

The TUI is a three-column layout — sessions (left), chat (middle), running
processes and cluster jobs (right). A few of the top-level keys:

| Key | Action |
|---|---|
| `c` | open the config editor |
| `m` | manage LLM backends |
| `a` | manage profiles & learnings (memories) |
| `ctrl+l` | switch the LLM backend for the current session |
| `shift+tab` | cycle the agent mode (see below) |
| `q` | quit (from the sessions column) |

Typing `/` (or `\`) in the chat entry lists the available slash commands and
your skills (invoke a skill with `/<skill>`; see *Skills* below).

## Features

* **Three-column TUI** — sessions on the left, the chat with the agent in the
  middle, and live local sub-processes and Slurm jobs on the right. Built on
  [Textual](https://textual.textualize.io/).
* **Concurrent per-session turns** — each session runs its own agent turn
  independently, with a per-session model line, working indicator, and context
  meter, so one session can be busy while you work in another.
* **Agent modes** — `manual`, `auto` and `full auto` control how much the agent
  does on its own versus asking first (see *Agent modes* below).
* **Single orchestrator + firewalled sub-loops** — one orchestrating agent holds
  the full tool registry and delegates two narrowly-scoped jobs to context-isolated
  sub-loops: a **doc-researcher** (RAG questions, via the `ask_docs` tool) and a
  **log-explainer** (failure diagnosis, inside `get_job_report`). Their raw
  retrieval and raw logs never enter the orchestrator's context.
* **Typed tool suite, no free-form shell** — every action (create/run scripts,
  submit/track jobs, file operations, doc lookups) goes through a typed,
  schema-validated tool and a single internal runner; even `run_bash` funnels
  through a syntax-checked throwaway script rather than a raw shell.
* **Dry-run & verification gates** — scripts and job submissions are syntax-checked
  natively (`bash -n`, `py_compile`, `snakemake -n`, `sbatch --test-only`) and then
  passed through a mostly-deterministic gate that checks CLI flags and API usage
  against indexed man pages, `--help` output, and source, before anything runs.
* **Destructive-operation safety net** — deletes, overwrites, kills, and cancels
  require explicit confirmation, and small files are hardlinked into a timestamped
  trash directory (with a TTL, cleaned on start) before being removed.
* **Job tracking** — a background poller watches cluster jobs (via `sacct`) and
  local subprocesses, records everything in an sqlite DB, and delivers terminal
  outcomes back into the conversation so the agent can react (a finished local
  process arrives with its exit code and a log tail; a cluster-job event announces
  the new state and points at the logs).
* **Log triage** — instead of dumping raw multi-MB logs at the model, a triage
  pipeline matches a user-extensible signature library, scores keyword candidates,
  and hands a compact structured report to the log-explainer, which explains why a
  job failed, its state, a suggested fix, and a finickiness estimate.
* **Grounded (RAG) answering** — the agent is steered to answer technical questions
  about external programs and APIs against indexed docs, man pages, and source
  rather than from model weights, routing them to the doc-researcher, which is
  asked to cite its sources and to mark answers it could not ground. (This policy
  is prompt-driven guidance to the model, not a hard code-enforced guarantee.)
* **Profiles & memory** — per-user profiles record durable learnings and
  preferences across two scopes: `system-prompt` memories injected into the
  orchestrator's prompt (under a token budget) and `rag` memories retrieved only
  when they match the current request. Memories are proposed for your approval via
  `/memorize` and `/conclude` and stored as hand-editable markdown.
* **Context compaction** — a long session is folded into a summary before it
  overflows the model's window (the meter above the chat shows how full it is),
  and you can fold it yourself with `/compact` — adding what the summary has to
  keep, or the step you are about to take, so it is written for what comes next.
  Only what the model receives is folded; the chat itself keeps every message.
* **Skills** — user-defined procedure files the agent follows for specific tasks
  (see *Skills* below).
* **Configurable LLM backends** — talk to any OpenAI-compatible endpoint (vLLM,
  llama.cpp-server, …); manage backends and switch per session from the TUI.
* **Clipboard that works under tmux** — system-clipboard copy via OSC 52 with a
  tmux paste-buffer fallback (see *Clipboard under tmux* below).

## Agent modes

Each session runs in one of three modes, shown on the line right above the chat
entry and cycled with `shift+tab` (`ctrl+m` also works in terminals whose
keyboard protocol can distinguish it from Enter — most cannot):

* **manual** — every script or command the agent wants to run is shown to you
  first; `y` runs it, `n` skips it (the agent is told you declined and will not
  retry it).
* **auto** — the agent works until the task is done without asking for
  confirmation. Destructive operations (delete, overwrite, kill, cancel) still
  require your approval.
* **full auto** — auto without the destructive-operation approvals: nothing
  pauses, nothing asks. The trash/backup layer still backs deletions and
  overwrites of small files, but this mode is otherwise on your own risk.

There was a fourth, `plan`, in which the agent executed nothing and drafted a
checklist for you to approve. It is gone: the `/plan` skill below does the same
job better — it settles the decisions with you first and writes the plan to a
file that outlives the session — with no mode to enter and leave.

New sessions start in `agent.default_mode` (settings, default `manual`); each
session remembers its own mode across restarts.

## Skills

Skills are user-defined procedure files — markdown (with optional YAML front
matter) or YAML — that tell the agent how to handle a specific kind of task. Each
skill has a `name`, a one-line `description`, optional `triggers`, and a `body`
(the procedure itself). The skill list is deliberately **not** put in the model's
prompt (no matter how many you define, the prompt cost stays flat): the system
prompt carries only a one-line note that skills exist and how to fetch one. A
skill's body reaches the model only when it is actually needed — see *Use a
skill* below.

### Where skills live (levels)

A skill is stored at one of four levels, which decides who sees it. On a name
collision the most specific level wins (**project > profile > global >
built-in**):

* **built-in** — shipped with HPCA, so a fresh install already has them (see
  *Built-in skills* below). Read-only: `/skill-remove` never offers one, and
  writing your own skill with the same name simply shadows it.
* **global** — visible to every profile. Stored under
  `~/.HolisticProcessingComputeAgent/skills/_shared/`.
* **profile** — the current profile only. Stored under
  `~/.HolisticProcessingComputeAgent/skills/<profile>/`.
* **project** — tied to the directory you launch `hpca` from, and only visible
  while running there. Stored under a hidden `.hpca/skills/` in that directory, so
  a repo can carry its own procedures without them leaking into other projects.

### Built-in skills

Two ship with HPCA, both aimed at settling a plan before any of it gets built:

* **`/grillme`** — the agent interviews you about a plan, decision or idea, one
  question at a time, recommending an answer for each and looking facts up itself.
  It does not act until you say you have reached a shared understanding.
* **`/plan`** — `/grillme` first, then the agent writes a `specs.md` in the working
  directory: the settled decisions and their reasoning, the files to touch, the
  steps in order, and how to verify them — detailed enough to implement in a fresh
  session that saw none of the conversation. It stops there; building is a separate
  session. This replaced the old `plan` mode.

### Create a skill

Run `/skill-creator` in the chat entry. A form collects the name, description,
level, and body (tab moves between fields); `esc` saves after a confirmation, or
cancels an empty form. You can also create skill files by hand — drop a `.md`,
`.yaml`, or `.yml` file into the appropriate directory above. A file without front
matter is still a valid skill (its filename becomes the name and its text the
body).

### Use a skill

There are two ways a skill runs, cheapest first:

* **Invoke it yourself** — type `/<skill>` (optionally with a prompt, e.g.
  `/grill-me review my sbatch script`). Skills show up in the same `/` menu as the
  built-in commands (↑/↓ to select, `⇥` to complete), and the skill's full
  procedure is handed to the model inline for that turn. This is the direct,
  predictable path. Only single-token skill names are reachable this way (the
  parser splits on the first space), and a built-in command wins a name clash.
* **Let the agent reach for it** — the model is told skills exist and can call the
  `read_skill` tool on its own when a request seems to match one (passing any name
  returns the available skills). Nothing is invoked automatically without either
  your `/<skill>` or the model deciding to load one.

Run `/skills-list` to see every skill the current profile can see, tagged with the
level each one resolves to (the profile's own are untagged).

### Remove a skill

Run `/skill-remove` and pick from the list. Only removable skills are offered —
the current profile's own skills and this project's skills. Global (`_shared`)
skills are intentionally not offered, since removing one would silently change
every other profile that sees it; delete those by removing the file directly.

## Clipboard under tmux

For the system-clipboard path (OSC 52) to work inside tmux you need one of:

```tmux
set -g set-clipboard on
# or, for tmux >= 3.3:
set -g allow-passthrough on
```

The tmux paste-buffer fallback (`prefix + ]`) works regardless of these settings.

## Development

```bash
uv run pytest        # or: pixi run -e dev pytest
```
