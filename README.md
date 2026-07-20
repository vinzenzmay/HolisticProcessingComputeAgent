# HolisticProcessingComputeAgent (HPCA)

A terminal-based AI agent that uses local or remote LLMs to help users of an HPC
Slurm cluster with biomedical data processing. It runs on a compute node, talks to
an OpenAI-compatible LLM backend (e.g. vLLM behind an SSH tunnel), and provides a
three-column TUI: sessions, chat, and running sub-processes/jobs.

See [project.md](project.md) for the full design document.

## Install

```bash
uv venv && uv pip install -e '.[dev]'
```

## Run

```bash
hpca
```

Configuration lives at `~/.HolisticProcessingComputeAgent/settings.json` and can be
edited from the in-app settings menu (`s`).

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
* **plan** — the agent executes nothing and instead drafts a checklist
  (look-around commands each ask first). When a plan is ready, a dialog lets you
  edit the checklist and hand it over for execution — on auto (`ctrl+r`) or
  step-by-step under manual approval (`ctrl+s`) — or keep refining it (`esc`).

New sessions start in `agent.default_mode` (settings, default `manual`); each
session remembers its own mode across restarts.

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
uv run pytest
```
