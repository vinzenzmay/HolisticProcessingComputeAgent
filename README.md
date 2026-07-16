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
