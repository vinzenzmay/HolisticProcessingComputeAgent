#!/bin/bash
# Run the embedding sidecar (for RAG / doc search in HPCA).
#
# This serves all-MiniLM-L6-v2 on port 20000 — tiny (~0.08 GiB), so it shares a
# GPU with the chat model rather than getting its own node. llm.a40-l40.sh
# launches it in the background, pinned to the first card of the allocation, so
# the two land on the *same* physical GPU and split its memory (LLM 0.90,
# embed 0.05). Nothing here runs on the login node.
#
# Runs in the foreground; the caller backgrounds it, e.g.:
#   ./embed.sh > embed-server.log 2>&1 &
#
# Env overrides the caller may set:
#   VENV_DIR          which venv to source (default ./venv)
#   EMBED_PORT        bind port, if 20000 is taken on the node
#   EMBED_EXTRA_ARGS  extra server flags (e.g. --dtype float16)
#   HF_HOME           weights cache; inherited from the parent job, which is
#                     what keeps both processes on one download
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

MODEL="sentence-transformers/all-MiniLM-L6-v2"
PORT="${EMBED_PORT:-20000}"
GPU_MEM_UTIL=0.05
# Unquoted on use below: this is a flag list, not one argument.
EXTRA_ARGS="${EMBED_EXTRA_ARGS:-}"

# Same default as llm.a40-l40.sh; normally inherited from it. ~90 MB here, so
# this one is not the reason to move the cache off $HOME.
export HF_HOME="${HF_HOME:-${HPCA_WEIGHTS_DIR:-$PWD/weights}}"

VENV_DIR=$(realpath "${VENV_DIR:-./venv}")
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV_DIR" >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo "Embedding sidecar: $MODEL on $(hostname -i):$PORT (gpu-mem $GPU_MEM_UTIL)"
echo "  venv: $VENV_DIR${EXTRA_ARGS:+  extra args: $EXTRA_ARGS}"

# Drop a discovery manifest for HPCA. This script's own EXIT trap is lost at
# the exec below — that's fine: the parent chat job's trap (glob on jobid)
# plus HPCA's stale-manifest reaping clean up this sidecar's manifest.
source "$(dirname "$(readlink -f "$0")")/write_manifest.sh" embedding "$MODEL" "$PORT" false

# The parent chat job exports VLLM_API_KEY for its own server; vLLM would pick
# it up here too and demand a key, contradicting needs_key=false above. The
# embedding endpoint stays open, so drop the inherited key.
unset VLLM_API_KEY

# exec so signals (job cancel / time limit) reach vLLM directly.
# $EXTRA_ARGS intentionally unquoted — it is a flag list (empty by default).
# shellcheck disable=SC2086
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    $EXTRA_ARGS
