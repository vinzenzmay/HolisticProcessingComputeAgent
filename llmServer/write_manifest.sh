# shellcheck shell=bash
# write_manifest.sh — drop a JSON discovery manifest for a vLLM endpoint.
#
# SOURCE this (don't execute it): the EXIT trap it registers must live in the
# caller's shell so the manifest is removed when the job's script exits.
#
#   source "$SCRIPT_DIR/write_manifest.sh" <role> <model> <port> <needs_key> [ctx_len]
#
#     role      "llm" or "embedding"
#     model     the id vLLM serves at /v1/models (the --model value)
#     port      the ACTUAL bound port (after any fallback)
#     needs_key "true" or "false" (display hint only — no secrets are written)
#     ctx_len   optional; omit/empty for embeddings
#
# Writes <dir>/<jobid>-<port>.json and registers a best-effort EXIT trap that
# removes ALL manifests for this job id (glob on "<jobid>-*"), so a chat job's
# trap also cleans up its co-located embedding sidecar's manifest (they share
# $SLURM_JOB_ID).
#
# The directory is the SHARED group location below, so endpoints are visible to
# everyone using HPCA, not just their owner. Job ids are unique cluster-wide, so
# manifests from different users cannot collide. Override with
# HPCA_ENDPOINTS_DIR (useful for testing without touching the shared dir).
#
# BEST-EFFORT BY DESIGN: because that directory is shared and outside any one
# user's control, a failed write must never take the server down with it. If the
# manifest cannot be written this warns and continues — you lose auto-discovery
# for that job, not the job. This matters because embed.sh sources this under
# `set -euo pipefail`.
#
# LIMITATION: a hard kill (OOM, NODE_FAIL, walltime SIGKILL) cannot run the
# EXIT trap and will leave a stale manifest behind. That is expected — HPCA
# reaps stale manifests via a SLURM liveness check.
#
# Safe under `set -euo pipefail`.

__wm_role="${1:?write_manifest: role required}"
__wm_model="${2:?write_manifest: model required}"
__wm_port="${3:?write_manifest: port required}"
__wm_needs_key="${4:?write_manifest: needs_key required}"
__wm_ctx_len="${5:-}"

__wm_jobid="${SLURM_JOB_ID:-manual-$$}"
__wm_user="${USER:-$(whoami)}"
__wm_node="$(hostname -s)"
__wm_ip="$(hostname -i | awk '{print $1}')"
__wm_started="$(date -u +%FT%TZ)"

__wm_dir="${HPCA_ENDPOINTS_DIR:-/data/cephfs-1/work/groups/cubi/tools/hpca_connections}"
__wm_file="$__wm_dir/${__wm_jobid}-${__wm_port}.json"
__wm_ok=1

# Every step guarded: see BEST-EFFORT above. `|| __wm_ok=0` also keeps `set -e`
# from killing the caller on a shared-filesystem hiccup.
mkdir -p "$__wm_dir" 2>/dev/null || __wm_ok=0

if [[ "$__wm_ok" == 1 ]]; then
    {
        printf '{\n'
        printf '  "role": "%s",\n'    "$__wm_role"
        printf '  "model": "%s",\n'   "$__wm_model"
        printf '  "jobid": "%s",\n'   "$__wm_jobid"
        printf '  "user": "%s",\n'    "$__wm_user"
        printf '  "node": "%s",\n'    "$__wm_node"
        printf '  "ip": "%s",\n'      "$__wm_ip"
        printf '  "port": %s,\n'      "$__wm_port"
        if [[ -n "$__wm_ctx_len" ]]; then
            printf '  "ctx_len": %s,\n' "$__wm_ctx_len"
        fi
        printf '  "needs_key": %s,\n' "$__wm_needs_key"
        printf '  "started": "%s"\n'  "$__wm_started"
        printf '}\n'
    } > "$__wm_file" 2>/dev/null || __wm_ok=0
fi

if [[ "$__wm_ok" == 1 ]]; then
    # World-readable: this is a shared discovery directory, and HPCA runs as
    # whoever is looking for an endpoint, not as the job's owner.
    chmod 644 "$__wm_file" 2>/dev/null || true
    echo "manifest written: $__wm_file"
    # Best-effort cleanup on exit. Glob on the job id so a chat job also removes
    # its embedding sidecar's manifest (shared $SLURM_JOB_ID). The glob cannot
    # match another user's file — job ids are unique cluster-wide.
    trap 'rm -f "$__wm_dir/${__wm_jobid}-"*.json 2>/dev/null' EXIT
else
    echo "WARNING: could not write manifest to $__wm_dir" >&2
    echo "         HPCA auto-discovery is off for this job; the server is fine." >&2
    echo "         Check the directory exists and is writable by $__wm_user." >&2
fi
