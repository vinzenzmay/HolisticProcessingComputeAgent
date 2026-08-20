#!/usr/bin/env bash
# A REAL turn: the Rust front-end → the protocol → hpca.core → the graph → the
# cluster's vLLM, and the reply back out as chat.append frames.
#
# This is the end-to-end case. HPCA_HOME is a scratch directory, so the real
# app dir and its sessions are untouched.
#
#   HPCA_LLM_KEY=... ./liveturn.sh ["a question for the agent"]
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
tmp="${TMPDIR:-/tmp}/hpca-live-$$"
mkdir -p "$tmp/home"
sock="$tmp/core.sock"
log="$tmp/core.log"

export PATH="$HOME/.cargo/bin:$HOME/.pixi/bin:$PATH"
export HPCA_HOME="$tmp/home"

base_url="${HPCA_LLM_URL:-http://localhost:20001/v1}"
key="${HPCA_LLM_KEY:-}"
model="${HPCA_LLM_MODEL:-Qwen3.8-27B-FP8}"
turn="${1:-In one sentence: what does the squeue command do?}"

cat >"$tmp/home/settings.json" <<JSON
{
  "llm": {
    "base_url": "$base_url",
    "api_key": "$key",
    "model": "$model"
  },
  "database": { "local_cache": false }
}
JSON

cd "$repo"
HPCA_PROBE_SESSION="$(pixi run -e dev python "$here/seed.py" | tail -1)"
export HPCA_PROBE_SESSION
export HPCA_PROBE_TURN="$turn"
export HPCA_PROBE_SECS="${HPCA_PROBE_SECS:-180}"
echo "session $HPCA_PROBE_SESSION"
echo "asking : $turn"
echo

pixi run -e dev python "$here/serve.py" "$sock" >"$log" 2>&1 &
core=$!
trap 'kill $core 2>/dev/null; rm -rf "$tmp"' EXIT

for _ in $(seq 1 200); do
  [ -S "$sock" ] && break
  kill -0 $core 2>/dev/null || break
  sleep 0.2
done
if [ ! -S "$sock" ]; then
  echo "the core never bound its socket:"
  tail -40 "$log"
  exit 1
fi

"$here/target/debug/hpca-ui" --probe "$sock"
status=$?

echo
echo "── the core's own log ───────────────────────────────────"
tail -30 "$log"
echo "─────────────────────────────────────────────────────────"
exit $status
