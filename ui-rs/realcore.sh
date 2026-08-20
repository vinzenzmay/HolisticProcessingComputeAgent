#!/usr/bin/env bash
# The Rust front-end against a REAL core: hpca.core.build_service on a socket,
# with the actual databases, session store, graph and pollers behind it.
#
# HPCA_HOME is pointed at a scratch directory, so this never touches the app
# dir the user's own runs keep their sessions in.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
tmp="${TMPDIR:-/tmp}/hpca-realcore-$$"
mkdir -p "$tmp/home"
sock="$tmp/core.sock"
log="$tmp/core.log"

export PATH="$HOME/.cargo/bin:$HOME/.pixi/bin:$PATH"
export HPCA_HOME="$tmp/home"

cd "$repo"
# The core cannot create a session over the protocol yet, so seed one first.
HPCA_PROBE_SESSION="$(pixi run -e dev python "$here/seed.py" | tail -1)"
export HPCA_PROBE_SESSION
echo "seeded session $HPCA_PROBE_SESSION"

pixi run -e dev python "$here/serve.py" "$sock" >"$log" 2>&1 &
core=$!
trap 'kill $core 2>/dev/null; rm -rf "$tmp"' EXIT

for _ in $(seq 1 150); do
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
tail -25 "$log"
echo "─────────────────────────────────────────────────────────"
exit $status
