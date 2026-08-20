#!/usr/bin/env bash
# End-to-end interop: the Rust front-end against a core running the real
# hpca.protocol / hpca.transport. Prints the core's view (which commands it
# accepted or rejected) and the front-end's view (what it decoded and drew).
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
tmp="${TMPDIR:-/tmp}/hpca-interop-$$"
mkdir -p "$tmp"
sock="$tmp/core.sock"
log="$tmp/core.log"

export PATH="$HOME/.cargo/bin:$HOME/.pixi/bin:$PATH"

cd "$repo"
pixi run -e dev python "$here/fakecore.py" "$sock" --script >"$log" 2>&1 &
core=$!
trap 'kill $core 2>/dev/null; rm -rf "$tmp"' EXIT

for _ in $(seq 1 100); do
  [ -S "$sock" ] && break
  sleep 0.2
done
if [ ! -S "$sock" ]; then
  echo "the core never bound its socket:"
  cat "$log"
  exit 1
fi

"$here/target/debug/hpca-ui" --probe "$sock"
status=$?

echo
echo "── what the core made of our commands ───────────────────"
cat "$log"
echo "─────────────────────────────────────────────────────────"
exit $status
