#!/usr/bin/env bash
set -euo pipefail

echo "=== GPU Nodes in Draining State ==="
echo ""
sinfo -N -o '%N %T %D %b %a %P' | grep -i 'drain' | grep -i 'gpu' || echo "No draining GPU nodes found."
echo ""
echo "=== All GPU Node States ==="
sinfo -N -o '%N %T %D %b %a %P' | grep -i 'gpu' | sort
