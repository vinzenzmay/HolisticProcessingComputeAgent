#!/bin/bash
# Create ./venv with mainline vLLM, for the A40 / L40 nodes (llm.a40-l40.sh).
#
# Ada and Ampere need nothing exotic: mainline vLLM ships manylinux wheels that
# load fine on Rocky 9, so a plain venv is enough — no container, no compiler,
# no CUDA toolkit (the wheels bundle their own CUDA runtime; only an NVIDIA
# driver on the GPU node is required, and the cluster already has one).
#
# Run this ON THE CLUSTER — it needs network, but no GPU. A login node is fine
# if it allows outbound HTTPS and has ~15 GB of space for the venv; otherwise
# grab an interactive slot: srun -p standard --mem=8G --time=2:00:00 --pty bash
#
#   ./setup_venv.sh
#   ./setup_venv.sh --upgrade     # bump an existing venv to the latest vLLM
#
# Python: 3.10-3.13, found automatically. Override with PYTHON_BIN=/path/to/python3.12
# if the cluster's default python is older (module load python/3.12, a conda or
# miniforge interpreter, anything — only the interpreter is borrowed, the venv
# is self-contained).
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

VENV_DIR="$(pwd)/venv"

# Pick an interpreter: an explicit PYTHON_BIN wins, else the newest supported
# python3.X on PATH. Checked here rather than at `venv` time because the failure
# is otherwise a wheel-resolution error 200 lines deep in pip.
find_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        [[ -x "$PYTHON_BIN" ]] || { echo "ERROR: PYTHON_BIN=$PYTHON_BIN is not executable" >&2; return 1; }
        echo "$PYTHON_BIN"; return 0
    fi
    local cand
    for cand in python3.12 python3.13 python3.11 python3.10 python3; do
        local path
        path=$(command -v "$cand" 2>/dev/null) || continue
        if "$path" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,14) else 1)' 2>/dev/null; then
            echo "$path"; return 0
        fi
    done
    echo "ERROR: no python 3.10-3.13 found on PATH." >&2
    echo "       Try 'module avail python' on this cluster, or install one" >&2
    echo "       (e.g. miniforge), then re-run with PYTHON_BIN=/path/to/python3.12" >&2
    return 1
}
PYTHON_BIN="$(find_python)"
echo "Python: $PYTHON_BIN ($("$PYTHON_BIN" -V))"

# Upgrading site-packages under a live server can crash it (vLLM imports some
# modules lazily). Warn if jobs from this directory are still running.
if command -v squeue >/dev/null 2>&1; then
    RUNNING=$(squeue -u "$USER" -h -o "%i %j" | grep -E "qwen" || true)
    if [[ -n "$RUNNING" ]]; then
        echo "WARNING: vLLM jobs appear to be running:"
        echo "$RUNNING"
        read -r -p "Continue? Running jobs may crash. [y/N] " ans
        [[ "$ans" == [yY]* ]] || { echo "Aborted."; exit 1; }
    fi
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Creating $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

OLD=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "not installed")
python -m pip install --quiet --upgrade pip setuptools wheel
if [[ "${1:-}" == "--upgrade" ]]; then
    python -m pip install --upgrade vllm
else
    python -m pip install vllm
fi
NEW=$(python -c "import vllm; print(vllm.__version__)")

echo
echo "vLLM: $OLD -> $NEW"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda)"
pip check

# The two things llm.a40-l40.sh needs from this venv, asked now rather than 20
# minutes into a model load: the server entrypoint imports, and this vLLM knows
# the architecture the checkpoint declares.
python -c "import vllm.entrypoints.openai.api_server" >/dev/null
python - <<'PY'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
missing = [a for a in ("Qwen3_5ForConditionalGeneration", "Qwen3_5MTP") if a not in archs]
if missing:
    raise SystemExit(
        "ERROR: this vLLM does not register " + ", ".join(missing) + ".\n"
        "       Re-run with --upgrade; Qwen3.8-27B needs vLLM >= 0.26.0\n"
        "       (Qwen3_5MTP is only needed for speculative decoding — set\n"
        "        NUM_SPEC_TOKENS=0 in llm.a40-l40.sh to serve without it)."
    )
print("architecture check OK: Qwen3_5ForConditionalGeneration + Qwen3_5MTP registered")
PY
echo "Import check OK."
echo
echo "Next: download the weights (~30.4 GB, needs >=16 GB of RAM), then submit:"
echo "  HF_HOME=$(pwd)/weights ./venv/bin/hf download Qwen/Qwen3.8-27B-FP8"
echo "  head -c 32 /dev/urandom | base64 > ~/.vllm_api_key && chmod 600 ~/.vllm_api_key"
echo "  sbatch llm.a40-l40.sh"
