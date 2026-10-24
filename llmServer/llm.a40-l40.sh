#!/bin/bash
#SBATCH --job-name=qwen38-27b-ada
#SBATCH --output=vllm.a40-l40.out
#SBATCH --gres=gpu:l40:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-0
#SBATCH --partition=gpu
#
# Qwen3.8-27B-INT4 (RedHatAI, 4-bit weights) on A40 or L40 (48GB), mainline
# vLLM, vision on. LLM on port 20001, with the embedding sidecar riding along
# on 20000.
#
#   sbatch llm.a40-l40.sh                     # 2x L40, TP=2 (the default)
#   sbatch --gres=gpu:a40:2 llm.a40-l40.sh    # 2x A40, TP=2
#   sbatch --gres=gpu:l40:1 llm.a40-l40.sh    # 1 card, 163840 window
#   PARALLEL_MODE=dp sbatch llm.a40-l40.sh    # 2 replicas instead of TP
#
# 4-BIT IS THE DEFAULT SINCE 2026-10-23, and that changed what one card is
# worth. At 8-bit, TP=2 was load-bearing: only sharding ~27.5 GiB of weights
# AND the 4 KV heads left room for the full window, and one card served
# 112000. At 4-bit one card has been run at 163840 with vision on (the
# 2026-10-23 job, `--gres=gpu:l40:1`); two cards serve the full 262144. THE
# CONTEXT WINDOW STILL FOLLOWS THE ALLOCATION, resolved from $TP and printed at
# startup. The speed and sizing figures in this file were MEASURED AT 8-BIT and
# are kept as that; 4-bit reads roughly half the weight bytes per token, so
# expect faster, and replace them once a 4-bit run has been measured.
#
# The served name is advertised at /v1/models and every client selects the model
# by it: SETUP.md, HPCA's model picker and any VS Code custom-model entry must
# all use the same string, or the client asks for a model id that does not
# exist. Change --served-model-name and you change all of them.
#
# ---- THE 4-BIT REVERT --------------------------------------------------------
# The 4-bit revision (7f5711d..ca2ce1c) SERVED CORRUPTED OUTPUT: 8-12 correct
# tokens, then one token repeated forever, deterministically at temperature 0.
#
#     'The capital of France is' -> ' Paris.\nThe capital of France is
#                                    Paris.\nWhat is the is is is is is...'
#
# THE CULPRIT WAS --kv-cache-dtype turboquant_4bit_nc, NOT the 4-bit weights.
# The evidence matters, because the obvious reading is wrong and acting on it
# costs tok/s for nothing:
#   - Teacher forcing (one token per request, so every token comes from a fresh
#     PREFILL) gave clean output from the same weights and prompt. The fault is
#     in decode alone; weights, Marlin and the chat template are all exonerated.
#   - It fails at the FIRST decode step — token 1 (prefill) right, token 2
#     already wrong. Not accumulation over a long window.
#   - TurboQuant was the only thing on that path: `Using TURBOQUANT attention
#     backend out of potential backends: ['TURBOQUANT']`. Prefill uses raw K/V,
#     decode reads the compressed cache — exactly the split the test isolated.
#   - MTP exonerated by its own metrics: per-position acceptance read 1.000
#     during the garbage, i.e. the 27B verifier agreed the drafts were its own
#     output. It reproduced a broken decode path rather than causing it.
#   - Prefix caching out: `Prefix cache hit rate: 0.0%` throughout.
#
# So turboquant_4bit_nc is broken on this hybrid architecture (16 full-attention
# layers of 64, head_dim 256), not merely lossy. Worth reporting upstream.
# The WEIGHTS were never implicated, which is why 4-bit weights could come back
# as the default — with the KV cache kept at fp8. FALLBACK 1 is the way back to
# 8-bit, the configuration known-good before that.
#
# ---- DECODE SPEED ------------------------------------------------------------
# Decode at batch 1 reads the whole model once per token, so it is pure memory
# bandwidth. ~27 GB moved per token at 8-bit; 17 tok/s measured gives ~459 GB/s
# actually achieved, which is the honest constant to predict with — it already
# contains vLLM's ~59% batch-1 efficiency, unlike the 780 GB/s headline.
#
# The KV cache is read every step too, so tok/s falls as a session FILLS its
# window. fp8 costs ~34.8 kB/token; TP=2 halves the per-card share:
#
#   context in use    1 card bare / MTP    2 cards bare / MTP
#      ~8k               16.8  /  ~30         ~27  /  ~47
#     112k               15.0  /  ~26         ~24  /  ~42
#     262k               12.8  /  ~23         ~21  /  ~36
#
# TP=2 keeps ~1.6x of the theoretical 2x over PCIe (L40 has no NVLink — Ada
# removed it), ~1.9x over an A40 bridge. The job measures and prints which.
#
# Levers: TP (~1.6x), MTP (~1.76x) and 4-bit weights (~1.5x predicted from
# the bytes, unmeasured) are all taken. A quantised KV cache is NOT a lever; it is what broke the
# endpoint. FlashAttention is unavailable at any --kv-cache-dtype (head_dim 256
# is past the bundled kernels), so do not pass --attention-backend: vLLM picks
# FLASHINFER on its own and is right to.

# Under sbatch, $0 is the spooled copy in /var/spool/slurmd, so derive the
# script dir from the submit dir instead ($0 fallback for direct runs).
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"

# WHERE THE WEIGHTS LIVE. Default: ./weights beside this script. That is tens
# of GB, so on a cluster whose $HOME has a quota, point it at work/scratch instead —
# export HF_HOME (or HPCA_WEIGHTS_DIR) before sbatch, or edit the default here.
# It must be the SAME directory the download used, or the job silently starts a
# second full download from the compute node.
export HF_HOME="${HF_HOME:-${HPCA_WEIGHTS_DIR:-$SCRIPT_DIR/weights}}"

# Download once, on a node with network and >=16GB RAM:
#   HF_HOME=$PWD/weights ./venv/bin/hf download RedHatAI/Qwen3.8-27B-INT4
# The MTP head comes with it and IS read here, because NUM_SPEC_TOKENS is
# non-zero (the 2026-10-23 job started with MTP on, so the head is there).
MODEL="RedHatAI/Qwen3.8-27B-INT4"
SERVED_NAME="Qwen3.8-27B-INT4"

# ---- CONTEXT WINDOW: FOLLOWS THE ALLOCATION ----------------------------------
# THE TABLE BELOW IS THE 8-BIT ONE. At 4-bit every column has more room; only
# 163840 on one card has been run (2026-10-23). The two allocations differed by
# 5.24x in capacity at 8-bit, not 2x, so no single value was defensible across
# both. Worst-case concurrent users at MAX_NUM_SEQS=6:
#
#     MAX_MODEL_LEN     1 card    2 cards
#        112000           2.25      11.4
#        131072           1.93       9.8
#        163840           1.59       8.1
#        262144           0.98       5.1     <- 1 card cannot start at all
#
# The modes ask different questions: on one card, does a SINGLE session fit
# (hard ceiling ~235k); on two, how many fit (below ~196k, MAX_NUM_SEQS binds
# instead of memory). One card at 8-bit is a 1-2 user machine at any long
# context. PARALLEL_MODE=dp resolves to TP=1 and so to the smaller window,
# correctly — DP replicas each hold whole weights and whole KV on one card.
#
# 262144 is exactly the native window (max_position_embeddings), the largest
# value needing no YaRN and no VLLM_ALLOW_LONG_MAX_MODEL_LEN.
#
# THE COST: this is no longer a static single source of truth for the configs a
# HUMAN maintains. write_manifest.sh gets the resolved value so HPCA is always
# right, but VS Code's maxInputTokens is a file and can only be set for one mode
# — set it for the two-card default. If you run one card REGULARLY, prefer a
# static 163840 in both slots: too-high in VS Code is a 400 at the worst
# possible moment, whereas too-low only silently caps.
MAX_MODEL_LEN_TP2=262144    # full native window,        ~5.1 users on two cards
MAX_MODEL_LEN_TP1=163840    # run at 4-bit with vision, 2026-10-23 (8-bit: 112000)

# Costs 144 MiB of pool per slot whether occupied or not — the 48
# linear-attention layers preallocate recurrent state per sequence, so this is
# real memory unlike on a pure-attention model. Also a hard cap on concurrency,
# so it must sit ABOVE the worst-case user count or oversubscription cannot pay.
# 6 is chosen against the two-card case (~5.1 at 262144). On one card at 112000
# the worst case is ~2.25, so 6 is heavy oversubscription — deliberately: that
# mode is the queue-time escape hatch, its users do not all pin full windows,
# and vLLM preempts and recomputes rather than failing when they do. Raising it
# is cheap in memory but not in speed: MTP pays best at low concurrency, and
# past ~8 concurrent decodes the ~1.76x shrinks noticeably.
MAX_NUM_SEQS=6

# 8-bit KV: ~33.4 kB/token, half of fp16 — independent of the weights' bit
# width, and what made the full window fit on two cards at 8-bit. DO NOT swap this for a quantised-KV option
# without reading THE 4-BIT REVERT — turboquant_4bit_nc loaded clean, logged
# nothing, and served garbage for a day. fp8 keeps FLASHINFER, which is the
# backend the working configuration used.
KV_CACHE_DTYPE=fp8

# MTP speculative decoding, and the reason the 8-bit roofline was survivable.
# The checkpoint's own mtp.safetensors drafts NUM_SPEC_TOKENS tokens per step
# and the 27B model verifies the block in ONE pass over its weights. Rejection
# sampling makes it exactly lossless — only speed is ever at risk. Worth ~1.76x
# MEASURED ON THE 8-BIT BUILD (17 -> 30 tok/s on one card); not yet re-measured
# at 4-bit. 3 is Qwen's recommendation; higher adds progressively worse guesses
# at the cost of a real forward pass each step.
#
# Costs ~0.5 GiB of draft weights plus a 17th cached layer (the draft layer is
# full_attention), i.e. ~6% more KV per token — both folded into SIZING. Set to
# 0 to remove it entirely; see FALLBACK 11 first.
NUM_SPEC_TOKENS=3

VENV_DIR="$SCRIPT_DIR/venv"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: $VENV_DIR missing — run ./setup_venv.sh" >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Ampere/Ada only: a V100 would land on Marlin, which needs sm_80+, and fail
# deep inside a kernel launch. Refuse early.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [[ "$GPU_NAME" == *V100* || -z "$GPU_NAME" ]]; then
    echo "ERROR: got '${GPU_NAME:-none}'. This script needs an A40 or L40." >&2
    echo "       V100 (sm_70) has no Marlin kernels for these weights." >&2
    exit 1
fi
echo "GPU: $GPU_NAME"

# The one place the two supported cards behave differently on these weights.
if [[ "$GPU_NAME" == *A40* && "$MODEL" == *FP8* ]]; then
    echo "NOTE: A40 (sm_86) has no native FP8 — weights run weight-only via"
    echo "      fp8-Marlin. Correct but no FP8 compute win."
fi

# A40 is also ~20% slower at the thing that sets tok/s: 696 GB/s vs L40's 864.
if [[ "$GPU_NAME" == *A40* ]]; then
    echo "NOTE: A40 memory bandwidth is ~696 GB/s vs L40 ~864 — expect roughly"
    echo "      80% of the tok/s quoted in DECODE SPEED (so ~38, not ~47)."
fi

# Refuse to start without an API key — with an empty one vLLM serves
# unauthenticated. Via env var, NOT --api-key: command lines are world-visible
# in `ps aux` on the node, the environment is not.
API_KEY_FILE="$HOME/.vllm_api_key"
if [[ ! -s "$API_KEY_FILE" ]]; then
    echo "ERROR: API key file missing or empty: $API_KEY_FILE" >&2
    exit 1
fi
export VLLM_API_KEY="$(cat "$API_KEY_FILE")"

# 0.88 here, and the embedding sidecar beside it. That was 0.90 while the
# sidecar was all-MiniLM-L6-v2 (0.7 GiB held); BAAI/bge-m3 holds ~2.3 GiB, and an
# LLM holds ~2.2 GiB beyond its own budget (measured on the 4-bit sibling of this
# job: 42.2 GiB in nvidia-smi against 39.98 budgeted), so 0.02 (~0.9 GiB per
# card, ~28k KV tokens on one card) moved to the sidecar. The SIZING figures
# below predate this and are 0.9 GiB per card high.
GPU_UTIL=0.88

# 20001 by convention, and every client assumes it: HPCA's SSH-tunnel template,
# the VS Code entry, SETUP.md. Only if it is already taken on the node does the
# job fall back to a random port — and then it logs the port it did get, which
# the tunnel has to match.
PREFERRED_PORT=20001
if python3 -c "import socket; s=socket.socket(); s.bind(('', $PREFERRED_PORT)); s.close()" 2>/dev/null; then
    PORT=$PREFERRED_PORT
else
    PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    echo "WARNING: Port $PREFERRED_PORT in use, using $PORT instead"
fi

echo "vLLM is running on: $(hostname -i)"
echo "vLLM port: $PORT"
export | grep CUDA_VISIBLE_DEVICES

# nvidia-smi honours CUDA_VISIBLE_DEVICES, so this counts what SLURM actually
# gave us, and changing --gres stays the only edit needed.
NGPU=$(nvidia-smi -L | wc -l)

# TENSOR parallel by default. At 8-bit both arguments point the same way:
#   - memory: TP shards weights AND the 4 KV heads 2/2. Without both, a card
#     cannot hold one 262144 session at all, so TP is what makes the full window
#     available rather than merely faster.
#   - speed: DP does nothing for a single user — each replica still reads all
#     ~27 GB per token. TP halves that read, the entire bottleneck.
# DP IS NOT THE THROUGHPUT OPTION, which is the intuitive-but-wrong call and was
# stated wrongly in an earlier revision. vLLM reads the weights ONCE PER STEP
# and that read serves the whole batch, so per step for 4 users on two cards:
#     DP=2   27 GB on EACH card -> 54 GB total -> 4 tokens
#     TP=2   13.5 GB on each card -> 27 GB total -> 4 tokens
# TP produces the same tokens for half the bytes, winning on aggregate
# throughput as well as latency. DP only pulls ahead once decode is compute-
# bound; on an L40 that crossover is near batch ~70, and MAX_NUM_SEQS is 6.
# So DP is for A/B measurement, and it silently costs you the window.
if [[ "${PARALLEL_MODE:-tp}" == "dp" ]]; then
    DP=$NGPU
    TP=1
else
    DP=1
    TP=$NGPU
fi
echo "GPUs allocated: $NGPU -> --data-parallel-size $DP --tensor-parallel-size $TP"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Resolve the window now that $TP is known — see CONTEXT WINDOW. This is why
# there is no pre-flight check: a 262144 window on a single card, which vLLM
# rejects three minutes into startup after torch.compile has run, is now
# unrepresentable rather than merely caught.
if [[ "$TP" -ge 2 ]]; then
    MAX_MODEL_LEN=$MAX_MODEL_LEN_TP2
else
    MAX_MODEL_LEN=$MAX_MODEL_LEN_TP1
fi
echo "Context window: $MAX_MODEL_LEN (TP=$TP) — this is what the HPCA manifest"
echo "  will advertise; the VS Code maxInputTokens is static, so check it agrees."

# Measured, not assumed: it sets what TP=2 is worth, and the card model alone
# does not tell you (an A40 pair may or may not have the bridge fitted).
if [[ "$NGPU" -gt 1 ]]; then
    NVL_ACTIVE=$(nvidia-smi nvlink --status 2>/dev/null | grep -c "GB/s" || true)
    if [[ "${NVL_ACTIVE:-0}" -gt 0 ]]; then
        echo "Interconnect: NVLink ACTIVE ($NVL_ACTIVE links) — expect TP=2 nearer 1.9x"
        nvidia-smi nvlink --status 2>/dev/null | head -8
    else
        echo "Interconnect: no active NVLink -> PCIe. Expect TP=2 nearer 1.6x."
        echo "  (Expected on L40/L40S — Ada has no NVLink at all. On an A40 pair"
        echo "   it means the bridge is not fitted, which is worth asking about:"
        echo "   112.5 vs 31.5 GB/s is most of the gap between 1.6x and 1.9x.)"
    fi
    nvidia-smi topo -m 2>/dev/null | head -6
fi

# Only does anything on A40, which reaches FP8 through fp8-Marlin for want of
# native FP8. Inert and harmless on L40, so it stays unconditional.
export VLLM_MARLIN_USE_ATOMIC_ADD=1

# Embedding sidecar on GPU 0 of the allocation, pinned explicitly so it cannot
# land on a different card than the one whose budget was reduced for it.
#
# STARTED FIRST, AND WAITED FOR. Both servers refuse to start unless their whole
# budget is free at that moment, and the LLM ends up holding more than its own
# budget (see GPU_UTIL). Launched side by side, which one fit was a race the
# sidecar won by luck while it was MiniLM; restarting it beside a running LLM
# found 1.76 GiB free. Up to 5 minutes: the LLM starts anyway after that, and a
# sidecar that failed leaves only document search down, not the chat model.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" EMBED_GPU_MEM_UTIL=0.07 \
    "$SCRIPT_DIR/embed.sh" > "$SCRIPT_DIR/embed-server.log" 2>&1 &
EMBED_PID=$!
echo "embedding sidecar launched in background (port 20000, see embed-server.log)"
for _ in $(seq 150); do
    if curl -sf -o /dev/null "http://127.0.0.1:${EMBED_PORT:-20000}/health"; then
        echo "embedding sidecar is up"
        break
    fi
    if ! kill -0 "$EMBED_PID" 2>/dev/null; then
        echo "WARNING: embedding sidecar exited — see embed-server.log" >&2
        break
    fi
    sleep 2
done

# Discovery manifest for HPCA, now that both $PORT and $MAX_MODEL_LEN are known.
source "$SCRIPT_DIR/write_manifest.sh" llm "$SERVED_NAME" "$PORT" true "$MAX_MODEL_LEN"

# ---- SIZING, PER 48GB CARD ---------------------------------------------------
# Anchored on the last real profile (vllm.a40-l40.out, 2026-08-22): 44.39 GiB on
# the card, 39.95 taken at util 0.90, of which 0.76 non-torch + 3.09 peak
# activation + 0.12 CUDA graphs = ~3.97 GiB of non-KV overhead. Materially worse
# than the 1.8 GiB an older, rosier profile suggested; prefer the pessimistic.
#
# Weights ~27.5 GiB resident (30.4 GB on disk, minus the ~0.9 GB vision tower
# --language-model-only never loaded, plus ~0.5 GiB of MTP). fp8 with MTP loaded
# costs ~34,800 B/token. THE FIGURES BELOW PREDATE VISION (see VISION): the
# tower now loads, ~0.84 GiB more on one card (~26k KV tokens), half that per
# card on two, plus whatever the encoder's profiling pass adds to peak
# activation — unmeasured until the first vision run's log.
#
# ALL OF THIS IS THE 8-BIT SIZING (27.5 GiB of weights); 4-bit weights free
# roughly half of that. Redo it from "Actual usage ... for weight" in a 4-bit log.
#
#   ONE CARD   39.95 - 27.5 - 3.97 - 0.84 slots = ~7.6 GiB  = ~236k tokens
#              112000 -> 2.1 users;  262144 does not fit, vLLM refuses to start
#   TWO CARDS  39.95 - 13.75 - 3.97 - 0.84 = ~21.4 GiB/card, and the per-card
#              cost halves to ~17,400 B    = ~1.32M tokens
#              262144 -> 5.0 users;  131072 -> 10.1
#
# READ THOSE AS A FLOOR, NOT A FORECAST: they assume every user pins the full
# window at once. Steady-state `GPU KV cache usage` on the earlier 8-bit run sat
# at 10.9%, and prefix caching shares the system prompt across sessions. Trust
# the real "GPU KV cache size" log line over this arithmetic.
SPEC_ARGS=()
if [[ "$NUM_SPEC_TOKENS" -gt 0 ]]; then
    SPEC_ARGS=(--speculative-config \
      "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}")
    echo "MTP speculative decoding: $NUM_SPEC_TOKENS draft tokens"
fi

# ---- VISION ------------------------------------------------------------------
# The checkpoint is Qwen3_5ForConditionalGeneration: one set of weights with a
# ~0.9 GB vision tower, which this job used to switch off with
# --language-model-only. HPCA's inspect_image sub-loop needs it on.
#
# MM_LIMIT caps what one request may carry. 4 images is what inspect_image
# sends at most (four paths per call); video stays 0 because nothing sends it,
# and a non-zero limit makes vLLM profile and reserve for a worst-case video.
#
# MM_MAX_PIXELS bounds one image's cost, and is the lever on the one-card
# margin. Image tokens scale with pixels (about one per 32x32 block on this
# encoder family), and vLLM sizes its encoder profiling pass — real peak
# activation memory, taken from the KV pool — for the LARGEST image the
# processor would accept. The processor's own ceiling is ~16 MP (~16k tokens);
# 1536x1536 (~2.4 MP, ~2.3k tokens) is plenty to read a plot, and HPCA scales
# images down to that before sending anyway. The cap here protects the box from
# every other client (VS Code with vision on) as well.
#
# CHECK AT STARTUP: if the log warns that `max_pixels` was dropped as an
# unknown processor kwarg, the processor wants
#   MM_PROCESSOR_KWARGS='{"size": {"longest_edge": 2359296, "shortest_edge": 3136}}'
# instead (both spellings mean total pixels). Then read "GPU KV cache size":
# against the last language-only run it shows what vision really cost.
MM_LIMIT='{"image": 4, "video": 0}'
MM_MAX_PIXELS=$((1536 * 1536))
if [[ -z "${MM_PROCESSOR_KWARGS:-}" ]]; then
    MM_PROCESSOR_KWARGS="{\"max_pixels\": $MM_MAX_PIXELS}"
fi
echo "Vision: limit $MM_LIMIT, processor kwargs $MM_PROCESSOR_KWARGS"

python -m vllm.entrypoints.openai.api_server \
   --model "$MODEL" \
   --served-model-name "$SERVED_NAME" \
   --trust-remote-code \
   --host 0.0.0.0 --port $PORT \
   --limit-mm-per-prompt "$MM_LIMIT" \
   --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
   --mm-encoder-tp-mode data \
   --data-parallel-size $DP \
   --tensor-parallel-size $TP \
   --gpu-memory-utilization $GPU_UTIL \
   --max-model-len $MAX_MODEL_LEN \
   --max-num-seqs $MAX_NUM_SEQS \
   --max-num-batched-tokens 8192 \
   --kv-cache-dtype "$KV_CACHE_DTYPE" \
   --enable-chunked-prefill \
   --enable-prefix-caching \
   "${SPEC_ARGS[@]}" \
   --enable-auto-tool-choice \
   --tool-call-parser qwen3_coder \
   --reasoning-parser qwen3

# ---- TUNING / FALLBACKS ------------------------------------------------------
# 1) BACK TO 8-BIT (the known-good configuration before 2026-10-23):
#      MODEL="Qwen/Qwen3.8-27B-FP8"
#      SERVED_NAME="Qwen3.8-27B-FP8"
#      MAX_MODEL_LEN_TP1=112000        # one card cannot hold more at 8-bit
#    The served name changes, so every client's model id changes with it.
#    Whichever way you switch, VERIFY WITH THE TEACHER-FORCING TEST FIRST — eyeballing a short reply will
#    not catch this class of bug, since the first ~8 tokens looked perfect
#    throughout the incident. /v1/completions, "Count from one to twenty in
#    words:", 40 tokens, temperature 0. Correct output counts; broken repeats.
# 2) OOM or "No available memory for the cache blocks": drop MAX_NUM_SEQS
#    (frees 144 MiB each) before touching the window. If it persists the weights
#    landed heavier than the ~27.5 GiB assumed — read "Actual usage is ... for
#    weight" in the log and redo SIZING with the real number. The margin is thin
#    on two cards and absent on one.
# 3) vLLM rejects the model or the `qwen3_5` architecture: ./venv too old ->
#    ./setup_venv.sh --upgrade. NOT expected — 0.26.0+ registers
#    Qwen3_5ForConditionalGeneration and Qwen3_5MTP. Verify before upgrading:
#    ./venv also serves the embedding sidecar, so an upgrade moves both.
# 4) More users rather than the full window: lower MAX_MODEL_LEN_TP2. 131072
#    takes the two-card worst case from ~5.1 to ~9.8, 112000 to ~11.4 — and
#    MAX_NUM_SEQS becomes the real cap, so raise it too or the extra capacity is
#    unreachable. Move SETUP.md and the VS Code maxInputTokens in the same edit;
#    the HPCA manifest needs none, being written from the resolved value.
# 5) IS MTP PAYING? vLLM logs mean acceptance length (ceiling
#    NUM_SPEC_TOKENS+1). >=2.0 working; ~1.5 marginal, try NUM_SPEC_TOKENS=2;
#    ~1.0 every draft rejected, and you are strictly slower than with it off.
#    Acceptance is workload-dependent — judge it on your traffic, not a demo.
# 6) TEMPTED BY A QUANTISED KV CACHE AGAIN: read THE 4-BIT REVERT. The failure
#    to expect is not a startup error — it starts clean and serves garbage, and
#    only the teacher-forcing test in FALLBACK 1 will tell you. Separately, do
#    NOT reach for `nvfp4`: it needs SM100 trtllm-gen kernels, i.e. Blackwell.
# 7) Reclaiming the sidecar's share: comment out the embed.sh launch (and its
#    wait loop) and raise GPU_UTIL to 0.93. Worth ~3.1 GiB per card, ~95k more
#    fp8 KV tokens on one card.
# 8) JOB STUCK PENDING — the two-card default is the likeliest reason on a busy
#    partition (`squeue -u $USER --start`). One card needs no edit but serves
#    112000; check the VS Code maxInputTokens before a long session.
# 9) Landed on 2 GPUs but only one used: check "GPUs allocated". It counts
#    `nvidia-smi -L`, which respects CUDA_VISIBLE_DEVICES, so a 1 there means
#    SLURM gave you one card regardless of what --gres asked for.
# 10) VISION COSTS TOO MUCH (OOM at startup on one card, or a window that no
#    longer fits): first lower MM_MAX_PIXELS — 1024x1024 roughly halves the
#    encoder's profiling peak. (`--mm-encoder-tp-mode data` replicates the
#    small encoder on each card instead of sharding it: faster at TP=2, a
#    no-op at TP=1, accepted by this model on 2026-10-23.) To go back to
#    text only, replace the three --*mm* lines with `--language-model-only`
#    (frees the tower), or with --limit-mm-per-prompt '{"image": 0, "video":
#    0}' where that flag is rejected. HPCA notices either way: its vision
#    probe gets a 400 and inspect_image says the backend cannot see.
# 11) MTP MISBEHAVING. NUM_SPEC_TOKENS=0 removes it; nothing else depends on it.
#    a) Startup fails on the speculative config — verify rather than upgrade:
#         ./venv/bin/python -c "from vllm.model_executor.models.registry import \
#           ModelRegistry; print('Qwen3_5MTP' in ModelRegistry.get_supported_archs())"
#    b) Illegal-memory-access crash deep into a long generation, with acceptance
#       jumping to exactly 100% beforehand: vllm-project/vllm#40756, seen on a
#       sibling Qwen3.x-27B-FP8 at long sequence lengths. Fix is
#       NUM_SPEC_TOKENS=0. NOT what the 4-bit corruption was, even though 100%
#       acceptance appeared there too — that one never crashed.
# 12) STILL TOO SLOW on two cards: confirm you got two ("GPUs allocated"), then
#    read "Interconnect:" — PCIe is expected on L40 and already priced into the
#    ~47 tok/s. VLLM_USE_V2_MODEL_RUNNER=1 exists but explicitly bails out for
#    HYBRID models, which this is (48 of 64 layers are linear attention), so
#    forcing it overrides that check rather than satisfying it. Benchmark it.
