# Serving the LLM that HPCA talks to

HPCA has no model of its own: it is a client for an OpenAI-compatible endpoint.
This directory is the server side of that — a Slurm job that runs **vLLM** on a
GPU node of your cluster, serving **Qwen3.8-27B-FP8** on port 20001, with a
small embedding model on port 20000 for HPCA's document search.

This guide assumes no prior knowledge of vLLM or Slurm. It takes about an hour,
most of it downloading.

| | |
|---|---|
| model | `Qwen3.8-27B-FP8` (27B parameters, 8-bit weights) |
| hardware | **2× L40** (48 GB each); 2× A40 also works, 1 card works with a smaller window |
| context window | **262 144 tokens** on two cards, 112 000 on one |
| speed | ~47 tokens/s on 2× L40, ~27 without speculative decoding |
| concurrent users | 6 (`MAX_NUM_SEQS`); ~5 can hold a *full* 262k window at once |
| ports | **20001** chat/completions (API key required), **20000** embeddings (open) |
| disk | ~15 GB runtime + ~30.4 GB weights |

---

## What you need before you start

* **A Slurm cluster with an L40 or A40 GPU node** and permission to submit to
  its GPU partition. The job asks for two 48 GB cards; see
  [Adapt it to your cluster](#4-adapt-it-to-your-cluster) — partition and GPU
  names differ everywhere and are the one thing you almost certainly must edit.
* **~45 GB of disk** somewhere you can write. If your `$HOME` has a small quota,
  read [Where things get written](#where-things-get-written) before step 1.
* **Python 3.10–3.13** available on the cluster (`python3 -V`, or
  `module avail python`). No compiler and no CUDA toolkit: the vLLM wheels
  bundle their own CUDA runtime, and the GPU nodes already have the driver.
* **Outbound HTTPS** from wherever you run the install and the download —
  usually a login node.

Nothing runs on the login node except the install and the download. The server
itself only ever runs inside a Slurm job on a GPU node.

---

## Quick start

If you know your cluster, this is the whole thing:

```bash
cd llmServer
./setup_venv.sh                                                   # ~15 GB of runtime
HF_HOME=$PWD/weights ./venv/bin/hf download Qwen/Qwen3.8-27B-FP8  # ~30 GB, needs 16 GB RAM
head -c 32 /dev/urandom | base64 > ~/.vllm_api_key && chmod 600 ~/.vllm_api_key
sbatch llm.a40-l40.sh                                             # 2× L40, TP=2
tail -f vllm.a40-l40.out                                          # wait for "Application startup complete"
```

Then start `hpca` on the cluster and it finds the endpoint by itself. The rest
of this file explains each of those lines, and what to do when one of them does
not behave.

---

## 1. Install the runtime

```bash
cd llmServer
./setup_venv.sh
```

This creates `./venv` and pip-installs mainline vLLM into it (~15 GB, mostly
PyTorch and CUDA libraries). Ada (L40) and Ampere (A40) need nothing exotic —
no container, no compiler — because vLLM ships prebuilt wheels for them.

The script picks a Python 3.10–3.13 off your `PATH` by itself. If the cluster's
default is older:

```bash
module load python/3.12        # whatever your cluster calls it
PYTHON_BIN=$(command -v python3.12) ./setup_venv.sh
```

It finishes by checking the two things the job actually needs: that the vLLM
server entrypoint imports, and that this vLLM knows the `Qwen3_5` architecture
the checkpoint declares. Both are asked now so you do not discover a version
problem twenty minutes into a model load.

Later, `./setup_venv.sh --upgrade` bumps an existing venv. Don't do it while a
server is running — vLLM imports some modules lazily and can crash mid-session;
the script warns you if it sees one of your jobs running.

## 2. Download the weights

```bash
HF_HOME=$PWD/weights ./venv/bin/hf download Qwen/Qwen3.8-27B-FP8
```

~30.4 GB, plus a separate 0.48 GB `mtp.safetensors` head that **is** used here
(it is what makes speculative decoding work — see
[Why it is fast](#why-it-is-fast)).

**Give this at least 16 GB of RAM.** `huggingface_hub` pulls the repo through
Xet, which buffers many chunks in parallel; on a 2 GB login-node limit the OOM
killer takes it after ~20 seconds with no error message, and each retry starts
from zero. If your login node is tight, do it in an interactive slot:

```bash
srun -p standard --mem=32G --time=4:00:00 --pty bash
```

It resumes if interrupted. If your `huggingface_hub` is older and has no `hf`
command, `./venv/bin/huggingface-cli download Qwen/Qwen3.8-27B-FP8` does the
same. The embedding model (~90 MB) is fetched automatically on the sidecar's
first run.

## 3. Make an API key

The job **refuses to start without one**, because vLLM with an empty key serves
unauthenticated to everyone who can reach the node:

```bash
head -c 32 /dev/urandom | base64 > ~/.vllm_api_key && chmod 600 ~/.vllm_api_key
```

The job reads that file and passes the key through the environment, never on
the command line — command lines are world-readable in `ps aux` on a shared
node, the environment is not.

The embedding endpoint on 20000 is deliberately *not* key-protected: it is a
90 MB sentence embedder, and HPCA's manifest advertises it as open.

## 4. Adapt it to your cluster

Open `llm.a40-l40.sh` and look at the `#SBATCH` lines at the top:

```bash
#SBATCH --gres=gpu:l40:2      # two L40 cards
#SBATCH --partition=gpu       # the GPU partition's name
#SBATCH --mem=96G             # host RAM, not GPU memory
#SBATCH --cpus-per-task=8
#SBATCH --time=5-0            # five days
```

**`--gres` and `--partition` are cluster-specific and are the usual reason a
first submit sits pending forever or is rejected outright.** Find out what
yours are called:

```bash
sinfo -o "%.20N %.10t %.15G %.20P"     # nodes, state, GRES, partition
```

The `%.15G` column prints things like `gpu:l40:8` or `gpu:a100:4` — the middle
field is the name to put in `--gres=gpu:<name>:2`. `./draining_gpu_nodes.sh` is
a small convenience that shows the same thing filtered to GPU nodes, plus which
of them are draining (a draining node will never start your job).

You may also need an account or QoS flag (`#SBATCH --account=...`) — ask your
cluster's documentation.

Everything *below* the `#SBATCH` block is tuned for a 48 GB card and does not
need editing to get started.

## 5. Submit the job

From inside this directory (the script resolves its own paths from
`$SLURM_SUBMIT_DIR`, so submit from here):

```bash
sbatch llm.a40-l40.sh                     # 2× L40, tensor-parallel — the default
sbatch --gres=gpu:a40:2 llm.a40-l40.sh    # 2× A40, same thing
sbatch --gres=gpu:l40:1 llm.a40-l40.sh    # 1 card: ~30 tok/s, 112k window, shorter queue
```

Output goes to `vllm.a40-l40.out` in this directory, and the embedding
sidecar's to `embed-server.log`. Startup takes a few minutes — the weights are
read off shared storage and CUDA graphs are compiled. You are up when the log
says:

```
Application startup complete.
```

To stop it: `scancel <jobid>`. The sidecar dies with the job.

**Two cards is not merely faster, it is what makes the full context window
fit.** Tensor parallelism splits the ~27.5 GiB of weights *and* the 4 KV heads
across both cards; only that combination leaves room for 262 144 tokens. One
card serves 112 000 instead. The script resolves this at startup and prints it
— there is nothing to edit either way, and HPCA reads the resolved value from
the discovery manifest.

## 6. Check it came up correctly

Four log lines explain nearly every disappointing result:

```bash
grep -E "GPUs allocated|Context window|Interconnect|Maximum concurrency" vllm.a40-l40.out
```

* **`GPUs allocated: 2 -> ... --tensor-parallel-size 2`** — counts what Slurm
  actually gave you (`nvidia-smi -L`), not what `--gres` asked for. A `1` here
  accounts for the entire gap between ~27 and ~47 tok/s.
* **`Context window: 262144 (TP=2)`** — what the server resolved and what the
  manifest advertises. `112000 (TP=1)` means you landed on one card.
* **`Interconnect: no active NVLink -> PCIe`** — expected on L40 (Ada has no
  NVLink at all) and already priced into the ~47 tok/s. On an **A40** pair it
  means the NVLink bridge is not fitted, which is worth asking your admins
  about: it is most of the difference between a 1.6× and a 1.9× TP speedup.
* **`Maximum concurrency for 262144 tokens per request: ~5x`** — the worst case,
  assuming every user pins the full window at once. Real sessions use far less.

Then ask the server what it is serving. The node name is in `squeue -u $USER`:

```bash
curl -H "Authorization: Bearer $(cat ~/.vllm_api_key)" http://<gpu-node>:20001/v1/models
```

The `id` it returns is `Qwen3.8-27B-FP8`, and **every client selects the model
by exactly that string**.

---

## Connecting HPCA

### On the cluster — nothing to configure

The job writes a small JSON manifest naming its node, port, model and resolved
context length into a shared directory
(`/data/cephfs-1/work/groups/cubi/tools/hpca_connections` by default, one file
per job id and port). HPCA reads those, checks with Slurm that the job is still
running, probes the endpoint, and connects straight to the GPU node — no tunnel.

Because the directory is shared, a server anyone on the team starts is
discoverable by everyone's HPCA. Job ids are unique cluster-wide, so manifests
cannot collide.

Two settings to know about, both in `<app dir>/settings.json`:

```json
{
  "endpoints": { "endpoints_dir": "/data/.../hpca_connections",
                 "login_host": "hpc-login-2.cubi.bihealth.org" },
  "rag":       { "embedding_base_url": "http://localhost:20000/v1" }
}
```

**The endpoints directory is named on both sides, and the two must agree.**
HPCA reads `endpoints.endpoints_dir`; the launch scripts write to
`HPCA_ENDPOINTS_DIR`, defaulting to the path hardcoded in `write_manifest.sh`.
Both defaults are our group's path on the BIH cluster, so **on any other cluster
you must change both** — pick a directory your group can write to, set it in
`settings.json`, and export `HPCA_ENDPOINTS_DIR` to the same value before
`sbatch` (or edit `__wm_dir` in `write_manifest.sh` once). Change only one and
nothing errors: the job writes its manifest where HPCA never looks, and
auto-discovery just finds nothing.

Writing the manifest is best-effort: if that directory is unwritable the job
warns and serves normally, you just lose auto-discovery and have to point HPCA
at the node by hand.

Manifests are deleted when the job exits cleanly; HPCA reaps the ones left
behind by a hard kill (walltime, OOM, node failure). To see what is live:
`cat <endpoints_dir>/*.json`.

### From your workstation — one SSH tunnel

Both services live on the GPU node, so both forwards target it. Get the node
name from `squeue -u $USER`:

```bash
ssh -fN -L 20001:<gpu-node>:20001 -L 20000:<gpu-node>:20000 <you>@<login-host>
```

Verify, then point HPCA at localhost:

```bash
curl -H "Authorization: Bearer $(cat ~/.vllm_api_key)" http://localhost:20001/v1/models
```

```json
{
  "llm": {
    "base_url": "http://localhost:20001/v1",
    "model": "Qwen3.8-27B-FP8",
    "api_key": "<contents of ~/.vllm_api_key>"
  },
  "rag": { "embedding_base_url": "http://localhost:20000/v1" }
}
```

HPCA also offers this in the UI (its manage-LLMs screen scans localhost and
prints a tunnel template built from `endpoints.login_host`/`login_user`), so
hand-editing the file is optional.

If port 20001 was already taken on the GPU node, the job picks a random one and
logs `WARNING: Port 20001 in use, using <PORT> instead`. Change the remote port
after `<gpu-node>:` in the tunnel to match; the local side can stay 20001.

### VS Code Copilot (optional)

The same endpoint serves Copilot's custom-model feature. Command Palette →
"Copilot: Configure Custom Language Models":

```json
{
    "name": "http://localhost:20001",
    "vendor": "customendpoint",
    "apiType": "chat-completions",
    "models": [
        {
            "id": "Qwen3.8-27B-FP8",
            "name": "Qwen3.8-27B-FP8",
            "url": "http://localhost:20001/v1/chat/completions",
            "toolCalling": true,
            "vision": false,
            "maxInputTokens": 262144,
            "maxOutputTokens": 32000
        }
    ]
}
```

Reload the window afterwards. `maxInputTokens` here is **static and set for the
two-card job**: on a one-card run the server serves 112 000 while VS Code still
believes 262 144, and an over-long request gets a 400 at the worst possible
moment. If you run one card regularly, set it to 112000 (too low only silently
caps, which is the safer failure). HPCA is immune — it reads the resolved value
from the manifest.

`vision: false` is correct: the job loads the language model only and refuses
images.

---

## Where things get written

| what | default | how to move it |
|---|---|---|
| runtime | `llmServer/venv` (~15 GB) | move the whole directory |
| weights | `llmServer/weights` (~30.4 GB) | `export HF_HOME=/work/.../weights` (or `HPCA_WEIGHTS_DIR`) before both the download **and** `sbatch` |
| API key | `~/.vllm_api_key` | `API_KEY_FILE` in `llm.a40-l40.sh` |
| job log | `llmServer/vllm.a40-l40.out` | `#SBATCH --output=` |
| manifests | the shared endpoints dir | `HPCA_ENDPOINTS_DIR` |

The weights are the one to think about: 30 GB in `$HOME` fails on many clusters,
and the download and the job must agree on the location or the job silently
starts a second 30 GB download from the compute node. Set `HF_HOME` in both
places, or edit the default in `llm.a40-l40.sh` once.

---

## Why it is fast

Decoding one token at batch size 1 reads the entire model out of GPU memory, so
tokens/second is pure memory bandwidth — not compute. Three things buy speed
here, and the script's comments explain each in full:

* **8-bit (FP8) weights** — ~27.5 GiB read per token instead of ~55.
* **Tensor parallelism over two cards** — each card reads half, ~1.6× on L40
  (PCIe), ~1.9× with an A40 NVLink bridge.
* **MTP speculative decoding** (~1.76×) — the checkpoint's own small draft head
  proposes 3 tokens per step and the 27B model verifies the whole block in one
  pass. Rejection sampling makes it **exactly lossless**: only speed is ever at
  risk, never output quality.

Throughput falls as a session fills its window, because the KV cache is re-read
every step: ~47 tok/s near-empty, ~36 at a genuinely full 262k on two cards.
That is the user's own context doing it, not a setting.

**Do not quantise the KV cache.** `KV_CACHE_DTYPE=fp8` is deliberate. A 4-bit KV
cache (`turboquant_4bit_nc`) loaded cleanly, logged nothing, and served
corrupted output for a day — 8–12 correct tokens, then one token repeated
forever. The full evidence is in the `THE 4-BIT REVERT` block at the top of
`llm.a40-l40.sh`, and it matters because the obvious reading of that failure
(blame the 4-bit *weights*) is wrong.

---

## Troubleshooting

**Job sits pending forever** — the two-card default on a busy partition.
`squeue -u $USER --start` estimates when it will run;
`sbatch --gres=gpu:l40:1 llm.a40-l40.sh` trades the big window and a third of
the speed for a slot. Also check `./draining_gpu_nodes.sh`: a draining node never
starts new jobs. If it is *rejected* rather than pending, `--gres`/`--partition`
do not match your cluster — see [step 4](#4-adapt-it-to-your-cluster).

**`ERROR: <dir>/venv missing — run ./setup_venv.sh`** — the venv is not where
the job looked, usually because you submitted from another directory. The script finds its files relative to where you ran `sbatch`.

**`ERROR: API key file missing or empty`** — [step 3](#3-make-an-api-key).

**`ERROR: got 'Tesla V100...'. This script needs an A40 or L40.`** — 8-bit needs
Marlin kernels, which need compute capability 8.0+. A V100 (7.0) cannot run this
model; that path needs a patched vLLM fork in a container and is out of scope
here.

**Model or `qwen3_5` architecture rejected** — the venv is too old:
`./setup_venv.sh --upgrade` (vLLM ≥ 0.26.0 registers it).

**A second 30 GB download starts on the GPU node** — `HF_HOME` at submit time
differs from the one used for the download. See
[Where things get written](#where-things-get-written).

**Only ~30 tok/s when expecting ~47** — you got one card. Check the
`GPUs allocated` line; `Context window: 112000 (TP=1)` confirms it.

**OOM, or "No available memory for the cache blocks"** — lower `MAX_NUM_SEQS`
(each slot costs 144 MiB whether used or not) before touching the window. If it
persists, read the `Actual usage is ... for weight` line in the log and redo the
arithmetic in the `SIZING` block.

**Repeated tokens / garbage after a good first few words** — a quantised KV
cache; check `KV_CACHE_DTYPE=fp8`. It starts clean and logs nothing, so the only
reliable check is teacher forcing: same prompt at temperature 0, one-shot versus
one-token-per-request. See `THE 4-BIT REVERT`.

**400 on a long request** — the client is configured for the two-card window but
the job resolved to 112 000 on one card.

**401** — the request needs `Authorization: Bearer <~/.vllm_api_key>`.

**Connection refused** — the tunnel is down, or the job picked a different port;
check the `vLLM port:` line in `vllm.a40-l40.out`.

**HPCA does not find the endpoint** — is the job `RUNNING` (`squeue`), does
`<endpoints_dir>/*.json` contain a file for it, and does HPCA's
`endpoints.endpoints_dir` match the directory the job wrote to? HPCA logs the
whole decision trail to `<app dir>/autoconnect.log`.

**Nothing here matches** — `llm.a40-l40.sh` ends with a numbered
`TUNING / FALLBACKS` block covering twelve failure modes in the order worth
trying, each with the evidence behind it.

---

## Files in this directory

| file | what it is |
|---|---|
| `llm.a40-l40.sh` | the Slurm job: vLLM serving Qwen3.8-27B-FP8 on 20001. Every non-obvious number in it is explained in place — read it before changing one |
| `setup_venv.sh` | one-time: creates `./venv` with mainline vLLM. `--upgrade` bumps it |
| `embed.sh` | the embedding sidecar on 20000, launched in the background by the job |
| `write_manifest.sh` | sourced by both, drops the JSON discovery manifest HPCA reads |
| `draining_gpu_nodes.sh` | convenience: which GPU nodes are draining, and what all of them look like |

Tuning that is deliberately *not* repeated here — context window versus user
count, the memory arithmetic per card, whether MTP is paying, going back to
4-bit weights — lives in the comment blocks of `llm.a40-l40.sh`, next to the
values it applies to.
