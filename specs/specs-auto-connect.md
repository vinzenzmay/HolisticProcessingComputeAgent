# Spec: automatic cluster endpoint discovery & connection ("auto-connect")

**Status:** design agreed, not yet implemented. This document is the hand-off
to a fresh implementation session — it is self-contained; you should not need
the conversation that produced it.

**One-line purpose:** on the HPC cluster, HPCA discovers running vLLM LLMs and
the embeddings sidecar and connects with zero setup; off the cluster (on a
workstation) it prints the exact SSH tunnel the user must create. This is a
**pre-agent layer** — it runs without any LLM and is never invoked by the
LLM-driven agent.

---

## 1. Context

HPCA is a TUI that talks to OpenAI-compatible vLLM backends. On this site every
backend is a vLLM server on a GPU node, launched via `sbatch` scripts that live
**outside this repo**, in the cluster GPU dir:

- Cluster path: `/data/cephfs-1/work/groups/cubi/users/mayv_c/gpu/`
- Mounted on the workstation at: `/vol/sshfs/vmay/bih_cluster/data/cephfs-1/work/groups/cubi/users/mayv_c/gpu/`
- Scripts there: `embed.sh` (embeddings sidecar), `llm.35B.sh`, `llm.35B.dp3.sh`,
  `llm.27B.sh`, `llm.gemma31B.sh`, plus `SETUP.md`, `write_manifest.sh` (to be
  created).

**Port scheme (already standardized), `localport == remote port`:**

| Instance | id | Port |
|---|---|---|
| embeddings sidecar (`embed.sh`, BAAI/bge-m3) | — | `20000` |
| Qwen3.6 35B (`llm.35B.sh`) | 1 | `20001` |
| Qwen3.6 35B dp3 (`llm.35B.dp3.sh`) | 2 | `20002` |
| Qwen3.6 27B (`llm.27B.sh`) | 3 | `20003` |
| Gemma (`llm.gemma31B.sh`) | 4 | `20004` |

Each `llm.*.sh` tries `PREFERRED_PORT = 20000 + id` and **falls back to a random
free port if taken** — so the bound port is *not* deterministic and must be
recovered at runtime. The scripts print `vLLM is running on: <ip>` (via
`hostname -i`) and `vLLM port: <port>` to their `.out` files. The embeddings
sidecar is launched **in the background from `llm.35B.sh` only** (co-located on
the 35B's GPU, gpu-util split 0.9/0.05) — so RAG is available only while the 35B
runs. All servers bind `--host 0.0.0.0`. The chat servers require an API key
(`VLLM_API_KEY`, from `~/.vllm_api_key`); the embeddings server needs none.

**Existing code to reuse (in this repo):**
- `src/hpca/discover.py` — `probe_endpoint(base_url, *, api_keys=()) -> list[DiscoveredBackend]`
  (probes `/v1/models`, returns model rows; a key-locked endpoint surfaces as
  `needs_key`/`KEY_REQUIRED` after trying the `api_keys` pool). `scan_local_ports(...)`
  scans `localhost`. `is_reachable(...)`. `DiscoveredBackend` has
  `base_url, model, max_model_len, needs_key, api_key`.
- `src/hpca/slurm.py` — `SlurmClient` with injectable `run` (for tests) and an
  optional `submit_host` SSH hop. Has `submit/test_only/status/cancel`. **No
  `squeue`/liveness method yet — add one.**
- `src/hpca/config.py` — `RagSettings.embedding_base_url`
  (default `http://localhost:20000/v1`), `LLMSettings.base_url`,
  `Settings.known_llm_ports`, `remember_llm_ports(...)`.
- `src/hpca/tui/manage_llms.py` — the backend discovery/selection screen
  (integration point). `src/hpca/tui/backend_form.py` — manual backend entry.

---

## 2. Goals and non-goals

**Goals**
- On the cluster, discover live LLM + embeddings endpoints and connect with no
  user action (subject to model choice and API keys).
- Off the cluster, tell the user exactly how to create the SSH tunnel.
- Correct in a **shared, multi-user** setting: connect to instances launched by
  *anyone* on the team, and never silently connect to the wrong model.
- Run entirely without an LLM; never called by the agent.

**Non-goals (explicitly out of scope — do not build)**
- HPCA does **not** spawn, manage, tear down, or ref-count SSH tunnels. The user
  owns tunnel lifecycle (their own `ssh -fN -L …`).
- **No changes to API-key handling.** The existing api-key registry (the
  `api_keys` pool) and the manual "this model needs a key → user provides one"
  flow stand unchanged. HPCA never reads `~/.vllm_api_key` or any key file
  itself.
- No periodic background rescanning (startup + manual rescan only).

---

## 3. Verified cluster facts

These were tested live and are load-bearing assumptions:

1. **Direct compute→GPU connection works.** From a compute node (`hpc-cpu-55`),
   `bash -c "exec 3<>/dev/tcp/<gpu-ip>/<port>"` returned `TCP OPEN`. Combined
   with the known-working login→GPU path (the existing tunnel relies on it),
   **direct connect works from any cluster node — login and compute.** So the
   tunnel fallback is genuinely workstation-only.
2. **Foreign job liveness is queryable.** `squeue -j <other-users-jobid>`
   returned the job's state/node even for a different user's job. So the
   shared-model liveness check works with plain `squeue`.

---

## 4. Design

### 4.1 Manifest (source of truth)

Each vLLM process writes one JSON file **once its real port is bound**:

- **Location:** `/data/cephfs-1/work/groups/cubi/tools/hpca_connections`
  — a **shared** group dir (`0755`, inside a group-only `tools/` parent), so
  everyone's HPCA sees everyone's endpoints. Manifests are written `0644`;
  only the dir owner can currently write, i.e. others *use* the servers but do
  not publish their own (give the dir `3775` — setgid + sticky — if teammates
  should host too). The path is configurable both ways: `endpoints_dir` in
  settings (§6) and `$HPCA_ENDPOINTS_DIR` for the launch scripts.
- **Filename / key:** `<jobid>-<port>.json`. The embeddings sidecar runs *inside*
  the 35B's job so it shares `$SLURM_JOB_ID`; the port disambiguates the two
  endpoints. (SLURM liveness still keys on `jobid`, so an LLM and its sidecar are
  reaped together — correct, they live and die together.)
- **Schema:**
  ```json
  {
    "role": "llm",                          // "llm" | "embedding"
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",    // must equal the served /v1/models id
    "jobid": "1234567",                     // $SLURM_JOB_ID — the liveness key
    "user": "mayv_c",
    "node": "hpc-gpu-8",                    // hostname -s
    "ip": "172.16.33.208",                  // hostname -i
    "port": 20001,                          // ACTUAL bound port (post-fallback)
    "ctx_len": 81920,                       // optional, for display
    "needs_key": true,                      // llm: true, embedding: false — a HINT only
    "started": "2026-07-23T13:30:00Z"
  }
  ```
- **No secrets in the manifest.** `needs_key` is a pre-probe hint for
  display/template; the authoritative answer comes from the probe.
- **Writer:** a shared `write_manifest.sh` in the GPU dir, called by `embed.sh`
  and each `llm.*.sh` after `$PORT` is known. It `mkdir -p`s the dir, writes the
  JSON, and sets `trap 'rm -f "$file"' EXIT` for best-effort cleanup (covers
  clean exit and `scancel`/SIGTERM; hard kills — OOM/NODE_FAIL/walltime SIGKILL —
  leave a stale file, handled by reaping below). `embed.sh` inherits
  `$SLURM_JOB_ID` because it is backgrounded inside the 35B job.

### 4.2 Discovery + two-layer liveness (on-cluster)

For each manifest file, in order:

1. **SLURM liveness (cheap pre-filter + safe reap).** `squeue -h -j "$jobid" -o '%T'`
   → `RUNNING` means keep; empty/absent means the job is gone → **skip and delete
   the manifest file** (reaping; idempotent across concurrent HPCA instances).
   Runs **locally** — manifests are only read on-cluster, so `squeue` is always
   local (no SSH-to-squeue path needed). If `squeue` itself fails transiently,
   fall back to probe-only (do not discard endpoints just because SLURM was
   briefly unreachable).
2. **Probe + model-id match (correctness).** Build `http://<ip>:<port>/v1`, call
   `probe_endpoint(...)`. Require a live `/v1/models` **whose served model id
   equals the manifest `model`.** This closes the port-reuse hole: SLURM says the
   *allocation* is alive, but vLLM could have died inside it and another user's
   process grabbed the port (ports are per-node, not per-cgroup). Mismatch or no
   answer ⇒ treat as stale, ignore.

Surviving entries become backends (direct `ip:port` base URLs, no tunnel).

### 4.3 Connection strategy

- **Direct-first**, assuming on-cluster: connect straight to `ip:port` (short
  timeout, e.g. the existing `TCP_TIMEOUT_S = 0.25`). Verified to work from login
  and compute nodes.
- **Off-cluster:** the manifest dir is a cephfs path that does not exist on the
  workstation, so this layer finds nothing and
  **falls through to the existing `localhost` scan** (`scan_local_ports`), which
  picks up a tunnel the user has already created. If that too is empty, show the
  help message (§4.6).

### 4.4 Startup behavior and selection

- Run discovery **at startup** + on **manual rescan (`r`)**. Re-probe the current
  backend on failure (already how `is_reachable` behaves). No periodic scan.
- **LLM selection = "auto if one, list if many" (rule (c)):**
  - Exactly one live LLM ⇒ auto-connect (set it as the active session backend),
    zero clicks.
  - Several ⇒ present them in the existing `manage_llms` list, pre-sorted
    (e.g. by context desc) and pre-selected (last-used), user picks.
  - Optional `preferred_models` config (ordered list of model-id substrings) ⇒
    full auto-connect to the top available match, for power users.
  - A locked LLM that no registry key unlocks **does not** auto-connect silently
    — it surfaces for the user to add a key (existing flow). Silent auto-connect
    only on an actual `200`.
- **Nothing connected ⇒ open the manage-LLMs screen** (v0.23.0). Once discovery
  and auto-connect have had their say, startup probes the *active* backend and,
  if it does not answer usably, says so and pushes `manage_llms`. "Usably"
  means real model rows for the key that backend carries — a `401` is up but
  not for us, and the first turn would fail exactly as if it were down. This
  covers the first run (nothing configured), the off-cluster case (tunnel down
  — the screen shows the §4.6 template by itself), and a cluster endpoint that
  needs a key. See `HpcaApp._ensure_backend_connected`; the check is a class
  attribute (`startup_backend_check`) so the unit suite can switch a startup
  modal off wholesale.

### 4.5 Embeddings wiring

- **Always auto-wired, never a prompt.** Point `rag.embedding_base_url` at the
  discovered embedding endpoint's direct `ip:port` (on-cluster). There is
  effectively one embeddings server, so no user choice. If none is live (e.g. the
  35B isn't running), RAG is simply unavailable — do not block LLM connection on
  it.

### 4.6 Off-cluster help message (generic template — decision (a))

When direct fails and the `localhost` scan is empty, print a generic template.
HPCA does **not** read manifests remotely and does **not** spawn SSH. It knows
the login host + username from config; node/port are placeholders the user fills
from the JSONs:

```
Couldn't connect to a backend directly.
On your workstation? Create a tunnel, then press 'r' to rescan.

  1. On the cluster, list running endpoints:
       cat /data/cephfs-1/work/groups/cubi/tools/hpca_connections/*.json
  2. From your workstation, forward BOTH the LLM and the embeddings server
     (they may be on different nodes) — one ssh, two -L forwards:

       ssh -fN \
         -L <llm_port>:<llm_node>:<llm_port> \
         -L 20000:<embed_node>:20000 \
         <you>@hpc-login-2.cubi.bihealth.org

  local port == remote port, so your app config just works.
```

---

## 5. Implementation plan (TDD, per project workflow)

Suggested order; write tests first for each Python unit.

1. **`write_manifest.sh`** (GPU dir, bash) + wire into `embed.sh`
   (`role=embedding`, `needs_key=false`, port `20000`) and the four `llm.*.sh`
   (`role=llm`, `needs_key=true`), called right after the `PORT` selection block.
   Add the `trap … EXIT`. No repo tests (shell, outside repo) — but keep it tiny
   and self-contained.
2. **`slurm.py` liveness method** — e.g.
   `async def job_states(self, job_ids: list[str]) -> dict[str, str]` using
   `squeue -h -j <ids> -o '%i|%T'` (or per-id), parsed deterministically; absent
   id ⇒ not running. Reuse the injectable `run` for tests. Handle the
   `squeue`-failed → "unknown" case distinctly from "not running".
3. **New discovery module** — e.g. `src/hpca/cluster_endpoints.py`:
   read manifest dir → JSON parse (tolerate malformed/partial files) →
   SLURM liveness (skip+reap dead) → `probe_endpoint` + model-id match →
   return `list[DiscoveredBackend]` split by role (llm vs embedding).
   Inject the manifest dir, the `SlurmClient`, and an httpx transport for tests.
4. **TUI startup integration** (`manage_llms.py` / app bootstrap): run the module
   → apply rule (c) → auto-wire embeddings → off-cluster fall-through to
   `scan_local_ports` → template. Keep it a clear pre-agent step.
5. **Config additions** (§6).
6. **Docs:** update the GPU dir `SETUP.md` to mention manifests + auto-connect.

---

## 6. Config additions

- `endpoints_dir: str` — manifest directory. Default
  `/data/cephfs-1/work/groups/cubi/tools/hpca_connections` (the shared group
  dir). Also names the dir in the off-cluster help text, so the command it
  prints matches what this app reads.
- `preferred_models: list[str]` — optional ordered model-id substrings for full
  auto-connect. Default `[]` (→ rule (c)).
- Login-host + username for the off-cluster template (there may already be a
  suitable config field — check before adding; default
  `hpc-login-2.cubi.bihealth.org`).

---

## 7. Decision log (do not re-litigate)

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Operating model | **Shared/team-wide** | 35B ties up an L40 for hours → instances get shared; discover anyone's. |
| 2 | Identification | **Manifest is source of truth** | Bound port is non-deterministic; `squeue` can't reveal it. We own the launch scripts. |
| 3 | Staleness guard | **Two layers: SLURM `RUNNING?` + probe/model-id** | SLURM prunes dead allocations & enables safe reap; probe catches alive-alloc/dead-vLLM/stolen-port. SLURM-unreachable → probe-only. |
| 4 | Connect strategy | **Direct-first, cluster is default** | Verified direct works from login *and* compute nodes. |
| 5 | Tunnel management | **Dropped entirely** | No HPCA-spawned ssh, no teardown, no ref-counting; user owns tunnels. Biggest complexity for least value. |
| 6 | Off-cluster message | **Generic template (a)** | Simplicity; HPCA reads no remote manifests, spawns no SSH. |
| 7 | Manifest schema/key | **`<jobid>-<port>.json`, both node+ip, shared helper** | jobid shared by llm+sidecar → port disambiguates; ip for direct, hostname for the tunnel. |
| 8 | Startup selection | **(c) auto-if-one/list-if-many; embeddings auto-wired** | Easy in the common case without guessing when ambiguous. |
| 9 | API keys | **Unchanged (existing registry)** | HPCA never reads a key file; locked endpoints use the existing pool/manual flow. |
| — | Embeddings operation | **(i) keep 35B-only sidecar** | Deferred; manifest treats embeddings as independent so this can change later. |

---

## 8. Open items / future

- ~~**Multi-user:** move `endpoints_dir` to a group-readable cephfs path.~~
  **Done (2026-08-05, v0.15.0):** default is
  `/data/cephfs-1/work/groups/cubi/tools/hpca_connections`, `0755` inside a
  group-only parent; `write_manifest.sh` writes there (`$HPCA_ENDPOINTS_DIR`
  overrides) and chmods each manifest `0644`. Still open underneath it:
  **multi-user hosting** — the dir is not group-writable, so only its owner
  can publish or reap manifests. Others' HPCA degrades cleanly (the reap
  `unlink` is already `except OSError: pass`), but a teammate's own vLLM job
  cannot announce itself until the dir gets `3775`.
- **Embeddings availability:** with decision (i), RAG needs the 35B up. Revisit
  as (ii) co-locate-on-every-chat-model or (iii) standalone job if RAG must be
  available under any model. No downstream change required — the manifest already
  models embeddings as an independent endpoint.
- **`preferred_models`** full-auto path can ship later; (c) is the MVP.
