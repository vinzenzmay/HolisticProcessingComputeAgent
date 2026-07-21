# Spec: LLM API-key registry and duplicate-scan fix

## Motivation

In the Manage-LLMs screen (`m`), the left "Discovered" panel lists endpoints
found by a localhost port scan. An endpoint that requires an API key comes back
from the scan as an uninformative sentinel row `(api key required)` — no model,
no context size — because the scan probes `/v1/models` without a key and reads
the 401/403.

Today the only place a key is stored is inline on a configured `LLMBackend` in
`settings.json`. A key you type when adding one endpoint does nothing for the
others. This spec introduces a **global key pool** ("key registry") so that keys
already known to the app are tried against key-locked endpoints, turning their
sentinel rows into informative ones (real model + context).

It also fixes a bug: after adding a key-locked endpoint, its `(api key required)`
sentinel stays in the Discovered list and can be added again.

## Design decisions (settled)

- **Global key pool**, not per-endpoint memory: any pooled key is tried against
  any key-locked endpoint.
- **Dedicated persisted list** `Settings.llm_api_keys`, unioned at probe time
  with keys already on configured backends. Durable and independent of which
  backends happen to be configured.
- **In-place re-probe** when a new key lands: re-probe only the endpoints already
  in the Discovered list (their ports are known/open) — no full TCP sweep.
- **The scan itself uses the pool**: a fresh scan / F5 rescan is directly
  informative for anything the pool unlocks.
- **Pool-unlocked entries add directly**, no form — we already validated the key
  against that exact endpoint during the probe.
- **Bug fix** via the re-probe (the sentinel upgrades to a real model, so normal
  `(base_url, model)` dedup hides it) **plus** a save-time guard for the
  force-save / bad-key path.
- **Append-only pool, no management UI** this iteration. Keys survive backend
  removal; a revoked key silently 401s until `settings.json` is hand-edited.
- Plaintext storage — same posture as the existing inline backend keys.

## Work items

The change splits by file into three slices. Slice A (`config.py`) and Slice B
(`discover.py`) touch disjoint files and are independent. Slice C
(`manage_llms.py`) depends on both. `backend_form.py` is **unchanged** — the form
is still used only to collect a key for a bare sentinel.

---

### Slice A — Key pool storage (`src/hpca/config.py`, `tests/test_config.py`)

1. Add field to `Settings`:
   ```python
   llm_api_keys: list[str] = []
   ```
   Place it near `backends` / `known_llm_ports`.

2. Add a helper method on `Settings`, mirroring `remember_llm_ports`:
   ```python
   def remember_llm_key(self, api_key: str | None) -> bool:
       """Add an API key to the pool if new. Returns whether it was newly
       learned (i.e. whether settings need saving)."""
   ```
   - `None`/empty string → no-op, returns `False`.
   - Dedup by exact string. Returns `True` only when a genuinely new key is
     appended.

3. Tests in `tests/test_config.py`:
   - `remember_llm_key` appends a new key and returns `True`; a duplicate returns
     `False` and does not grow the list; `None`/`""` is a no-op returning `False`.
   - `llm_api_keys` round-trips through `save()` / `load()`.

**Do not** touch `discover.py` or `manage_llms.py`.

---

### Slice B — Keyed probing (`src/hpca/discover.py`, `tests/test_discover.py`)

1. Extract the sentinel model string to a module constant, e.g.:
   ```python
   KEY_REQUIRED = "(api key required)"
   ```
   Use it where the 401/403 branch currently builds
   `model="(api key required)"`.

2. Add an `api_key` field to `DiscoveredBackend`:
   ```python
   api_key: str | None = None  # the pool key that unlocked this endpoint
   ```
   Only set when a key succeeded; `describe()`/`details()` behaviour is
   unchanged (they already show `key: yes` from `needs_key`).

3. `probe_endpoint(base_url, *, api_key=None, api_keys=(), ...)`:
   - Keep the existing single-`api_key` parameter and behaviour (used by the
     form's validation) intact.
   - Add `api_keys: Sequence[str] = ()`. New behaviour when the **unauthenticated
     or single-key** probe yields a 401/403 and `api_keys` is non-empty: try each
     key in `api_keys` in order; the first that returns 200 wins — return its
     real model rows with `needs_key=True` and `api_key=<that key>` set on each.
   - If no key in `api_keys` works (or the list is empty), return the
     `(api key required)` sentinel exactly as today.
   - A key already passed as `api_key=` should not need duplicating in
     `api_keys`; keep the logic simple and correct, order not significant beyond
     "first success wins".

4. `scan_local_ports(..., api_keys: Sequence[str] = ())`: thread `api_keys`
   through to the `probe_endpoint` calls so the scan resolves key-locked
   endpoints inline.

5. Tests in `tests/test_discover.py` (follow the existing `httpx.MockTransport`
   style already in the file):
   - `probe_endpoint` with `api_keys` where one key is accepted (200 with a model
     list) and others 401 → returns the real model rows, each with `api_key` set
     to the accepted key and `needs_key=True`.
   - `probe_endpoint` with `api_keys` where none work → the `(api key required)`
     sentinel, `api_key is None`.
   - `probe_endpoint` with empty `api_keys` on a 401 → sentinel (unchanged
     behaviour).
   - `scan_local_ports` passes `api_keys` through and surfaces the resolved
     model for a key-locked port.

**Do not** touch `config.py` or `manage_llms.py`.

---

### Slice C — UI integration (`src/hpca/tui/manage_llms.py`, `tests/test_tui_llm_mgmt.py`)

Depends on Slices A and B being merged first.

1. **Effective key set** — a helper that returns the pool unioned with keys on
   configured backends (deduped, order-stable):
   ```python
   def _effective_keys(self) -> list[str]:
       s = self.app.settings
       keys = list(s.llm_api_keys)
       for b in s.backends:
           if b.api_key and b.api_key not in keys:
               keys.append(b.api_key)
       return keys
   ```

2. **Scan integration** — pass `_effective_keys()` into `scan_local_ports` in
   `scan_worker` so fresh scans / F5 rescans resolve key-locked endpoints inline.

3. **In-place re-probe** — a new async method, e.g. `_reprobe_discovered()`, that
   re-probes the current `self._discovered` entries that are still sentinels
   (`backend.model == discover.KEY_REQUIRED`) using `_effective_keys()`, replaces
   any that resolve with their real rows, and refreshes the Discovered panel. It
   must **not** run a TCP sweep — only `probe_endpoint` against the known
   base_urls. Run it off the UI thread appropriately (a worker), mirroring the
   existing async patterns.

4. **Direct add for unlocked entries** (`_add_backend`): if
   `discovered.api_key is not None` (pool-unlocked), save the `LLMBackend`
   directly with `api_key=discovered.api_key` — no form. A bare sentinel
   (`discovered.needs_key` and no `api_key`) still opens `BackendFormScreen` as
   today. A keyless endpoint is unchanged.

5. **On save** (`_save_backend`):
   - After appending the backend, call
     `settings.remember_llm_key(backend.api_key)`; if it returns `True` (a new
     key), fire the in-place re-probe.
   - **Save-time guard (bug fix):** drop from `self._discovered` any entry that
     is still a sentinel (`model == KEY_REQUIRED`) sharing the just-saved
     backend's `base_url`, then refresh Discovered. This closes the duplicate
     even on the force-save / bad-key path where no key unlocks the sentinel.
   - Persist via the existing `settings.save()`.

6. Tests in `tests/test_tui_llm_mgmt.py` (follow the existing Textual test
   harness in that file):
   - In-place re-probe upgrades a `(api key required)` row to its real model when
     a pooled key works.
   - After adding a key-locked endpoint (valid key), it disappears from
     Discovered (no duplicate).
   - Save-time guard: force-saving a backend at a base_url removes any remaining
     sentinel at that base_url from Discovered (bug fixed without a working key).
   - A pool-unlocked discovered entry adds directly, without pushing
     `BackendFormScreen`.
   - A manually-added backend's key feeds the pool (`llm_api_keys` gains it) and
     triggers a re-probe.

## Acceptance

- `uv run pytest` is green (whole suite, not just the touched files).
- Manual sanity: with two key-locked localhost endpoints sharing a key, adding
  one and providing the key makes the other's row informative; the added one no
  longer appears in Discovered; force-saving also removes the sentinel.
