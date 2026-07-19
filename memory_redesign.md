# Memory & self-learning redesign (per-profile, Hermes-inspired)

Status: **implemented** (all six phases, branch `feature/memory-redesign`).
Extends project.md §4.4 (self-reflection) and §6 (profiles & memory).

Implementation notes, where the built thing differs from the plan below:

* **Struggle notes live in tier 3, not tier 2** (§5 anticipated this; it is
  what shipped). They are situational and were the main tier-2 bloat source.
* **`session_search` ranks with OR over distinctive terms**, not the AND the
  first cut used — an AND over every word of a natural request never matches.
  Stop words are dropped so an OR match does not pull in every session.
* **Skill patches append under a `## Corrections` heading** in the profile's
  own copy rather than rewriting the skill: a small model asked to restate a
  procedure drops steps, and one profile's correction must not change
  another's.
* **The `memory` tool gained a `demote` operation** (§6 planned demotion only
  as a UI flow). A `replace` cannot move tiers, so freeing tier 2 by demoting
  to tier 3 needed its own verb.
* **Compaction folds the view, not the stored history** — the transcript keeps
  everything; only what the model sees is folded.
* **Pre-eviction review is deferred to after the reply**, not run inline: a
  modal opened mid-round would suspend the turn behind an unrequested dialog.
* **The curator is deterministic only.** The LLM consolidation pass (§7) was
  not built; ageing and archival cover the decay problem, and merge proposals
  from a 27B are worth revisiting only once there is real accumulated data.

The one deliberate omission: the LLM curator pass. Everything else in §2–§8
is in place, with 1010 tests passing.

## 0. Motivation and sources

The current system (project.md §6) is wired end-to-end but underdelivers:

* All tier-2 memories are injected into every orchestrator prompt regardless of
  relevance; the token caps are soft warnings only (`check_memory_caps`,
  `tui/app.py`), so a growing profile silently bloats every prompt — the worst
  possible failure mode for 27B/35B models.
* Struggle matching (`agent/struggle.py:matching_struggles`) only produces a TUI
  toast; the *model* is never told about past struggles, defeating §4.4's purpose.
* Tier 3 (retrieved memory) is format-reserved but unimplemented.
* Skills, the symbol index, and RAG are global despite docstrings claiming
  per-profile scoping; only the path registry is actually profile-scoped.
* The `backend` tag on memories and `default_backend` front-matter are written
  but never read.
* There is no context compaction: `graph.py` sends the full message history
  every round and never consults `LLMBackend.max_model_len`.

This plan adapts the memory/learning architecture of Nous Research's
**hermes-agent** (github.com/nousresearch/hermes-agent). Its core ideas, and
whether we adopt them:

| Hermes idea | Adopt? | Adaptation for HPCA |
|---|---|---|
| Split memory by mutation rate: tiny always-injected curated store, unbounded searchable episodic DB, skills as load-on-demand procedures | Yes | Maps onto tier1/tier2 (curated), a new `session_search` tool (episodic), the existing skills dir (procedural), and tier 3 (retrieved overflow) |
| Hard **character** budgets on injected memory, enforced at write time; over-budget writes rejected with the full inventory ("consolidate or drop") | Yes | Char budgets replace the soft token warnings; but consolidation is HITL (small models write memories, humans approve — §6.3 stays) |
| `memory` tool: add/replace/remove + atomic `operations` batch, substring addressing (no IDs to hallucinate) | Yes | Implemented with pydantic + constrained decoding; writes route through the existing `MemoryProposalScreen` approval gate |
| Counter-based background self-review after the reply is delivered (N user turns → memory review, N tool rounds → skill review) | Yes | Runs as a firewalled sub-LLM call (same pattern as `conclude.py`); replaces the failure-marker heuristic as the *only* trigger |
| Anti-capture rules in reflection prompts (no transient failures, no "tool X is broken" claims, declarative facts not imperatives) | Yes | Verbatim-adapted; this is cheap and directly targets small-model self-poisoning |
| Frozen memory snapshot per session (prefix-cache stability) | Yes | vLLM has automatic prefix caching; a stable system-prompt prefix matters for our throughput too |
| Episodic recall: SQLite FTS5 + BM25, snippet + ±5-message window + first/last-3 "bookends", lineage dedup | Yes | Over a new `messages` table in `hpca.db`, profile-scoped. No embeddings needed for v1 |
| Fenced recall injection (`<memory-context>` block marked "NOT new user input") | Yes | Used for struggle warnings and tier-3 prefetch |
| Weekly idle-triggered curator (consolidate into umbrella skills, archive-never-delete) | Later (Phase 6) | HITL via the §6.4 external-editor proposal flow |
| External memory-provider plugin bus (mem0, honcho, …) | No | Patient-data environment: everything stays local; no plugin bus |
| Threat-pattern scanning of memory entries | Slim version | Small regex list; memories are HITL-approved anyway |
| Background review in a forked full agent with tool whitelist | Simplified | Our firewalled sub-calls are toolless single calls with JSON-schema output — cheaper and safer for small models than an agentic fork |

Design principle throughout: **Hermes trusts a frontier model to maintain its own
memory; we do not trust a 27B to do that unsupervised.** Wherever Hermes lets the
model mutate state freely, HPCA keeps the human-approval gate and uses
constrained decoding to make proposals structurally valid. Wherever Hermes
spends model attention (agentic review forks), HPCA uses single firewalled calls
with fixed JSON schemas.

## 1. Target memory taxonomy

One set per profile. Profiles are entirely user-defined — there are no
built-in ones beyond `default`, which exists only as the fallback for
sessions whose profile was deleted. A user creates blank profiles, copies an
existing one (all memory tiers and its own skills come along; the two then
diverge, and the copy records `copied_from`), and deletes them, taking their
memories, retrieval index and skills with them.

A profile remains "a named memory namespace", now covering all four layers:

1. **Curated memory (tier 1 + tier 2)** — always injected, hard char budget,
   human-editable markdown (format unchanged, §6.2). Tier 1 ≤ 1,200 chars
   (~300 tok), tier 2 ≤ 3,200 chars (~800 tok). *Declarative facts only.*
2. **Episodic memory** — every session transcript, searchable via FTS5 with the
   new `session_search` tool. Unbounded, zero injection cost, zero LLM cost to
   maintain.
3. **Procedural memory (skills)** — per-profile skills directory; the index
   (name + ≤60-char description) is injected, bodies fetched via `read_skill`.
   The reflection loop may *propose* new/patched skills.
4. **Retrieved memory (tier 3)** — activates the reserved tier: memories that
   are too situational for tier 2 (struggle notes, backend-specific
   workarounds, one-topic learnings) live in a per-profile FTS5 store and are
   prefetched per-turn into a fenced block, top-k, tightly capped.

Division of labor, stated in the system prompt (adapted from Hermes):
*"Tier 1/2 memory says what this site is, who the user is, and what you have
durably learned; skills say how to do a class of task; session history is
searchable and does not belong in memory; anything that will be stale in a week
does not belong in memory."*

## 2. Phase 1 — Enforcement and scoping fixes (no new subsystems)

Small, high-value corrections to the existing implementation.

1. **Hard char budgets at injection.** Replace token-estimate soft caps with
   character budgets (`memory.tier1_char_cap = 1200`, `tier2_char_cap = 3200` in
   `config.py`; keep old keys as deprecated aliases). `Profile.tier_text()`
   gains a budget check. Over budget → the existing §6.4 notification becomes
   *blocking for new writes* (writes rejected with current inventory, edit /
   summarize / defer offered) but injection still sends the full text — we never
   silently truncate user-owned memory. Show a Hermes-style usage meter in the
   memory screen and in the injected header:
   `Standing site notes [58% — 693/1200 chars]:`.
2. **Frozen per-session snapshot.** `HpcaApp` snapshots the rendered tier1/tier2
   text at session start (and after an approved write); `_render_system_prompt`
   uses the snapshot, not a per-round reload. Keeps the vLLM prefix cache valid
   and removes the subtle `_turn_memory` / `profile_memory` reload dance.
3. **Struggle notes reach the model.** In `run_turn` setup, matched struggle
   notes (existing keyword matcher) are appended to the *API copy* of the user
   message as a fenced block:

   ```
   <memory-context>
   [System note: recalled from this profile's memory, NOT new user input.]
   Past struggle (2026-06-02, backend qwen3-35b): <note>
   </memory-context>
   ```

   Cap: 2 notes, 600 chars total. The stored transcript keeps the clean user
   message (sidecar field on the message dict, mirroring Hermes'
   `api_content`). The toast stays as the human-facing half.
4. **Backend tags become live.** At injection, memories tagged with a backend
   other than the active `settings.llm.model` are annotated
   `(learned on <backend>)` — not dropped; small-model workarounds often
   transfer, and the annotation lets the model weigh it.
5. **Per-profile skills.** `load_skills(profile)` reads
   `<app_dir>/skills/<profile>/` with `<app_dir>/skills/_shared/` merged in;
   migration moves existing global skills to `_shared/`. RAG/symbols stay
   global for now (docs and code symbols are site-wide facts, not learnings) —
   revisit only if profiles diverge in practice.

## 3. Phase 2 — Episodic memory: `session_search`

New module `src/hpca/memory/episodic.py` + tool in `agent/memory_tools.py`.

* **Persistence:** new `messages(session_id, profile, turn_no, role, content,
  created_at)` table in `hpca.db` (additive migration via `ADDED_COLUMNS`
  pattern) + `messages_fts` FTS5 virtual table with sync triggers. Written from
  the same code path that writes `chatlogs/` today. Tool results and thinking
  are excluded; only user/assistant turns are indexed (keeps BM25 vocabulary
  clean — Hermes learned this the hard way with cron sessions).
* **Tool `session_search`** (profile-scoped by default):
  - `query=` → BM25 top-k (k=5), each hit returns: highlighted snippet,
    ±5-message window, first-3 and last-3 messages of the session ("bookends":
    goal and resolution without paying for the transcript).
  - `session_id=` + optional `around_turn=` → windowed scroll/read.
  - Result budget: ≤ 2,500 chars per call, head/tail-clipped like other tools.
* **Prompt guidance** (added to `prompts.py`): *"When the user references
  something from a past session, use session_search before asking them to
  repeat themselves. session_search shows what was said in past sessions — it
  is not evidence about the current state of files or the cluster."*
* FTS5 ships in the bundled sqlite on our targets; add a startup capability
  check with a graceful "episodic search unavailable" fallback.

## 4. Phase 3 — The `memory` tool (curated store, agent-writable with HITL)

Today only `/memorize`, `/conclude`, and struggle proposals write memory. Give
the orchestrator a first-class tool so it can capture facts the moment a user
states a preference or correction — Hermes' highest-value pattern.

* **Tool `memory`** in `agent/memory_tools.py`, pydantic schema:
  `operations: list[{op: add|replace|remove, tier: 1|2|3, match?: str,
  text?: str}]` — atomic batch, char budget checked only on the final state,
  `replace`/`remove` address entries by unique substring (multi-match →
  disambiguation error listing candidates; no IDs for the model to
  hallucinate).
* **HITL:** the tool call itself is the proposal. Executing it opens the
  existing `MemoryProposalScreen` (extended to show remove/replace diffs); the
  tool result reports what the user approved/rejected. Config
  `memory.auto_approve_tiers: []` allows opting tier 3 into auto-approval
  later; tiers 1–2 stay gated (destructive-tool precedent, §5.3).
* **Over-budget behavior** (Hermes' forced consolidation, HITL-adapted): the
  tool returns the full current inventory plus *"the write was rejected; either
  reissue ONE batch that also removes/shortens stale entries, or tell the user
  the memory tier is full"*. One retry max per turn (a 27B will loop; Hermes
  caps at 3 even for frontier models), then the §6.4 summarize flow is offered
  to the human.
* **Standing guidance block** in the system prompt, adapted verbatim from
  Hermes (`MEMORY_GUIDANCE`): save proactively on user preferences /
  corrections / durable environment facts; priority = corrections > environment
  facts > procedures; skip task progress, completed-work logs, anything stale
  in a week; *"write declarative facts, not instructions to yourself —
  imperative phrasing gets re-read as a directive in later sessions"*;
  procedures belong in skills.
* **Write hygiene:** dedup exact adds as no-ops; scan added text against a
  small threat-pattern list (`memory/threat_patterns.py`: "ignore previous
  instructions", role-injection markers, etc.) and flag matches in the approval
  screen; before rewriting `<name>.md`, round-trip the on-disk file through the
  parser and refuse (with a `.bak.<ts>` snapshot) if it no longer matches the
  loaded state (protects hand edits made mid-session).

## 5. Phase 4 — Reflection loop (replaces the struggle heuristic as sole trigger)

New module `src/hpca/agent/reflect.py`, following the `conclude.py` firewalled
pattern: single LLM call, bounded transcript slice, JSON-schema-constrained
output, never fed back into the session context.

* **Triggers** (checked in `_after_turn`, where `maybe_propose_struggle_note`
  runs today):
  - `turns_since_memory_review >= memory.review_interval` (default 8 user
    turns) → memory review.
  - `tool_rounds_since_skill_review >= memory.skill_review_interval` (default
    12 tool rounds) → skill review.
  - Failure markers / user abort (the existing `turn_struggled` heuristic) →
    immediate struggle review, bypassing counters. The heuristic is kept but
    demoted from "only trigger" to "priority trigger".
* **Execution:** after the reply is delivered (never competing with the user's
  turn), on the existing worker pattern. Input is a digest: last 24 messages
  verbatim, older turns collapsed to one synthetic summary line each
  (Hermes' digest replay — we cannot afford full-history replay on small
  contexts). Output schema:
  `{proposals: [{kind: memory|skill_patch|skill_new|struggle, tier?, text,
  keywords?, skill_name?}], nothing_to_save: bool}`.
* **Prompt** adapted from Hermes' `_MEMORY_REVIEW_PROMPT` /
  `_SKILL_REVIEW_PROMPT`, including the full anti-capture list: no
  environment-transient failures; no negative capability claims ("tool X does
  not work" hardens into a refusal cited for months); if a retry worked, the
  lesson is the retry pattern, not the failure; no one-off task narratives;
  skill names must be class-level, never a specific job ID / error string.
  Unlike Hermes we keep *"Nothing to save"* as a fully acceptable default —
  an eager 27B reviewer producing junk proposals every 8 turns would erode
  trust in the approval screen.
* **Delivery:** proposals go through `MemoryProposalScreen` / a new
  `SkillProposalScreen` (diff view for patches). A one-line toast summarizes:
  `Self-review: 1 memory + 1 skill patch proposed`.
* **Struggle notes** get `kind: struggle` + keywords as today, but are stored
  in **tier 3** (Phase 5) instead of tier 2 — they are situational by nature
  and were the main tier-2 bloat source.

## 6. Phase 5 — Tier 3: retrieved memory

Activates the format-reserved tier using infrastructure already in the repo.

* **Store:** `profile_memories(profile, tier, kind, backend, created, keywords,
  text)` in `hpca.db` + FTS5 mirror. Tier-3 blocks in `<name>.md` remain the
  human-editable source of truth; the DB is an index rebuilt on parse (same
  lenient-markdown philosophy — vim edits survive, index follows).
* **Retrieval:** per-turn prefetch, not tool-mediated — a small model won't
  reliably remember to call a search tool, and Hermes' own finding was that
  BM25 + pragmatic ranking beats embeddings. Query = the raw user message
  (skip Hermes' LLM query-rewrite step; not worth a model call here). Top-3
  hits, ≤ 800 chars total, injected in the same fenced `<memory-context>`
  block as struggle matches (Phase 1.3), ranked by BM25 with recency and
  same-backend bonuses. Optional embedding rerank via the existing
  `EmbeddingClient`/sqlite-vec behind `memory.tier3_embeddings: false`.
* **Overflow path:** when tier 2 is full, the §6.4 flow gains a fourth option:
  **(3) demote to tier 3** — move selected blocks under `## [tier3]` instead of
  deleting them. Curated memory stays small; nothing is lost.

## 7. Phase 6 — Curator and context compaction

* **Curator** (`memory/curator.py`): triggered on app idle (no cron daemon),
  ≥ 2 h idle and > 7 days since last run, state in
  `profiles/.curator_state.json`. Deterministic pass first: mark tier-3
  memories and skills untouched for 30 days as stale, 90 days as
  archive-candidates (archived = moved to `profiles/<name>.archive.md` /
  `skills/<profile>/.archive/` — *never deleted*, Hermes hard rule). Optional
  LLM pass proposes consolidations ("merge these 4 qwen-workaround notes into
  one") written to a proposal file and opened in the §6.4 external editor —
  never auto-applied. Pinning: a `pinned` metadata flag exempts entries.
* **Context compaction** (closes the biggest raw gap): a compaction step in
  `graph.py`'s orchestrator when estimated prompt size exceeds
  `0.8 × max_model_len` (finally consuming that config field): summarize the
  oldest two-thirds of messages via a firewalled call into one synthetic
  context message. **Pre-compress extraction hook** (Hermes
  `on_pre_compress`): before discarding, the reflection reviewer (Phase 4)
  runs over the soon-to-be-dropped slice so durable facts can be proposed
  before the context that evidences them disappears.

## 8. Config additions (`settings.json`, `memory` section)

As implemented:

```json
"memory": {
  "tier1_char_cap": 1200,
  "tier2_char_cap": 3200,
  "tier3_prefetch_chars": 800,
  "tier3_prefetch_count": 3,
  "cross_profile_search": false,
  "review_interval": 8,
  "propose_new_skills": true,
  "curator_interval_days": 7,
  "curator_stale_days": 30,
  "curator_archive_days": 90
}
```

`tier1_token_cap`/`tier2_token_cap` are still honored (×4 chars) when the
char caps are left at their defaults, so a hand-tuned settings.json keeps
working.

Compaction has no setting of its own: it activates when the active backend
in the catalog records a `max_model_len`, and stays off otherwise rather
than guessing a window.

## 9. Build order, effort, and dependencies

All six phases are implemented, one commit each:

| Phase | Contents | Key modules |
|---|---|---|
| 1 | Budgets, frozen snapshot, fenced struggle injection, backend tags, per-profile skills | `profiles.py`, `agent/memory_context.py`, `skills.py` |
| 2 | messages table + FTS5 + `session_search` | `episodic.py`, `agent/memory_tools.py` |
| 3 | `memory` tool + guidance + write hygiene | `memory_ops.py`, `tui/memory_screens.py` |
| 4 | Reflection loop + proposal screens | `agent/reflect.py` |
| 5 | Tier 3 store + prefetch + demotion | `memory_index.py` |
| 6 | Compaction + pre-eviction extraction + curator | `agent/compact.py`, `curator.py` |

## 10. Testing strategy

* Unit: budget enforcement incl. batch-final-state check; substring
  disambiguation; lenient parse round-trip with hand-edited files; FTS5
  ranking/bookends on fixture sessions; threat-pattern flags; digest builder.
* Integration (existing `HPCA_HOME` + injected-LLM test harness): a scripted
  session where a fake LLM proposes memories → approval → next session's
  system prompt contains them within budget; struggle → fenced block appears
  in API copy but not the stored transcript; compaction preserves the answer
  to a question about early-session content.
* Adversarial fixtures: a memory entry containing "ignore previous
  instructions" is flagged; an on-disk profile edited mid-session triggers the
  drift guard, not data loss.

## 11. Open questions — resolved (2026-07-19)

1. Tier-3 auto-prefetch **is visible**: a collapsed line in the TUI transcript
   showing what was recalled.
2. Cross-profile `session_search` exists behind
   `memory.cross_profile_search: false` (default off).
3. Skill-creation proposals from the reflection loop are gated behind
   `memory.propose_new_skills: true` (default on, so quality can be evaluated
   in practice; set to `false` to disable).
