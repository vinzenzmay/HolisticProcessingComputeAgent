"""Prompt building blocks (§4.3 prompt assembly).

Empirically validated against the live Qwen3.6 backend: without the
respond-vs-tool guidance the model routes its own words through action tools
(e.g. wrapping a greeting in an ``echo`` call); with it, conversational turns
reliably become direct responses.

Every block here is paid on every turn of every session, so the standing budget
is kept to facts a model cannot guess: what is site-specific (tools not on
PATH), what is counter-intuitive (scripts run fail-fast, ``{key}`` expansion,
run_bash is not for writing files), and what the tool schemas cannot say. The
rationale that once explained each rule to a *reader* lives in these comments
instead, where it costs no tokens; repetition of what a tool's own description
already states was removed rather than restated. The blocks stay separate and
in this order — the assembly below is unchanged — so a block can be measured
by shortening it, not by deleting it.
"""

# The opening clause is not throat-clearing, it is the load-bearing half.
# Measured on the live 27B (edit_eval shift tier, 2026-08-13): shortened to
# "Call a tool only to perform a real action; answer ... directly", the model
# stopped acting on a plain "create this file at /abs/path" request in 5 of 6
# generations — narrating the plan, asking the user to register the path,
# saying "I cannot perform file operations at this time", and twice reporting
# a file it had never written. Restoring "You are an assistant with tools"
# brought the tool calls back. Same failure signature as the one recorded in
# middleware.format_instruction: strip the assertion that it HAS tools and the
# 27B concludes it has none. A restriction on when to call a tool only reads
# correctly to a model that already believes it can.
RESPOND_VS_TOOL_GUIDANCE = (
    "You are an assistant with tools. Tools perform real actions. Call a tool "
    "whenever the user asks for an action a tool performs; answer conversation, "
    'questions and greetings directly with {"action": "respond", ...} — never '
    "route your own words through a tool."
)


# Scoped deliberately: the earlier blanket "never answer from memory" made the
# agent research its OWN tools before calling them, which is pure waste — their
# schemas are already in this prompt. The hallucination risk is in the *external*
# programs it drives from scripts (samtools, minimap2, ...), not in its toolbox.
GROUNDED_ANSWERING_GUIDANCE = (
    "Never state an EXTERNAL program's flags or syntax from memory — the "
    "command-line tools you drive from scripts (samtools, minimap2, ...), "
    "libraries, file formats, error messages: call ask_docs and relay its "
    "cited answer. Your own tools need no research; their schemas are here, "
    "so just call them."
)

# The key-first drilling that used to open this block is gone: every file tool
# now takes a literal absolute path wherever it takes a key
# (PathRegistry.resolve_or_register), so "register first, then call" is no
# longer true, and the register_path round-trip it mandated was pure tax.
# What survives is what the schemas cannot say: that keys exist at all, and
# the brace expansion in run_bash lines.
PATH_WORKFLOW_GUIDANCE = (
    "File tools take a registry key or an absolute path. When you are given a "
    "path, pass it straight to the tool — you do not need to register it "
    "first. register_path only gives a long path a short key for later. In a "
    "run_bash line a key is written in braces — `head -2 "
    "{ref_fasta}` — and expands to the real path; that is also how you run a "
    "script you created: `{my_script} --flag`."
)

# Without this the model has no idea it may look around: every file tool takes
# a key, and keys came from the user's message, so "find X somewhere on this
# system" looked impossible and it narrated instead of acting. (Literal paths
# are accepted now, but nothing else tells it that looking is allowed.) The
# package-manager probe this used to carry lives in ENVIRONMENT_TOOL_GUIDANCE,
# which is the block about tools that are not on PATH.
DISCOVERY_GUIDANCE = (
    "You can look around this system, and you should rather than guess or ask: "
    "`find <dirs> -maxdepth <n> -iname '<pattern>' 2>/dev/null | head -20` "
    "locates files, `command -v samtools` checks a program. Keep searches "
    "bounded so they finish in seconds — likely roots rather than /, capped "
    "depth, piped through head."
)

# The single most common way the agent burns its tool budget: it cannot run an
# environment-managed tool and thrashes trying to activate one. The idiom that
# ends that thrashing lives in ENVIRONMENT_TOOL_GUIDANCE below.
#
# This block is also the one home of the skeleton-then-fill protocol for a long
# write (measured in v0.19.0: a 27B loses count filling ten sections, hence the
# `TBD:` marker the tool result then ratchets down). create_file's own
# description used to repeat it; it says only what create_file is for now.
SCRIPT_GUIDANCE = (
    "Run real work with create_script then start_background_script; use "
    "run_bash when you want the output now. Bash scripts run fail-fast (`set "
    "-euo pipefail` is added), so a failed command stops the script: never end "
    "one with an unconditional `echo \"Done\"`, let the exit code report "
    "success. Write each command plainly on one line: no added quotes, commas "
    "or `\\` line-continuations between arguments. "
    "To WRITE a file that is not a script — specs, notes, a config — call "
    "create_file, never `echo` lines or a `cat << EOF` heredoc in run_bash, "
    "which is for looking around and not for writing. Past ~150 lines, "
    "create_file a skeleton — headings, each with one `TBD: ...` line — then "
    "fill one section per edit_file call. To CHANGE a file that exists, read "
    "the region and call edit_file with just the lines to replace."
)


# Retired the conda-specific idioms. They were wrong on the very machine this
# runs on, where conda is not installed at all (mamba is), and wrong again on
# any site using modules, Spack or containers. A live session followed them and
# failed twice — `conda run` then `mamba run`, both "command not found" — and
# only succeeded once it called the binary by its full path. So teach the form
# that works whatever installed the tool, and teach checking before assuming.
# The real fix is a provider registry rendering these facts per site; until
# then this says only what is true regardless of packaging system.
ENVIRONMENT_TOOL_GUIDANCE = (
    "Scientific tools here are usually NOT on PATH: they live in package "
    "environments or module trees, and which system manages them differs per "
    "site. Assume nothing — not that a tool name works as typed, not that any "
    "given package manager exists. Check (`command -v conda mamba micromamba "
    "spack module apptainer`), then call the executable by its full binary "
    "path in your script: the one form that works whatever installed it. A "
    "`conda run -n <env>` wrapper needs that wrapper installed; do not "
    "`source` an activate script."
)


def environment_facts() -> str:
    """Volatile facts, rendered per turn — never stored as memories (§4.3).

    These ride the *tail* of the turn (the user message's ``api_content``
    sidecar), NOT the system prompt: the minute-granularity timestamp used to
    sit at the front of the prompt and invalidated the backend's prefix KV
    cache for the whole conversation every time the clock ticked over a minute.
    Keeping volatile content after the stable history preserves the cache.
    """
    from datetime import datetime

    return f"Current date and time: {datetime.now():%Y-%m-%d %H:%M} (local)."


# The tool schema alone does not teach *when* to reach for recall; small
# models ask the user to repeat themselves instead. The second half is the
# Hermes "source-first limit": recall is what was SAID, not what currently IS.
SESSION_SEARCH_GUIDANCE = (
    "Past sessions are searchable with session_search: when the user refers to "
    "an earlier conversation ('like last time', 'the pipeline we set up'), "
    "search before asking them to repeat it. It shows what was SAID then — "
    "never evidence about the current state of files, jobs, or the cluster; "
    "check the system itself for that."
)


# Adapted from Hermes' MEMORY_GUIDANCE. The declarative-vs-imperative rule is
# the load-bearing one: an imperative memory ("always use --no-mmap") is
# re-read as a standing order in later sessions and overrides what the user is
# actually asking for now — a failure mode small models are especially prone
# to, since they weight instructions in context over the current request.
MEMORY_GUIDANCE = (
    "You have memory across sessions but cannot write it: use the memory tool "
    "to FLAG durable facts, which the user reviews at /conclude. Flag "
    "proactively — corrections and preferences first, then site facts, then "
    "workarounds — whenever the user states a preference, corrects you, or "
    "tells you a durable fact about this site. Write them as declarative "
    "FACTS, not instructions to yourself: “the user prefers R over Python” is "
    "right, “always answer in R” is wrong — an instruction gets re-read as a "
    "standing order in a later session and overrides what is being asked then. "
    "Do NOT flag task progress, file names, job ids, or anything stale in a "
    "week; past sessions are searchable instead."
)


# The tool schemas say what watch_log/watch_job do, not when they earn their
# keep. Without this the model reports a job id and a log path in chat and
# moves on, which leaves the user doing exactly the squeue-then-tail loop the
# panel exists to end.
WATCH_GUIDANCE = (
    "When you find a running job the user cares about call watch_job with its "
    "id, and when you find the log a running tool is writing call watch_log "
    "with its path. Both pin a live box in the right-hand panel — Slurm state, "
    "time since the last write — so the user sees whether the work is alive "
    "without asking you. Do it as soon as you have the id or path, unasked, "
    "and say that you did."
)


SKILLS_GUIDANCE = (
    "The user has defined skills: written procedures for specific tasks, whose "
    "names are not listed here. When a request seems to call for one, call "
    "read_skill (any name returns the available skills) and follow the "
    "procedure it gives back. A skill the user invokes with \"/<skill>\" is "
    "handed to you inline; follow that."
)


def build_skill_directive(name: str, description: str, body: str) -> str:
    """Directive placed on the API copy of a user message when the user
    invoked a skill by name (typed ``/<skill>``).

    Unlike the on-demand read_skill path, an explicit invocation drops the full
    procedure straight into the turn so a small model follows it without a tool
    round-trip. Rides the sidecar (``compose_api_content``) so the transcript
    keeps the clean ``/<skill> …`` the user typed.
    """
    head = f"The user invoked the skill {name!r} directly."
    if description:
        head += f" ({description})"
    body = body.strip()
    if not body:
        return f"{head} Follow that skill for this request."
    return f"{head} Follow its procedure for this request:\n\n{body}"


def orchestrator_system_prompt(
    *,
    system_prompt_memories: str = "",
    memory_meter: str = "",
    has_skills: bool = False,
    session_search: bool = False,
    memory_tool: bool = False,
    watch_tools: bool = False,
) -> str:
    """System prompt for the orchestrator; the cacheable prompt *prefix*.

    Everything here is stable across a session so the backend's prefix KV
    cache survives from turn to turn: only memory approvals, a skill change or
    a mode switch move it. Volatile facts (the date/time) deliberately live at
    the tail instead — see ``environment_facts``.

    Skills are *not* listed here — only a one-line note that they exist and how
    to fetch one (``read_skill``), so the prompt stays small no matter how many
    the user has defined. Bodies reach the model on demand, or inline when the
    user invokes ``/<skill>``. The meter shows how full the system-prompt
    memory scope is against its token budget.
    """
    parts = [
        "You are HPCA, a terminal assistant helping a scientist with data "
        "processing on an HPC cluster.",
        RESPOND_VS_TOOL_GUIDANCE,
        PATH_WORKFLOW_GUIDANCE,
        DISCOVERY_GUIDANCE,
        ENVIRONMENT_TOOL_GUIDANCE,
        SCRIPT_GUIDANCE,
        GROUNDED_ANSWERING_GUIDANCE,
    ]
    if watch_tools:
        parts.append(WATCH_GUIDANCE)
    if session_search:
        parts.append(SESSION_SEARCH_GUIDANCE)
    if memory_tool:
        parts.append(MEMORY_GUIDANCE)
    if has_skills:
        parts.append(SKILLS_GUIDANCE)
    if system_prompt_memories:
        label = _metered("Memory (site facts, preferences, learnings)", memory_meter)
        parts.append(f"{label}:\n{system_prompt_memories}")
    return "\n\n".join(parts)


def _metered(label: str, meter: str) -> str:
    return f"{label} [{meter}]" if meter else label
