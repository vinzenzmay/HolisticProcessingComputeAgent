"""Prompt building blocks (§4.3 prompt assembly).

Empirically validated against the live Qwen3.6 backend: without the
respond-vs-tool guidance the model routes its own words through action tools
(e.g. wrapping a greeting in an ``echo`` call); with it, conversational turns
reliably become direct responses.
"""

RESPOND_VS_TOOL_GUIDANCE = (
    "You are an assistant with tools. Tools perform real actions. "
    "Use a tool only when the user asks for an action a tool performs. "
    "For conversation, questions, and greetings, always answer directly "
    'with {"action": "respond", ...} — never route your own words through a tool.'
)

# The same rule for the native tool-calling protocol
# (``settings.llm.tool_protocol``), where an answer is ordinary assistant text
# and ``{"action": "respond", ...}`` names a format the model cannot emit —
# leaving that clause in tells it to produce something the channel has no room
# for. The last sentence is not a rewording: measured on the live 27B, the
# native protocol's characteristic failure is narrating the edit instead of
# making it (one read_file call, then "I have updated the file", with the file
# untouched), which the envelope's explicit two-branch listing suppresses by
# making "call a tool" the visibly available option.
RESPOND_VS_TOOL_GUIDANCE_NATIVE = (
    "You are an assistant with tools. Tools perform real actions. "
    "Use a tool only when the user asks for an action a tool performs. "
    "For conversation, questions, and greetings, answer directly in your own "
    "words — never route them through a tool. Never describe an action you "
    "have not performed: if the request needs a change made, call the tool "
    "that makes it, and answer only once the tool has reported it done."
)


# Scoped deliberately: the earlier blanket "never answer from memory" made the
# agent research its OWN tools before calling them, which is pure waste — their
# schemas are already in this prompt. The hallucination risk is in the *external*
# programs it drives from scripts (samtools, minimap2, ...), not in its toolbox.
GROUNDED_ANSWERING_GUIDANCE = (
    "Your own tools need no research: you already have their schemas, so never "
    "look up documentation before calling one — just call it. Grounding applies "
    "to EXTERNAL software instead: command-line programs you invoke from scripts "
    "(samtools, minimap2, bcftools, ...), libraries, file formats, and error "
    "messages. Never state their flags or syntax from memory — call ask_docs and "
    "relay its cited answer. Answer directly only for conversation and for "
    "information already present in this conversation."
)

# What is left of the path block once the registry is gone
# (specs/specs-path-registry.md).
# It says the one thing a path-taking interface still needs said — where a
# relative path is anchored — and the one handle that is still not a path: a
# kept script's name.
PATH_WORKFLOW_GUIDANCE = (
    "File tools take a path. Use the path the user gave you, or the one you "
    "just saw in output, exactly as it stands — copy it, do not retype it "
    "from memory. A relative path is taken from the working directory this "
    "session was started in, so '.' is that directory. A script you created "
    "with create_script is named, not pathed: in a run_bash line write its "
    "name in braces — `{my_script} --flag` — and it expands to its path."
)

# Without this the model has no idea it may look around: it narrated instead
# of acting when a path was not already in the conversation.
DISCOVERY_GUIDANCE = (
    "You can look around this system, and you should rather than guess or ask. "
    "Use run_bash for a one-shot check — it writes, runs and returns the "
    "output of a small bash script in one step. Useful lines: "
    "`find <dirs> -maxdepth <n> -iname '<pattern>' 2>/dev/null | head -20` to "
    "locate files; `command -v samtools` to check a program. For a tool that "
    "is not on PATH, first see which package managers this site actually has "
    "(`command -v conda mamba micromamba spack module apptainer`) and query "
    "only the ones that answered, or search the likely install roots directly "
    "with find. Keep searches "
    "bounded so they finish in seconds: start from likely roots rather than /, "
    "cap the depth, pipe through head. Then use the paths it printed as they "
    "were printed."
)

# The single most common way the agent burns its tool budget: it cannot run an
# environment-managed tool and thrashes trying to activate one. The idiom that
# ends that thrashing lives in ENVIRONMENT_TOOL_GUIDANCE below.
#
# The create_file half used to be the negative branch — "a file that is not a
# script", then a list of prose genres — under an opening line that offered "a
# pipeline" as something create_script makes. Asked for a snakemake workflow,
# the live model read both sentences exactly as written and called
# create_script with kind=bash; `bash -n` refused it, and kind=python would
# have been refused too, because a Snakefile is neither. There is no
# create_script call that writes one, so the classification had to be made
# before the tool was picked, and by the axis that actually decides it: what
# LANGUAGE the file is, not whether it runs. Hence the split stated as bash and
# python against everything else, with the executable examples named on the
# create_file side, where prose examples alone could never put them.
#
# The two-file shape is spelled out for the same reason: a workflow written by
# create_file has a path and no script *name*, so start_background_script — which
# takes names — cannot reach it, and the bash wrapper is what closes that.
SCRIPT_GUIDANCE = (
    "Running real work — driving a tool, launching a pipeline — means "
    "create_script then start_background_script; those bash scripts run "
    "fail-fast (`set -euo pipefail` is added), so a failed command stops the "
    "script and is reported as failed. Because of that, never end a script "
    "with an unconditional `echo \"Done\"`: let the exit code report success. Choose between the two "
    "run tools by when you need the answer, not by what the script is: "
    "start_background_script for work that outlives this turn (it returns a pid "
    "and tells you later how it ended), run_bash when you want the output now — "
    "a look-around check, or a kept script run with `{its_name}`. "
    "Build each command plainly on one line — real flags and paths separated "
    "by single spaces, nothing else. Do NOT insert quotes, commas or `\\` "
    "line-continuations between arguments; a stray `\",` turns your command "
    "into garbage the tool rejects. Write paths out literally; do not leave a "
    "shell variable unset. "
    "create_script writes bash and python, and those two only. Every "
    "other file is a create_file, with the path of the new file and the "
    "content one line per array element: prose (a specs or design "
    "document, notes, a README), data (a config, a sample sheet), and "
    "equally anything that RUNS in a language create_script does not "
    "have — a Snakefile, a Makefile, a nextflow .nf, an R script. Being "
    "real work does not make a file a script here; being bash or python "
    "does. To then run one of those, create_file it at the path it needs "
    "to live at, then create_script a short bash script that calls it "
    "(`snakemake -s <path> ...`) and start that. "
    "Never build a file out of `echo` lines or a `cat << EOF` heredoc in "
    "run_bash: the content then has to survive bash quoting, a single stray "
    "line costs the whole file, and run_bash refuses a script that long "
    "anyway — it is for looking around, not for writing. A file longer than "
    "~150 lines does not fit in one call: create_file a skeleton instead — "
    "headings, each with one `TBD: ...` placeholder line — then fill one "
    "section per edit_file call (placeholder in old_lines, content in "
    "new_lines). "
    "To CHANGE a file that already exists — a script of yours, a config, a "
    "sample sheet — call edit_file with just the lines to replace. Do not "
    "re-send the whole file through create_script or create_file: create_file "
    "refuses an existing path, and create_script's name must be new, so it "
    "would leave the old file sitting there beside a second copy. Read the "
    "file first and copy the lines into old_lines exactly as they appear, "
    "spacing and all. On a long file read_file shows one window at a time: "
    "page through with start_line, as each result suggests, until you have "
    "seen the exact region you will edit."
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
    "site (conda, mamba, micromamba, Spack, Lmod, containers). Assume nothing "
    "— neither that a tool name works as typed, nor that any particular "
    "package manager exists. Find the executable first (see below), then call "
    "it by its full binary path in your script: that is the one form that "
    "works whatever installed it. A wrapper like `conda run -n <env> <tool>` "
    "only works if that wrapper is itself installed, so check with "
    "`command -v conda` before relying on one. Do not try to `source` an "
    "activate script: it is rarely where you expect and wastes a step."
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
    "Past sessions are searchable with session_search. When the user refers "
    "to something from an earlier conversation ('like last time', 'the "
    "pipeline we set up'), search before asking them to repeat it. "
    "session_search shows what was said back then — never treat it as "
    "evidence about the current state of files, jobs, or the cluster; check "
    "the system itself for that."
)


# Adapted from Hermes' MEMORY_GUIDANCE. The declarative-vs-imperative rule is
# the load-bearing one: an imperative memory ("always use --no-mmap") is
# re-read as a standing order in later sessions and overrides what the user is
# actually asking for now — a failure mode small models are especially prone
# to, since they weight instructions in context over the current request.
MEMORY_GUIDANCE = (
    "You have memory that persists across sessions. You cannot write it "
    "directly: use the memory tool to FLAG durable facts as you notice them, "
    "and they are collected for the user to review together when the session "
    "is concluded. Flag proactively when the user states a preference, "
    "corrects you, or tells you a durable fact about this site — the best "
    "memory is one that stops the user having to repeat themselves. "
    "Priority: corrections and preferences first, then site facts, then "
    "workarounds. Write memories as declarative FACTS, not instructions to "
    "yourself: “the user prefers minimap2 over bwa” is right, “always align "
    "with minimap2” is wrong — an instruction gets re-read as a standing order in a "
    "later session and overrides what is being asked then. Do NOT flag task "
    "progress, what you did this session, file names, job ids, or anything "
    "that will be stale in a week; past sessions are searchable instead. "
    "Flag rather than agonize — the user has the final say at /conclude."
)


# The tool schemas say what watch_log/watch_job do, not when they earn their
# keep. Without this the model reports a job id and a log path in chat and
# moves on, which leaves the user doing exactly the squeue-then-tail loop the
# panel exists to end.
WATCH_GUIDANCE = (
    "The right-hand panel can pin things for the user to keep an eye on. "
    "When you find a running job the user cares about, call watch_job with "
    "its id; when you find the log a running tool is writing (a pipeline "
    "run's log, a tool's own log file), call watch_log with its path. Both "
    "give the user a live box — a job's Slurm state, a log's time since the "
    "last write — so they can see at a glance whether the work is still "
    "alive instead of asking you to check. Each takes an array: pin "
    "everything you found in ONE call — every id in one watch_job, every "
    "path in one watch_log — rather than a call per target. Do this as soon "
    "as you have the ids or the paths in hand, without being asked, and say "
    "that you did. "
    "Watching costs the user nothing: they drop a box with one keypress."
)


SKILLS_GUIDANCE = (
    "The user has defined skills: written procedures for specific tasks. "
    "The list is kept out of this prompt to stay small — you are not shown "
    "the names. When a request seems to call for a skill, call read_skill "
    "(passing any name returns the available skills) and follow the procedure "
    "it gives back. When the user invokes one directly with \"/<skill>\", its "
    "procedure is handed to you inline; follow that."
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
    native_tools: bool = False,
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
        # Which protocol carries a call changes what "answer directly" looks
        # like, so this one block varies with it; everything else is the same
        # text either way.
        RESPOND_VS_TOOL_GUIDANCE_NATIVE if native_tools else RESPOND_VS_TOOL_GUIDANCE,
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
