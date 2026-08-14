#!/usr/bin/env python
"""Standalone eval: how well does the live LLM drive HPCA's edit_file /
read_file / create_file tools?

Runs each task in a fresh temp workspace with a real ToolContext (real
PathRegistry, TrashManager, ProcessRunner), a bounded decision loop through
hpca.agent.middleware.decide, and tool handlers called DIRECTLY (no HITL
gating). Designed to be run twice — once from a worktree of main (baseline)
and once from the treatment branch — so it touches only stable HPCA APIs and
degrades gracefully when minor details differ.

Usage:
    python evals/edit_eval.py --out results.json --repeats 3 [--tasks N]
                              [--label name] [--dry-run]

Exit codes: 0 ok, 2 backend unreachable/unauthorized.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Make `import hpca` work when run from a repo checkout without installation.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

try:
    import httpx
except ImportError:
    httpx = None  # only needed for live runs

from hpca.agent.context import ToolContext
from hpca.agent.middleware import DecisionError, DirectResponse, ToolCall, decide
from hpca.agent.tools import ToolRegistry
from hpca.config import Settings
from hpca.registry import PathRegistry
from hpca.runner import ProcessRunner
from hpca.trash import TrashManager

LIVE_URL = os.environ.get("HPCA_TEST_LLM_URL", "http://localhost:20001/v1")
LIVE_KEY = os.environ.get("HPCA_TEST_LLM_KEY")
# How long the /models reachability probe waits; see discover_backend.
PROBE_TIMEOUT_S = float(os.environ.get("HPCA_TEST_LLM_PROBE_TIMEOUT", "30"))

MAX_DECISIONS = 8

# ------------------------------------------------------------------ prompts

_FALLBACK_GUIDANCE = (
    "You are an assistant with tools. Tools perform real actions. "
    "Use a tool only when the user asks for an action a tool performs. "
    "For conversation, questions, and greetings, always answer directly "
    'with {"action": "respond", ...} — never route your own words through a tool.'
)


# The task-facing half of the system prompt. Byte-identical to what the core
# and hard tiers have always been measured with — changing a character of it
# moves their baselines, so a tier that needs different standing text says so
# per task (``Task.system_note``) instead of editing this.
_DEFAULT_SYSTEM_NOTE = (
    "You are working on files in a workspace. The relevant paths are "
    "already registered in the path registry under the keys named in the "
    "task; do not call register_path. Read a file with read_file before "
    "editing it, make the change with edit_file (or create_file for a new "
    "file), and when the change is done answer directly with a short "
    "confirmation. Copy old_lines exactly as they appear in the file, "
    "including indentation and spacing, without line numbers."
)

# The 'shift' tier's note: there, a key may be wrong or absent and the task
# hands the model a literal path, so forbidding register_path would make the
# tasks unwinnable rather than measuring what they cost.
_SHIFT_SYSTEM_NOTE = (
    "You are working on files in a workspace. Some paths are already "
    "registered in the path registry under the keys named in the task, but "
    "the task may also give you a literal absolute path, or a key that turns "
    "out to be wrong — you may call register_path to give a path you were "
    "given a key. Read a file with read_file before editing it, make the "
    "change with edit_file (or create_file for a new file), and when the "
    "change is done answer directly with a short confirmation. Copy old_lines "
    "exactly as they appear in the file, including indentation and spacing, "
    "without line numbers."
)


def _system_prompt(note: str = "", native: bool = False) -> str:
    """Match HPCA's own prompt style; fall back if the constant moved."""
    # Production picks this block by protocol (orchestrator_system_prompt):
    # under native an answer is plain text, so the envelope's
    # {"action": "respond", ...} names a format the model cannot emit.
    guidance = None
    if native:
        try:
            from hpca.agent.prompts import RESPOND_VS_TOOL_GUIDANCE_NATIVE as guidance
        except Exception:
            guidance = None
    if guidance is None:
        try:
            from hpca.agent.prompts import RESPOND_VS_TOOL_GUIDANCE as guidance
        except Exception:
            guidance = _FALLBACK_GUIDANCE
    # Production always carries the file-writing guidance
    # (orchestrator_system_prompt includes SCRIPT_GUIDANCE); measuring
    # without it handicaps whichever side is checked out. Each side gets its
    # own version's text — that IS the production difference under test.
    try:
        from hpca.agent.prompts import SCRIPT_GUIDANCE as script_guidance
    except Exception:
        script_guidance = ""
    # Likewise PATH_WORKFLOW_GUIDANCE. Leaving it out was a real measurement
    # bug, not a simplification: it is the block that says what a file tool's
    # key argument accepts, so without it the shift tier judged the path
    # changes with the guidance about paths removed — and the model, told
    # nothing, sometimes answered instead of acting. Production always carries
    # it (orchestrator_system_prompt), and each checkout supplies its own text.
    try:
        from hpca.agent.prompts import PATH_WORKFLOW_GUIDANCE as path_guidance
    except Exception:
        path_guidance = ""
    return f"{guidance}\n\n{path_guidance}\n\n{script_guidance}\n\n" + (
        note or _DEFAULT_SYSTEM_NOTE
    )


# ----------------------------------------------------------- history shape

# What one completed tool call adds to the conversation. Production is moving
# from "a bare [tool result] user turn" to "the model's own call echoed back as
# an assistant turn, then the result" — the shape agent-trained models are
# post-trained on. The import decides which shape a run measures: on a baseline
# checkout without hpca.agent.history it fails and the fallback below
# reproduces today's single user message verbatim.
try:
    from hpca.agent.history import tool_exchange
except Exception:

    def tool_exchange(
        tool_name: str, arguments: dict, result: str, *, call_id: str = ""
    ) -> list[dict]:
        return [{"role": "user", "content": f"[tool result] {tool_name}: {result}"}]


def _not_native() -> bool:
    return False


def _exchange(tool_name: str, arguments: dict, result: str, call_id: str) -> list[dict]:
    """tool_exchange, tolerant of a baseline checkout that has no call_id."""
    try:
        return tool_exchange(tool_name, arguments, result, call_id=call_id)
    except TypeError:
        return tool_exchange(tool_name, arguments, result)


def _arguments_dict(arguments) -> dict:
    """decision.arguments as a plain dict, whatever it is on this branch."""
    try:
        return arguments.model_dump()
    except Exception:
        try:
            return dict(arguments)
        except Exception:
            return {}


# ------------------------------------------------------------- tool wiring


def _eval_tool_registry() -> ToolRegistry:
    """read_file + edit_file + create_file + register_path, via the same add_*
    wiring HPCA uses. register_path is what a model reaches for when a key is
    wrong or missing (the 'shift' tier); the core and hard tiers still tell it
    not to, in prompt text that has not changed."""
    from hpca.agent.builtin_tools import default_tool_registry
    from hpca.agent.file_tools import add_file_tools

    registry = add_file_tools(default_tool_registry())
    wanted = ["read_file", "edit_file", "create_file", "register_path"]
    try:
        return registry.subset(wanted)
    except Exception:
        # subset() may differ across branches; rebuild by hand.
        slim = ToolRegistry()
        for name in wanted:
            slim.register(registry.get(name))
        return slim


def _filtered_kwargs(cls, **kwargs):
    """Drop kwargs the dataclass/init on this branch does not know about."""
    params = inspect.signature(cls).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _make_context(workdir: Path) -> ToolContext:
    from hpca.db import connect, init_db

    conn = connect(workdir / "hpca.db")
    init_db(conn)
    ctx = ToolContext(
        **_filtered_kwargs(
            ToolContext,
            registry=PathRegistry(
                conn, **_filtered_kwargs(PathRegistry, profile="eval", session_id="eval")
            ),
            runner=ProcessRunner(
                conn,
                **_filtered_kwargs(
                    ProcessRunner, session_id="eval", log_dir=workdir / "logs"
                ),
            ),
            settings=Settings(),
            scripts_dir=workdir / "scripts",
            session_id="eval",
            trash=TrashManager(workdir / "trash", backup_limit_bytes=1024 * 1024),
        )
    )
    ctx._eval_conn = conn  # keep alive; closed with the workspace
    return ctx


# ------------------------------------------------------------------- tasks


@dataclass
class Task:
    name: str
    prompt: str
    # registry key -> (file name, content). Content written verbatim (bytes
    # when bytes are given, e.g. CRLF fixtures).
    files: dict[str, tuple[str, str | bytes]]
    # predicate on the workspace dir: did the task succeed?
    check: Callable[[Path], bool]
    # registry key -> relative name, registered but NOT created. A key may
    # legitimately point at nothing: register_path takes a path before it
    # exists, so the model meets one whenever it names a file it is about to
    # write. Kept separate from `files` because the whole point is the absence.
    missing: dict[str, str] = field(default_factory=dict)
    # relative name -> content, written into the workspace and deliberately
    # NOT registered under any key. The only way to set up a file the model
    # can reach solely by its literal absolute path.
    unregistered: dict[str, str | bytes] = field(default_factory=dict)
    # scripted decisions for --dry-run: the fake model emits these in order,
    # then responds "DONE".
    fake_calls: list[dict] = field(default_factory=list)
    # decision budget override; None = the global MAX_DECISIONS. Multi-part
    # writes (skeleton + one edit per section) legitimately need more calls.
    max_decisions: int | None = None
    # opt-in prompt templating: when True the prompt is .format()ed with the
    # run's workspace path ({workspace}). Opt-in because several prompts carry
    # literal braces in shell/config snippets, and a formatting crash would
    # silently zero a task's success rate. Also substitutes {workspace} inside
    # the fake_calls arguments, so a scripted route can name a literal path.
    templated: bool = False
    # per-task replacement for the task-facing half of the system prompt
    # (_DEFAULT_SYSTEM_NOTE). Empty = the text core and hard have always used.
    system_note: str = ""


def _edit(key: str, old: list[str], new: list[str]) -> dict:
    return {
        "action": "tool_call",
        "tool": "edit_file",
        "arguments": {"registry_key": key, "old_lines": old, "new_lines": new},
    }


def _read(key: str) -> dict:
    return {"action": "tool_call", "tool": "read_file", "arguments": {"registry_key": key}}


def _register(key: str, path: str) -> dict:
    return {
        "action": "tool_call",
        "tool": "register_path",
        "arguments": {"key": key, "path": path},
    }


def _create(dir_key: str, name: str, lines: list[str]) -> dict:
    return {
        "action": "tool_call",
        "tool": "create_file",
        "arguments": {"dir_key": dir_key, "name": name, "content_lines": lines},
    }


BIG_FILE_HEADER = "# run manifest — generated, do not hand-edit sections A/B\n"


def _big_file() -> str:
    lines = [BIG_FILE_HEADER.rstrip("\n")]
    for i in range(1, 299):
        if i == 150:
            lines.append("threads = 4        # section C: tunables")
        else:
            lines.append(f"entry_{i:03d} = value_{i:03d}")
    return "\n".join(lines) + "\n"


def build_tasks() -> list[Task]:
    tasks: list[Task] = []

    # 1. simple one-line replacement in a small script
    small_sh = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'echo "starting run"\n'
        "sleep 1\n"
        'echo "done"\n'
    )
    tasks.append(
        Task(
            name="simple_replace",
            prompt=(
                "The bash script is registered as 'runner'. Change the line "
                "echo \"starting run\" so it says \"starting run v2\" instead."
            ),
            files={"runner": ("runner.sh", small_sh)},
            check=lambda ws: 'echo "starting run v2"' in (ws / "runner.sh").read_text(),
            fake_calls=[
                _read("runner"),
                _edit("runner", ['echo "starting run"'], ['echo "starting run v2"']),
            ],
        )
    )

    # 2. replacement in the middle of a ~300-line file
    tasks.append(
        Task(
            name="middle_of_large_file",
            prompt=(
                "The manifest is registered as 'manifest'. Somewhere in the "
                "middle there is a line setting threads = 4. Change it to "
                "threads = 16, keeping the trailing comment exactly as it is."
            ),
            files={"manifest": ("manifest.cfg", _big_file())},
            check=lambda ws: "threads = 16        # section C: tunables"
            in (ws / "manifest.cfg").read_text(),
            fake_calls=[
                _read("manifest"),
                _edit(
                    "manifest",
                    ["threads = 4        # section C: tunables"],
                    ["threads = 16        # section C: tunables"],
                ),
            ],
        )
    )

    # 3. edit in a CRLF file
    crlf = b"[settings]\r\nretries = 2\r\ntimeout = 30\r\nverbose = false\r\n"
    tasks.append(
        Task(
            name="crlf_file",
            prompt=(
                "The Windows-style config is registered as 'winconf'. Change "
                "retries = 2 to retries = 5. The file uses CRLF line endings."
            ),
            files={"winconf": ("settings.ini", crlf)},
            check=lambda ws: b"retries = 5" in (ws / "settings.ini").read_bytes(),
            fake_calls=[
                _read("winconf"),
                # read_text() applies universal newlines, so handlers see LF
                _edit("winconf", ["retries = 2"], ["retries = 5"]),
            ],
        )
    )

    # 4. edit near unicode (smart quotes / en-dash in comments)
    uni = (
        "# “Results” for samples 3–7 (en-dash), don’t touch the header\n"
        "sample_range = 3-7\n"
        "normalize = false\n"
    )
    tasks.append(
        Task(
            name="unicode_context",
            prompt=(
                "The file is registered as 'unifile'. Change normalize = false "
                "to normalize = true. Leave the comment line untouched."
            ),
            files={"unifile": ("analysis.cfg", uni)},
            check=lambda ws: "normalize = true" in (ws / "analysis.cfg").read_text()
            and "don’t touch" in (ws / "analysis.cfg").read_text(),
            fake_calls=[
                _read("unifile"),
                _edit("unifile", ["normalize = false"], ["normalize = true"]),
            ],
        )
    )

    # 5. append-to-end idiom
    hosts = "node01\nnode02\nnode03\n"
    tasks.append(
        Task(
            name="append_to_end",
            prompt=(
                "The host list is registered as 'hosts'. Add a new line "
                "node04 at the end of the file, after node03."
            ),
            files={"hosts": ("hosts.txt", hosts)},
            check=lambda ws: (ws / "hosts.txt").read_text().rstrip("\n").split("\n")
            == ["node01", "node02", "node03", "node04"],
            fake_calls=[
                _read("hosts"),
                _edit("hosts", ["node03"], ["node03", "node04"]),
            ],
        )
    )

    # 6. multi-line block replacement
    block_sh = (
        "#!/bin/bash\n"
        "module load samtools\n"
        "samtools sort in.bam -o sorted.bam\n"
        "samtools index sorted.bam\n"
        "samtools flagstat sorted.bam > stats.txt\n"
        'echo "pipeline done"\n'
    )
    tasks.append(
        Task(
            name="multiline_block",
            prompt=(
                "The script is registered as 'pipeline'. Replace the three "
                "samtools lines (sort, index, flagstat) with a single line: "
                "samtools sort -@ 8 in.bam -o sorted.bam && samtools index sorted.bam"
            ),
            files={"pipeline": ("pipeline.sh", block_sh)},
            check=lambda ws: (
                "samtools sort -@ 8 in.bam -o sorted.bam && samtools index sorted.bam"
                in (ws / "pipeline.sh").read_text()
                and "flagstat" not in (ws / "pipeline.sh").read_text()
            ),
            fake_calls=[
                _read("pipeline"),
                _edit(
                    "pipeline",
                    [
                        "samtools sort in.bam -o sorted.bam",
                        "samtools index sorted.bam",
                        "samtools flagstat sorted.bam > stats.txt",
                    ],
                    [
                        "samtools sort -@ 8 in.bam -o sorted.bam && samtools index sorted.bam"
                    ],
                ),
            ],
        )
    )

    # 7. duplicated similar blocks — needs disambiguation
    dup = (
        "[stage: align]\n"
        "threads = 8\n"
        "mem_gb = 16\n"
        "\n"
        "[stage: call]\n"
        "threads = 8\n"
        "mem_gb = 16\n"
    )
    def _check_dup(ws: Path) -> bool:
        text = (ws / "stages.cfg").read_text()
        call_part = text.split("[stage: call]")[-1]
        align_part = text.split("[stage: call]")[0]
        return "mem_gb = 64" in call_part and "mem_gb = 16" in align_part

    tasks.append(
        Task(
            name="duplicate_blocks",
            prompt=(
                "The stage config is registered as 'stages'. Both stages "
                "currently have mem_gb = 16. Change mem_gb to 64 for the "
                "'call' stage ONLY; leave the 'align' stage at 16."
            ),
            files={"stages": ("stages.cfg", dup)},
            check=_check_dup,
            fake_calls=[
                _read("stages"),
                _edit(
                    "stages",
                    ["[stage: call]", "threads = 8", "mem_gb = 16"],
                    ["[stage: call]", "threads = 8", "mem_gb = 64"],
                ),
            ],
        )
    )

    # 8. indentation-sensitive Python edit (goes through py_compile gate)
    py = (
        "def process(items):\n"
        "    results = []\n"
        "    for item in items:\n"
        "        if item.valid:\n"
        "            results.append(item.value)\n"
        "    return results\n"
    )
    tasks.append(
        Task(
            name="python_indent",
            prompt=(
                "The Python file is registered as 'pyfile'. Inside the loop, "
                "change results.append(item.value) to "
                "results.append(item.value * 2), keeping the code valid."
            ),
            files={"pyfile": ("process.py", py)},
            check=lambda ws: "            results.append(item.value * 2)"
            in (ws / "process.py").read_text(),
            fake_calls=[
                _read("pyfile"),
                _edit(
                    "pyfile",
                    ["            results.append(item.value)"],
                    ["            results.append(item.value * 2)"],
                ),
            ],
        )
    )

    # 9. delete-lines edit
    dbg = (
        "input = load()\n"
        "print('DEBUG: loaded', input)\n"
        "print('DEBUG: type', type(input))\n"
        "result = transform(input)\n"
        "save(result)\n"
    )
    tasks.append(
        Task(
            name="delete_lines",
            prompt=(
                "The file is registered as 'debugfile'. Delete the two DEBUG "
                "print lines; keep everything else exactly as it is."
            ),
            files={"debugfile": ("job.txt", dbg)},
            check=lambda ws: (ws / "job.txt").read_text()
            == "input = load()\nresult = transform(input)\nsave(result)\n",
            fake_calls=[
                _read("debugfile"),
                _edit(
                    "debugfile",
                    [
                        "print('DEBUG: loaded', input)",
                        "print('DEBUG: type', type(input))",
                    ],
                    [],
                ),
            ],
        )
    )

    # 10. create a new small file with create_file
    tasks.append(
        Task(
            name="create_new_file",
            prompt=(
                "The workspace directory is registered as 'workspace'. Create "
                "a new file named NOTES.md in it with exactly two lines: a "
                "heading '# Run notes' and a line 'Started 2026-08-06.'"
            ),
            files={},
            check=lambda ws: (ws / "NOTES.md").is_file()
            and "# Run notes" in (ws / "NOTES.md").read_text()
            and "Started 2026-08-06." in (ws / "NOTES.md").read_text(),
            fake_calls=[
                {
                    "action": "tool_call",
                    "tool": "create_file",
                    "arguments": {
                        "dir_key": "workspace",
                        "name": "NOTES.md",
                        "content_lines": ["# Run notes", "Started 2026-08-06."],
                    },
                }
            ],
        )
    )

    # 11. edit a config value
    yaml = (
        "cluster:\n"
        "  partition: gpu\n"
        "  gres: gpu:1\n"
        "run:\n"
        "  epochs: 10\n"
        "  batch_size: 32\n"
    )
    tasks.append(
        Task(
            name="config_value",
            prompt=(
                "The YAML config is registered as 'runconf'. Change epochs "
                "from 10 to 50. Keep the YAML indentation intact."
            ),
            files={"runconf": ("run.yaml", yaml)},
            check=lambda ws: "  epochs: 50" in (ws / "run.yaml").read_text(),
            fake_calls=[
                _read("runconf"),
                _edit("runconf", ["  epochs: 10"], ["  epochs: 50"]),
            ],
        )
    )

    # 12. trailing-space discrepancy planted in the file: the natural copy of
    # old_lines has no trailing space, so the first attempt near-misses.
    trap = "alpha = 1\nbeta = 2 \ngamma = 3\n"  # note trailing space on beta
    tasks.append(
        Task(
            name="trailing_space_trap",
            prompt=(
                "The file is registered as 'trapfile'. Change beta = 2 to "
                "beta = 7."
            ),
            files={"trapfile": ("params.txt", trap)},
            check=lambda ws: "beta = 7" in (ws / "params.txt").read_text()
            and "beta = 2" not in (ws / "params.txt").read_text(),
            fake_calls=[
                _read("trapfile"),
                _edit("trapfile", ["beta = 2 "], ["beta = 7"]),
            ],
        )
    )

    return tasks


def _deep_file() -> str:
    # 700 lines; the only interesting line sits at ~400, past any window the
    # old head/tail read_file could show for a file this long. Its exact value
    # and inline comment are NOT in the prompt, so the model must actually see
    # the line to copy it.
    lines = ["# pipeline stage table — generated"]
    for i in range(2, 701):
        if i == 400:
            lines.append("chunk_size = 4096      # stage K: io tuning, keep power of two")
        else:
            lines.append(f"stage_{i:03d} = pass")
    return "\n".join(lines) + "\n"


def build_hard_tasks() -> list[Task]:
    """Tier 'hard': realistic shapes the old tooling structurally mishandled.

    Same tasks run against both baseline and treatment — the point is that
    these are ordinary cluster files (long generated configs, Windows-edited
    ini, prose with typographic quotes), not synthetic gotchas.
    """
    tasks: list[Task] = []

    # H1. Edit deep inside a 700-line file. The old read_file showed
    # head/tail with the middle omitted and had no offset, so the region
    # around line 400 was unreachable; paging makes it reachable.
    tasks.append(
        Task(
            name="deep_edit_700",
            prompt=(
                "The generated pipeline config is registered as 'pipeline'. "
                "Somewhere in it a line sets chunk_size. Double the value on "
                "that line, and keep its inline comment exactly as it is. The "
                "file is long — page through it until you have actually seen "
                "the line before editing."
            ),
            files={"pipeline": ("pipeline.cfg", _deep_file())},
            check=lambda ws: "chunk_size = 8192      # stage K: io tuning, keep power of two"
            in (ws / "pipeline.cfg").read_text(),
            fake_calls=[
                _read("pipeline"),
                _edit(
                    "pipeline",
                    ["chunk_size = 4096      # stage K: io tuning, keep power of two"],
                    ["chunk_size = 8192      # stage K: io tuning, keep power of two"],
                ),
            ],
        )
    )

    # H2. CRLF file where success requires the line endings to SURVIVE the
    # edit byte-for-byte (the old write path rewrote the file with LF).
    crlf_body = (
        b"[cluster]\r\n"
        b"queue = short\r\n"
        b"max_jobs = 8\r\n"
        b"notify = none\r\n"
    )
    tasks.append(
        Task(
            name="crlf_preserved",
            prompt=(
                "The config is registered as 'clusterconf'. Change max_jobs = 8 "
                "to max_jobs = 32. The file comes from a Windows tool; it must "
                "keep its CRLF line endings."
            ),
            files={"clusterconf": ("cluster.ini", crlf_body)},
            check=lambda ws: (
                b"max_jobs = 32\r\n" in (ws / "cluster.ini").read_bytes()
                and b"queue = short\r\n" in (ws / "cluster.ini").read_bytes()
            ),
            fake_calls=[
                _read("clusterconf"),
                _edit("clusterconf", ["max_jobs = 8"], ["max_jobs = 32"]),
            ],
        )
    )

    # H4. Write a LONG specs document — the real-session failure this tier
    # exists for (see specs_test transcript, 2026-08-06): a ~12-decision plan
    # is 250+ lines, past what one create_file call can carry under the
    # 4096-token decision cap, so a one-shot write truncates and dies. The
    # winning shape is skeleton-then-fill. Modeled on the cut_locus session.
    specs_sections = [
        "Purpose and scope",
        "Command-line interface",
        "Region parsing",
        "Read selection",
        "Trimming logic",
        "Sequence collection",
        "Output format",
        "Error handling",
        "Build system",
        "Test strategy",
    ]
    # The decisions the document must carry — in the real session these were
    # agreed one by one in the grilling; the write-down is transcription plus
    # elaboration, not invention. Each entry: (decision text for the prompt,
    # phrase list of which at least one must appear in the file).
    specs_decisions = [
        ("flags -a/--alignments, -f/--reference, -r/--region", ["--region"]),
        (
            "region like chr3:24,489,125-24,501,081 — commas are stripped",
            ["comma"],
        ),
        (
            "FASTA to stdout, ONE entry per unique QNAME, sorted "
            "lexicographically by read name",
            ["lexicograph"],
        ),
        (
            "trim extents = union over ALL alignments of a QNAME that "
            "overlap the region (supplementary/secondary only widen the "
            "window), walked via the CIGAR strings",
            ["CIGAR"],
        ),
        (
            "insertions on the region border are included; soft-clipped "
            "bases outside the kept span are excluded",
            ["soft-clip", "soft clip"],
        ),
        (
            "sequence comes from the primary alignment; if no primary is in "
            "the region, follow the SA tag to fetch it",
            ["SA tag"],
        ),
        (
            "reverse-complement reverse-strand reads back to original "
            "sequencing orientation",
            ["reverse-complement", "reverse complement"],
        ),
        (
            "no reads in region: empty stdout, a warning on stderr, exit 0",
            ["stderr"],
        ),
        (
            "build with CMake + GCC against htslib from the samtools conda "
            "env",
            ["CMake"],
        ),
        (
            "Catch2 unit tests for the CIGAR walker and region parser, bash "
            "integration tests with minimap2-aligned synthetic reads",
            ["Catch2"],
        ),
    ]
    specs_prompt = (
        "Write the implementation plan specs.md for the cut_locus tool into "
        "the workspace directory (registered as 'workspace'). cut_locus is a "
        "C++ tool that extracts trimmed read sequences from CRAM alignments "
        "for one genomic region. The plan was already agreed as follows — "
        "every one of these decisions must be in the document, elaborated "
        "with enough implementation detail to build from (data flow, edge "
        "cases, examples): "
        + "; ".join(text for text, _ in specs_decisions)
        + ". Structure it as exactly these ten '## ' sections in this "
        "order: "
        + ", ".join(specs_sections)
        + ". A plan this size is far too long for one call — write it "
        "incrementally, and finish only when every section is filled."
    )

    def _specs_ok(ws: Path) -> bool:
        path = ws / "specs.md"
        if not path.is_file():
            return False
        text = path.read_text()
        if len(text.split("\n")) < 150 or "TBD" in text:
            return False
        lowered = text.lower()
        # every agreed decision landed (any of its marker phrases), and every
        # section exists — case-insensitive; the tooling is under test, not
        # heading capitalization or prose stamina
        decisions_ok = all(
            any(p.lower() in lowered for p in phrases)
            for _, phrases in specs_decisions
        )
        sections_ok = all(
            f"## {title.lower()}" in lowered for title in specs_sections
        )
        return decisions_ok and sections_ok

    def _specs_fake_calls() -> list[dict]:
        skeleton = ["# cut_locus — implementation plan"]
        for title in specs_sections:
            skeleton += [f"## {title}", f"TBD: {title}"]
        calls = [
            {
                "action": "tool_call",
                "tool": "create_file",
                "arguments": {
                    "dir_key": "workspace",
                    "name": "specs.md",
                    "content_lines": skeleton,
                },
            }
        ]
        for title, (decision, _) in zip(specs_sections, specs_decisions):
            body = [f"- {decision}"] + [
                f"- {title} detail line {i + 1}" for i in range(15)
            ]
            calls.append(
                {
                    "action": "tool_call",
                    "tool": "edit_file",
                    "arguments": {
                        "registry_key": "workspace",
                        "subpath": "specs.md",
                        "old_lines": [f"TBD: {title}"],
                        "new_lines": body,
                    },
                }
            )
        return calls

    tasks.append(
        Task(
            name="long_specs_write",
            prompt=specs_prompt,
            files={},
            check=_specs_ok,
            fake_calls=_specs_fake_calls(),
            # production parity: graph.MAX_TOOL_ROUNDS is 30 — a 16 cap sat
            # exactly where skeleton + ten fills + a retry lands, so runs
            # died on the harness, not on the tooling
            max_decisions=30,
        )
    )

    # H3. The line to replace contains typographic quotes and an en-dash.
    # A model that transcribes them as ASCII used to get "NOT edited" and
    # loop; fuzzy matching absorbs the transcription.
    smart = (
        "# report strings\n"
        "title = “Weekly QC – node health”\n"
        "footer = plain\n"
    )
    tasks.append(
        Task(
            name="smart_quote_line",
            prompt=(
                "The file is registered as 'report'. On the title line, "
                "change the word Weekly to Daily, leaving the rest of the "
                "line as it is."
            ),
            files={"report": ("report.cfg", smart)},
            check=lambda ws: "Daily QC" in (ws / "report.cfg").read_text()
            and "Weekly" not in (ws / "report.cfg").read_text(),
            fake_calls=[
                _read("report"),
                _edit(
                    "report",
                    ["title = “Weekly QC – node health”"],
                    ["title = “Daily QC – node health”"],
                ),
            ],
        )
    )

    # H5. The target directory is registered but has never been created — what
    # a path the user names for output that does not exist yet looks like.
    # create_file makes missing parents anyway, so the only question is whether
    # the tool lets the call through or bounces it into the retry loop.
    tasks.append(
        Task(
            name="write_into_unmade_dir",
            prompt=(
                "The results directory for this run is registered as "
                "'results'. Write a README.md in it with exactly these two "
                "lines:\n"
                "# Results\n"
                "Populated by the nightly QC run."
            ),
            files={},
            missing={"results": "results"},
            check=lambda ws: (ws / "results" / "README.md").is_file()
            and "nightly QC" in (ws / "results" / "README.md").read_text(),
            fake_calls=[
                {
                    "action": "tool_call",
                    "tool": "create_file",
                    "arguments": {
                        "dir_key": "results",
                        "name": "README.md",
                        "content_lines": [
                            "# Results",
                            "Populated by the nightly QC run.",
                        ],
                    },
                }
            ],
        )
    )

    # H6. A file key that resolves to nothing. The model has to notice the file
    # is not there and write it, rather than retry the read or invent content.
    tasks.append(
        Task(
            name="read_key_pointing_at_nothing",
            prompt=(
                "The run notes are registered as 'notes', and the workspace "
                "directory is registered as 'workspace'. Check the notes, and "
                "if they have not been written yet create notes.txt in the "
                "workspace containing exactly this line:\n"
                "no output yet"
            ),
            files={},
            missing={"notes": "notes.txt"},
            check=lambda ws: (ws / "notes.txt").is_file()
            and (ws / "notes.txt").read_text().strip() == "no output yet",
            fake_calls=[
                _read("notes"),
                {
                    "action": "tool_call",
                    "tool": "create_file",
                    "arguments": {
                        "dir_key": "workspace",
                        "name": "notes.txt",
                        "content_lines": ["no output yet"],
                    },
                },
            ],
        )
    )

    return tasks


def build_shift_tasks() -> list[Task]:
    """Tier 'shift': what HPCA's key-only tool interface costs a model that was
    post-trained on file paths.

    Every task here is winnable on the old code — the friction shows up as
    extra tool calls and `[tool error]` results (the `tool_errors` metric), not
    as a rigged 0% baseline. Each `fake_calls` script therefore drives the
    route that works on BOTH sides: register a fresh key, then act.
    """
    tasks: list[Task] = []

    readme_lines = ["# Results", "Populated by the nightly QC run."]

    # S1. The key is there but points at a typo'd directory that does not
    # exist; the user's message has the real path. Old code: register_path on
    # an existing key raises RegistryError ("pick a different key"), which
    # comes back as [tool error], and the model has to invent a second key for
    # the same directory. New code: the repoint just works.
    tasks.append(
        Task(
            name="repoint_stale_key",
            prompt=(
                "The results directory for this run is registered as 'outdir', "
                "but that key points at a misspelling of the name and there is "
                "nothing there. The real directory is {workspace}/results. "
                "Write a README.md in the real results directory with exactly "
                "these two lines:\n"
                "# Results\n"
                "Populated by the nightly QC run."
            ),
            files={},
            missing={"outdir": "reslts"},
            templated=True,
            system_note=_SHIFT_SYSTEM_NOTE,
            check=lambda ws: (ws / "results" / "README.md").is_file()
            and "nightly QC" in (ws / "results" / "README.md").read_text()
            and "# Results" in (ws / "results" / "README.md").read_text(),
            fake_calls=[
                _register("results_dir", "{workspace}/results"),
                _create("results_dir", "README.md", readme_lines),
            ],
        )
    )

    # S2. A file nobody registered, named by its literal absolute path — the
    # shape every agent-trained model expects. Old code: register_path first,
    # then edit_file by key, two calls minimum. New code: edit_file takes the
    # path where a key is expected, one call.
    sampler = (
        "[sampler]\n"
        "seed = 17\n"
        "max_reads = 1000\n"
        "keep_duplicates = false\n"
    )
    tasks.append(
        Task(
            name="edit_by_literal_path",
            prompt=(
                "The sampler config is at {workspace}/conf/sampler.cfg. "
                "Change max_reads = 1000 to max_reads = 5000, leaving the rest "
                "of the file exactly as it is."
            ),
            files={},
            unregistered={"conf/sampler.cfg": sampler},
            templated=True,
            system_note=_SHIFT_SYSTEM_NOTE,
            check=lambda ws: "max_reads = 5000"
            in (ws / "conf" / "sampler.cfg").read_text()
            and "seed = 17" in (ws / "conf" / "sampler.cfg").read_text(),
            fake_calls=[
                _register("sampler", "{workspace}/conf/sampler.cfg"),
                _edit("sampler", ["max_reads = 1000"], ["max_reads = 5000"]),
            ],
        )
    )

    # S3. Same shift for create_file: the target directory exists (an
    # unregistered sibling file puts it on disk) and is named literally, so the
    # only question is whether the write needs a key ceremony first.
    tasks.append(
        Task(
            name="create_by_literal_path",
            prompt=(
                "The analysis outputs live in {workspace}/analysis. Create a "
                "file named SUMMARY.md in that directory with exactly these "
                "two lines:\n"
                "# Analysis summary\n"
                "All samples passed QC."
            ),
            files={},
            unregistered={"analysis/inputs.txt": "sample_a\nsample_b\n"},
            templated=True,
            system_note=_SHIFT_SYSTEM_NOTE,
            check=lambda ws: (ws / "analysis" / "SUMMARY.md").is_file()
            and "# Analysis summary"
            in (ws / "analysis" / "SUMMARY.md").read_text()
            and "All samples passed QC."
            in (ws / "analysis" / "SUMMARY.md").read_text(),
            fake_calls=[
                _register("analysis_dir", "{workspace}/analysis"),
                _create(
                    "analysis_dir",
                    "SUMMARY.md",
                    ["# Analysis summary", "All samples passed QC."],
                ),
            ],
        )
    )

    return tasks


# --------------------------------------------------------------- fake model


class FakeLLM:
    """Scripted model for --dry-run: emits each queued decision once, then
    responds DONE. Satisfies the two methods decide() uses."""

    def __init__(self, calls: list[dict]):
        self._queue = list(calls)

    def bind_workspace(self, workspace: Path) -> None:
        """Substitute {workspace} inside the scripted arguments.

        Plain replacement, not .format(): scripted content lines carry braces
        of their own and must survive untouched.
        """

        def _sub(value):
            if isinstance(value, str):
                return value.replace("{workspace}", str(workspace))
            if isinstance(value, list):
                return [_sub(v) for v in value]
            if isinstance(value, dict):
                return {k: _sub(v) for k, v in value.items()}
            return value

        self._queue = [_sub(call) for call in self._queue]

    async def supports_constrained_decoding(self) -> bool:
        return False

    async def chat(self, messages, json_schema=None, max_tokens=None):
        if self._queue:
            payload = self._queue.pop(0)
        else:
            payload = {"action": "respond", "response": "DONE"}

        class _Resp:
            content = json.dumps(payload)
            reasoning = ""
            usage = {"completion_tokens": 0}

        return _Resp()


# ---------------------------------------------------------------- run loop


def _is_failed_edit(tool_name: str, result: str) -> bool:
    if tool_name not in ("edit_file", "create_file"):
        return False
    return result.startswith("NOT ") or "[tool error]" in result


def _render_prompt(task: Task, workspace: Path) -> str:
    """The prompt as the model sees it, with {workspace} filled in.

    Opt-in via Task.templated, and belt-and-braces even then: a stray brace in
    a future prompt must not crash the run and silently zero its success rate.
    """
    if not getattr(task, "templated", False):
        return task.prompt
    try:
        return task.prompt.format(workspace=str(workspace))
    except (KeyError, IndexError, ValueError):
        return task.prompt.replace("{workspace}", str(workspace))


async def run_task(task: Task, llm, tools: ToolRegistry, keep: bool = False) -> dict:
    workdir = Path(tempfile.mkdtemp(prefix=f"hpca_eval_{task.name}_"))
    workspace = workdir / "workspace"
    workspace.mkdir()
    metrics = {
        "task": task.name,
        "success": False,
        "tool_calls": 0,
        "failed_edits": 0,
        # results that came back as [tool error]: the tool refused or raised.
        # Broader than failed_edits (any tool, not just edit/create) — this is
        # the friction a wrong-shaped interface generates, and it used to be
        # invisible in the numbers.
        "tool_errors": 0,
        # Malformed calls the middleware silently absorbed (_strip_key_echo).
        # Not a failure — a repair is a failure that did not happen — but the
        # only place a change to that layer is visible at all.
        "repairs": 0,
        "decisions": 0,
        "completion_tokens": 0,
        "wall_s": 0.0,
        "error": "",
    }
    started = time.monotonic()
    try:
        ctx = _make_context(workdir)
        ctx.registry.register("workspace", workspace)
        for key, (fname, content) in task.files.items():
            path = workspace / fname
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content)
            ctx.registry.register(key, path)
        for key, fname in getattr(task, "missing", {}).items():
            ctx.registry.register(key, workspace / fname)
        for name, content in getattr(task, "unregistered", {}).items():
            path = workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content)
        bind = getattr(llm, "bind_workspace", None)
        if bind is not None:
            bind(workspace)

        messages = [
            {
                "role": "system",
                "content": _system_prompt(
                    getattr(task, "system_note", ""),
                    native=bool(getattr(llm, "uses_native_tools", _not_native)()),
                ),
            },
            {"role": "user", "content": _render_prompt(task, workspace)},
        ]
        for _ in range(task.max_decisions or MAX_DECISIONS):
            try:
                decision = await decide(llm, messages, tools)
            except DecisionError as exc:
                metrics["error"] = f"DecisionError: {exc}"
                break
            metrics["decisions"] += 1
            usage = getattr(decision, "usage", None) or {}
            metrics["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            if isinstance(decision, DirectResponse):
                break
            assert isinstance(decision, ToolCall)
            metrics["tool_calls"] += 1
            try:
                # handlers called directly: no HITL gate in this harness
                result = await decision.tool.handler(decision.arguments, ctx)
                # mirror graph.py: a repaired (e.g. salvaged) call's result
                # must tell the model what actually happened
                for repair in getattr(decision, "repairs", None) or []:
                    result = f"{result}\n[repaired] {repair}"
                    # Counted, not just shown: a repair is a malformed call the
                    # middleware absorbed, so it is the only visible measure of
                    # a change to _strip_key_echo. Without this the swallowed
                    # `new_lines: [` class moves nothing in the summary.
                    metrics["repairs"] += 1
            except Exception as exc:
                result = f"[tool error] {decision.tool.name}: {type(exc).__name__}: {exc}"
            if _is_failed_edit(decision.tool.name, result):
                metrics["failed_edits"] += 1
            if result.startswith("[tool error]"):
                metrics["tool_errors"] += 1
            messages = messages + _exchange(
                decision.tool.name,
                _arguments_dict(decision.arguments),
                result,
                # empty unless the backend answered on its native tool channel,
                # which is what makes the history use the tool role there
                getattr(decision, "call_id", "") or "",
            )
            if task.check(workspace):
                # model already succeeded; one more decision would only spend
                # tokens, so stop here.
                break
        metrics["success"] = bool(task.check(workspace))
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        metrics["wall_s"] = round(time.monotonic() - started, 2)
        conn = getattr(locals().get("ctx"), "_eval_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if keep:
            # diagnosis mode: leave the workspace, dump the conversation
            try:
                (workdir / "transcript.json").write_text(
                    json.dumps(locals().get("messages") or [], indent=1)
                )
            except Exception:
                pass
            metrics["workdir"] = str(workdir)
            print(f"      kept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    return metrics


# ------------------------------------------------------------ live backend


def discover_backend() -> tuple[str, str]:
    """Return (url, model) or exit(2) with a clear message."""
    if httpx is None:
        print("httpx is not installed; cannot reach the backend.", file=sys.stderr)
        sys.exit(2)
    headers = {"Authorization": f"Bearer {LIVE_KEY}"} if LIVE_KEY else {}
    try:
        # Generous on purpose. A cluster backend reached through an SSH tunnel
        # answers /models in milliseconds when warm and in 3-8 seconds when the
        # tunnel has been idle or the server is loaded — measured on the 27B
        # box, 2026-08-13. A 5s probe turned that into "backend unreachable"
        # and killed whole eval stages at the door while generations (180s
        # timeout) would have been perfectly fine.
        resp = httpx.get(f"{LIVE_URL}/models", timeout=PROBE_TIMEOUT_S, headers=headers)
    except Exception as exc:
        print(
            f"LLM backend at {LIVE_URL} is unreachable ({exc}). "
            "Set HPCA_TEST_LLM_URL, or use --dry-run.",
            file=sys.stderr,
        )
        sys.exit(2)
    if resp.status_code == 401:
        print(
            f"LLM backend at {LIVE_URL} answered 401 Unauthorized. "
            "Export HPCA_TEST_LLM_KEY with a valid key.",
            file=sys.stderr,
        )
        sys.exit(2)
    if resp.status_code != 200:
        print(
            f"LLM backend at {LIVE_URL} answered {resp.status_code}.",
            file=sys.stderr,
        )
        sys.exit(2)
    model = os.environ.get("HPCA_TEST_LLM_MODEL")
    if not model:
        data = resp.json().get("data", [])
        if not data:
            print(f"Backend at {LIVE_URL} serves no models.", file=sys.stderr)
            sys.exit(2)
        model = data[0]["id"]
    return LIVE_URL, model


def make_live_llm(tool_protocol: str = "envelope"):
    from hpca.config import LLMSettings
    from hpca.llm import LLMClient

    url, model = discover_backend()
    print(f"Backend: {url}  model: {model}  protocol: {tool_protocol}")
    # _filtered_kwargs drops what this checkout's LLMSettings does not have, so
    # a baseline without tool_protocol still runs (as envelope, which it is).
    return LLMClient(
        LLMSettings(
            **_filtered_kwargs(
                LLMSettings,
                base_url=url,
                model=model,
                api_key=LIVE_KEY,
                request_timeout_s=180,
                enable_thinking=False,
                tool_protocol=tool_protocol,
            )
        )
    )


# --------------------------------------------------------------------- CLI


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results.json", help="JSON output path")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=None, help="run only the first N tasks")
    parser.add_argument("--label", default="", help="label recorded in the JSON (e.g. baseline)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every task once against a scripted fake model (no backend)",
    )
    parser.add_argument(
        "--only", default="", help="run only tasks whose name contains this"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep each run's workdir and dump transcript.json into it",
    )
    parser.add_argument(
        "--tier",
        choices=["core", "hard", "shift", "all"],
        default="core",
        help="core: the original 12 tasks; hard: shapes the old tooling "
        "structurally mishandled; shift: what the key-only interface costs a "
        "model post-trained on paths; all: every tier",
    )
    parser.add_argument(
        "--tool-protocol",
        choices=["envelope", "native"],
        default="envelope",
        help="how a tool call travels: the hand-rolled JSON envelope under a "
        "grammar, or the backend's own tool-calling channel (needs the server "
        "started with --enable-auto-tool-choice --tool-call-parser)",
    )
    args = parser.parse_args()

    tasks = {
        "core": build_tasks(),
        "hard": build_hard_tasks(),
        "shift": build_shift_tasks(),
        "all": build_tasks() + build_hard_tasks() + build_shift_tasks(),
    }[args.tier]
    if args.tasks:
        tasks = tasks[: args.tasks]
    if args.only:
        tasks = [t for t in tasks if args.only in t.name]
        if not tasks:
            print(f"no task matches --only {args.only!r}")
            return 2
    tools = _eval_tool_registry()

    live = None
    if not args.dry_run:
        live = make_live_llm(args.tool_protocol)
    repeats = 1 if args.dry_run else args.repeats

    runs: list[dict] = []
    try:
        for task in tasks:
            for rep in range(repeats):
                llm = FakeLLM(task.fake_calls) if args.dry_run else live
                metrics = await run_task(task, llm, tools, keep=args.keep)
                metrics["repeat"] = rep
                runs.append(metrics)
                status = "ok " if metrics["success"] else "FAIL"
                print(
                    f"[{status}] {task.name:<24} rep {rep}: "
                    f"{metrics['tool_calls']} calls, "
                    f"{metrics['failed_edits']} failed edits, "
                    f"{metrics['tool_errors']} tool errors, "
                    f"{metrics['decisions']} decisions, "
                    f"{metrics['wall_s']}s"
                    + (f"  error: {metrics['error']}" if metrics["error"] else "")
                )
    finally:
        if live is not None:
            try:
                await live.close()
            except Exception:
                pass

    n = len(runs)
    successes = sum(1 for r in runs if r["success"])
    summary = {
        "label": args.label or ("dry-run" if args.dry_run else "live"),
        "tool_protocol": args.tool_protocol,
        "n_runs": n,
        "success_rate": round(successes / n, 3) if n else 0.0,
        "mean_failed_edits": round(sum(r["failed_edits"] for r in runs) / n, 3) if n else 0.0,
        "mean_tool_errors": round(sum(r["tool_errors"] for r in runs) / n, 3) if n else 0.0,
        "mean_repairs": round(sum(r.get("repairs", 0) for r in runs) / n, 3) if n else 0.0,
        "mean_tool_calls": round(sum(r["tool_calls"] for r in runs) / n, 3) if n else 0.0,
        "mean_decisions": round(sum(r["decisions"] for r in runs) / n, 3) if n else 0.0,
        "total_completion_tokens": sum(r["completion_tokens"] for r in runs),
        "total_wall_s": round(sum(r["wall_s"] for r in runs), 1),
    }
    out = {"summary": summary, "runs": runs}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== summary ===")
    for key, value in summary.items():
        print(f"{key:>24}: {value}")
    print(f"\nWrote {args.out}")
    return 0 if successes == n or not args.dry_run else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
