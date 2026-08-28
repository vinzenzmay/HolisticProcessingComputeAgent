"""Watch tools (§5.1): watch_log, watch_job, list_watches, unwatch.

The user's own description of the problem these solve: an sbatch job submits a
pipeline, the pipeline spawns sniffles, sniffles writes a log — and finding out
whether any of that is still alive means squeue, then ssh to the node, then
tail. The agent can already *find* the job and the log while working
on the task; these tools let it pin what it found to the right column, where a
glance answers the question from then on.

Two separate register tools rather than one with a ``kind`` switch: a small
model picks between "watch_log" and "watch_job" from the names alone, and the
parameters genuinely differ (a path versus a Slurm id).

Both register tools take an *array*, and so does ``unwatch``. What the agent
finds, it finds in bulk — a squeue listing is a dozen ids, a pipeline directory
is a dozen logs — and one call per target meant a dozen decisions, a dozen
round trips, and a dozen squeue processes for one question. An array is one
decision, and for jobs one ``squeue`` plus at most one ``sacct`` for the whole
batch.

An array is resolved element by element and applied in part: what resolved
cleanly happens, and what did not is reported by name. A name that matches two
watches is still refused rather than guessed at — the array is exactly how the
model says which two it meant.

Nothing here polls. Registering only records the target; the TUI's timers
refresh every box (see :mod:`hpca.watches`).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from hpca.agent import hints
from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry
from hpca.paths import PathError, resolve_path
from hpca.watches import (
    KIND_JOB,
    KIND_LOG,
    LOG_GONE,
    Watch,
    WatchStore,
    job_fields,
    log_fields,
    watch_lines,
)


def _require_watches(ctx: ToolContext) -> WatchStore:
    if ctx.watches is None:
        raise RuntimeError("Watches are not configured in this session")
    return ctx.watches


def _resolve_path(target: str, ctx: ToolContext) -> Path:
    """The log file the model named, anchored like every other path argument."""
    return resolve_path(target, ctx.workdir)


def _render_list(watches: list[Watch]) -> str:
    if not watches:
        return "Nothing is being watched."
    lines = [f"{len(watches)} watch(es) in the panel:"]
    for watch in watches:
        lines.append(f"- [{watch.kind}] {watch.title} ({watch.target}): "
                     + " / ".join(watch_lines(watch)))
    return "\n".join(lines)


def _as_string_list(value: object) -> object:
    """A bare scalar where an array belongs, wrapped; a JSON number, stringified.

    Two repairs, both for the same class of thing the backend actually does.
    Handing a list-typed argument one string instead of a list of them is the
    dominant shape error on the native channel (measured for ``edit_file`` at
    6 of 6 generations — see the ``middleware`` module docstring). edit_file
    cannot repair its version of it, because a script's line breaks are gone by
    the time it arrives; one path or one job id is unambiguously one element,
    so it is repaired here rather than bounced back as a validation error.

    And a Slurm id is written as a number as readily as as a string — the user
    reads it as an integer, so the model emits one — while pydantic v2 does not
    coerce int to str. Refusing that would be a retry spent on punctuation.
    """
    items = value if isinstance(value, list) else [value]
    if not isinstance(items, list):  # pragma: no cover - defensive
        return value
    return [
        str(item) if isinstance(item, int) and not isinstance(item, bool) else item
        for item in items
    ]


def _unique(values: list[str]) -> list[str]:
    """Order-preserving dedupe. The same target twice is one box, not two."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _blank_note(refused: int, hint: str) -> str:
    """What became of array elements that named nothing at all.

    Reported rather than dropped in silence: a blank element is the model
    losing an argument mid-array, and a count it can see is what tells it the
    call was short one target.
    """
    if not refused:
        return ""
    entries = "entry" if refused == 1 else "entries"
    return f"\n{refused} blank {entries} ignored. {hint}".rstrip()


# ------------------------------------------------------------------ watch_log


class WatchLogParams(BaseModel):
    # Arrays come last in every params model (``middleware`` module docstring),
    # so the label goes first even though the paths are the point.
    label: str = Field(
        default="",
        description="Short name for the panel box when watching ONE file, "
        "e.g. 'sniffles'; with several paths each file is named after itself",
    )
    paths: list[str] = Field(
        min_length=1,
        description="Paths of the log files to watch, one string per path "
        "(watching a single log is an array of one)",
    )

    _coerce_paths = field_validator("paths", mode="before")(_as_string_list)


def _log_result(watched: list[tuple[Path, str, str]], refused: int) -> str:
    if len(watched) == 1:
        path, state, head = watched[0]
        if state == LOG_GONE:
            body = (
                f"Watching {path} — it does not exist yet, so the panel box "
                f"will say so until something creates it. "
                f"{hints.WATCH_TARGET_MISSING}"
            )
        else:
            body = (
                f"Watching {path} in the Processes panel ({head}, {state}). "
                "The box shows how long ago it was last written to; the user "
                "can press Enter on it for the tail, or D to stop watching."
            )
        return body + _blank_note(refused, hints.WATCH_NEEDS_A_PATH)
    lines = [f"Watching {len(watched)} logs in the Processes panel:"]
    for path, state, head in watched:
        if state == LOG_GONE:
            lines.append(f"- {path} — does not exist yet")
        else:
            lines.append(f"- {path} ({head}, {state})")
    lines.append(
        "Each box shows how long ago its file was last written to; the user "
        "can press Enter on one for the tail, or D to stop watching."
    )
    if any(state == LOG_GONE for _, state, _ in watched):
        lines.append(hints.WATCH_TARGET_MISSING)
    return "\n".join(lines) + _blank_note(refused, hints.WATCH_NEEDS_A_PATH)


async def watch_log(args: WatchLogParams, ctx: ToolContext) -> str:
    store = _require_watches(ctx)
    paths: list[str] = []
    refused = 0
    for raw in args.paths:
        try:
            paths.append(str(_resolve_path(raw, ctx)))
        except PathError:
            refused += 1
    # Deduped after resolution: "run.log" and "./run.log" are one file.
    paths = _unique(paths)
    if not paths:
        return f"Not watched: no path given. {hints.WATCH_NEEDS_A_PATH}"
    # A label names one box. Given several files it would name all of them the
    # same thing, so it is dropped and each file is named after itself.
    label = args.label if len(paths) == 1 else ""
    watched: list[tuple[Path, str, str]] = []
    for target in paths:
        watch = store.add(
            kind=KIND_LOG,
            target=target,
            label=label,
            profile=ctx.profile,
            session_id=ctx.session_id,
        )
        state, head, changed_at = log_fields(target)
        store.update(watch.id, state=state, head=head, changed_at=changed_at or None)
        watched.append((Path(target), state, head))
    return _log_result(watched, refused)


# ------------------------------------------------------------------ watch_job


class WatchJobParams(BaseModel):
    label: str = Field(
        default="",
        description="Short name for the panel box when watching ONE job; "
        "with several ids each job is named after itself",
    )
    job_ids: list[str] = Field(
        min_length=1,
        description="Slurm job ids as squeue reports them, one per element "
        "(watching a single job is an array of one)",
    )

    _coerce_job_ids = field_validator("job_ids", mode="before")(_as_string_list)


def _job_result(
    watched: list[tuple[str, str]], unknown: list[str], refused: int
) -> str:
    """``watched`` is (job_id, state) with "" for a job squeue no longer lists."""
    unknown_note = ""
    if unknown:
        ids = ", ".join(unknown)
        plural = "job" if len(unknown) == 1 else "jobs"
        unknown_note = (
            f"Not watched: Slurm does not know {plural} {ids} (neither squeue "
            f"nor sacct). {hints.WATCH_JOB_UNKNOWN}"
        )
    # The same hint twice in one result reads as a stutter, so the blank note
    # goes without it when the unknown-id line has already given it.
    blanks = _blank_note(refused, "" if unknown_note else hints.WATCH_JOB_UNKNOWN)
    if not watched:
        return (unknown_note or "Not watched: no job id given.") + blanks
    if len(watched) == 1:
        job_id, state = watched[0]
        if state:
            body = (
                f"Watching job {job_id} ({state}) in the Processes panel. It "
                "is refreshed from squeue; when it leaves the queue the box "
                "shows the final state from sacct."
            )
        else:
            body = (
                f"Watching job {job_id} in the Processes panel. It is no "
                "longer queued, so the box will show its final state from "
                "sacct."
            )
        return "\n".join(filter(None, [body, unknown_note])) + blanks
    lines = [f"Watching {len(watched)} jobs in the Processes panel:"]
    for job_id, state in watched:
        lines.append(
            f"- {job_id} {state}"
            if state
            else f"- {job_id} — no longer queued; the box will show its final "
            "state from sacct"
        )
    lines.append(
        "The boxes are refreshed from squeue; when a job leaves the queue its "
        "box shows the final state from sacct."
    )
    if unknown_note:
        lines.append(unknown_note)
    return "\n".join(lines) + blanks


async def watch_job(args: WatchJobParams, ctx: ToolContext) -> str:
    store = _require_watches(ctx)
    if ctx.slurm is None:
        return "Not watched: this session has no Slurm connection."
    refused = sum(1 for raw in args.job_ids if not raw.strip())
    job_ids = _unique([raw.strip() for raw in args.job_ids if raw.strip()])
    if not job_ids:
        return f"Not watched: no job id given. {hints.WATCH_JOB_UNKNOWN}"
    # One squeue for the whole array, then one sacct for whatever it did not
    # know: batching the Slurm calls is what the array is for.
    details = await ctx.slurm.job_details(job_ids)
    missing = [job_id for job_id in job_ids if job_id not in details]
    # Not queued. Such a job may still be a real one that finished a minute ago
    # — worth a box that says COMPLETED — but a typo is not, and refusing the
    # typo is what stops a dead box sitting there forever.
    accounted = await ctx.slurm.status(missing) if missing else {}
    label = args.label if len(job_ids) == 1 else ""
    watched: list[tuple[str, str]] = []
    unknown: list[str] = []
    for job_id in job_ids:
        detail = details.get(job_id)
        if detail is None and job_id not in accounted:
            unknown.append(job_id)
            continue
        watch = store.add(
            kind=KIND_JOB,
            target=job_id,
            label=label or (detail.name if detail is not None else ""),
            profile=ctx.profile,
            session_id=ctx.session_id,
        )
        if detail is None:
            watched.append((job_id, ""))
            continue
        state, head, line = job_fields(detail)
        store.update(watch.id, state=state, head=head, detail=line)
        watched.append((job_id, state))
    return _job_result(watched, unknown, refused)


# --------------------------------------------------------------- list/unwatch


class ListWatchesParams(BaseModel):
    pass


async def list_watches(args: ListWatchesParams, ctx: ToolContext) -> str:
    store = _require_watches(ctx)
    # The agent sees its own session's watches, matching the panel the
    # user is looking at while it answers.
    return _render_list(store.list(session_id=ctx.session_id))


class UnwatchParams(BaseModel):
    targets: list[str] = Field(
        min_length=1,
        description="Which watches to drop — each element a label, a job id, "
        "or a path (dropping a single watch is an array of one)",
    )

    _coerce_targets = field_validator("targets", mode="before")(_as_string_list)


def match_watches(watches: list[Watch], needle: str) -> list[Watch]:
    """Watches a user's or model's shorthand could mean.

    Matched loosely — label, target, file name, or a substring of any of them
    — because the name in the conversation is rarely the absolute path in the
    row, and an ambiguous match is reported rather than guessed at.
    """
    needle = needle.strip()
    if not needle:
        return []
    lowered = needle.lower()
    exact = [
        w
        for w in watches
        if needle in (w.target, w.label)
        or Path(w.target).name == needle
        or str(w.id) == needle
    ]
    if exact:
        return exact
    return [
        w
        for w in watches
        if lowered in w.target.lower() or lowered in w.label.lower()
    ]


def _named(watches: list[Watch]) -> str:
    return ", ".join(f"{w.title} ({w.target})" for w in watches)


def _unwatch_result(
    removed: list[Watch],
    missing: list[str],
    ambiguous: list[tuple[str, list[Watch]]],
    remaining: list[Watch],
    refused: int,
) -> str:
    """One element's outcome per line, and what is left in the panel.

    Every element is resolved against the same snapshot and reported on its
    own: an array where one name misses must not cost the names around it
    their removal (partial application), and a name that matches two watches
    is still refused rather than guessed at — the array is precisely how the
    model says which two it meant.
    """
    if len(removed) + len(missing) + len(ambiguous) == 1:
        # One target: the wording it had before there was an array.
        if removed:
            watch = removed[0]
            body = f"Stopped watching {watch.title} ({watch.target})."
            return body + _blank_note(refused, "")
        if ambiguous:
            needle, matches = ambiguous[0]
            return (
                f"{needle!r} matches several watches: {_named(matches)}. "
                f"{hints.WATCH_AMBIGUOUS}"
            ) + _blank_note(refused, "")
        return (
            f"No watch matches {missing[0]!r}.\n{_render_list(remaining)}"
        ) + _blank_note(refused, "")
    lines: list[str] = []
    if removed:
        lines.append(f"Stopped watching {len(removed)}:")
        lines += [f"- {w.title} ({w.target})" for w in removed]
    for needle in missing:
        lines.append(f"No watch matches {needle!r}.")
    for needle, matches in ambiguous:
        lines.append(f"{needle!r} matches several watches: {_named(matches)}.")
    if ambiguous:
        # Once, at the end: the same twelve-word hint after every ambiguous
        # element reads as a stutter and says nothing the first one did not.
        lines.append(hints.WATCH_AMBIGUOUS)
    if missing or ambiguous:
        lines.append(_render_list(remaining))
    return "\n".join(lines) + _blank_note(refused, "")


async def unwatch(args: UnwatchParams, ctx: ToolContext) -> str:
    store = _require_watches(ctx)
    watches = store.list(session_id=ctx.session_id)
    refused = sum(1 for raw in args.targets if not raw.strip())
    needles = _unique([raw.strip() for raw in args.targets if raw.strip()])
    if not needles:
        return (
            f"Nothing unwatched: no watch named.\n{_render_list(watches)}"
        )
    removed: dict[int, Watch] = {}
    missing: list[str] = []
    ambiguous: list[tuple[str, list[Watch]]] = []
    for needle in needles:
        # Matched against the snapshot, not against what is left: two names
        # for one watch would otherwise make the second one a phantom miss.
        matches = match_watches(watches, needle)
        if not matches:
            missing.append(needle)
        elif len(matches) > 1:
            ambiguous.append((needle, matches))
        else:
            removed.setdefault(matches[0].id, matches[0])
    for watch_id in removed:
        store.remove(watch_id)
    return _unwatch_result(
        list(removed.values()),
        missing,
        ambiguous,
        store.list(session_id=ctx.session_id),
        refused,
    )


def add_watch_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="watch_log",
            description=(
                "Pin one or more log files to the Processes panel so the user "
                "can see at a glance whether they are still being written to"
            ),
            params=WatchLogParams,
            handler=watch_log,
        )
    )
    registry.register(
        Tool(
            name="watch_job",
            description=(
                "Pin one or more running Slurm jobs to the Processes panel, "
                "refreshed from squeue"
            ),
            params=WatchJobParams,
            handler=watch_job,
        )
    )
    registry.register(
        Tool(
            name="list_watches",
            description="What is currently pinned to the Processes panel",
            params=ListWatchesParams,
            handler=list_watches,
        )
    )
    registry.register(
        Tool(
            name="unwatch",
            description=(
                "Stop watching one or more logs or jobs; removes their "
                "panel boxes"
            ),
            params=UnwatchParams,
            handler=unwatch,
        )
    )
    return registry
