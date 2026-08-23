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

Nothing here polls. Registering only records the target; the TUI's timers
refresh every box (see :mod:`hpca.watches`).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

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


# ------------------------------------------------------------------ watch_log


class WatchLogParams(BaseModel):
    path: str = Field(
        description="Path of the log file to watch"
    )
    label: str = Field(
        default="",
        description="Short name for the panel box, e.g. 'sniffles'; "
        "defaults to the file name",
    )


async def watch_log(args: WatchLogParams, ctx: ToolContext) -> str:
    store = _require_watches(ctx)
    try:
        path = _resolve_path(args.path, ctx)
    except PathError:
        return f"Not watched: no path given. {hints.WATCH_NEEDS_A_PATH}"
    watch = store.add(
        kind=KIND_LOG,
        target=str(path),
        label=args.label,
        profile=ctx.profile,
        session_id=ctx.session_id,
    )
    state, head, changed_at = log_fields(path)
    store.update(watch.id, state=state, head=head, changed_at=changed_at or None)
    if state == LOG_GONE:
        return (
            f"Watching {path} — it does not exist yet, so the panel box will "
            f"say so until something creates it. {hints.WATCH_TARGET_MISSING}"
        )
    return (
        f"Watching {path} in the Processes panel ({head}, {state}). The box "
        "shows how long ago it was last written to; the user can press Enter "
        "on it for the tail, or D to stop watching."
    )


# ------------------------------------------------------------------ watch_job


class WatchJobParams(BaseModel):
    job_id: str = Field(description="Slurm job id, as squeue reports it")
    label: str = Field(
        default="",
        description="Short name for the panel box; defaults to the job name",
    )


async def watch_job(args: WatchJobParams, ctx: ToolContext) -> str:
    store = _require_watches(ctx)
    job_id = args.job_id.strip()
    if ctx.slurm is None:
        return "Not watched: this session has no Slurm connection."
    details = await ctx.slurm.job_details([job_id])
    detail = details.get(job_id)
    label = args.label
    if detail is None:
        # Not queued. It may still be a real job that finished a minute ago —
        # worth a box that says COMPLETED — but a typo is not, and refusing
        # the typo is what stops a dead box sitting there forever.
        statuses = await ctx.slurm.status([job_id])
        if job_id not in statuses:
            return (
                f"Not watched: Slurm does not know job {job_id} (neither "
                f"squeue nor sacct). {hints.WATCH_JOB_UNKNOWN}"
            )
    elif not label:
        label = detail.name
    watch = store.add(
        kind=KIND_JOB,
        target=job_id,
        label=label,
        profile=ctx.profile,
        session_id=ctx.session_id,
    )
    if detail is not None:
        state, head, line = job_fields(detail)
        store.update(watch.id, state=state, head=head, detail=line)
        return (
            f"Watching job {job_id} ({state}) in the Processes panel. It is "
            "refreshed from squeue; when it leaves the queue the box shows "
            "the final state from sacct."
        )
    return (
        f"Watching job {job_id} in the Processes panel. It is no longer "
        "queued, so the box will show its final state from sacct."
    )


# --------------------------------------------------------------- list/unwatch


class ListWatchesParams(BaseModel):
    pass


async def list_watches(args: ListWatchesParams, ctx: ToolContext) -> str:
    store = _require_watches(ctx)
    # The agent sees its own session's watches, matching the panel the
    # user is looking at while it answers.
    return _render_list(store.list(session_id=ctx.session_id))


class UnwatchParams(BaseModel):
    target: str = Field(
        description="Which watch to drop: its label, its job id, or its path"
    )


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


async def unwatch(args: UnwatchParams, ctx: ToolContext) -> str:
    store = _require_watches(ctx)
    watches = store.list(session_id=ctx.session_id)
    matches = match_watches(watches, args.target)
    if not matches:
        return f"No watch matches {args.target!r}.\n{_render_list(watches)}"
    if len(matches) > 1:
        names = ", ".join(f"{w.title} ({w.target})" for w in matches)
        return (
            f"{args.target!r} matches several watches: {names}. "
            f"{hints.WATCH_AMBIGUOUS}"
        )
    watch = matches[0]
    store.remove(watch.id)
    return f"Stopped watching {watch.title} ({watch.target})."


def add_watch_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="watch_log",
            description=(
                "Pin a log file to the Processes panel so the user can see at "
                "a glance whether it is still being written to"
            ),
            params=WatchLogParams,
            handler=watch_log,
        )
    )
    registry.register(
        Tool(
            name="watch_job",
            description=(
                "Pin a running Slurm job to the Processes panel, refreshed "
                "from squeue"
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
            description="Stop watching a log or job; removes its panel box",
            params=UnwatchParams,
            handler=unwatch,
        )
    )
    return registry
