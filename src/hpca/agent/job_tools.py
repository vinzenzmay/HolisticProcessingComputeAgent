"""Cluster job tools (§5.1): submit_job, job_status, cancel_job.

Submission always passes the ``sbatch --test-only`` gate first (§5.2), pins
stdout/stderr to generated ``%j`` log paths so every log location is known to
the job DB (§5.4), and registers the resolved paths in the path registry.
``cancel_job`` is destructive and therefore HITL-gated by the graph (§5.3).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.explainer import explain_failure
from hpca.agent.tools import Tool, ToolRegistry
from hpca.slurm import JobStatus
from hpca.triage import load_signatures, triage_job


def _require_cluster(ctx: ToolContext):
    if ctx.slurm is None or ctx.jobs is None or ctx.job_log_dir is None:
        raise RuntimeError("Cluster job tools are not configured in this session")
    return ctx.slurm, ctx.jobs, ctx.job_log_dir


class SubmitJobParams(BaseModel):
    registry_key: str = Field(description="Registry key of the sbatch script")
    args: str = Field(
        default="", description="Extra sbatch options, e.g. '--mem=8G --time=01:00:00'"
    )


async def submit_job(args: SubmitJobParams, ctx: ToolContext) -> str:
    slurm, jobs, job_log_dir = _require_cluster(ctx)
    script = ctx.registry.resolve(args.registry_key)
    job_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_tpl = str(job_log_dir / f"{args.registry_key}-%j.out")
    stderr_tpl = str(job_log_dir / f"{args.registry_key}-%j.err")
    sbatch_args = ["-o", stdout_tpl, "-e", stderr_tpl]
    if args.args:
        sbatch_args += args.args.split()

    ok, message = await slurm.test_only(str(script), sbatch_args)
    if not ok:
        return (
            "Job NOT submitted: sbatch --test-only rejected it — fix and retry:\n"
            f"{message}"
        )

    job_id = await slurm.submit(str(script), sbatch_args)
    stdout_path = stdout_tpl.replace("%j", job_id)
    stderr_path = stderr_tpl.replace("%j", job_id)
    jobs.add(
        job_id=job_id,
        kind="sbatch",
        session_id=ctx.session_id,
        profile=ctx.profile,
        script_key=args.registry_key,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    out_key = ctx.registry.register_auto(
        stdout_path, hint=f"{args.registry_key}_job_stdout"
    )
    err_key = ctx.registry.register_auto(
        stderr_path, hint=f"{args.registry_key}_job_stderr"
    )
    return (
        f"Submitted job {job_id} ({args.registry_key!r}). It is tracked in the "
        f"background; logs: {out_key}, {err_key}. Check with job_status."
    )


class JobStatusParams(BaseModel):
    job_id: str = Field(description="Slurm job id")


def _format_status(status: JobStatus) -> str:
    parts = [f"state {status.state}"]
    if status.elapsed_s is not None:
        parts.append(f"elapsed {status.elapsed_s}s")
    if status.max_rss_bytes is not None:
        parts.append(f"max RSS {status.max_rss_bytes // (1024 * 1024)}M")
    if status.reqmem:
        parts.append(f"requested {status.reqmem}")
    if status.exit_code is not None and status.is_terminal:
        parts.append(f"exit {status.exit_code}")
    if status.raw_state != status.state:
        parts.append(f"({status.raw_state})")
    return ", ".join(parts)


async def job_status(args: JobStatusParams, ctx: ToolContext) -> str:
    slurm, jobs, _ = _require_cluster(ctx)
    row = jobs.get(args.job_id)
    statuses = await slurm.status([args.job_id])
    status = statuses.get(args.job_id)
    if status is None:
        if row is None:
            return f"Job {args.job_id} is unknown (not in the job DB, not in sacct)."
        return (
            f"Job {args.job_id}: state {row.state} — sacct does not report it "
            "yet (accounting lag is normal right after submission)."
        )
    if row is not None:
        jobs.update_status(status)
    return f"Job {args.job_id}: {_format_status(status)}"


class GetJobReportParams(BaseModel):
    job_id: str = Field(description="Slurm job id to triage")


async def get_job_report(args: GetJobReportParams, ctx: ToolContext) -> str:
    """Triaged failure report (§5.5): deterministic scan + firewalled explainer."""
    slurm, jobs, _ = _require_cluster(ctx)
    row = jobs.get(args.job_id)
    if row is None:
        return f"Job {args.job_id} is not in the job DB."
    statuses = await slurm.status([args.job_id])
    status = statuses.get(args.job_id)
    if status is not None:
        jobs.update_status(status)
    extra_logs = [Path(log.log_path) for log in jobs.logs(args.job_id)]
    report = triage_job(
        row, status, signatures=load_signatures(), extra_logs=extra_logs
    )
    matched = ", ".join(m.title for m in report.matches) or "no known signature"
    header = f"Job {args.job_id} ({row.script_key}): {report.state} — {matched}."
    if ctx.llm is None:
        return header
    explanation = await explain_failure(ctx.llm, report, tier1=ctx.tier1_text)
    return f"{header}\n{explanation.render()}"


class CancelJobParams(BaseModel):
    job_id: str = Field(description="Slurm job id to cancel")


async def cancel_job(args: CancelJobParams, ctx: ToolContext) -> str:
    slurm, jobs, _ = _require_cluster(ctx)
    await slurm.cancel(args.job_id)
    if jobs.get(args.job_id) is not None:
        # sacct will report the final CANCELLED state on the next poll
        jobs.mark(args.job_id, "CANCELLING")
    return f"Requested cancellation of job {args.job_id}."


def add_job_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="submit_job",
            description="Submit a registered script to Slurm via sbatch "
            "(dry-run gated)",
            params=SubmitJobParams,
            handler=submit_job,
        )
    )
    registry.register(
        Tool(
            name="job_status",
            description="Current state of a cluster job",
            params=JobStatusParams,
            handler=job_status,
        )
    )
    registry.register(
        Tool(
            name="get_job_report",
            description="Triaged failure report for a job: why it failed, "
            "suggested fix",
            params=GetJobReportParams,
            handler=get_job_report,
        )
    )
    registry.register(
        Tool(
            name="cancel_job",
            description="Cancel a cluster job",
            params=CancelJobParams,
            handler=cancel_job,
            destructive=True,
        )
    )
    return registry
