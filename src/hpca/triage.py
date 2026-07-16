"""Deterministic log triage (§5.5) — the model never sees raw log dumps.

Pipeline: collect the job's logs → scan their tails against the signature
library (built-in YAML + user overrides) → extract a bounded excerpt per
matched signature → assemble a compact structured report. Terminal sacct
states (OUT_OF_MEMORY, TIMEOUT) contribute synthetic matches even when the
logs show nothing, so accounting knowledge is never lost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hpca.config import app_dir
from hpca.jobs import JobRow
from hpca.slurm import JobStatus

TAIL_LINES = 5000
EXCERPT_BEFORE = 9
EXCERPT_AFTER = 20  # before + match + after ≤ 30 lines (§5.5)
FALLBACK_TAIL = 30

STATE_SIGNATURES = {"OUT_OF_MEMORY": "oom_kill", "TIMEOUT": "time_limit"}


class SignatureError(Exception):
    """The signature library could not be parsed."""


@dataclass
class Signature:
    id: str
    title: str
    patterns: list[str]
    hint: str = ""
    _compiled: list[re.Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def match_line(self, line: str) -> bool:
        return any(p.search(line) for p in self._compiled)


@dataclass
class SignatureMatch:
    signature_id: str
    title: str
    hint: str
    file: str
    matched_line: str
    excerpt: str


@dataclass
class TriageReport:
    job_id: str
    state: str
    exit_info: str | None
    elapsed_s: int | None
    max_rss_bytes: int | None
    reqmem: str
    timelimit: str
    script_key: str
    matches: list[SignatureMatch]
    fallback_excerpt: str | None = None


def builtin_signatures_path() -> Path:
    return Path(__file__).parent / "data" / "signatures.yaml"


def user_signatures_path() -> Path:
    return app_dir() / "signatures.yaml"


def _parse_signature_file(path: Path) -> list[Signature]:
    try:
        entries = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise SignatureError(f"Malformed signature file {path}: {e}") from e
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise SignatureError(f"Signature file {path} must be a YAML list")
    signatures = []
    for entry in entries:
        try:
            signatures.append(
                Signature(
                    id=entry["id"],
                    title=entry.get("title", entry["id"]),
                    patterns=list(entry["patterns"]),
                    hint=entry.get("hint", ""),
                )
            )
        except (KeyError, TypeError, re.error) as e:
            raise SignatureError(
                f"Invalid signature entry in {path}: {entry!r} ({e})"
            ) from e
    return signatures


def load_signatures(user_file: Path | None = None) -> list[Signature]:
    """Built-in library plus user entries; user ids override built-ins."""
    by_id = {s.id: s for s in _parse_signature_file(builtin_signatures_path())}
    user_file = user_file if user_file is not None else user_signatures_path()
    if user_file.exists():
        for signature in _parse_signature_file(user_file):
            by_id[signature.id] = signature
    return list(by_id.values())


def scan_log(
    path: Path,
    signatures: list[Signature],
    *,
    tail_lines: int = TAIL_LINES,
) -> list[SignatureMatch]:
    """Scan a log tail; one match (the last occurrence) per signature."""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()[-tail_lines:]
    except OSError:
        return []
    matches: list[SignatureMatch] = []
    for signature in signatures:
        hit_index = None
        for i, line in enumerate(lines):
            if signature.match_line(line):
                hit_index = i
        if hit_index is None:
            continue
        start = max(0, hit_index - EXCERPT_BEFORE)
        excerpt_lines = lines[start : hit_index + 1 + EXCERPT_AFTER]
        matches.append(
            SignatureMatch(
                signature_id=signature.id,
                title=signature.title,
                hint=signature.hint,
                file=str(path),
                matched_line=lines[hit_index].strip(),
                excerpt="\n".join(excerpt_lines),
            )
        )
    return matches


def triage_job(
    job: JobRow,
    status: JobStatus | None,
    *,
    signatures: list[Signature],
    extra_logs: list[Path] | None = None,
) -> TriageReport:
    log_paths = [Path(job.sbatch_stdout_path), Path(job.sbatch_stderr_path)]
    if job.snakemake_log_path:
        log_paths.append(Path(job.snakemake_log_path))
    log_paths.extend(extra_logs or [])

    matches: list[SignatureMatch] = []
    for path in log_paths:
        matches.extend(scan_log(path, signatures))

    state = status.state if status else job.state
    state_sig_id = STATE_SIGNATURES.get(state)
    if state_sig_id and state_sig_id not in {m.signature_id for m in matches}:
        signature = next((s for s in signatures if s.id == state_sig_id), None)
        if signature:
            matches.append(
                SignatureMatch(
                    signature_id=signature.id,
                    title=signature.title,
                    hint=signature.hint,
                    file="(sacct)",
                    matched_line=f"job state {state}",
                    excerpt=f"sacct reports terminal state {state}",
                )
            )

    fallback = None
    if not matches:
        try:
            tail = (
                Path(job.sbatch_stderr_path)
                .read_text(errors="replace")
                .splitlines()[-FALLBACK_TAIL:]
            )
            fallback = "\n".join(tail) if tail else None
        except OSError:
            fallback = None

    return TriageReport(
        job_id=job.job_id,
        state=state,
        exit_info=job.exit_info
        or (f"exit {status.exit_code}" if status and status.exit_code is not None else None),
        elapsed_s=status.elapsed_s if status else None,
        max_rss_bytes=status.max_rss_bytes if status else None,
        reqmem=status.reqmem if status else "",
        timelimit=status.timelimit if status else "",
        script_key=job.script_key,
        matches=matches,
        fallback_excerpt=fallback,
    )


def format_report(report: TriageReport) -> str:
    """Compact text form of the report, bounded for a small model's context."""
    lines = [
        f"job {report.job_id} ({report.script_key}): {report.state}"
        + (f", {report.exit_info}" if report.exit_info else "")
    ]
    meta = []
    if report.elapsed_s is not None:
        meta.append(f"elapsed {report.elapsed_s}s")
    if report.max_rss_bytes is not None:
        meta.append(f"max RSS {report.max_rss_bytes // (1024 * 1024)}M")
    if report.reqmem:
        meta.append(f"requested {report.reqmem}")
    if report.timelimit:
        meta.append(f"time limit {report.timelimit}")
    if meta:
        lines.append(", ".join(meta))
    for match in report.matches:
        lines += [
            "",
            f"## {match.title} [{match.signature_id}] in {match.file}",
            f"hint: {match.hint}" if match.hint else "",
            "```",
            match.excerpt[:4000],
            "```",
        ]
    if not report.matches:
        lines.append("")
        lines.append("No known error signature matched.")
        if report.fallback_excerpt:
            lines += ["stderr tail:", "```", report.fallback_excerpt[:4000], "```"]
    return "\n".join(line for line in lines if line is not None)
