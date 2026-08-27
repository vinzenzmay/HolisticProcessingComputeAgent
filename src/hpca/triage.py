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
from hpca.filetail import read_tail
from hpca.jobs import JobRow
from hpca.slurm import JobStatus

TAIL_LINES = 5000
# The byte window those lines are read out of. A job log is the one file here
# that is routinely enormous — a gigabyte of progress bars is an ordinary
# afternoon — and "the last 5000 lines" used to be reached by loading all of
# it. 4 MiB holds 5000 lines at up to ~800 bytes each, which no ordinary log
# exceeds; a log whose lines are wider than that is scanned over fewer of them
# but still over its most recent 4 MiB, which is where a failure that just
# happened is.
TAIL_BYTES = 4 << 20
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
    return Path(__file__).parent / "data" / "error_signatures.yaml"


def user_signatures_path() -> Path:
    return app_dir() / "error_signatures.yaml"


def legacy_user_signatures_path() -> Path:
    """Pre-rename location; still read so nobody's entries vanish silently."""
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
    if user_file is not None:
        candidates = [user_file]
    else:
        candidates = [legacy_user_signatures_path(), user_signatures_path()]
    for path in candidates:
        if path.exists():
            for signature in _parse_signature_file(path):
                by_id[signature.id] = signature
    return list(by_id.values())


def tail_of(path: Path | str, tail_lines: int = TAIL_LINES) -> list[str]:
    """The last ``tail_lines`` lines of a log, read from the end of the file.

    An unreadable log is not an error here — triage is a best effort over
    whatever the job left behind, and every caller's answer to "no lines" is
    already "no findings".

    The first line of a window that did not reach the start of the file is
    dropped: the seek landed in the middle of it, and half a line is not a
    line to match a signature against.
    """
    try:
        text, whole = read_tail(path, TAIL_BYTES)
    except OSError:
        return []
    lines = text.splitlines()
    if not whole and lines:
        del lines[0]
    return lines[-tail_lines:]


def scan_log(
    path: Path,
    signatures: list[Signature],
    *,
    tail_lines: int = TAIL_LINES,
) -> list[SignatureMatch]:
    """Scan a log tail; one match (the last occurrence) per signature."""
    lines = tail_of(path, tail_lines)
    if not lines:
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


# ------------------------------------------------------- tier 2: keyword scan
#
# When no signature matches, a broad keyword sweep still finds most real
# failures. The trap is that it also finds statistics: on a chatty log,
# "Reads failing QC filter: 1423" and "Error rate: 0.012%" match just as
# readily as the actual cause, and taking the last hit lands on a cleanup
# line. So candidates are scored rather than picked by position, and several
# are kept — choosing between them is judgment, which is tier 3's job.

GENERIC_CONTEXT = 3
GENERIC_TOP_N = 5

# A structured error marker: tools emit these deliberately when something
# broke, rather than in passing.
STRONG_MARKERS = re.compile(
    r"(?:^|\W)(?:error|fatal|traceback|exception|abort(?:ed)?|core dumped|"
    r"segmentation fault|panic)\s*:|"
    r"\[E::|^E:|"
    r"(?:failed|unable) to |cannot |could not |no such file|not found|"
    r"permission denied|command not found",
    re.IGNORECASE,
)
WEAK_MARKERS = re.compile(
    r"error|fail(?:ed|ure)?|invalid|denied|broken|missing|refus|corrupt",
    re.IGNORECASE,
)
# Counters and rates. "0 errors" and "Error rate: 0.01%" are reports of
# health, not failures, and they are extremely common in aligner output.
STATISTIC_MARKERS = re.compile(
    r"\d\s*%|"
    r"\b0\s+(?:errors?|failures?|failed)\b|"
    r"\b(?:error|failure)\s+rate\b|"
    r"\b\d+\s+(?:reads?|records?|sequences?|entries)\b",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    """A line that might be the cause, with the lines around it."""

    line_no: int  # 1-based, within the scanned tail
    line: str
    score: float
    excerpt: str


def scan_generic(
    path: Path,
    *,
    context: int = GENERIC_CONTEXT,
    top_n: int = GENERIC_TOP_N,
    tail_lines: int = TAIL_LINES,
) -> list[Candidate]:
    """Scored error-ish lines from a log, best first, each with context.

    Deliberately deterministic and exhaustive: it proposes, something else
    disposes. Returning several candidates instead of one guess is the point
    — a regex cannot tell a cleanup message from the failure that caused it.
    """
    lines = tail_of(path, tail_lines)
    scored: list[Candidate] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        score = 0.0
        if STRONG_MARKERS.search(line):
            score += 3.0
        elif WEAK_MARKERS.search(line):
            score += 1.0
        else:
            continue
        if STATISTIC_MARKERS.search(line):
            score -= 4.0  # a health report, not a failure
        if score <= 0:
            continue
        # Later lines are somewhat more likely to be the cause, but never
        # enough to outrank a genuine error marker further up.
        score += (index / max(len(lines) - 1, 1)) * 0.5
        start = max(0, index - context)
        scored.append(
            Candidate(
                line_no=index + 1,
                line=line.strip(),
                score=round(score, 3),
                excerpt="\n".join(lines[start : index + 1 + context]),
            )
        )
    scored.sort(key=lambda c: (-c.score, -c.line_no))
    return scored[:top_n]


# ------------------------------------------------------ tiered log analysis


@dataclass
class LogFinding:
    """What the deterministic tiers could establish about a failed log."""

    tier: int  # 1 = signature matched, 2 = keyword candidates, 3 = nothing
    title: str = ""
    hint: str = ""
    matched_line: str = ""
    excerpt: str = ""
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def conclusive(self) -> bool:
        """Tier 1 names the failure class; the rest only narrow it down."""
        return self.tier == 1


def analyse_log(
    path: Path, signatures: list[Signature], *, tail_lines: int = TAIL_LINES
) -> LogFinding:
    """Tier 1 then tier 2; tier 3 (the model) is the caller's business."""
    matches = scan_log(path, signatures, tail_lines=tail_lines)
    if matches:
        # Last match wins: signatures scan in library order, and the failure
        # that actually stopped the run is the one nearest the end.
        best = matches[-1]
        return LogFinding(
            tier=1,
            title=best.title,
            hint=best.hint,
            matched_line=best.matched_line,
            excerpt=best.excerpt,
        )
    candidates = scan_generic(path, tail_lines=tail_lines)
    if candidates:
        return LogFinding(
            tier=2,
            title="Possible error lines",
            matched_line=candidates[0].line,
            excerpt=candidates[0].excerpt,
            candidates=candidates,
        )
    return LogFinding(tier=3)


def append_user_signature(
    signature: Signature, *, user_file: Path | None = None
) -> Path:
    """Add a signature to the user library, so tier 3 teaches tiers 1–2.

    Every tier-3 explanation is evidence that a signature is missing: the
    expensive path ran because the cheap one had nothing to say. Writing the
    result back means a tool that fails oddly on this cluster costs a model
    call once rather than every time. Validated before it is written — a bad
    regex here would break loading for every later triage.
    """
    path = user_file if user_file is not None else user_signatures_path()
    for pattern in signature.patterns:
        re.compile(pattern)  # raises re.error on a malformed proposal
    entry = {
        "id": signature.id,
        "title": signature.title,
        "patterns": list(signature.patterns),
        "hint": signature.hint,
    }
    existing = []
    if path.exists():
        existing = [
            e
            for e in (yaml.safe_load(path.read_text()) or [])
            if e.get("id") != signature.id
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(existing + [entry], sort_keys=False))
    return path


def triage_job(
    job: JobRow,
    status: JobStatus | None,
    *,
    signatures: list[Signature],
    extra_logs: list[Path] | None = None,
) -> TriageReport:
    log_paths = [Path(job.sbatch_stdout_path), Path(job.sbatch_stderr_path)]
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
        tail = tail_of(job.sbatch_stderr_path, FALLBACK_TAIL)
        fallback = "\n".join(tail) if tail else None

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
