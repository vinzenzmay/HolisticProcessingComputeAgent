"""Tests for hpca.triage: signature library and deterministic log triage (§5.5)."""

import pytest

from hpca.jobs import JobRow
from hpca.slurm import JobStatus
from hpca.triage import (
    SignatureError,
    builtin_signatures_path,
    format_report,
    load_signatures,
    scan_log,
    triage_job,
)

EXPECTED_BUILTIN_IDS = {
    "oom_kill",
    "time_limit",
    "command_not_found",
    "missing_input",
    "permission_denied",
    "quota_exceeded",
}


@pytest.fixture
def signatures():
    return load_signatures()


class TestLoadSignatures:
    def test_builtin_library_covers_spec_list(self, signatures):
        assert EXPECTED_BUILTIN_IDS <= {s.id for s in signatures}

    def test_builtin_file_exists(self):
        assert builtin_signatures_path().exists()

    def test_user_file_merged(self, tmp_path):
        user_file = tmp_path / "signatures.yaml"
        user_file.write_text(
            "- id: my_tool_error\n"
            "  title: MyTool crashed\n"
            "  patterns: ['MYTOOL FATAL']\n"
            "  hint: rerun with --safe-mode\n"
        )
        sigs = load_signatures(user_file)
        ids = {s.id for s in sigs}
        assert "my_tool_error" in ids
        assert EXPECTED_BUILTIN_IDS <= ids

    def test_user_file_overrides_builtin_id(self, tmp_path):
        user_file = tmp_path / "signatures.yaml"
        user_file.write_text(
            "- id: oom_kill\n"
            "  title: Custom OOM\n"
            "  patterns: ['oom-kill']\n"
            "  hint: custom hint\n"
        )
        sigs = load_signatures(user_file)
        oom = next(s for s in sigs if s.id == "oom_kill")
        assert oom.title == "Custom OOM"

    def test_malformed_user_file_raises_clear_error(self, tmp_path):
        user_file = tmp_path / "signatures.yaml"
        user_file.write_text("- id: broken\n  patterns: 'not a list")
        with pytest.raises(SignatureError, match=str(user_file)):
            load_signatures(user_file)

    def test_missing_user_file_ignored(self, tmp_path):
        sigs = load_signatures(tmp_path / "nope.yaml")
        assert EXPECTED_BUILTIN_IDS <= {s.id for s in sigs}


class TestScanLog:
    def write_log(self, tmp_path, lines):
        path = tmp_path / "job.err"
        path.write_text("\n".join(lines))
        return path

    def test_oom_matched(self, tmp_path, signatures):
        log = self.write_log(
            tmp_path,
            ["loading data", "slurmstepd: error: Detected 1 oom-kill event(s)"],
        )
        matches = scan_log(log, signatures)
        assert [m.signature_id for m in matches] == ["oom_kill"]
        assert "oom-kill" in matches[0].matched_line

    def test_excerpt_contains_the_error_line(self, tmp_path, signatures):
        lines = ["setup ok"] * 5 + [
            "minimap2: command not found",
            "align.sh: line 3: exiting",
        ]
        log = self.write_log(tmp_path, lines)
        matches = scan_log(log, signatures)
        ids = [m.signature_id for m in matches]
        assert "command_not_found" in ids
        excerpt = next(
            m for m in matches if m.signature_id == "command_not_found"
        ).excerpt
        assert "minimap2: command not found" in excerpt

    def test_time_limit(self, tmp_path, signatures):
        log = self.write_log(
            tmp_path, ["...", "slurmstepd: error: *** JOB 1 CANCELLED DUE TO TIME LIMIT ***"]
        )
        assert "time_limit" in [m.signature_id for m in scan_log(log, signatures)]

    def test_multiple_signatures_in_one_file(self, tmp_path, signatures):
        log = self.write_log(
            tmp_path,
            ["bash: samtols: command not found", "cp: cannot stat 'x': No such file or directory"],
        )
        ids = {m.signature_id for m in scan_log(log, signatures)}
        assert "command_not_found" in ids
        assert "missing_input" in ids

    def test_one_match_per_signature_last_occurrence(self, tmp_path, signatures):
        log = self.write_log(
            tmp_path,
            ["Permission denied: /a", "other stuff", "Permission denied: /b"],
        )
        matches = [m for m in scan_log(log, signatures) if m.signature_id == "permission_denied"]
        assert len(matches) == 1
        assert "/b" in matches[0].matched_line

    def test_excerpt_bounded_to_30_lines(self, tmp_path, signatures):
        lines = [f"noise {i}" for i in range(100)] + ["Disk quota exceeded"] + [
            f"post {i}" for i in range(100)
        ]
        log = self.write_log(tmp_path, lines)
        match = next(
            m for m in scan_log(log, signatures) if m.signature_id == "quota_exceeded"
        )
        assert len(match.excerpt.splitlines()) <= 30

    def test_tail_first_old_matches_beyond_tail_ignored(self, tmp_path, signatures):
        lines = ["Permission denied early"] + [f"noise {i}" for i in range(6000)]
        log = self.write_log(tmp_path, lines)
        assert scan_log(log, signatures, tail_lines=5000) == []

    def test_unreadable_file_skipped(self, tmp_path, signatures):
        assert scan_log(tmp_path / "missing.log", signatures) == []

    def test_clean_log_no_matches(self, tmp_path, signatures):
        log = self.write_log(tmp_path, ["all good", "done"])
        assert scan_log(log, signatures) == []


def make_job(tmp_path, job_id="1"):
    return JobRow(
        job_id=job_id,
        kind="sbatch",
        session_id="s1",
        profile="default",
        submit_time="2026-07-16T10:00:00",
        state="FAILED",
        script_key="align",
        sbatch_stdout_path=str(tmp_path / "job.out"),
        sbatch_stderr_path=str(tmp_path / "job.err"),
        last_checked=None,
        exit_info="FAILED, exit 1",
    )


class TestTriageJob:
    def test_report_combines_meta_and_matches(self, tmp_path, signatures):
        (tmp_path / "job.out").write_text("starting\n")
        (tmp_path / "job.err").write_text("minimap2: command not found\n")
        job = make_job(tmp_path)
        status = JobStatus(
            job_id="1", state="FAILED", raw_state="FAILED", exit_code=1,
            elapsed_s=120, max_rss_bytes=2 * 1024**3, reqmem="4G",
        )
        report = triage_job(job, status, signatures=signatures)
        assert report.job_id == "1"
        assert report.state == "FAILED"
        assert "command_not_found" in [m.signature_id for m in report.matches]

    def test_state_derived_oom_without_log_match(self, tmp_path, signatures):
        (tmp_path / "job.err").write_text("no obvious error text\n")
        job = make_job(tmp_path)
        status = JobStatus(
            job_id="1", state="OUT_OF_MEMORY", raw_state="OUT_OF_MEMORY", signal=125
        )
        report = triage_job(job, status, signatures=signatures)
        assert "oom_kill" in [m.signature_id for m in report.matches]

    def test_state_derived_timeout(self, tmp_path, signatures):
        job = make_job(tmp_path)
        status = JobStatus(job_id="1", state="TIMEOUT", raw_state="TIMEOUT")
        report = triage_job(job, status, signatures=signatures)
        assert "time_limit" in [m.signature_id for m in report.matches]

    def test_fallback_stderr_tail_when_nothing_matched(self, tmp_path, signatures):
        (tmp_path / "job.err").write_text("mysterious one-off failure text\n")
        job = make_job(tmp_path)
        status = JobStatus(job_id="1", state="FAILED", raw_state="FAILED", exit_code=1)
        report = triage_job(job, status, signatures=signatures)
        assert report.matches == []
        assert "mysterious one-off failure" in report.fallback_excerpt

    def test_extra_logs_scanned(self, tmp_path, signatures):
        tool_log = tmp_path / "align.log"
        tool_log.write_text("bwa: command not found\n")
        job = make_job(tmp_path)
        status = JobStatus(job_id="1", state="FAILED", raw_state="FAILED", exit_code=1)
        report = triage_job(
            job, status, signatures=signatures, extra_logs=[tool_log]
        )
        assert "command_not_found" in [m.signature_id for m in report.matches]


class TestFormatReport:
    def test_contains_meta_signature_and_hint(self, tmp_path, signatures):
        (tmp_path / "job.err").write_text(
            "slurmstepd: error: Detected 1 oom-kill event(s)\n"
        )
        job = make_job(tmp_path)
        status = JobStatus(
            job_id="1", state="FAILED", raw_state="FAILED", exit_code=1,
            elapsed_s=120, max_rss_bytes=4 * 1024**3, reqmem="4G",
        )
        text = format_report(triage_job(job, status, signatures=signatures))
        assert "job 1" in text
        assert "FAILED" in text
        assert "oom" in text.lower()
        assert "--mem" in text  # deterministic hint from the signature library

    def test_bounded_size(self, tmp_path, signatures):
        (tmp_path / "job.err").write_text(
            "\n".join(["x" * 200] * 100 + ["minimap2: command not found"] * 5)
        )
        job = make_job(tmp_path)
        status = JobStatus(job_id="1", state="FAILED", raw_state="FAILED", exit_code=1)
        text = format_report(triage_job(job, status, signatures=signatures))
        assert len(text) < 15000  # never a raw multi-MB dump (§5.5)
