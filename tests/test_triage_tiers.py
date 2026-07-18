"""Tests for the three-tier error identification (§5.5).

Tier 1 signature library, tier 2 scored keyword candidates, tier 3 the model.
Each tier only runs when the cheaper one above it found nothing.
"""

import json

import pytest

from hpca.agent.explainer import (
    ProcessExplanation,
    explain_process_failure,
    format_candidates,
)
from hpca.llm import ChatResponse
from hpca.triage import (
    Signature,
    analyse_log,
    append_user_signature,
    load_signatures,
    scan_generic,
)

# A chatty aligner log: the real cause sits mid-file, and the naive "last
# keyword hit" lands on the cleanup line instead.
NOISY = """\
[M::main] reading reference...
[M::worker] mapped 120000 sequences
Reads failing QC filter: 1423 (0.4%)
Error rate: 0.012%
[E::hts_open_format] Failed to open file 'sample.bam' : No such file or directory
samtools sort: failed to read header
[M::cleanup] removing temporary files
Dropped 0 reads, 0 errors during cleanup
[M::main] finished in 42.1s
"""

CLEAN = """\
[M::main] reading reference
[M::worker] mapped 120000 sequences
[M::main] finished in 42.1s
"""


@pytest.fixture
def log(tmp_path):
    def write(text, name="run.err"):
        path = tmp_path / name
        path.write_text(text)
        return path

    return write


class TestTier2KeywordScan:
    def test_statistics_are_not_mistaken_for_failures(self, log):
        candidates = scan_generic(log(NOISY))
        found = {c.line for c in candidates}
        assert not any("Error rate" in line for line in found)
        assert not any("Reads failing QC" in line for line in found)
        assert not any("0 errors during cleanup" in line for line in found)

    def test_real_errors_rank_above_everything_else(self, log):
        candidates = scan_generic(log(NOISY))
        assert len(candidates) == 2
        assert "failed to read header" in candidates[0].line
        assert "hts_open_format" in candidates[1].line

    def test_a_late_cleanup_line_does_not_outrank_a_real_error(self, log):
        """Position is one signal, never enough to beat an error marker."""
        top = scan_generic(log(NOISY))[0]
        assert "cleanup" not in top.line

    def test_candidates_carry_surrounding_context(self, log):
        candidate = scan_generic(log(NOISY), context=3)[0]
        lines = candidate.excerpt.splitlines()
        assert candidate.line in "\n".join(lines)
        assert len(lines) <= 7  # 3 before + the line + 3 after
        assert "Error rate: 0.012%" in candidate.excerpt  # 3 lines above

    def test_context_width_is_configurable(self, log):
        wide = scan_generic(log(NOISY), context=1)[0]
        assert len(wide.excerpt.splitlines()) <= 3

    def test_healthy_log_yields_nothing(self, log):
        assert scan_generic(log(CLEAN)) == []

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert scan_generic(tmp_path / "nope.err") == []


class TestTierSelection:
    def test_signature_match_is_tier_1(self, log):
        finding = analyse_log(
            log("sniffles: error: unrecognized arguments: out.vcf\n"),
            load_signatures(user_file=None if False else None),
        )
        assert finding.tier == 1
        assert finding.title == "Wrong command-line arguments"
        assert finding.hint  # tier 1's real value: names the class of mistake
        assert finding.conclusive

    def test_keywords_only_is_tier_2(self, log, tmp_path):
        # no signature covers this, but it is plainly an error line
        finding = analyse_log(log("wibble: catastrophic flange failure\n"), [])
        assert finding.tier == 2
        assert not finding.conclusive
        assert finding.candidates

    def test_nothing_found_is_tier_3(self, log):
        finding = analyse_log(log(CLEAN), [])
        assert finding.tier == 3
        assert not finding.candidates

    def test_tier_1_wins_over_tier_2_in_the_same_log(self, log):
        finding = analyse_log(log(NOISY), load_signatures())
        assert finding.tier == 1  # htslib signature, not the keyword sweep


class TestSignatureLibraryFile:
    def test_new_and_legacy_user_files_both_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        legacy = tmp_path / "signatures.yaml"
        legacy.write_text(
            "- id: legacy_one\n  title: Legacy\n  patterns: ['zzz']\n"
        )
        ids = {s.id for s in load_signatures()}
        assert "legacy_one" in ids  # a rename must not drop existing entries

    def test_appending_a_signature_makes_it_match(self, tmp_path):
        user_file = tmp_path / "error_signatures.yaml"
        signature = Signature(
            id="flange_failure",
            title="Flange failure",
            patterns=[r"catastrophic flange failure"],
            hint="Replace the flange.",
        )
        append_user_signature(signature, user_file=user_file)
        loaded = load_signatures(user_file=user_file)
        assert "flange_failure" in {s.id for s in loaded}
        log = tmp_path / "run.err"
        log.write_text("wibble: catastrophic flange failure\n")
        # what was tier 2 before is tier 1 now: the loop closes
        assert analyse_log(log, loaded).tier == 1

    def test_rewriting_an_id_replaces_it(self, tmp_path):
        user_file = tmp_path / "error_signatures.yaml"
        first = Signature(id="x", title="First", patterns=["aaa"], hint="")
        second = Signature(id="x", title="Second", patterns=["bbb"], hint="")
        append_user_signature(first, user_file=user_file)
        append_user_signature(second, user_file=user_file)
        loaded = [s for s in load_signatures(user_file=user_file) if s.id == "x"]
        assert len(loaded) == 1
        assert loaded[0].title == "Second"

    def test_a_bad_regex_never_reaches_the_file(self, tmp_path):
        """A model-proposed pattern that does not compile would break loading
        for every later triage, so it must not be persisted."""
        user_file = tmp_path / "error_signatures.yaml"
        with pytest.raises(Exception):
            append_user_signature(
                Signature(id="bad", title="Bad", patterns=["([unclosed"], hint=""),
                user_file=user_file,
            )
        assert not user_file.exists()

    def test_a_valid_library_survives_a_rejected_proposal(self, tmp_path):
        user_file = tmp_path / "error_signatures.yaml"
        append_user_signature(
            Signature(id="good", title="Good", patterns=["ok"], hint=""),
            user_file=user_file,
        )
        with pytest.raises(Exception):
            append_user_signature(
                Signature(id="bad", title="Bad", patterns=["(["], hint=""),
                user_file=user_file,
            )
        assert {s.id for s in load_signatures(user_file=user_file)} >= {"good"}


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    async def chat(self, messages, **kwargs):
        self.prompts.append(messages)
        return ChatResponse(content=json.dumps(self.payload))


class TestTier3Explainer:
    @pytest.fixture
    def candidates(self, log):
        return scan_generic(log(NOISY))

    async def test_picks_the_causing_line(self, candidates):
        llm = FakeLLM({
            "why": "The input BAM does not exist.",
            "cause_line": "[E::hts_open_format] Failed to open file 'sample.bam' : No such file or directory",
            "suggested_fix": "Check the path to sample.bam.",
            "proposed_signature": None,
        })
        result = await explain_process_failure(
            llm, name="job", exit_code=1, candidates=candidates
        )
        assert result.conclusive
        assert "hts_open_format" in result.cause_line

    async def test_an_invented_cause_line_is_rejected(self, candidates):
        """The failure mode this tier risks: a plausible line that is not
        in the log at all. Quote-or-admit, enforced in code."""
        llm = FakeLLM({
            "why": "Ran out of memory.",
            "cause_line": "slurmstepd: error: Detected 1 oom-kill event",
            "suggested_fix": "Request more memory.",
            "proposed_signature": None,
        })
        result = await explain_process_failure(
            llm, name="job", exit_code=1, candidates=candidates
        )
        assert result.cause_line == ""
        assert not result.conclusive
        assert "unverified" in result.why

    async def test_admitting_ignorance_is_allowed(self, candidates):
        llm = FakeLLM({
            "why": "None of these lines explains the exit code.",
            "cause_line": "",
            "suggested_fix": "Re-run with verbose logging.",
            "proposed_signature": None,
        })
        result = await explain_process_failure(
            llm, name="job", exit_code=1, candidates=candidates
        )
        assert not result.conclusive
        assert "unverified" not in result.why  # honesty is not punished

    async def test_proposed_signature_is_returned(self, candidates):
        llm = FakeLLM({
            "why": "Input missing.",
            "cause_line": "samtools sort: failed to read header",
            "suggested_fix": "Check the input.",
            "proposed_signature": {
                "id": "samtools_header",
                "title": "samtools could not read a header",
                "patterns": ["failed to read header"],
                "hint": "The BAM is truncated or not a BAM.",
            },
        })
        result = await explain_process_failure(
            llm, name="job", exit_code=1, candidates=candidates
        )
        assert result.proposed_signature.id == "samtools_header"

    async def test_only_candidates_are_shown_to_the_model(self, candidates):
        """The firewall: the model judges retrieved lines, it does not get
        handed the raw log to search."""
        prompt = format_candidates("job", 1, candidates)
        assert "failed to read header" in prompt
        assert "mapped 120000 sequences" not in prompt  # noise stayed out
