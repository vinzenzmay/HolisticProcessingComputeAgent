"""Tests for hpca.verify_code: the semantic code-vs-docs gate (§5.2)."""

import pytest

from hpca.db import connect, init_db
from hpca.symbols import Symbol, SymbolIndex
from hpca.verify_code import (
    commands_needing_docs,
    extract_bash,
    extract_python,
    verify_script,
)


class TestExtractBash:
    def test_commands_and_flags(self):
        usages = extract_bash(
            "#!/bin/bash\n"
            "samtools view -b -q 20 in.bam | grep -v chrM > out.sam\n"
        )
        assert ("samtools", ["view"], ["-b", "-q"]) in usages
        assert ("grep", [], ["-v"]) in usages

    def test_assignments_keywords_comments_skipped(self):
        usages = extract_bash(
            "# a comment\n"
            "THREADS=4\n"
            "if [ -f x ]; then\n"
            "  sort -k1,1 x\n"
            "fi\n"
        )
        commands = [u[0] for u in usages]
        assert "sort" in commands
        assert "THREADS=4" not in commands
        assert "if" not in commands and "then" not in commands

    def test_long_flag_with_value_normalized(self):
        usages = extract_bash("bcftools call --output-type=z in.vcf\n")
        assert ("bcftools", ["call"], ["--output-type"]) in usages

    def test_command_after_and_and(self):
        usages = extract_bash("mkdir -p out && cd out\n")
        assert ("mkdir", [], ["-p"]) in usages


class TestComments:
    def test_full_line_comments_and_shebang_ignored(self):
        usages = extract_bash("#!/bin/bash\n# dedupe with -x speed\nsort -u f.txt\n")
        assert usages == [("sort", [], ["-u"])]

    def test_trailing_comment_contributes_no_flags(self):
        usages = extract_bash("sort -u f.txt  # not a --invented-flag\n")
        assert usages == [("sort", [], ["-u"])]

    def test_quoted_hash_is_data_not_a_comment(self):
        # VCF headers start with '#'; treating it as a comment would drop the
        # rest of the pipeline from verification entirely
        usages = extract_bash("grep '#CHROM' in.vcf | cut -f1\n")
        assert usages == [("grep", [], []), ("cut", [], ["-f1"])]

    def test_hash_inside_a_regex_survives(self):
        usages = extract_bash('grep -v "^#" in.vcf | sort -k1,1\n')
        assert usages == [("grep", [], ["-v"]), ("sort", [], ["-k1,1"])]


class TestPipelines:
    """Every stage of a pipeline is its own command with its own flags."""

    def test_three_stage_pipeline(self):
        usages = extract_bash("grep -v chrM in.sam | sort -k1,1 | gzip -c > out.gz\n")
        assert usages == [
            ("grep", [], ["-v"]),
            ("sort", [], ["-k1,1"]),
            ("gzip", [], ["-c"]),
        ]

    def test_unspaced_pipe_still_splits(self):
        # shlex.split leaves "in.bam|gzip" as one token, which merged both
        # stages and checked gzip's flags against samtools
        usages = extract_bash("samtools view -b in.bam|gzip -c > out.gz\n")
        assert usages == [("samtools", ["view"], ["-b"]), ("gzip", [], ["-c"])]

    def test_pipe_inside_quotes_is_data_not_a_separator(self):
        usages = extract_bash("awk -F'|' '{print $1}' f.txt\n")
        assert usages == [("awk", [], ["-F|"])]

    def test_each_stage_may_be_separately_wrapped(self):
        usages = extract_bash(
            "conda run -n bio minimap2 -ax map-ont ref.fa r.fq | "
            "conda run -n bio samtools sort -o out.bam -\n"
        )
        assert usages == [("minimap2", [], ["-ax"]), ("samtools", ["sort"], ["-o"])]

    def test_bare_dash_operand_is_not_a_flag(self):
        # `-` means stdin/stdout; treating it as a flag blocked every
        # `samtools sort -o out.bam -` pipeline
        usages = extract_bash("samtools sort -o out.bam -\n")
        assert usages == [("samtools", ["sort"], ["-o"])]

    def test_end_of_options_marker_stops_flag_collection(self):
        usages = extract_bash("grep -- -weird file.txt\n")
        assert usages == [("grep", [], [])]


class TestUnwrapWrappers:
    """ENVIRONMENT_TOOL_GUIDANCE tells the agent to write `conda run -n env
    <tool>`; without unwrapping, every such line reads as a call to conda and
    the tool's flags get checked against conda's."""

    def test_conda_run_attributes_flags_to_the_inner_tool(self):
        usages = extract_bash("conda run -n bio minimap2 -x map-ont ref.fa r.fq\n")
        assert usages == [("minimap2", [], ["-x"])]

    def test_long_form_env_flag_consumed_with_its_value(self):
        usages = extract_bash("micromamba run --name bio bwa mem -t 4 ref.fa\n")
        assert usages == [("bwa", ["mem"], ["-t"])]

    def test_transparent_prefixes_stripped(self):
        usages = extract_bash("time nohup samtools sort -o out.bam in.bam\n")
        assert usages == [("samtools", ["sort"], ["-o"])]

    def test_absolute_path_keeps_its_path_for_probing(self):
        # the literal token is preserved so the auto-indexer can execute it;
        # the index key is the basename (see TestVerifyBash)
        usages = extract_bash("/opt/conda/envs/bio/bin/samtools view -b in.bam\n")
        assert usages == [("/opt/conda/envs/bio/bin/samtools", ["view"], ["-b"])]


class TestExtractPython:
    def test_calls_with_kwargs(self):
        calls = extract_python(
            "import pysam\n"
            "af = pysam.AlignmentFile('x.bam', mode='rb')\n"
            "align(reads, reference, threads=4, min_quality=20)\n"
        )
        by_name = {c[0]: c[1] for c in calls}
        assert by_name["AlignmentFile"] == ["mode"]
        assert by_name["align"] == ["threads", "min_quality"]

    def test_syntax_error_returns_empty(self):
        assert extract_python("def broken(:") == []


@pytest.fixture
def index(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    idx = SymbolIndex(conn)
    idx.add(
        [
            Symbol(name="-b", kind="cli-flag", parent="samtools-view", source="man"),
            Symbol(name="-q", kind="cli-flag", parent="samtools-view", source="man"),
            Symbol(name="-o", kind="cli-flag", parent="samtools-view", source="man"),
            Symbol(name="-v", kind="cli-flag", parent="grep", source="man"),
            Symbol(
                name="align",
                kind="function",
                parent="tools",
                params=["reads", "reference", "threads", "min_quality"],
                signature="align(reads, reference, *, threads=1, min_quality=20)",
                source="tools.py",
            ),
        ]
    )
    yield idx
    conn.close()


class TestVerifyBash:
    def test_known_flags_confirmed(self, index):
        reports = verify_script(
            "bash", "samtools view -b -q 20 in.bam\n", index=index
        )
        assert all(r.status == "confirmed" for r in reports)
        assert {r.symbol for r in reports} == {
            "samtools view -b",
            "samtools view -q",
        }

    def test_invented_flag_is_mismatch(self, index):
        reports = verify_script("bash", "samtools view -e in.bam\n", index=index)
        mismatch = next(r for r in reports if r.status == "mismatch")
        assert "-e" in mismatch.symbol
        assert "-b" in mismatch.detail  # known flags listed for the fix loop

    def test_unindexed_command_reported_once(self, index):
        reports = verify_script(
            "bash", "bwa mem -t 4 ref.fa reads.fq\nbwa index ref.fa\n", index=index
        )
        not_indexed = [r for r in reports if r.status == "not_indexed"]
        assert len(not_indexed) == 1
        assert "bwa" in not_indexed[0].symbol

    def test_subcommand_resolution(self, index):
        # index stores "samtools-view"; the script says "samtools view"
        reports = verify_script("bash", "samtools view -o out.bam in.bam\n", index=index)
        assert reports[0].status == "confirmed"

    def test_absolute_path_resolves_to_the_basename_key(self, index):
        reports = verify_script(
            "bash", "/opt/conda/envs/bio/bin/samtools view -b in.bam\n", index=index
        )
        assert reports[0].status == "confirmed"

    def test_attached_short_option_value_is_not_a_mismatch(self, index):
        # `-q20` and `-q 20` are the same flag; an exact-match test would
        # block a correct script (this bit `sort -k1,1` in a live run)
        reports = verify_script("bash", "samtools view -q20 in.bam\n", index=index)
        assert reports[0].status == "confirmed"

    def test_clustered_short_options_accepted(self, index):
        reports = verify_script("bash", "samtools view -bq in.bam\n", index=index)
        assert reports[0].status == "confirmed"

    def test_invented_long_flag_still_blocked(self, index):
        reports = verify_script("bash", "grep --notaflag x f\n", index=index)
        assert reports[0].status == "mismatch"


class TestVerifyPython:
    def test_valid_kwargs_confirmed(self, index):
        reports = verify_script(
            "python", "align(r, ref, threads=8)\n", index=index
        )
        assert reports[0].status == "confirmed"

    def test_typo_kwarg_is_mismatch(self, index):
        reports = verify_script(
            "python", "align(r, ref, min_qualty=20)\n", index=index
        )
        mismatch = next(r for r in reports if r.status == "mismatch")
        assert "min_qualty" in mismatch.detail
        assert "min_quality" in mismatch.detail  # signature shown for the fix

    def test_unindexed_call_with_kwargs_reported(self, index):
        reports = verify_script("python", "mystery(x, mode='rb')\n", index=index)
        assert reports[0].status == "not_indexed"

    def test_unindexed_call_without_kwargs_skipped(self, index):
        # print(x), range(n), ... — nothing checkable, stay quiet (§5.2.4)
        assert verify_script("python", "print(align_result)\n", index=index) == []


# ------------------------------------------------- create_script integration

from hpca.agent.builtin_tools import default_tool_registry  # noqa: E402
from hpca.agent.builtin_tools import script_names, script_path  # noqa: E402
from hpca.agent.context import ToolContext  # noqa: E402
from hpca.agent.doc_tools import safe_to_execute  # noqa: E402
from hpca.config import Settings  # noqa: E402
from hpca.runner import ProcessRunner  # noqa: E402


@pytest.fixture
def ctx(tmp_path, index):
    conn = connect(tmp_path / "ctx.db")
    init_db(conn)
    yield ToolContext(
        workdir=tmp_path,
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
        settings=Settings(),
        scripts_dir=tmp_path / "scripts",
        symbols=index,
    )
    conn.close()


async def create(ctx, key, lines):
    tools = default_tool_registry()
    tool = tools.get("create_script")
    args = tool.params.model_validate(
        {"kind": "bash", "name": key, "content_lines": lines}
    )
    return await tool.handler(args, ctx)


class TestCreateScriptGate:
    async def test_invented_flag_blocks_creation(self, ctx):
        result = await create(ctx, "bad", ["samtools view -e in.bam"])
        assert "NOT created" in result
        assert "-e" in result
        assert "bad" not in script_names(ctx)

    async def test_valid_flags_pass(self, ctx):
        result = await create(ctx, "good", ["samtools view -b -q 20 in.bam"])
        assert "ok" in result.lower()
        assert "good" in script_names(ctx)

    async def test_unindexed_command_warns_but_creates(self, ctx):
        result = await create(ctx, "warned", ["bwa mem -t 4 ref.fa reads.fq"])
        assert "warned" in script_names(ctx)
        assert "not indexed" in result.lower()

    async def test_no_index_no_gate(self, ctx):
        ctx.symbols = None
        result = await create(ctx, "ungated", ["samtools view -e in.bam"])
        assert "ungated" in script_names(ctx)


@pytest.fixture
def faketool(tmp_path):
    """A real executable with a real --help, so the probe path runs for real.

    Deliberately *not* in a bin/ directory: an installed tool may sit
    anywhere, and probing must not depend on where it was put.
    """
    path = tmp_path / "faketool"
    path.write_text(
        "#!/bin/bash\n"
        "cat <<'EOF'\n"
        "Usage: faketool [options] <in>\n"
        "Options:\n"
        "  -a           enable the first thing\n"
        "  -b INT       the second thing\n"
        "  -c STR       the third thing\n"
        "  --verbose    say more\n"
        "EOF\n"
    )
    path.chmod(0o755)
    return path


class TestAutoIndexing:
    """The gate is useless on commands nobody indexed, and index_docs is
    explicit-only — so create_script learns them itself (§5.2)."""

    async def test_unknown_command_is_learned_from_its_help(self, ctx, faketool):
        assert not ctx.symbols.has_command("faketool")
        await create(ctx, "ok", [f"{faketool} -a -b 3 in.txt"])
        assert ctx.symbols.has_command("faketool")
        assert set(ctx.symbols.flags_for("faketool")) == {"-a", "-b", "-c", "--verbose"}

    async def test_correct_usage_of_a_learned_command_passes(self, ctx, faketool):
        result = await create(ctx, "ok", [f"{faketool} -a --verbose in.txt"])
        assert "NOT created" not in result
        assert "ok" in script_names(ctx)

    async def test_invented_flag_blocks_once_the_command_is_learned(
        self, ctx, faketool
    ):
        result = await create(ctx, "bad", [f"{faketool} -z in.txt"])
        assert "NOT created" in result
        assert "-z" in result
        assert "bad" not in script_names(ctx)

    async def test_unprobeable_command_warns_and_is_not_retried(self, ctx):
        result = await create(ctx, "warned", ["no-such-program-xyz -q in.txt"])
        assert "warned" in script_names(ctx)  # unverifiable is not a failure
        assert "not indexed" in result.lower()
        assert "no-such-program-xyz" in ctx.doc_probe_failed

    async def test_every_stage_of_a_pipeline_is_learned(self, ctx, faketool):
        await create(ctx, "piped", [f"cat in.txt | {faketool} -a | gzip -c > o.gz"])
        assert ctx.symbols.has_command("faketool")
        assert ctx.symbols.has_command("gzip")  # downstream stage, learned too

    async def test_bad_flag_in_a_downstream_stage_blocks(self, ctx, faketool):
        result = await create(ctx, "bad", [f"cat in.txt | {faketool} -z"])
        assert "NOT created" in result
        assert "bad" not in script_names(ctx)

    async def test_flagless_commands_are_not_probed(self, ctx):
        await create(ctx, "plain", ["no-such-program-xyz in.txt"])
        assert "no-such-program-xyz" not in ctx.doc_probe_failed


class TestProbeSafety:
    """The probe runs a program to read its --help, so what it refuses to run
    matters. Location is not one of the tests: requiring a bin/ directory
    refused real tools at /software/<tool>-<version>/<tool> while still
    running anything dropped in ~/bin, so it cost coverage and bought no
    guarantee."""

    def test_installed_software_is_probeable(self):
        assert safe_to_execute("sort")

    def test_a_tool_outside_bin_is_probeable(self, tmp_path):
        """The HPC layout the old bin/ rule silently refused."""
        tool = tmp_path / "software" / "minimap2-2.24" / "minimap2"
        tool.parent.mkdir(parents=True)
        tool.write_text("#!/bin/bash\necho hi\n")
        tool.chmod(0o755)
        assert safe_to_execute(str(tool))

    def test_destructive_commands_are_never_probed(self):
        assert not safe_to_execute("rm")
        assert not safe_to_execute("/bin/rm")

    def test_unknown_command_is_not_probeable(self):
        assert not safe_to_execute("no-such-program-xyz")

    def test_the_agents_own_scripts_are_never_executed(self, tmp_path):
        """A generated script has no --help to read, and running one before
        the §5.3 approval gate has seen it would invert that gate."""
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        generated = scripts / "pipeline.sh"
        generated.write_text("#!/bin/bash\necho hi\n")
        generated.chmod(0o755)
        assert not safe_to_execute(str(generated), scripts_dir=scripts)
        assert safe_to_execute(str(generated))  # the rule is the directory

    async def test_a_generated_script_is_not_run_by_the_gate(self, ctx, tmp_path):
        marker = tmp_path / "side-effect.txt"
        ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
        helper = ctx.scripts_dir / "helper.sh"
        helper.write_text(f"#!/bin/bash\ntouch {marker}\n")
        helper.chmod(0o755)

        result = await create(ctx, "wrapper", [f"{helper} in.bam -x 5"])

        assert not marker.exists(), "the gate executed a script the agent wrote"
        assert "wrapper" in script_names(ctx)  # unverifiable is not a failure
        assert "not indexed" in result.lower()


class TestCommandsNeedingDocs:
    def test_only_unindexed_commands_used_with_flags(self, index):
        pending = commands_needing_docs(
            "bash",
            "grep -v x f\nbwa mem -t 4 ref.fa\ncat plain.txt\n",
            index=index,
        )
        assert pending == [("bwa", "mem")]  # grep is indexed, cat has no flags

    def test_deduplicated_by_command(self, index):
        pending = commands_needing_docs(
            "bash", "bwa index ref.fa\nbwa mem -t 4 ref.fa\nbwa aln -n 2 r.fq\n",
            index=index,
        )
        assert [name for name, _ in pending] == ["bwa"]

    def test_non_bash_kinds_are_skipped(self, index):
        assert commands_needing_docs("python", "subprocess.run(['bwa'])", index=index) == []
