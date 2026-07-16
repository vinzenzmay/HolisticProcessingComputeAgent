"""Tests for hpca.verify_code: the semantic code-vs-docs gate (§5.2)."""

import pytest

from hpca.db import connect, init_db
from hpca.symbols import Symbol, SymbolIndex
from hpca.verify_code import (
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


class TestVerifyOtherKinds:
    def test_r_returns_empty_v1(self, index):
        assert verify_script("R", "library(dplyr)\n", index=index) == []


# ------------------------------------------------- create_script integration

from hpca.agent.builtin_tools import default_tool_registry  # noqa: E402
from hpca.agent.context import ToolContext  # noqa: E402
from hpca.config import Settings  # noqa: E402
from hpca.registry import PathRegistry  # noqa: E402
from hpca.runner import ProcessRunner  # noqa: E402


@pytest.fixture
def ctx(tmp_path, index):
    conn = connect(tmp_path / "ctx.db")
    init_db(conn)
    yield ToolContext(
        registry=PathRegistry(conn, profile="default", session_id="s1"),
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
        {"kind": "bash", "registry_key": key, "content_lines": lines}
    )
    return await tool.handler(args, ctx)


class TestCreateScriptGate:
    async def test_invented_flag_blocks_creation(self, ctx):
        result = await create(ctx, "bad", ["samtools view -e in.bam"])
        assert "NOT created" in result
        assert "-e" in result
        assert "bad" not in ctx.registry.list()

    async def test_valid_flags_pass(self, ctx):
        result = await create(ctx, "good", ["samtools view -b -q 20 in.bam"])
        assert "ok" in result.lower()
        assert "good" in ctx.registry.list()

    async def test_unindexed_command_warns_but_creates(self, ctx):
        result = await create(ctx, "warned", ["bwa mem -t 4 ref.fa reads.fq"])
        assert "warned" in ctx.registry.list()
        assert "not indexed" in result.lower()

    async def test_no_index_no_gate(self, ctx):
        ctx.symbols = None
        result = await create(ctx, "ungated", ["samtools view -e in.bam"])
        assert "ungated" in ctx.registry.list()
