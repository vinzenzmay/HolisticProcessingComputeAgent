"""Tests for hpca.skills: user-defined procedure files (§5.1, milestone 12)."""

import pytest

from hpca.skills import Skill, load_skills, skills_dir, summarize_skills


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def write_skill(name: str, content: str) -> None:
    skills_dir().mkdir(parents=True, exist_ok=True)
    (skills_dir() / name).write_text(content)


SKILL_MD = """\
---
name: bam-subset
description: Subset a BAM file by genomic region
triggers: [bam, subset, region]
---

1. Ensure the BAM is indexed (samtools index).
2. Run samtools view with the region argument.
3. Verify the output is non-empty.
"""


class TestLoadSkills:
    def test_missing_dir_is_empty(self, hpca_home):
        assert load_skills() == []

    def test_loads_markdown_skill(self, hpca_home):
        write_skill("bam.md", SKILL_MD)
        skills = load_skills()
        assert len(skills) == 1
        skill = skills[0]
        assert skill.name == "bam-subset"
        assert skill.description == "Subset a BAM file by genomic region"
        assert skill.triggers == ["bam", "subset", "region"]
        assert "samtools index" in skill.body

    def test_loads_yaml_skill(self, hpca_home):
        write_skill(
            "qc.yaml",
            "name: fastq-qc\n"
            "description: Run FastQC on reads\n"
            "triggers: [fastq, qc]\n"
            "body: |\n"
            "  1. Run fastqc on each file.\n"
            "  2. Collect reports with multiqc.\n",
        )
        skill = load_skills()[0]
        assert skill.name == "fastq-qc"
        assert "multiqc" in skill.body

    def test_name_defaults_to_filename(self, hpca_home):
        write_skill("my-procedure.md", "Just a body, no front matter.\n")
        skill = load_skills()[0]
        assert skill.name == "my-procedure"
        assert "Just a body" in skill.body

    def test_sorted_by_name(self, hpca_home):
        write_skill("b.md", "---\nname: bravo\n---\nbody\n")
        write_skill("a.md", "---\nname: alpha\n---\nbody\n")
        assert [s.name for s in load_skills()] == ["alpha", "bravo"]

    def test_malformed_front_matter_still_loads_body(self, hpca_home):
        write_skill("broken.md", "---\nname: [unclosed\n---\nthe body\n")
        skills = load_skills()
        assert len(skills) == 1
        assert "the body" in skills[0].body
        assert skills[0].problems

    def test_empty_file_skipped(self, hpca_home):
        write_skill("empty.md", "")
        assert load_skills() == []

    def test_non_skill_files_ignored(self, hpca_home):
        write_skill("notes.txt", "not a skill")
        write_skill("real.md", "---\nname: real\n---\nbody\n")
        assert [s.name for s in load_skills()] == ["real"]

    def test_string_triggers_normalized(self, hpca_home):
        write_skill("s.md", "---\nname: s\ntriggers: bam\n---\nbody\n")
        assert load_skills()[0].triggers == ["bam"]


class TestMatching:
    def test_trigger_match_case_insensitive(self):
        skill = Skill(name="s", description="", triggers=["BAM"], body="x")
        assert skill.matches("subset my bam file")
        assert not skill.matches("run a fastqc report")

    def test_name_matches(self):
        skill = Skill(name="bam-subset", description="", triggers=[], body="x")
        assert skill.matches("do a bam-subset please")

    def test_word_boundary_not_substring(self):
        skill = Skill(name="s", description="", triggers=["qc"], body="x")
        assert skill.matches("run qc now")
        assert not skill.matches("this is a qcircuit thing")


class TestSummarize:
    def test_lists_names_and_descriptions(self):
        skills = [
            Skill(name="a", description="Do A", triggers=[], body="x"),
            Skill(name="b", description="Do B", triggers=[], body="y"),
        ]
        text = summarize_skills(skills)
        assert "a: Do A" in text
        assert "b: Do B" in text

    def test_empty(self):
        assert summarize_skills([]) == ""


# ------------------------------------------------------------- skill tool

from hpca.agent.context import ToolContext  # noqa: E402
from hpca.agent.skill_tools import add_skill_tools  # noqa: E402
from hpca.agent.tools import ToolRegistry  # noqa: E402
from hpca.config import Settings  # noqa: E402
from hpca.db import connect, init_db  # noqa: E402
from hpca.registry import PathRegistry  # noqa: E402
from hpca.runner import ProcessRunner  # noqa: E402


@pytest.fixture
def ctx(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield ToolContext(
        registry=PathRegistry(conn, profile="default", session_id="s1"),
        runner=ProcessRunner(conn, session_id="s1", log_dir=tmp_path / "logs"),
        settings=Settings(),
        scripts_dir=tmp_path / "scripts",
        skills=[
            Skill(
                name="bam-subset",
                description="Subset a BAM",
                triggers=["bam"],
                body="1. index\n2. view",
            )
        ],
    )
    conn.close()


async def call(tools, tool_name, ctx, **kwargs):
    tool = tools.get(tool_name)
    return await tool.handler(tool.params.model_validate(kwargs), ctx)


class TestReadSkillTool:
    async def test_returns_body(self, ctx):
        tools = add_skill_tools(ToolRegistry())
        result = await call(tools, "read_skill", ctx, name="bam-subset")
        assert "1. index" in result

    async def test_unknown_skill_lists_available(self, ctx):
        tools = add_skill_tools(ToolRegistry())
        result = await call(tools, "read_skill", ctx, name="nope")
        assert "bam-subset" in result

    async def test_no_skills_configured(self, ctx):
        ctx.skills = []
        tools = add_skill_tools(ToolRegistry())
        result = await call(tools, "read_skill", ctx, name="x")
        assert "no skills" in result.lower()
