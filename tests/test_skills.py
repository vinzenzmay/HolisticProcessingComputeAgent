"""Tests for hpca.skills: user-defined procedure files (§5.1, milestone 12)."""

import pytest

from hpca.skills import (
    SHARED_SKILLS_DIR,
    Skill,
    load_skills,
    skills_dir,
    summarize_skills,
)


@pytest.fixture
def app_home(monkeypatch, tmp_path):
    """An empty app dir. The shipped skills stay visible."""
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def hpca_home(app_home, monkeypatch, tmp_path):
    """An empty app dir *and* no shipped skills, so the tests below can assert
    exact skill sets. The shipped ones are HPCA's own and always visible; they
    are exercised in :class:`TestBuiltinSkills`, which uses ``app_home``."""
    monkeypatch.setattr("hpca.skills.BUILTIN_SKILLS_DIR", tmp_path / "no-builtins")
    return app_home


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


class TestOwnSkills:
    """The profile's own skills — what /skill-remove may delete, as opposed to
    shared (_shared/) or legacy flat skills, which one profile must not remove."""

    def test_own_skills_are_only_the_profiles_own_dir(self, hpca_home):
        from hpca.skills import load_own_skills, write_skill as write

        write(Skill(name="mine", description="", triggers=[], body="b"), "default")
        # a shared skill and a legacy flat skill are visible but not "own"
        (skills_dir() / SHARED_SKILLS_DIR).mkdir(parents=True, exist_ok=True)
        (skills_dir() / SHARED_SKILLS_DIR / "shared.md").write_text(
            "---\nname: shared\n---\nb\n"
        )
        (skills_dir() / "legacy.md").write_text("---\nname: legacy\n---\nb\n")
        assert [s.name for s in load_own_skills("default")] == ["mine"]
        # all three are still visible to the profile
        assert {s.name for s in load_skills("default")} == {
            "mine",
            "shared",
            "legacy",
        }

    def test_delete_own_skill_removes_the_file(self, hpca_home):
        from hpca.skills import delete_own_skill, load_own_skills
        from hpca.skills import write_skill as write

        write(Skill(name="mine", description="d", triggers=[], body="b"), "default")
        skill = load_own_skills("default")[0]
        assert delete_own_skill(skill, "default") is True
        assert load_own_skills("default") == []

    def test_delete_own_skill_when_missing_returns_false(self, hpca_home):
        from hpca.skills import delete_own_skill

        ghost = Skill(name="ghost", description="", triggers=[], body="b", source="ghost.md")
        assert delete_own_skill(ghost, "default") is False


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
from hpca.runner import ProcessRunner  # noqa: E402


@pytest.fixture
def ctx(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield ToolContext(
        workdir=tmp_path,
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


class TestPerProfileSkills:
    """Redesign Phase 1: skills/_shared/ + skills/<profile>/, flat files
    counting as shared, profile winning name collisions."""

    def write_in(self, subdir, name, content):
        directory = skills_dir() / subdir if subdir else skills_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(content)

    def test_profile_sees_shared_and_own(self, hpca_home):
        self.write_in("_shared", "a.md", "---\nname: shared-a\n---\nbody")
        self.write_in("genetics", "b.md", "---\nname: gen-b\n---\nbody")
        self.write_in("hpc-admin", "c.md", "---\nname: admin-c\n---\nbody")
        names = [s.name for s in load_skills("genetics")]
        assert names == ["gen-b", "shared-a"]

    def test_flat_files_count_as_shared(self, hpca_home):
        self.write_in(None, "legacy.md", "---\nname: legacy\n---\nbody")
        assert [s.name for s in load_skills("genetics")] == ["legacy"]

    def test_profile_wins_name_collision(self, hpca_home):
        self.write_in("_shared", "a.md", "---\nname: align\n---\nshared body")
        self.write_in("genetics", "a.md", "---\nname: align\n---\nprofile body")
        skills = load_skills("genetics")
        assert len(skills) == 1
        assert skills[0].body == "profile body"

    def test_level_is_stamped_by_the_loader(self, hpca_home):
        self.write_in("_shared", "a.md", "---\nname: shared-a\n---\nbody")
        self.write_in("genetics", "b.md", "---\nname: gen-b\n---\nbody")
        levels = {s.name: s.level for s in load_skills("genetics")}
        assert levels == {"shared-a": "global", "gen-b": "profile"}


class TestSkillWriting:
    """Redesign Phase 4: the self-review loop writes skills back."""

    def test_write_and_reload_round_trip(self, hpca_home):
        from hpca.skills import write_skill

        write_skill(
            Skill(
                name="read-qc",
                description="Run fastqc then multiqc",
                triggers=["fastqc", "qc"],
                body="1. fastqc\n2. multiqc",
            ),
            "genetics",
        )
        loaded = load_skills("genetics")
        assert len(loaded) == 1
        assert loaded[0].name == "read-qc"
        assert loaded[0].description == "Run fastqc then multiqc"
        assert loaded[0].triggers == ["fastqc", "qc"]
        assert "multiqc" in loaded[0].body

    def test_write_goes_under_the_profile(self, hpca_home):
        from hpca.skills import skills_dir, write_skill

        write_skill(Skill("a", "d", [], "body"), "genetics")
        assert (skills_dir() / "genetics" / "a.md").exists()

    def test_unsafe_names_are_sanitized(self, hpca_home):
        from hpca.skills import skill_path

        path = skill_path("../../etc/passwd", "genetics")
        assert path.name == "etc-passwd.md"  # no separators, no leading dots
        assert path.parent.name == "genetics"

    def test_empty_name_falls_back(self, hpca_home):
        from hpca.skills import skill_path

        assert skill_path("...", "genetics").name == "skill.md"

    def test_patched_body_appends_under_a_heading(self):
        from hpca.skills import patched_body

        skill = Skill("a", "d", [], "1. do the thing")
        patched = patched_body(skill, "check the index first")
        assert "1. do the thing" in patched  # original steps survive
        assert "## Corrections" in patched
        assert "- check the index first" in patched

    def test_second_patch_reuses_the_heading(self):
        from hpca.skills import patched_body

        skill = Skill("a", "d", [], "1. do the thing")
        once = patched_body(skill, "first correction")
        skill.body = once
        twice = patched_body(skill, "second correction")
        assert twice.count("## Corrections") == 1
        assert "- first correction" in twice and "- second correction" in twice


class TestSkillLevels:
    """Three explicit levels: global (``_shared/``), profile, and project
    (``<cwd>/.hpca/skills``). Precedence on a collision: project > profile >
    global."""

    def test_project_skills_dir_convention(self, tmp_path):
        from hpca.skills import project_skills_dir

        assert project_skills_dir(tmp_path) == tmp_path / ".hpca" / "skills"

    def test_write_lands_at_the_chosen_level(self, hpca_home, tmp_path):
        from hpca.skills import (
            SHARED_SKILLS_DIR,
            project_skills_dir,
            skills_dir,
            write_skill,
        )

        write_skill(Skill("g", "d", [], "b"), "genetics", level="global")
        write_skill(Skill("p", "d", [], "b"), "genetics", level="profile")
        write_skill(
            Skill("j", "d", [], "b"),
            "genetics",
            level="project",
            project_root=tmp_path,
        )
        assert (skills_dir() / SHARED_SKILLS_DIR / "g.md").exists()
        assert (skills_dir() / "genetics" / "p.md").exists()
        assert (project_skills_dir(tmp_path) / "j.md").exists()

    def test_default_level_is_profile(self, hpca_home):
        from hpca.skills import skills_dir, write_skill

        write_skill(Skill("p", "d", [], "b"), "genetics")
        assert (skills_dir() / "genetics" / "p.md").exists()

    def test_project_skills_visible_with_project_root(self, hpca_home, tmp_path):
        from hpca.skills import write_skill

        write_skill(
            Skill("proj", "d", [], "b"),
            "genetics",
            level="project",
            project_root=tmp_path,
        )
        names = [s.name for s in load_skills("genetics", project_root=tmp_path)]
        assert names == ["proj"]

    def test_project_hidden_from_other_directories(self, hpca_home, tmp_path):
        from hpca.skills import write_skill

        write_skill(
            Skill("proj", "d", [], "b"),
            "genetics",
            level="project",
            project_root=tmp_path,
        )
        elsewhere = tmp_path / "elsewhere"
        assert load_skills("genetics", project_root=elsewhere) == []

    def test_project_wins_profile_wins_global(self, hpca_home, tmp_path):
        from hpca.skills import write_skill

        write_skill(Skill("align", "d", [], "global body"), "genetics", level="global")
        write_skill(Skill("align", "d", [], "profile body"), "genetics", level="profile")
        write_skill(
            Skill("align", "d", [], "project body"),
            "genetics",
            level="project",
            project_root=tmp_path,
        )
        skills = load_skills("genetics", project_root=tmp_path)
        assert len(skills) == 1
        assert skills[0].body == "project body"

    def test_profile_wins_over_global(self, hpca_home, tmp_path):
        from hpca.skills import write_skill

        write_skill(Skill("align", "d", [], "global body"), "genetics", level="global")
        write_skill(Skill("align", "d", [], "profile body"), "genetics", level="profile")
        skills = load_skills("genetics", project_root=tmp_path)
        assert skills[0].body == "profile body"

    def test_global_is_shared_across_profiles(self, hpca_home, tmp_path):
        from hpca.skills import write_skill

        write_skill(Skill("common", "d", [], "b"), "genetics", level="global")
        names = [s.name for s in load_skills("hpc-admin", project_root=tmp_path)]
        assert names == ["common"]

    def test_load_project_skills_lists_only_project(self, hpca_home, tmp_path):
        from hpca.skills import load_project_skills, write_skill

        write_skill(Skill("mine", "d", [], "b"), "genetics")  # profile level
        write_skill(
            Skill("proj", "d", [], "b"),
            "genetics",
            level="project",
            project_root=tmp_path,
        )
        assert [s.name for s in load_project_skills(project_root=tmp_path)] == ["proj"]

    def test_delete_removes_a_project_skill(self, hpca_home, tmp_path):
        from hpca.skills import delete_own_skill, load_project_skills, write_skill

        write_skill(
            Skill("proj", "d", [], "b"),
            "genetics",
            level="project",
            project_root=tmp_path,
        )
        skill = load_project_skills(project_root=tmp_path)[0]
        assert delete_own_skill(skill, "genetics", project_root=tmp_path) is True
        assert load_project_skills(project_root=tmp_path) == []

    def test_project_skill_path_is_sanitized_and_hidden(self, hpca_home, tmp_path):
        from hpca.skills import project_skills_dir, skill_path

        path = skill_path(
            "../../etc/passwd", "genetics", level="project", project_root=tmp_path
        )
        assert path.name == "etc-passwd.md"
        assert path.parent == project_skills_dir(tmp_path)


class TestBuiltinSkills:
    """Skills shipped with HPCA (``hpca/data/skills``): visible to every profile
    on a fresh install, ranked below every user level, never removable."""

    def test_shipped_skills_are_visible_on_a_fresh_install(self, app_home):
        names = [s.name for s in load_skills("genetics")]
        assert "grillme" in names
        assert "plan" in names

    def test_every_shipped_skill_parses(self, app_home):
        from hpca.skills import load_builtin_skills

        shipped = load_builtin_skills()
        assert shipped, "HPCA ships no skills — the package data is missing"
        for skill in shipped:
            assert not skill.problems
            assert skill.description and skill.body
            assert skill.level == "builtin"

    def test_plan_grills_first_then_writes_specs(self, app_home):
        """The point of /plan: grill to a shared understanding, then leave a
        specs.md a fresh session can implement from."""
        plan = next(s for s in load_skills("genetics") if s.name == "plan")
        assert "grillme" in plan.body
        assert "specs.md" in plan.body

    def test_a_user_skill_shadows_a_shipped_one(self, app_home):
        from hpca.skills import write_skill

        write_skill(Skill("plan", "mine", [], "my own planning procedure"), "genetics")
        plan = next(s for s in load_skills("genetics") if s.name == "plan")
        assert plan.body == "my own planning procedure"
        assert plan.level == "profile"

    def test_shipped_skills_are_not_removable(self, app_home):
        """They live outside the profile and project dirs, so /skill-remove
        never offers them and deleting one cannot reach the package."""
        from hpca.skills import delete_own_skill, load_builtin_skills

        shipped = load_builtin_skills()[0]
        assert delete_own_skill(shipped, "genetics") is False
        assert shipped.name in [s.name for s in load_builtin_skills()]


class TestProfileSkillLifecycle:
    """Skills follow their profile when it is copied or deleted."""

    def write_in(self, subdir, name, content):
        directory = skills_dir() / subdir if subdir else skills_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(content)

    def test_copy_takes_the_profiles_own_skills(self, hpca_home):
        from hpca.skills import copy_profile_skills

        self.write_in("base", "a.md", "---\nname: align\n---\nbase body")
        assert copy_profile_skills("base", "variants") == 1
        assert [s.name for s in load_skills("variants")] == ["align"]

    def test_shared_skills_are_not_duplicated(self, hpca_home):
        """_shared is already visible to both; copying it would turn one
        procedure into two that drift apart silently."""
        from hpca.skills import copy_profile_skills

        self.write_in("_shared", "s.md", "---\nname: shared\n---\nbody")
        self.write_in("base", "a.md", "---\nname: align\n---\nbody")
        assert copy_profile_skills("base", "variants") == 1
        assert not (skills_dir() / "variants" / "s.md").exists()
        # but the copy still sees the shared one through _shared
        assert sorted(s.name for s in load_skills("variants")) == ["align", "shared"]

    def test_copied_skills_diverge(self, hpca_home):
        from hpca.skills import copy_profile_skills, write_skill

        self.write_in("base", "a.md", "---\nname: align\n---\nbase body")
        copy_profile_skills("base", "variants")
        write_skill(Skill("align", "d", [], "changed in the copy"), "variants")
        assert "base body" in load_skills("base")[0].body
        assert "changed in the copy" in load_skills("variants")[0].body

    def test_copying_a_profile_without_skills(self, hpca_home):
        from hpca.skills import copy_profile_skills

        assert copy_profile_skills("base", "variants") == 0

    def test_delete_removes_only_that_profiles_skills(self, hpca_home):
        from hpca.skills import delete_profile_skills

        self.write_in("_shared", "s.md", "---\nname: shared\n---\nbody")
        self.write_in("base", "a.md", "---\nname: align\n---\nbody")
        delete_profile_skills("base")
        assert [s.name for s in load_skills("base")] == ["shared"]
        assert not (skills_dir() / "base").exists()

    def test_deleting_skills_of_an_unknown_profile_is_quiet(self, hpca_home):
        from hpca.skills import delete_profile_skills

        delete_profile_skills("never-existed")  # must not raise
