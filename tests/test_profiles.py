"""Tests for hpca.profiles: two-scope markdown memory store (§6)."""

import pytest

from hpca.profiles import (
    MemoryScope,
    Profile,
    estimate_tokens,
    profiles_dir,
)

SP = MemoryScope.SYSTEM_PROMPT
RAG = MemoryScope.RAG


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


class TestNewProfile:
    def test_load_missing_creates_empty(self, hpca_home):
        profile = Profile.load("default")
        assert profile.name == "default"
        assert profile.memories == []
        assert profile.problems == []

    def test_save_creates_file(self, hpca_home):
        Profile.load("default").save()
        assert (profiles_dir() / "default.md").exists()

    def test_list_profiles(self, hpca_home):
        Profile.load("alpha").save()
        Profile.load("beta").save()
        # the default is always present and always listed first: it is the
        # fallback for sessions whose profile is deleted
        assert Profile.list_profiles() == ["default", "alpha", "beta"]

    def test_default_is_listed_even_when_no_file_exists(self, hpca_home):
        assert Profile.list_profiles() == ["default"]


class TestRoundTrip:
    def test_memories_survive_save_load(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory(
            "Cluster is called cubi, scheduler is Slurm 25.05.",
            scope=SP, backend="qwen3-6b", kind="fact",
        )
        profile.add_memory(
            "STAR alignments need at least 40G of memory here.",
            scope=SP, backend="qwen3-6b", kind="learning",
        )
        profile.save()
        loaded = Profile.load("default")
        assert len(loaded.memories) == 2
        assert all(m.scope is SP for m in loaded.memories)
        facts = [m for m in loaded.memories if m.kind == "fact"]
        assert facts[0].text == "Cluster is called cubi, scheduler is Slurm 25.05."
        assert facts[0].backend == "qwen3-6b"
        learnings = [m for m in loaded.memories if m.kind == "learning"]
        assert "STAR" in learnings[0].text

    def test_front_matter_preserved(self, hpca_home):
        profile = Profile.load("default")
        profile.default_backend = "qwen3-6b"
        profile.save()
        loaded = Profile.load("default")
        assert loaded.default_backend == "qwen3-6b"
        assert loaded.created  # stamped on creation

    def test_multiline_memory_text(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("line one\nline two\nline three", scope=SP)
        profile.save()
        loaded = Profile.load("default")
        assert loaded.memories[0].text == "line one\nline two\nline three"


class TestLenientParsing:
    def write(self, hpca_home, content):
        profiles_dir().mkdir(parents=True, exist_ok=True)
        (profiles_dir() / "edited.md").write_text(content)
        return Profile.load("edited")

    def test_hand_written_block_without_metadata(self, hpca_home):
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## [system-prompt]\n\n"
            "The site firewall blocks outgoing traffic.\n",
        )
        assert len(profile.memories) == 1
        assert profile.memories[0].scope is SP
        assert profile.memories[0].backend == ""
        assert "firewall" in profile.memories[0].text

    def test_moving_block_between_headings_changes_scope(self, hpca_home):
        # a block filed under the rag heading is retrieved-only, not injected
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## [system-prompt]\n\n## [rag]\n\n"
            "<!-- backend: q, created: 2026-01-01, kind: fact -->\nmoved here\n",
        )
        assert profile.memories[0].scope is RAG

    def test_missing_front_matter_reports_problem_but_parses(self, hpca_home):
        profile = self.write(
            hpca_home, "## [system-prompt]\n\nremember this\n"
        )
        assert profile.problems  # reported
        assert profile.memories[0].text == "remember this"

    def test_unknown_heading_reports_problem(self, hpca_home):
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## random notes\n\nnot a memory\n\n"
            "## [system-prompt]\n\na real memory\n",
        )
        assert any("random notes" in p for p in profile.problems)
        assert [m.text for m in profile.memories] == ["a real memory"]

    def test_rag_scope_accepted_in_format(self, hpca_home):
        # the retrieved scope: parsed, stored, indexed rather than injected
        profile = self.write(
            hpca_home, "---\nname: edited\n---\n\n## [rag]\n\nfuture memory\n"
        )
        assert profile.memories[0].scope is RAG

    def test_empty_file(self, hpca_home):
        profile = self.write(hpca_home, "")
        assert profile.memories == []

    def test_malformed_metadata_comment_kept_as_text_problem(self, hpca_home):
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## [system-prompt]\n\n"
            "<!-- backend qwen no colons here -->\nsome memory text\n",
        )
        assert profile.memories[0].text == "some memory text"


class TestScopeAccess:
    def test_scope_text_joins_memories(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("fact one", scope=SP)
        profile.add_memory("fact two", scope=SP)
        profile.add_memory("rag thing", scope=RAG)
        text = profile.scope_text(SP)
        assert "fact one" in text and "fact two" in text
        assert "rag thing" not in text

    def test_system_prompt_tokens_positive_and_scales(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("word " * 400, scope=SP)
        # rag memories do not count against the injected budget
        profile.add_memory("word " * 400, scope=RAG)
        assert profile.system_prompt_tokens() > 100
        assert Profile.load("default").system_prompt_tokens() == 0

    def test_over_budget(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x" * 4000, scope=SP)  # well over any small cap
        used = profile.system_prompt_tokens()
        assert profile.over_budget(used - 1) is True
        assert profile.over_budget(used) is False


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_reasonable_scale(self):
        text = "the quick brown fox jumps over the lazy dog " * 50
        tokens = estimate_tokens(text)
        assert 200 < tokens < 1200


class TestProfileManagement:
    def test_validate_name_accepts_reasonable_names(self, hpca_home):
        assert Profile.validate_name("  bam work ") == "bam work"
        assert Profile.validate_name("proj-1.2_v3") == "proj-1.2_v3"

    def test_validate_name_rejects_empty(self, hpca_home):
        with pytest.raises(ValueError):
            Profile.validate_name("   ")

    def test_validate_name_rejects_path_characters(self, hpca_home):
        for bad in ("../escape", "a/b", "with\ttab", ".hidden"):
            with pytest.raises(ValueError):
                Profile.validate_name(bad)

    def test_validate_name_rejects_duplicates(self, hpca_home):
        Profile.create("alpha")
        with pytest.raises(ValueError):
            Profile.validate_name("alpha")
        with pytest.raises(ValueError):
            Profile.validate_name("default")  # always exists

    def test_create_writes_an_empty_profile(self, hpca_home):
        profile = Profile.create("alpha")
        assert profile.name == "alpha"
        assert Profile.path_for("alpha").exists()
        assert Profile.load("alpha").memories == []

    def test_delete_removes_the_file(self, hpca_home):
        Profile.create("alpha")
        Profile.delete("alpha")
        assert not Profile.path_for("alpha").exists()
        assert "alpha" not in Profile.list_profiles()

    def test_default_cannot_be_deleted(self, hpca_home):
        Profile.load("default").save()
        with pytest.raises(ValueError):
            Profile.delete("default")
        assert "default" in Profile.list_profiles()

    def test_deleting_a_missing_profile_is_quiet(self, hpca_home):
        Profile.delete("never-existed")  # must not raise


class TestTokenBudgets:
    """Redesign §6.4: a hard, write-time token budget on the injected scope."""

    def test_system_prompt_tokens_counts_injected_text(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("abcd", scope=SP)
        profile.add_memory("efgh", scope=SP)
        assert profile.system_prompt_tokens() == estimate_tokens("abcd\n\nefgh")

    def test_usage_meter_format(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("word " * 100, scope=SP)
        used = profile.system_prompt_tokens()
        assert profile.usage_meter(used * 2) == f"50% — {used}/{used * 2} tokens"

    def test_usage_meter_zero_cap_does_not_divide(self, hpca_home):
        assert Profile.load("default").usage_meter(0) == "0% — 0/0 tokens"

    def test_would_exceed_counts_the_joiner(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("alpha", scope=SP)
        # the "\n\n" joiner between the existing and the new text is budgeted
        combined = estimate_tokens("alpha\n\nbeta")
        assert not profile.would_exceed("beta", cap=combined)
        assert profile.would_exceed("beta", cap=combined - 1)

    def test_would_exceed_empty_scope_has_no_joiner(self, hpca_home):
        profile = Profile.load("default")
        assert not profile.would_exceed("beta", cap=estimate_tokens("beta"))

    def test_over_budget_uses_tokens(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("word " * 100, scope=SP)
        used = profile.system_prompt_tokens()
        assert profile.over_budget(used - 1) is True
        assert profile.over_budget(used) is False


class TestBackendAnnotation:
    """Memories from another backend are annotated at injection, not dropped —
    a workaround for one small model often transfers."""

    def test_other_backend_annotated(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("Use --no-mmap here.", scope=SP, backend="qwen3-6b")
        text = profile.system_prompt_text(active_backend="gemma3-27b")
        assert text == "(learned on qwen3-6b) Use --no-mmap here."

    def test_same_backend_unannotated(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("Use --no-mmap here.", scope=SP, backend="qwen3-6b")
        text = profile.system_prompt_text(active_backend="qwen3-6b")
        assert text == "Use --no-mmap here."

    def test_untagged_memory_unannotated(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("Cluster is cubi.", scope=SP)
        assert (
            profile.system_prompt_text(active_backend="qwen3-6b")
            == "Cluster is cubi."
        )


class TestDuplication:
    """A copy starts from everything the original learned, then diverges —
    the workflow is a general base profile forked per specialism."""

    def base(self, name="base"):
        profile = Profile.create(name)
        profile.default_backend = "qwen3-35b"
        profile.add_memory("Cluster is cubi, scheduler is Slurm.", scope=SP)
        profile.add_memory("The user prefers R.", scope=SP)
        profile.add_memory(
            "Snakemake dry-runs fail here.", scope=RAG, kind="struggle"
        )
        profile.save()
        return profile

    def test_all_scopes_copied(self, hpca_home):
        self.base()
        copy = Profile.duplicate("base", "variants")
        assert [(m.scope, m.text) for m in copy.memories] == [
            (SP, "Cluster is cubi, scheduler is Slurm."),
            (SP, "The user prefers R."),
            (RAG, "Snakemake dry-runs fail here."),
        ]

    def test_metadata_preserved_and_provenance_recorded(self, hpca_home):
        self.base()
        copy = Profile.duplicate("base", "variants")
        assert copy.default_backend == "qwen3-35b"
        assert copy.copied_from == "base"
        assert copy.copied_on
        # struggle notes stay struggle notes, so matching still works
        assert copy.memories[2].kind == "struggle"

    def test_provenance_survives_a_round_trip(self, hpca_home):
        self.base()
        Profile.duplicate("base", "variants")
        assert Profile.load("variants").copied_from == "base"

    def test_ordinary_profiles_carry_no_provenance(self, hpca_home):
        Profile.create("plain").save()
        assert Profile.load("plain").copied_from == ""
        assert "copied_from" not in Profile.path_for("plain").read_text()

    def test_the_copy_is_written_to_disk(self, hpca_home):
        self.base()
        Profile.duplicate("base", "variants")
        assert "variants" in Profile.list_profiles()

    def test_they_diverge(self, hpca_home):
        self.base()
        Profile.duplicate("base", "variants")

        original = Profile.load("base")
        original.add_memory("Learned later by the base.", scope=SP)
        original.save()

        copy = Profile.load("variants")
        copy.add_memory("Learned later by the copy.", scope=SP)
        copy.save()

        base_texts = [m.text for m in Profile.load("base").memories]
        copy_texts = [m.text for m in Profile.load("variants").memories]
        assert "Learned later by the base." in base_texts
        assert "Learned later by the base." not in copy_texts
        assert "Learned later by the copy." in copy_texts
        assert "Learned later by the copy." not in base_texts
        # and the shared base is still in both
        assert "The user prefers R." in base_texts
        assert "The user prefers R." in copy_texts

    def test_editing_a_copied_memory_does_not_touch_the_original(self, hpca_home):
        self.base()
        copy = Profile.duplicate("base", "variants")
        copy.memories[1].text = "The user prefers Python now."
        copy.save()
        assert Profile.load("base").memories[1].text == "The user prefers R."

    def test_copying_an_empty_profile(self, hpca_home):
        Profile.create("blank")
        copy = Profile.duplicate("blank", "also-blank")
        assert copy.memories == []
        assert copy.copied_from == "blank"

    def test_copy_of_a_copy_records_its_immediate_source(self, hpca_home):
        self.base()
        Profile.duplicate("base", "variants")
        grandchild = Profile.duplicate("variants", "variants-wgs")
        assert grandchild.copied_from == "variants"
