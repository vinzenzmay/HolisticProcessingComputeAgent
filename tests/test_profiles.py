"""Tests for hpca.profiles: two-tier markdown memory store (§6)."""

import pytest

from hpca.profiles import Memory, Profile, estimate_tokens, profiles_dir


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
            tier=1, backend="qwen3-6b", kind="fact",
        )
        profile.add_memory(
            "STAR alignments need at least 40G of memory here.",
            tier=2, backend="qwen3-6b", kind="learning",
        )
        profile.save()
        loaded = Profile.load("default")
        assert len(loaded.memories) == 2
        tier1 = [m for m in loaded.memories if m.tier == 1]
        assert tier1[0].text == "Cluster is called cubi, scheduler is Slurm 25.05."
        assert tier1[0].backend == "qwen3-6b"
        assert tier1[0].kind == "fact"
        tier2 = [m for m in loaded.memories if m.tier == 2]
        assert "STAR" in tier2[0].text

    def test_front_matter_preserved(self, hpca_home):
        profile = Profile.load("default")
        profile.default_backend = "qwen3-6b"
        profile.save()
        loaded = Profile.load("default")
        assert loaded.default_backend == "qwen3-6b"
        assert loaded.created  # stamped on creation

    def test_multiline_memory_text(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("line one\nline two\nline three", tier=2)
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
            "---\nname: edited\n---\n\n## [tier1]\n\n"
            "The site firewall blocks outgoing traffic.\n",
        )
        assert len(profile.memories) == 1
        assert profile.memories[0].tier == 1
        assert profile.memories[0].backend == ""
        assert "firewall" in profile.memories[0].text

    def test_moving_block_between_headings_changes_tier(self, hpca_home):
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## [tier1]\n\n## [tier2]\n\n"
            "<!-- backend: q, created: 2026-01-01, kind: fact -->\nmoved here\n",
        )
        assert profile.memories[0].tier == 2

    def test_missing_front_matter_reports_problem_but_parses(self, hpca_home):
        profile = self.write(
            hpca_home, "## [tier2]\n\nremember this\n"
        )
        assert profile.problems  # reported
        assert profile.memories[0].text == "remember this"

    def test_unknown_heading_reports_problem(self, hpca_home):
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## random notes\n\nnot a memory\n\n"
            "## [tier2]\n\na real memory\n",
        )
        assert any("random notes" in p for p in profile.problems)
        assert [m.text for m in profile.memories] == ["a real memory"]

    def test_tier3_accepted_in_format(self, hpca_home):
        # deferred tier (§6.1): parsed, stored, not injected anywhere yet
        profile = self.write(
            hpca_home, "---\nname: edited\n---\n\n## [tier3]\n\nfuture memory\n"
        )
        assert profile.memories[0].tier == 3

    def test_empty_file(self, hpca_home):
        profile = self.write(hpca_home, "")
        assert profile.memories == []

    def test_malformed_metadata_comment_kept_as_text_problem(self, hpca_home):
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## [tier2]\n\n"
            "<!-- backend qwen no colons here -->\nsome memory text\n",
        )
        assert profile.memories[0].text == "some memory text"


class TestTierAccess:
    def test_tier_text_joins_memories(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("fact one", tier=1)
        profile.add_memory("fact two", tier=1)
        profile.add_memory("tier2 thing", tier=2)
        text = profile.tier_text(1)
        assert "fact one" in text and "fact two" in text
        assert "tier2 thing" not in text

    def test_tier_tokens_positive_and_scales(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("word " * 400, tier=2)
        assert profile.tier_tokens(2) > 100
        assert profile.tier_tokens(1) == 0

    def test_over_cap_tiers(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x" * 4000, tier=1)  # ~1000 tokens
        assert profile.over_cap_tiers(tier1_cap=300, tier2_cap=800) == [1]


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


class TestCharBudgets:
    """Redesign Phase 1: hard, model-independent character budgets."""

    def test_tier_chars_counts_injected_text(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("abcd", tier=1)
        profile.add_memory("efgh", tier=1)
        assert profile.tier_chars(1) == len("abcd\n\nefgh")

    def test_usage_meter_format(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x" * 600, tier=2)
        assert profile.usage_meter(2, 1200) == "50% — 600/1200 chars"

    def test_usage_meter_zero_cap_does_not_divide(self, hpca_home):
        assert Profile.load("default").usage_meter(1, 0) == "0% — 0/0 chars"

    def test_would_exceed_counts_the_joiner(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x" * 10, tier=1)
        # 10 used + 2 joiner + 5 new = 17
        assert not profile.would_exceed(1, "y" * 5, cap=17)
        assert profile.would_exceed(1, "y" * 5, cap=16)

    def test_would_exceed_empty_tier_has_no_joiner(self, hpca_home):
        profile = Profile.load("default")
        assert not profile.would_exceed(1, "y" * 5, cap=5)

    def test_over_cap_tiers_uses_chars(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x" * 100, tier=2)
        assert profile.over_cap_tiers(tier1_cap=50, tier2_cap=99) == [2]
        assert profile.over_cap_tiers(tier1_cap=50, tier2_cap=100) == []


class TestBackendAnnotation:
    """Memories from another backend are annotated at injection, not dropped —
    a workaround for one small model often transfers."""

    def test_other_backend_annotated(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("Use --no-mmap here.", tier=2, backend="qwen3-6b")
        text = profile.tier_prompt_text(2, active_backend="gemma3-27b")
        assert text == "(learned on qwen3-6b) Use --no-mmap here."

    def test_same_backend_unannotated(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("Use --no-mmap here.", tier=2, backend="qwen3-6b")
        text = profile.tier_prompt_text(2, active_backend="qwen3-6b")
        assert text == "Use --no-mmap here."

    def test_untagged_memory_unannotated(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("Cluster is cubi.", tier=1)
        assert (
            profile.tier_prompt_text(1, active_backend="qwen3-6b")
            == "Cluster is cubi."
        )


class TestDuplication:
    """A copy starts from everything the original learned, then diverges —
    the workflow is a general base profile forked per specialism."""

    def base(self, name="base"):
        profile = Profile.create(name)
        profile.default_backend = "qwen3-35b"
        profile.add_memory("Cluster is cubi, scheduler is Slurm.", tier=1)
        profile.add_memory("The user prefers R.", tier=2)
        profile.add_memory("Snakemake dry-runs fail here.", tier=3, kind="struggle")
        profile.save()
        return profile

    def test_all_tiers_copied(self, hpca_home):
        self.base()
        copy = Profile.duplicate("base", "variants")
        assert [(m.tier, m.text) for m in copy.memories] == [
            (1, "Cluster is cubi, scheduler is Slurm."),
            (2, "The user prefers R."),
            (3, "Snakemake dry-runs fail here."),
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
        original.add_memory("Learned later by the base.", tier=2)
        original.save()

        copy = Profile.load("variants")
        copy.add_memory("Learned later by the copy.", tier=2)
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
